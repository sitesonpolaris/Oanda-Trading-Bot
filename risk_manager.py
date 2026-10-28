"""Risk manager: every trade signal passes through here before an order can be placed.

`RiskManager.evaluate(signal, account)` returns a Decision: either an approved
Order with a position size, or a rejection with a reason. It never talks to the
broker; the caller supplies the current account state.

Checks, in order (first failure wins):
    halted                     bot was halted earlier and hasn't been reset
    daily_loss_limit           today's loss reached the halt line -> halts the bot
    max_drawdown_limit         drawdown from starting balance reached the halt line -> halts
    invalid_signal             stop/target on the wrong side, zero stop distance, bad side
    max_entries_per_setup      already at the max entries on this setup
    no_bos_confirmation        scale-in without a favourable BOS after the previous entry
    setup_risk_cap             no risk room left on this setup for a scale-in
    max_open_positions         too many open positions
    position_too_small         risk budget buys less than the minimum units
    max_open_risk              total risk across open positions would exceed the cap
    would_breach_daily_loss    if every open stop and this one hit, today's loss passes the halt line
    would_breach_max_drawdown  same, for total drawdown

Halt lines sit `safety_margin_pct` inside the firm's limits, so the bot stops
before the firm's rule is broken. A halt is saved to disk and survives
restarts until `reset()` (or `python risk_manager.py reset`).

Usage:
    python risk_manager.py status
    python risk_manager.py reset
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "state" / "risk_state.json"


@dataclass
class RiskLimits:
    """All values are percentages. PLACEHOLDERS for the prop firm values - replace with the real rules."""
    risk_per_trade_pct: float = 0.5        # loss at stop, % of current balance; also the per-setup cap
    firm_daily_loss_pct: float = 5.0       # PLACEHOLDER: firm's max daily loss, % of starting balance
    firm_max_drawdown_pct: float = 10.0    # PLACEHOLDER: firm's max total loss, % of starting balance
    safety_margin_pct: float = 1.0         # halt this many points before the firm's limits
    max_open_positions: int = 3
    max_open_risk_pct: float = 1.5         # combined risk to stop across all open positions
    max_entries_per_setup: int = 3         # first entry + two scale-ins
    min_units: int = 1

    def __post_init__(self):
        for name in ("firm_daily_loss_pct", "firm_max_drawdown_pct"):
            if getattr(self, name) <= self.safety_margin_pct:
                raise ValueError(f"safety_margin_pct must be smaller than {name}")

    @property
    def daily_halt_pct(self):
        return self.firm_daily_loss_pct - self.safety_margin_pct

    @property
    def drawdown_halt_pct(self):
        return self.firm_max_drawdown_pct - self.safety_margin_pct


@dataclass
class Signal:
    instrument: str
    side: str                       # "long" or "short"
    entry: float
    stop: float
    target: float
    setup_id: str                   # same id for every entry on one setup (e.g. the zone)
    time: datetime
    bos_time: datetime | None = None  # latest break of structure in the trade's favour
    quote_to_account: float = 1.0     # account-currency value of 1 unit of quote currency


@dataclass
class Position:
    instrument: str
    side: str
    units: int
    entry: float
    stop: float
    setup_id: str
    opened_at: datetime
    quote_to_account: float = 1.0

    @property
    def risk_amount(self):
        """Loss if the stop is hit now; zero once the stop is at or beyond entry."""
        per_unit = self.entry - self.stop if self.side == "long" else self.stop - self.entry
        return max(0.0, per_unit * self.units * self.quote_to_account)


@dataclass
class AccountState:
    balance: float                  # closed-trade balance
    equity: float                   # balance + floating P&L
    day_start_balance: float        # balance at the firm's daily reset time
    starting_balance: float         # challenge starting balance
    open_positions: list = field(default_factory=list)


@dataclass
class Order:
    instrument: str
    side: str
    units: int
    entry: float
    stop: float
    target: float
    setup_id: str
    entry_number: int
    risk_amount: float
    risk_pct: float

    @property
    def oanda_units(self):
        return self.units if self.side == "long" else -self.units


@dataclass
class Decision:
    approved: bool
    reason: str
    detail: str = ""
    order: Order | None = None


class RiskManager:
    def __init__(self, limits=None, state_path=DEFAULT_STATE_PATH):
        self.limits = limits or RiskLimits()
        self.state_path = Path(state_path)
        self._state = self._load()

    # --- halt state ---------------------------------------------------------

    @property
    def halted(self):
        return self._state["halted"]

    @property
    def halt_reason(self):
        return self._state["reason"]

    def _load(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"halted": False, "reason": "", "detail": "", "since": None}

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def _halt(self, reason, detail):
        self._state = {"halted": True, "reason": reason, "detail": detail,
                       "since": datetime.now(timezone.utc).isoformat()}
        self._save()
        return Decision(False, reason, detail)

    def reset(self):
        """Manual reset after a halt. Nothing in the bot calls this automatically."""
        self._state = {"halted": False, "reason": "", "detail": "", "since": None}
        self._save()

    # --- limits -------------------------------------------------------------

    def _loss_pcts(self, account, extra_loss=0.0):
        """Today's loss and total drawdown, % of starting balance, after an optional extra loss."""
        equity = account.equity - extra_loss
        daily = 100 * (account.day_start_balance - equity) / account.starting_balance
        total = 100 * (account.starting_balance - equity) / account.starting_balance
        return daily, total

    def check_account(self, account):
        """Halt if a daily or total limit has been reached. Safe to call on every price update."""
        if self.halted:
            return Decision(False, "halted", f"{self.halt_reason}: {self._state['detail']}")
        daily, total = self._loss_pcts(account)
        if daily >= self.limits.daily_halt_pct:
            return self._halt("daily_loss_limit", f"down {daily:.2f}% today, halt line "
                              f"{self.limits.daily_halt_pct}% (firm limit {self.limits.firm_daily_loss_pct}%)")
        if total >= self.limits.drawdown_halt_pct:
            return self._halt("max_drawdown_limit", f"down {total:.2f}% from start, halt line "
                              f"{self.limits.drawdown_halt_pct}% (firm limit {self.limits.firm_max_drawdown_pct}%)")
        return None

    # --- evaluation ---------------------------------------------------------

    def evaluate(self, signal, account):
        lim = self.limits
        if rejection := self.check_account(account):
            return rejection
        if problem := validate(signal):
            return Decision(False, "invalid_signal", problem)

        same_setup = [p for p in account.open_positions if p.setup_id == signal.setup_id]
        budget = account.balance * lim.risk_per_trade_pct / 100
        if same_setup:
            if len(same_setup) >= lim.max_entries_per_setup:
                return Decision(False, "max_entries_per_setup",
                                f"{len(same_setup)} entries already open on {signal.setup_id}")
            last_entry = max(p.opened_at for p in same_setup)
            if signal.bos_time is None or signal.bos_time <= last_entry:
                return Decision(False, "no_bos_confirmation",
                                f"scale-in needs a favourable BOS after {last_entry:%Y-%m-%d %H:%M}")
            budget -= sum(p.risk_amount for p in same_setup)
            if budget <= 0:
                return Decision(False, "setup_risk_cap",
                                f"setup {signal.setup_id} already risks {lim.risk_per_trade_pct}% of balance")

        if len(account.open_positions) >= lim.max_open_positions:
            return Decision(False, "max_open_positions", f"{len(account.open_positions)} positions open")

        risk_per_unit = abs(signal.entry - signal.stop) * signal.quote_to_account
        units = math.floor(budget / risk_per_unit + 1e-9)
        if units < lim.min_units:
            return Decision(False, "position_too_small", f"budget {budget:.2f} buys {units} units")
        risk = units * risk_per_unit

        open_risk = sum(p.risk_amount for p in account.open_positions)
        open_risk_pct = 100 * (open_risk + risk) / account.balance
        if open_risk_pct > lim.max_open_risk_pct + 1e-9:
            return Decision(False, "max_open_risk",
                            f"open risk would be {open_risk_pct:.2f}%, cap {lim.max_open_risk_pct}%")

        daily, total = self._loss_pcts(account, extra_loss=open_risk + risk)
        if daily > lim.daily_halt_pct + 1e-9:
            return Decision(False, "would_breach_daily_loss",
                            f"all stops hit would put today at -{daily:.2f}%, halt line {lim.daily_halt_pct}%")
        if total > lim.drawdown_halt_pct + 1e-9:
            return Decision(False, "would_breach_max_drawdown",
                            f"all stops hit would put drawdown at {total:.2f}%, halt line {lim.drawdown_halt_pct}%")

        order = Order(signal.instrument, signal.side, units, signal.entry, signal.stop, signal.target,
                      signal.setup_id, len(same_setup) + 1, round(risk, 2),
                      round(100 * risk / account.balance, 3))
        return Decision(True, "approved", f"{units} units risking {risk:.2f}", order)


def validate(signal):
    if signal.side not in ("long", "short"):
        return f"side must be long or short, got {signal.side!r}"
    sign = 1 if signal.side == "long" else -1
    if sign * (signal.entry - signal.stop) <= 0:
        return "stop must be beyond entry on the losing side"
    if sign * (signal.target - signal.entry) <= 0:
        return "target must be beyond entry on the winning side"
    if signal.quote_to_account <= 0:
        return "quote_to_account must be positive"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Show or reset the risk manager's halt state.")
    parser.add_argument("action", choices=["status", "reset"])
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    args = parser.parse_args()

    rm = RiskManager(state_path=args.state)
    if args.action == "reset":
        was = rm.halt_reason
        rm.reset()
        print(f"Reset. Bot was {'halted (' + was + ')' if was else 'not halted'}; new signals allowed.")
        return
    lim = rm.limits
    print(f"Halted: {rm.halted}" + (f" - {rm.halt_reason}: {rm._state['detail']} (since {rm._state['since']})"
                                    if rm.halted else ""))
    print(f"Risk per trade {lim.risk_per_trade_pct}% | daily halt at {lim.daily_halt_pct}% "
          f"(firm {lim.firm_daily_loss_pct}%) | drawdown halt at {lim.drawdown_halt_pct}% "
          f"(firm {lim.firm_max_drawdown_pct}%) | max {lim.max_open_positions} positions, "
          f"{lim.max_open_risk_pct}% open risk")


if __name__ == "__main__":
    main()
