"""Main loop: once per H1 candle close, pull D1/H4/H1 data, find zones, trend and
momentum, generate signals (strategy.py rules, same as the backtest), pass them
through the risk manager, and hand approved orders to the executioner.
Practice account only.

Dry run is the default: everything runs, including reads from the practice
account, but no orders are placed or cancelled. Use --send-orders to place them.

Usage:
    python bot.py --once                 # one dry-run cycle now
    python bot.py                        # dry run, every candle close
    python bot.py --send-orders          # place orders on the practice account
    python bot.py --as-of "2025-11-19 01:00"   # dry-run replay of one past candle close
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from backtest import TIMEFRAMES, BacktestConfig, build_market, NEWS_PATH
from executioner import Executioner, RefuseToRun, client_order_id, make_client, resolve_environment
from fetch_candles import fetch_candles
from risk_manager import AccountState, Position, RiskLimits, RiskManager
from signals import add_signals, evaluate_zones
from strategy import load_news
from structure import StructureConfig
from zones import ZoneConfig, pip_size

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "bot.log"
STATE_PATH = ROOT / "state" / "bot_state.json"

log = logging.getLogger("bot")


@dataclass
class BotConfig:
    instrument: str = "EUR_USD"
    lookback_days: dict = field(default_factory=lambda: {"H1": 90, "H4": 180, "D": 400})
    max_order_age_candles: int = 24        # cancel unfilled limit orders after this many H1 candles
    daily_reset_tz: str = "UTC"            # PLACEHOLDER: the prop firm's daily reset time zone
    starting_balance: float | None = None  # PLACEHOLDER: challenge start; None = balance on first run
    candle_close_delay_s: int = 20         # wait after the close so OANDA marks the candle complete
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    trade: BacktestConfig = field(default_factory=BacktestConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)


def setup_logging(path=LOG_PATH):
    path.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s UTC %(levelname)-7s %(name)-11s %(message)s")
    fmt.converter = time.gmtime
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in (file_handler, console):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("urllib3", "fetch_candles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # oandapyV20 logs every non-2xx as ERROR, including the expected 404 from the duplicate-order check.
    # The executioner logs real failures itself.
    logging.getLogger("oandapyV20").setLevel(logging.CRITICAL)


class Bot:
    def __init__(self, cfg, client, account_id, dry_run, state_path=STATE_PATH):
        self.cfg = cfg
        self.client = client
        self.ex = Executioner(client, account_id, dry_run=dry_run)
        self.rm = RiskManager(cfg.risk)
        self.state_path = state_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        self.pip = pip_size(cfg.instrument)

    def save_state(self):
        self.state_path.parent.mkdir(exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    # --- account ---------------------------------------------------------------

    def account_state(self, book, now):
        if "starting_balance" not in self.state:
            self.state["starting_balance"] = self.cfg.starting_balance or book.balance
            log.info("Recorded challenge starting balance %.2f", self.state["starting_balance"])
        today = now.astimezone(ZoneInfo(self.cfg.daily_reset_tz)).date().isoformat()
        if self.state.get("day") != today:
            self.state.update(day=today, day_start_balance=book.balance)
            log.info("New trading day %s (%s): day-start balance %.2f", today, self.cfg.daily_reset_tz, book.balance)
        self.save_state()
        positions = [t.as_position() for t in book.trades] + [o.as_position() for o in book.pending]
        return AccountState(book.balance, book.equity, self.state["day_start_balance"],
                            self.state["starting_balance"], positions)

    # --- one cycle --------------------------------------------------------------

    def cycle(self, now=None):
        cfg = self.cfg
        now = now or datetime.now(timezone.utc)
        log.info("===== Cycle %s | %s | D1 trend, H4 momentum, H1 zones =====", f"{now:%Y-%m-%d %H:%M}", self.ex.mode)

        book = self.ex.sync()
        account = self.account_state(book, now)
        daily = 100 * (account.day_start_balance - account.equity) / account.starting_balance
        total = 100 * (account.starting_balance - account.equity) / account.starting_balance
        log.info("Risk: today %+.2f%% (halt at -%.1f%%), from start %+.2f%% (halt at -%.1f%%)",
                 -daily, cfg.risk.daily_halt_pct, -total, cfg.risk.drawdown_halt_pct)

        if halt := self.rm.check_account(account):
            log.error("HALTED (%s). No new trades; cancelling pending orders. Reset with: "
                      "python risk_manager.py reset", halt.detail)
            for o in book.pending:
                self.ex.cancel(o, "bot halted")
            return

        end = now.replace(minute=0, second=0, microsecond=0)
        candles = {g: closed_by(fetch_candles(self.client, cfg.instrument, g, end - timedelta(days=days), end), g, now)
                   for g, days in cfg.lookback_days.items()}
        log.info("Data: %s", ", ".join(f"{len(df)} {g} (last {df['time'].iat[-1]:%Y-%m-%d %H:%M})"
                                       for g, df in candles.items()))

        news = load_news(NEWS_PATH)
        market, zones = build_market(candles["H1"], candles["H4"], candles["D"], cfg.zones, cfg.instrument,
                                     self.pip, news, cfg.structure)
        for name, structure in (("D1 trend", market.d1), ("H4 momentum", market.h4)):
            last = structure.iloc[-1]
            log.info("%s: %s | last swing high %.5f, low %.5f | last BOS: %s", name, last.trend,
                     last.last_swing_high, last.last_swing_low, _last_bos(structure))
        if news is None:
            log.info("News filter off: no %s", NEWS_PATH.relative_to(ROOT))

        verdicts = evaluate_zones(zones, market, cfg.trade, self.pip, now)
        counts = {}
        for v in verdicts:
            counts[v.status] = counts.get(v.status, 0) + 1
        log.info("H1 zones: %d in %d days | %s", len(zones), cfg.lookback_days["H1"], counts)
        for v in verdicts:
            log.debug("  zone %s %s: %s", v.setup_id, v.status, v.detail)

        self.manage_pending(book, {v.setup_id: v for v in verdicts}, now)
        self.handle_signals([v for v in verdicts if v.status == "signal"], book, account)
        self.handle_adds(add_signals(book.trades, market, cfg.trade, now), account)

    def manage_pending(self, book, verdicts, now):
        max_age = TIMEFRAMES["H1"] * self.cfg.max_order_age_candles
        for o in book.pending:
            v = verdicts.get(o.setup_id)
            if v is None or v.status != "signal":
                self.ex.cancel(o, f"zone invalidated: {v.status if v else 'zone no longer detected'}")
            elif now - o.created >= max_age:
                self.ex.cancel(o, f"unfilled after {self.cfg.max_order_age_candles} H1 candles")
            else:
                log.info("Keeping %s: zone still valid, age %s", o.client_id, now - o.created)

    def handle_signals(self, signals, book, account):
        taken = {o.setup_id for o in book.pending} | {t.setup_id for t in book.trades}
        quote = self.cfg.instrument.split("_")[1]
        if signals and quote != book.currency:
            log.error("Account currency %s differs from quote currency %s; conversion not implemented, "
                      "skipping %d signal(s)", book.currency, quote, len(signals))
            return
        for v in signals:
            s = v.signal
            if s.setup_id in taken:
                log.info("Signal %s: order or trade already active for this setup", s.setup_id)
                continue
            if previous := self.ex.find_order(client_order_id(s.setup_id, 1)):
                log.info("Signal %s: already traded this setup before (order %s, %s); fresh zones only",
                         s.setup_id, previous.get("id"), previous.get("state"))
                continue
            decision = self.rm.evaluate(s, account)
            log.info("Signal %s %s %s -> risk manager: %s (%s)", s.setup_id, s.side, v.detail,
                     decision.reason.upper(), decision.detail)
            if decision.approved:
                o = decision.order
                self.ex.place(o)
                account.open_positions.append(
                    Position(o.instrument, o.side, o.units, o.entry, o.stop, o.setup_id, s.time))

    def handle_adds(self, signals, account):
        for s in signals:
            decision = self.rm.evaluate(s, account)
            log.info("Scale-in %s %s at market ~%.5f after H1 BOS %s -> risk manager: %s (%s)", s.setup_id, s.side,
                     s.entry, f"{s.bos_time:%Y-%m-%d %H:%M}", decision.reason.upper(), decision.detail)
            if decision.approved:
                o = decision.order
                self.ex.place(o)
                account.open_positions.append(
                    Position(o.instrument, o.side, o.units, o.entry, o.stop, o.setup_id, s.time))


def closed_by(candles, granularity, now):
    """Only candles that had closed at `now` (matters for --as-of replays)."""
    return candles[candles["time"] + TIMEFRAMES[granularity] <= now].reset_index(drop=True)


def _last_bos(structure):
    hits = structure[structure["bos"] != ""]
    if hits.empty:
        return "none"
    row = hits.iloc[-1]
    return f"{row.bos} at {row.time:%Y-%m-%d %H:%M}"


def sleep_until_next_close(cfg):
    period = TIMEFRAMES["H1"]
    now = datetime.now(timezone.utc)
    next_close = now.replace(minute=0, second=0, microsecond=0) + period
    wake = next_close + timedelta(seconds=cfg.candle_close_delay_s)
    log.info("Sleeping until %s UTC", f"{wake:%Y-%m-%d %H:%M:%S}")
    time.sleep((wake - now).total_seconds())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--send-orders", action="store_true",
                        help="actually place/cancel orders on the practice account (default: dry run)")
    parser.add_argument("--once", action="store_true", help="run a single cycle now and exit")
    parser.add_argument("--as-of", help="dry-run one cycle as if it were this past UTC candle close "
                                        "(market data only; account data is current)")
    args = parser.parse_args()
    if args.as_of and args.send_orders:
        parser.error("--as-of is a dry-run replay and can't be combined with --send-orders")

    setup_logging()
    token, account_id = os.environ.get("OANDA_API_TOKEN"), os.environ.get("OANDA_ACCOUNT_ID")
    if not token or not account_id:
        log.error("OANDA_API_TOKEN and OANDA_ACCOUNT_ID must both be set")
        sys.exit(1)
    try:
        environment = resolve_environment()
    except RefuseToRun as e:
        log.error("Refusing to run: %s", e)
        sys.exit(1)

    cfg = BotConfig()
    log.info("Starting bot: environment=%s account=%s mode=%s", environment, account_id,
             "SEND ORDERS" if args.send_orders else "DRY RUN")
    client = make_client(token, environment)
    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
        log.info("Replay as of %s UTC (separate state file; nothing is sent)", f"{as_of:%Y-%m-%d %H:%M}")
        Bot(cfg, client, account_id, dry_run=True, state_path=STATE_PATH.with_name("replay_state.json")).cycle(as_of)
        return
    bot = Bot(cfg, client, account_id, dry_run=not args.send_orders)

    while True:
        try:
            bot.cycle()
        except Exception:
            log.exception("Cycle failed; will try again next candle")
        if args.once:
            break
        try:
            sleep_until_next_close(cfg)
        except KeyboardInterrupt:
            log.info("Stopped by user")
            break


if __name__ == "__main__":
    main()
