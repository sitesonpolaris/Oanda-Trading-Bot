from datetime import datetime, timedelta, timezone

import pytest

from risk_manager import AccountState, Position, RiskLimits, RiskManager, Signal

T0 = datetime(2026, 5, 4, 9, tzinfo=timezone.utc)
LIMITS = RiskLimits(
    risk_per_trade_pct=0.5, firm_daily_loss_pct=5.0, firm_max_drawdown_pct=10.0,
    safety_margin_pct=1.0, max_open_positions=3, max_open_risk_pct=1.5,
)


def account(balance=100_000.0, equity=None, day_start=None, positions=()):
    return AccountState(balance=balance, equity=balance if equity is None else equity,
                        day_start_balance=balance if day_start is None else day_start,
                        starting_balance=100_000.0, open_positions=list(positions))


def signal(setup="zone-1", entry=1.1000, stop=1.0980, target=1.1060, side="long", **kw):
    return Signal(instrument="EUR_USD", side=side, entry=entry, stop=stop, target=target,
                  setup_id=setup, time=kw.pop("time", T0), **kw)


def position(setup="zone-x", units=250_000, entry=1.1000, stop=1.0980, opened=T0 - timedelta(hours=5)):
    return Position(instrument="EUR_USD", side="long", units=units, entry=entry, stop=stop,
                    setup_id=setup, opened_at=opened)


@pytest.fixture
def rm(tmp_path):
    return RiskManager(LIMITS, state_path=tmp_path / "risk_state.json")


# --- approvals and sizing -------------------------------------------------

def test_approves_and_sizes_to_risk_percent(rm):
    d = rm.evaluate(signal(), account())
    assert d.approved and d.reason == "approved"
    assert d.order.units == 250_000  # 0.5% of 100k = $500 over a 20-pip stop ($200 per 100k units)
    assert d.order.risk_amount == pytest.approx(500)
    assert d.order.oanda_units == 250_000


def test_short_orders_have_negative_oanda_units(rm):
    d = rm.evaluate(signal(side="short", entry=1.1000, stop=1.1020, target=1.0940), account())
    assert d.approved and d.order.oanda_units == -250_000


def test_sizes_off_current_balance_and_converts_quote_currency(tmp_path):
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0), state_path=tmp_path / "s.json")
    d = rm.evaluate(signal(quote_to_account=0.5), account(balance=96_000))
    assert d.order.units == 960_000  # 1% of current $96k = $960, / (0.0020 * 0.5)


# --- every rejection reason ----------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(stop=1.1010),                         # stop on wrong side of a long
    dict(target=1.0990),                       # target on wrong side
    dict(stop=1.1000),                         # zero stop distance
    dict(side="sideways"),
])
def test_rejects_invalid_signal(rm, kw):
    d = rm.evaluate(signal(**kw), account())
    assert not d.approved and d.reason == "invalid_signal"


def test_rejects_when_position_too_small(tmp_path):
    rm = RiskManager(RiskLimits(min_units=1_000), state_path=tmp_path / "s.json")
    d = rm.evaluate(signal(stop=0.6), account())  # $500 over a 5000-pip stop = 1,000 units: just fits
    assert d.approved
    d = rm.evaluate(signal(stop=0.5), account())  # 833 units
    assert d.reason == "position_too_small"


def test_rejects_at_max_open_positions(rm):
    open_ = [position(f"z{i}", units=1_000) for i in range(3)]
    assert rm.evaluate(signal(), account(positions=open_)).reason == "max_open_positions"


def test_rejects_when_total_open_risk_cap_exceeded(rm):
    open_ = [position("a"), position("b", units=200_000)]  # 0.5% + 0.4% open; +0.5% fits, so push it
    assert rm.evaluate(signal(), account(positions=open_)).approved
    open_ = [position("a"), position("b", units=300_000)]  # 0.5% + 0.6% + 0.5% = 1.6% > 1.5%
    assert rm.evaluate(signal(), account(positions=open_)).reason == "max_open_risk"


def test_rejects_trade_whose_stop_would_breach_daily_halt(rm):
    # Down 3.7% today; a 0.5% stop-out would reach 4.2%, past the 4% halt line.
    d = rm.evaluate(signal(), account(balance=96_300, day_start=100_000))
    assert d.reason == "would_breach_daily_loss" and not rm.halted


def test_rejects_trade_whose_stop_would_breach_drawdown_halt(rm):
    # Down 8.7% overall but flat today; 0.5% more would pass the 9% halt line.
    d = rm.evaluate(signal(), account(balance=91_300, day_start=91_300))
    assert d.reason == "would_breach_max_drawdown" and not rm.halted


# --- halting ----------------------------------------------------------------

def test_daily_loss_halts_before_firm_limit(rm):
    d = rm.evaluate(signal(), account(balance=100_000, equity=95_900, day_start=100_000))
    assert d.reason == "daily_loss_limit"  # 4.1% loss: past the 4% halt, short of the firm's 5%
    assert rm.halted and rm.halt_reason == "daily_loss_limit"


def test_max_drawdown_halts_before_firm_limit(rm):
    d = rm.evaluate(signal(), account(balance=91_000, day_start=91_000))
    assert d.reason == "max_drawdown_limit" and rm.halted


def test_floating_losses_count_toward_limits(rm):
    d = rm.evaluate(signal(), account(balance=100_000, equity=90_900, day_start=91_000))
    assert d.reason == "max_drawdown_limit"


def test_halted_bot_rejects_everything_even_after_recovery(rm):
    rm.evaluate(signal(), account(equity=95_000))
    d = rm.evaluate(signal(setup="zone-2"), account())
    assert not d.approved and d.reason == "halted"


def test_halt_survives_restart_until_manual_reset(rm, tmp_path):
    rm.evaluate(signal(), account(equity=95_000))
    again = RiskManager(LIMITS, state_path=tmp_path / "risk_state.json")
    assert again.halted and again.evaluate(signal(), account()).reason == "halted"
    again.reset()
    assert not again.halted and again.evaluate(signal(), account()).approved
    assert not RiskManager(LIMITS, state_path=tmp_path / "risk_state.json").halted


def test_safety_margin_must_leave_room():
    with pytest.raises(ValueError):
        RiskLimits(firm_daily_loss_pct=1.0, safety_margin_pct=1.0)


# --- scaling in ---------------------------------------------------------------

def test_scale_in_requires_bos_after_previous_entry(rm):
    first = position("zone-1", stop=1.1000)  # stop already moved to breakeven: no risk left
    d = rm.evaluate(signal(entry=1.1030, stop=1.1010, target=1.1090), account(positions=[first]))
    assert d.reason == "no_bos_confirmation"
    stale = signal(entry=1.1030, stop=1.1010, target=1.1090, bos_time=first.opened_at - timedelta(hours=1))
    assert rm.evaluate(stale, account(positions=[first])).reason == "no_bos_confirmation"


def test_scale_in_approved_after_bos_within_setup_cap(rm):
    first = position("zone-1", stop=1.1000)
    add = signal(entry=1.1030, stop=1.1010, target=1.1090, bos_time=T0 - timedelta(hours=1))
    d = rm.evaluate(add, account(positions=[first]))
    assert d.approved and d.order.entry_number == 2 and d.order.units == 250_000


def test_scale_in_sized_down_to_remaining_setup_risk(rm):
    first = position("zone-1", stop=1.0990)  # still risking $250 of the $500 setup cap
    add = signal(entry=1.1030, stop=1.1010, target=1.1090, bos_time=T0 - timedelta(hours=1))
    d = rm.evaluate(add, account(positions=[first]))
    assert d.approved and d.order.risk_amount == pytest.approx(250)


def test_scale_in_rejected_when_setup_cap_used(rm):
    first = position("zone-1")  # full $500 still at risk
    add = signal(entry=1.1030, stop=1.1010, target=1.1090, bos_time=T0 - timedelta(hours=1))
    assert rm.evaluate(add, account(positions=[first])).reason == "setup_risk_cap"


def test_no_fourth_entry_on_a_setup(rm):
    entries = [position("zone-1", units=1_000, stop=1.1000, opened=T0 - timedelta(hours=h)) for h in (9, 6, 3)]
    add = signal(entry=1.1030, stop=1.1010, target=1.1090, bos_time=T0 - timedelta(hours=1))
    assert rm.evaluate(add, account(positions=entries)).reason == "max_entries_per_setup"
