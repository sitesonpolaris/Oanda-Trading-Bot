import pandas as pd
import pytest

from backtest import BacktestConfig, simulate, summarize
from helpers import AWAY, LOSS, SETUP, T0, TOUCH, WIN, bars, lifecycle_zone, market, zone

PIP = 0.0001
NO_SPREAD = BacktestConfig(spread_pips=0)


def run(rows, cfg=NO_SPREAD, z=None, **market_kw):
    b = bars(rows)
    zones = pd.DataFrame([z or zone()])
    return simulate(zones, market(b, **market_kw), cfg, PIP).iloc[0]


def test_first_entry_hits_target_measured_from_origin():
    t = run(SETUP + [TOUCH, WIN])
    assert t.status == "filled" and t.result == "win"
    assert t.order_time == T0 + pd.Timedelta(hours=6)  # after the impulse high is confirmed
    assert (t.entry, t.stop, t.target) == (1.1000, pytest.approx(1.0989), pytest.approx(1.1038))
    assert t.r_multiple == pytest.approx(38 / 11 / 3, abs=1e-3)  # one-third position


def test_stop_out_with_first_entry_only_loses_a_third():
    t = run(SETUP + [TOUCH, LOSS])
    assert t.result == "loss" and t.r_multiple == pytest.approx(-1 / 3, abs=1e-3)


def test_add_on_bullish_h1_bos_after_fill():
    rows = SETUP + [TOUCH, WIN]
    bos = [""] * 7 + ["bullish", ""]  # BOS at the close of the fill candle -> add at next open (1.1010)
    t = run(rows, h1_bos=bos)
    assert t.adds == 1
    assert t.r_multiple == pytest.approx(38 / 11 / 3 + 28 / 21 / 3, abs=1e-3)


def test_touch_before_impulse_confirmed_burns_zone():
    early = (1.1045, 1.1046, 1.0999, 1.1035)  # dips into the zone at bar 4, before the order exists
    assert run(SETUP[:4] + [early] + SETUP[5:] + [TOUCH, WIN]).status == "touched_before_order"


def test_needs_d1_trend_and_h4_momentum():
    assert run(SETUP + [TOUCH, WIN], d1_trend="down").status == "skipped_trend"
    assert run(SETUP + [TOUCH, WIN], h4_trend="sideways").status == "skipped_trend"


def test_countertrend_allowed_after_d1_bos():
    t = run(SETUP + [TOUCH, WIN], d1_trend="sideways", d1_bos="bullish")
    assert t.status == "filled" and "countertrend" in t.reason
    assert run(SETUP + [TOUCH, WIN], d1_trend="sideways", d1_bos="bearish").status == "skipped_trend"
    no_ct = BacktestConfig(spread_pips=0, allow_countertrend_with_bos=False)
    assert run(SETUP + [TOUCH, WIN], no_ct, d1_trend="sideways", d1_bos="bullish").status == "skipped_trend"


def test_opposing_zone_too_close_is_skipped():
    supply = lifecycle_zone("supply", top=1.1020, bottom=1.1010)  # 10 pips above entry, stop is 11 pips
    t = run(SETUP + [TOUCH, WIN], zones=[supply])
    assert t.status == "skipped_room" and t.room_r == pytest.approx(10 / 11, abs=0.01)
    broken = lifecycle_zone("supply", top=1.1020, bottom=1.1010, broken=T0)
    assert run(SETUP + [TOUCH, WIN], zones=[broken]).status == "filled"


def test_news_window_skips_trade():
    news = pd.DataFrame({"time": [T0 + pd.Timedelta(hours=7, minutes=10)], "currency": ["USD"], "impact": ["high"]})
    assert run(SETUP + [TOUCH, WIN], news=news).status == "skipped_news"
    other = news.assign(currency="JPY")
    assert run(SETUP + [TOUCH, WIN], news=other).status == "filled"


def test_erratic_market_is_skipped():
    assert run(SETUP + [TOUCH, WIN], BacktestConfig(spread_pips=0, max_big_wicks=0)).status == "skipped_erratic"


def test_low_reward_to_risk_is_skipped():
    assert run(SETUP + [TOUCH, WIN], BacktestConfig(spread_pips=0, min_reward_risk=4)).status == "skipped_rr"


def test_spread_can_prevent_fill():
    graze = (1.1010, 1.1012, 1.1000, 1.1005)  # mid low exactly at entry; ask stays above it
    assert run(SETUP + [graze]).status == "filled"
    assert run(SETUP + [graze], BacktestConfig(spread_pips=1.0)).status == "untouched"


def test_summary_metrics():
    times = pd.date_range("2025-01-06", periods=4, freq="h", tz="UTC")
    trades = pd.DataFrame({
        "status": ["filled"] * 4, "exit_time": times,
        "r_multiple": [2.0, -1.0, -1.0, 3.0], "planned_rr": [2.0, 2.0, 2.0, 3.0], "adds": [0, 1, 2, 0],
    })
    s = summarize(trades, BacktestConfig(risk_per_trade=0.01))
    assert s["trades"] == 4 and s["win_rate_%"] == 50.0 and s["avg_adds"] == 0.75
    assert s["total_return_%"] == pytest.approx((1.02 * 0.99 * 0.99 * 1.03 - 1) * 100, abs=0.01)
    assert s["max_drawdown_%"] == pytest.approx(1.99, abs=0.01)


def test_h4_momentum_is_checked_when_zone_forms_not_at_touch():
    b = bars(SETUP + [TOUCH, WIN])
    m = market(b)
    turned = T0 + pd.Timedelta(hours=5)  # momentum turns down during the pullback, before the touch
    m.h4 = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1), turned], "trend": ["up", "down"]})
    assert simulate(pd.DataFrame([zone()]), m, NO_SPREAD, PIP).iloc[0].status == "filled"
    m.h4 = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1), turned], "trend": ["down", "up"]})
    assert simulate(pd.DataFrame([zone()]), m, NO_SPREAD, PIP).iloc[0].status == "skipped_trend"
