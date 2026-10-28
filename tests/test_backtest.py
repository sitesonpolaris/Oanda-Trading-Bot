import pandas as pd
import pytest

from backtest import BacktestConfig, simulate, summarize
from helpers import LOSS, SETUP, T0, TOUCH, WIN, bars, lifecycle_zone, market, pivot, zone

PIP = 0.0001
NO_SPREAD = BacktestConfig(spread_pips=0)


def run(rows, cfg=NO_SPREAD, z=None, **market_kw):
    b = bars(rows)
    zones = pd.DataFrame([z or zone()])
    return simulate(zones, market(b, **market_kw), cfg, PIP).iloc[0]


# --- stop and target ------------------------------------------------------------

def test_stop_buffer_is_ten_percent_of_daily_atr():
    t = run(SETUP + [TOUCH, WIN])
    assert t.order_time == T0 + pd.Timedelta(hours=3)  # as soon as the zone is valid
    assert (t.entry, t.stop) == (1.1000, pytest.approx(1.0990 - 0.0007))
    assert run(SETUP + [TOUCH, WIN], daily_atr=0.0100).stop == pytest.approx(1.0980)


def test_target_at_opposing_zone_edge_when_no_nearer_pivot():
    t = run(SETUP + [TOUCH, WIN])
    assert t.status == "filled" and t.result == "win"
    assert (t.target, t.target_source) == (1.1050, "zone")
    assert t.r_multiple == pytest.approx(50 / 17 / 3, abs=1e-3)  # one-third position


def test_target_at_prior_pivot_when_nearer():
    t = run(SETUP + [TOUCH, WIN], pivots=[pivot("high", 1.1045)])
    assert (t.target, t.target_source) == (1.1045, "pivot")


def test_most_recent_pivot_above_entry_counts():
    pivots = [pivot("high", 1.1030, T0 - pd.Timedelta(hours=5)), pivot("high", 1.1048, T0 - pd.Timedelta(hours=2)),
              pivot("high", 1.0995, T0 - pd.Timedelta(hours=1)),  # below entry: not a target
              pivot("low", 1.1040)]                                # lows don't count for longs
    assert run(SETUP + [TOUCH, WIN], pivots=pivots).target == 1.1048


def test_pivot_confirmed_after_the_touch_is_ignored():
    late = pivot("high", 1.1045, T0 + pd.Timedelta(hours=8))
    assert run(SETUP + [TOUCH, WIN], pivots=[late]).target_source == "zone"


def test_target_uses_zones_active_when_price_returns():
    broken = lifecycle_zone("supply", 1.1030, 1.1025, broken=T0)
    after = lifecycle_zone("supply", 1.1035, 1.1030, available=T0 + pd.Timedelta(hours=8))
    far = lifecycle_zone("supply", 1.1110, 1.1100)
    assert run(SETUP + [TOUCH, WIN], zones=[broken, after, far]).target == 1.1100


def test_no_pivot_and_no_opposing_zone_means_no_trade():
    assert run(SETUP + [TOUCH, WIN], zones=[]).status == "no_target"


def test_target_too_close_fails_two_to_one():
    t = run(SETUP + [TOUCH, WIN], pivots=[pivot("high", 1.1020)])  # 20 pips vs a 17-pip stop
    assert t.status == "skipped_rr" and t.planned_rr == pytest.approx(20 / 17, abs=0.01)


def test_short_targets_nearer_of_pivot_low_and_demand_zone():
    sz = zone(kind="supply", top=1.1010, bottom=1.1000)
    rows = [(1.0990, 1.0995, 1.0985, 1.0992)] * 2 + [(1.1000, 1.1002, 1.0970, 1.0972)] + \
        [(1.0975, 1.0980, 1.0965, 1.0970)] * 4 + [(1.0990, 1.1005, 1.0988, 1.0995), (1.0990, 1.0992, 1.0945, 1.0948)]
    kw = dict(z=sz, zones=[lifecycle_zone("demand", 1.0950, 1.0940)], d1_trend="down", h4_trend="down")
    t = run(rows, **kw)
    assert (t.entry, t.stop, t.target) == (1.1000, pytest.approx(1.1017), 1.0950)
    assert t.status == "filled" and t.result == "win"
    assert run(rows, pivots=[pivot("low", 1.0960)], **kw).target == 1.0960


# --- outcome ---------------------------------------------------------------------

def test_stop_out_with_first_entry_only_loses_a_third():
    t = run(SETUP + [TOUCH, LOSS])
    assert t.result == "loss" and t.r_multiple == pytest.approx(-1 / 3, abs=1e-3)


def test_add_on_bullish_h1_bos_after_fill():
    bos = [""] * 7 + ["bullish", ""]  # BOS at the close of the fill candle -> add at next open (1.1010)
    t = run(SETUP + [TOUCH, WIN], h1_bos=bos)
    assert t.adds == 1
    assert t.r_multiple == pytest.approx(50 / 17 / 3 + 40 / 27 / 3, abs=1e-3)


def test_immediate_return_fills_the_order():
    t = run(SETUP[:3] + [TOUCH, WIN])
    assert t.status == "filled" and t.entry_time == T0 + pd.Timedelta(hours=3)


def test_order_waits_for_zone_to_be_valid():
    late = zone(valid_time=T0 + pd.Timedelta(hours=6))  # validity candle closes at 07:00 UTC
    assert run(SETUP + [TOUCH, WIN], z=late).order_time == T0 + pd.Timedelta(hours=7)
    assert run(SETUP + [TOUCH, WIN], z=zone(valid_time=pd.NaT)).status == "not_valid"


def test_all_four_types_traded_by_default():
    assert run(SETUP + [TOUCH, WIN], z=zone(type="DBR")).status == "filled"
    b = bars(SETUP + [TOUCH, WIN])
    only = BacktestConfig(spread_pips=0, trade_types=("RBR", "DBD"))
    assert simulate(pd.DataFrame([zone(type="DBR")]), market(b), only, PIP).empty


# --- filters ---------------------------------------------------------------------

def test_d1_stage_and_h4_momentum_must_align():
    assert run(SETUP + [TOUCH, WIN], d1_trend="down").status == "skipped_trend"
    assert run(SETUP + [TOUCH, WIN], h4_trend="sideways").status == "skipped_trend"


def test_countertrend_off_by_default():
    assert run(SETUP + [TOUCH, WIN], d1_trend="sideways", d1_bos="bullish").status == "skipped_trend"
    ct = BacktestConfig(spread_pips=0, allow_countertrend_with_bos=True)
    t = run(SETUP + [TOUCH, WIN], ct, d1_trend="sideways", d1_bos="bullish")
    assert t.status == "filled" and "countertrend" in t.reason


def test_h4_momentum_is_checked_when_zone_becomes_valid():
    b = bars(SETUP + [TOUCH, WIN])
    m = market(b)
    turned = T0 + pd.Timedelta(hours=5)  # momentum turns down during the pullback, before the touch
    m.h4 = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1), turned], "trend": ["up", "down"]})
    assert simulate(pd.DataFrame([zone()]), m, NO_SPREAD, PIP).iloc[0].status == "filled"
    m.h4 = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1), turned], "trend": ["down", "up"]})
    assert simulate(pd.DataFrame([zone()]), m, NO_SPREAD, PIP).iloc[0].status == "skipped_trend"


def test_news_window_skips_trade():
    news = pd.DataFrame({"time": [T0 + pd.Timedelta(hours=7, minutes=10)], "currency": ["USD"], "impact": ["high"]})
    assert run(SETUP + [TOUCH, WIN], news=news).status == "skipped_news"
    assert run(SETUP + [TOUCH, WIN], news=news.assign(currency="JPY")).status == "filled"


def test_erratic_market_is_skipped():
    assert run(SETUP + [TOUCH, WIN], BacktestConfig(spread_pips=0, max_big_wicks=0)).status == "skipped_erratic"


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
