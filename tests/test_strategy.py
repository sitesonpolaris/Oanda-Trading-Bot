from types import SimpleNamespace

import pandas as pd
import pytest

from helpers import T0, bars, lifecycle_zone, pivot
from strategy import (StrategyConfig, direction_check, entry_stop, h1_pivots, level_on_level, news_conflict,
                      opposing_edge, prior_pivot, trade_target, zone_lifecycle)
from structure import market_structure

CFG = StrategyConfig()


def z(kind, top, bottom, **kw):
    return SimpleNamespace(kind=kind, top=top, bottom=bottom, atr_pips=10, start_time=T0,
                           legout_time=T0 + pd.Timedelta(hours=1), **kw)


def test_stop_beyond_far_edge_by_ten_percent_of_daily_atr():
    assert entry_stop(z("demand", 1.1000, 1.0990), 0.0070, CFG) == (1.1000, pytest.approx(1.0983))
    assert entry_stop(z("supply", 1.1010, 1.1000), 0.0070, CFG) == (1.1000, pytest.approx(1.1017))


def test_target_is_nearer_of_prior_pivot_and_opposing_zone():
    zones = pd.DataFrame([lifecycle_zone("supply", 1.1060, 1.1050), lifecycle_zone("demand", 1.0950, 1.0940)])
    pivots = pd.DataFrame([pivot("high", 1.1040), pivot("low", 1.0970)])
    assert trade_target(True, 1.1000, T0, zones, pivots) == (1.1040, "pivot")
    assert trade_target(False, 1.1000, T0, zones, pivots) == (1.0970, "pivot")
    far_pivots = pd.DataFrame([pivot("high", 1.1080), pivot("low", 1.0900)])
    assert trade_target(True, 1.1000, T0, zones, far_pivots) == (1.1050, "zone")
    assert trade_target(False, 1.1000, T0, zones, far_pivots) == (1.0950, "zone")
    assert trade_target(True, 1.1100, T0, zones, far_pivots) == (None, "")


def test_prior_pivot_is_most_recent_beyond_entry_and_confirmed():
    pivots = pd.DataFrame([pivot("high", 1.1030, T0 - pd.Timedelta(hours=3)),
                           pivot("high", 1.1060, T0 - pd.Timedelta(hours=1)),
                           pivot("high", 1.1020, T0 + pd.Timedelta(hours=1))])
    assert prior_pivot(True, 1.1000, T0, pivots) == 1.1060


def test_h1_pivots_record_confirmation_time():
    closes = [1.10, 1.105, 1.11, 1.115, 1.12, 1.1175, 1.115, 1.1125, 1.11, 1.115, 1.12]
    times = pd.date_range(T0, periods=len(closes), freq="h")
    df = pd.DataFrame({"open": closes, "high": [c + 0.0002 for c in closes], "low": [c - 0.0003 for c in closes],
                       "close": closes, "time": times})
    p = h1_pivots(market_structure(df), pd.Timedelta("1h"), 2)
    high = p[p["kind"] == "high"].iloc[0]
    assert high.level == pytest.approx(1.1202) and high.confirmed_time == times[6] + pd.Timedelta("1h")


@pytest.mark.parametrize("long,d1,bos,h4,ok", [
    (True, "up", "", "up", True),
    (True, "up", "", "sideways", False),        # momentum must agree
    (True, "down", "bullish", "up", False),     # opposite D1 trend still in force
    (True, "sideways", "bullish", "up", False),  # countertrend off: all timeframes must align
    (True, "sideways", "", "up", False),
    (False, "down", "", "down", True),
    (False, "sideways", "bearish", "down", False),
])
def test_direction_check(long, d1, bos, h4, ok):
    assert direction_check(long, d1, bos, h4, CFG)[0] is ok


def test_countertrend_after_d1_bos_when_enabled():
    ct = StrategyConfig(allow_countertrend_with_bos=True)
    assert direction_check(True, "sideways", "bullish", "up", ct)[0]
    assert direction_check(False, "sideways", "bearish", "down", ct)[0]


def test_zone_lifecycle_breaks_on_close_beyond_far_edge():
    b = bars([(1.1000, 1.1010, 1.0995, 1.1005), (1.1005, 1.1008, 1.0985, 1.0988), (1.0988, 1.0990, 1.0980, 1.0985)])
    zones = pd.DataFrame([{"kind": "demand", "top": 1.1000, "bottom": 1.0990,
                           "valid_time": T0 - pd.Timedelta(hours=1)}])
    out = zone_lifecycle(zones, b, pd.Timedelta("1h"))
    assert out["available_time"].iat[0] == T0
    assert out["broken_time"].iat[0] == T0 + pd.Timedelta(hours=2)  # bar 1 closed below 1.0990


def test_opposing_edge_is_nearest_active_zone_beyond_entry():
    zones = pd.DataFrame([lifecycle_zone("supply", 1.1040, 1.1030), lifecycle_zone("supply", 1.1025, 1.1020),
                          lifecycle_zone("supply", 1.1015, 1.1012, broken=T0 - pd.Timedelta(minutes=1)),
                          lifecycle_zone("demand", 1.1010, 1.1005), lifecycle_zone("demand", 1.0990, 1.0980)])
    assert opposing_edge(True, 1.1000, T0, zones) == 1.1020      # bottom of the nearest supply above
    assert opposing_edge(True, 1.1050, T0, zones) is None
    assert opposing_edge(False, 1.1000, T0, zones) == 1.0990     # top of the nearest demand below
    assert opposing_edge(False, 1.0970, T0, zones) is None


def test_level_on_level_needs_stacked_same_side_zone():
    me = z("demand", 1.1000, 1.0990)
    stacked = pd.DataFrame([lifecycle_zone("demand", 1.0992, 1.0980)])
    far = pd.DataFrame([lifecycle_zone("demand", 1.0950, 1.0940)])
    opposite = pd.DataFrame([lifecycle_zone("supply", 1.0995, 1.0985)])
    assert level_on_level(me, stacked, 0.0010, CFG)
    assert not level_on_level(me, far, 0.0010, CFG)
    assert not level_on_level(me, opposite, 0.0010, CFG)


def test_news_window():
    events = pd.DataFrame({"time": [T0], "currency": ["EUR"], "impact": ["high"]})
    assert news_conflict(events, "EUR_USD", T0 + pd.Timedelta(minutes=20), T0 + pd.Timedelta(minutes=80), CFG) is not None
    assert news_conflict(events, "EUR_USD", T0 + pd.Timedelta(minutes=40), T0 + pd.Timedelta(minutes=100), CFG) is None
    assert news_conflict(None, "EUR_USD", T0, T0, CFG) is None
