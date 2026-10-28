from types import SimpleNamespace

import pandas as pd
import pytest

from helpers import T0, bars, lifecycle_zone
from strategy import (StrategyConfig, direction_check, level_on_level, news_conflict, opposing_edge, plan_trade,
                      zone_lifecycle)

CFG = StrategyConfig()


def z(kind, top, bottom, **kw):
    return SimpleNamespace(kind=kind, top=top, bottom=bottom, atr_pips=10, start_time=T0,
                           legout_time=T0 + pd.Timedelta(hours=1), **kw)


def test_target_is_80pct_of_the_way_to_the_opposing_zone():
    entry, stop, target = plan_trade(z("demand", 1.1000, 1.0990), 1.1050, CFG, 0.0001)
    assert (entry, stop, target) == (1.1000, pytest.approx(1.0989), pytest.approx(1.1040))
    entry, stop, target = plan_trade(z("supply", 1.1010, 1.1000), 1.0950, CFG, 0.0001)
    assert (entry, stop, target) == (1.1000, pytest.approx(1.1011), pytest.approx(1.0960))


@pytest.mark.parametrize("long,d1,bos,h4,ok", [
    (True, "up", "", "up", True),
    (True, "up", "", "sideways", False),        # momentum must agree
    (True, "down", "bullish", "up", False),     # opposite D1 trend still in force
    (True, "sideways", "bullish", "up", True),  # countertrend after D1 BOS
    (True, "sideways", "", "up", False),
    (False, "down", "", "down", True),
    (False, "sideways", "bearish", "down", True),
])
def test_direction_check(long, d1, bos, h4, ok):
    assert direction_check(long, d1, bos, h4, CFG)[0] is ok


def test_zone_lifecycle_breaks_on_close_beyond_far_edge():
    b = bars([(1.1000, 1.1010, 1.0995, 1.1005), (1.1005, 1.1008, 1.0985, 1.0988), (1.0988, 1.0990, 1.0980, 1.0985)])
    zones = pd.DataFrame([{"kind": "demand", "top": 1.1000, "bottom": 1.0990,
                           "breakout_time": T0 - pd.Timedelta(hours=1)}])
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
