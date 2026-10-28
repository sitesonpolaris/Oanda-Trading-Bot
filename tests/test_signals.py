from datetime import timedelta

import pandas as pd

from backtest import BacktestConfig
from executioner import OpenTrade
from helpers import SETUP, T0, TOUCH, bars, market, zone
from signals import add_signals, evaluate_zones

PIP = 0.0001
CFG = BacktestConfig()


def now_after(b):
    return b["time"].iat[-1] + timedelta(hours=1)


def verdict(rows, **kw):
    b = bars(rows)
    return evaluate_zones(pd.DataFrame([zone()]), market(b, **kw), CFG, PIP, now_after(b))[0]


def test_fresh_confirmed_trend_aligned_zone_is_a_signal():
    v = verdict(SETUP)
    assert v.status == "signal" and v.signal.order_type == "LIMIT"
    assert (v.signal.entry, round(v.signal.target, 5)) == (1.1000, 1.1038)
    assert v.setup_id == "EUR_USD-H1-202601050900"


def test_touched_and_unconfirmed_and_trend_verdicts():
    assert verdict(SETUP + [TOUCH]).status == "touched"
    assert verdict(SETUP[:4]).status == "waiting_impulse"  # swing high not confirmed yet
    assert verdict(SETUP, h4_trend="down").status == "skipped_trend"


def test_level_on_level_signals_come_first():
    b = bars(SETUP)
    stacked = {"start_time": T0 - pd.Timedelta(hours=5), "kind": "demand", "top": 1.0992, "bottom": 1.0985,
               "available_time": T0 - pd.Timedelta(hours=3), "broken_time": pd.Timestamp.max.tz_localize("UTC")}
    zones = pd.DataFrame([zone(top=1.1000, bottom=1.0990), zone(top=1.0960, bottom=1.0950, start=T0)])
    v = evaluate_zones(zones, market(b, zones=[stacked]), CFG, PIP, now_after(b))
    assert v[0].level_on_level and v[0].zone.top == 1.1000


def test_scale_in_signal_after_bos_since_last_entry():
    b = bars(SETUP)
    trade = OpenTrade("7", "EUR_USD-H1-x", "EUR_USD", 30_000, 1.1000, 1.0989, T0 + timedelta(hours=2), 0.0, 1.1038)
    bos = [""] * 5 + ["bullish", ""]  # BOS at the close of bar 5, after the entry at bar 2
    adds = add_signals([trade], market(b, h1_bos=bos), CFG, now_after(b))
    assert len(adds) == 1 and adds[0].order_type == "MARKET"
    assert (adds[0].stop, adds[0].target, adds[0].entry) == (1.0989, 1.1038, b["close"].iat[-1])
    old_bos = [""] * 1 + ["bullish"] + [""] * 5  # before the entry: no add
    assert add_signals([trade], market(b, h1_bos=old_bos), CFG, now_after(b)) == []


def test_live_momentum_check_uses_zone_formation_time():
    b = bars(SETUP)
    m = market(b)
    m.h4 = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1), T0 + pd.Timedelta(hours=5)],
                         "trend": ["up", "down"]})
    assert evaluate_zones(pd.DataFrame([zone()]), m, CFG, PIP, now_after(b))[0].status == "signal"
