"""Hand-built H1 market for strategy, backtest and signal tests.

Demand zone 1.0990-1.1000 (ATR 10 pips), leg-out at bar 2. Bar 3 makes the impulse
high (1.1050), confirmed as a swing at the close of bar 5, so the order goes in at
bar 6. Plan: entry 1.1000, stop 1.0989, target 1.0990 + 0.8 x 60 pips = 1.1038
(3.45R). Bar 7 touches the zone, bar 8 reaches the target.
"""

import pandas as pd

from backtest import Market

T0 = pd.Timestamp("2026-01-05 08:00", tz="UTC")
LEAD = (1.1010, 1.1015, 1.1005, 1.1012)
LEGOUT = (1.1000, 1.1030, 1.0998, 1.1028)
PEAK = (1.1028, 1.1050, 1.1025, 1.1045)
DIP1 = (1.1045, 1.1046, 1.1030, 1.1035)
DIP2 = (1.1035, 1.1040, 1.1020, 1.1025)
AWAY = (1.1025, 1.1030, 1.1015, 1.1020)
TOUCH = (1.1010, 1.1012, 1.0995, 1.1005)
WIN = (1.1010, 1.1045, 1.1005, 1.1040)
LOSS = (1.1000, 1.1002, 1.0980, 1.0985)
SETUP = [LEAD, LEAD, LEGOUT, PEAK, DIP1, DIP2, AWAY]


def bars(rows):
    times = pd.date_range(T0, periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(time=times)


def zone(kind="demand", top=1.1000, bottom=1.0990, start=None, legout_bar=2, **extra):
    start = start if start is not None else T0 + pd.Timedelta(hours=legout_bar - 1)
    return {"start_time": start, "legout_time": T0 + pd.Timedelta(hours=legout_bar), "type": "RBR" if
            kind == "demand" else "DBD", "kind": kind, "top": top, "bottom": bottom, "base_candles": 1,
            "impulse_pips": 30, "impulse_ratio": 3.0, "impulse_atr": 3.0, "atr_pips": 10, **extra}


def lifecycle_zone(kind, top, bottom, available=T0 - pd.Timedelta(hours=1), broken=pd.Timestamp.max.tz_localize("UTC"),
                   start=None):
    return {"start_time": start or available - pd.Timedelta(hours=2), "kind": kind, "top": top,
            "bottom": bottom, "available_time": available, "broken_time": broken}


def market(b, d1_trend="up", d1_bos="", h4_trend="up", h1_bos=None, zones=(), news=None):
    state = pd.DataFrame({"available_time": [T0 - pd.Timedelta(days=1)]})
    return Market(
        bars=b,
        d1=state.assign(trend=d1_trend, last_bos=d1_bos, last_swing_high=1.11, last_swing_low=1.09, bos=""),
        h4=state.assign(trend=h4_trend, last_bos="", last_swing_high=1.11, last_swing_low=1.09, bos=""),
        h1=pd.DataFrame({"available_time": b["time"] + pd.Timedelta(hours=1),
                         "bos": h1_bos or [""] * len(b)}),
        all_zones=pd.DataFrame(list(zones), columns=["start_time", "kind", "top", "bottom", "available_time",
                                                     "broken_time"]),
        news=news,
        instrument="EUR_USD",
    )
