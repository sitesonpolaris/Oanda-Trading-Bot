"""Supply/demand zone detection on saved candle data.

Pure analysis: reads OHLC dataframes, returns zones. No API access, no orders.

A zone is a base of 1-5 basing candles followed by an impulse candle (the
leg-out): the first non-basing candle that closes outside the previous candle's
range. A close above the previous high is a bullish impulse (demand zone), a
close below the previous low a bearish one (supply zone). Optionally the impulse
must also be at least `legout_multiple` times the average range of the candles
before it. The candle before the base is the leg-in. Zone types:

    RBR  rally-base-rally  (demand)     DBR  drop-base-rally  (demand)
    DBD  drop-base-drop    (supply)     RBD  rally-base-drop  (supply)

Zone edges: the far edge (where the stop goes) is always the base's wick
extreme. The near edge (entry) is the wick for a single-candle base and the
body for a multi-candle base: top of the bodies for demand, bottom for supply.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ZONE_COLUMNS = [
    "start_time", "legout_time", "type", "kind", "top", "bottom", "base_candles",
    "impulse_pips", "impulse_ratio", "impulse_atr", "atr_pips",
]


@dataclass
class ZoneConfig:
    # Optional strength filter on the impulse: range >= legout_multiple x mean range of the
    # previous legout_lookback candles. None = off (any close outside the previous range counts).
    legout_multiple: float | None = None
    legout_lookback: int = 10
    # Base length. A run of basing candles longer than max_base_candles is not a base.
    min_base_candles: int = 1
    max_base_candles: int = 5
    # "Small" base candle: body <= base_max_body_ratio of its range and range <= base_max_atr x ATR.
    base_max_body_ratio: float = 0.5
    base_max_atr: float = 1.0
    atr_period: int = 14


def pip_size(instrument):
    return 0.01 if instrument.endswith("JPY") else 0.0001


def atr(df, period=14):
    """Average True Range with Wilder's smoothing."""
    prev_close = df["close"].shift()
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def detect_zones(df, config=None, pip=0.0001):
    """Return a dataframe of zones (one per impulse candle), oldest first."""
    cfg = config or ZoneConfig()
    df = df.reset_index(drop=True)
    o, h, l, c = (df[col].to_numpy() for col in ("open", "high", "low", "close"))
    rng = h - l
    body = abs(c - o)
    body_ratio = np.divide(body, rng, out=np.zeros_like(rng), where=rng > 0)  # zero-range candles -> 0
    atr_vals = atr(df, cfg.atr_period).to_numpy()
    avg_prev_range = pd.Series(rng).rolling(cfg.legout_lookback).mean().shift().to_numpy()
    small = (body_ratio <= cfg.base_max_body_ratio) & (rng <= cfg.base_max_atr * atr_vals)

    zones = []
    first = max(cfg.legout_lookback, cfg.atr_period) + cfg.max_base_candles + 1
    for i in range(first, len(df)):
        if c[i] > h[i - 1]:
            leg_out_dir = "R"
        elif c[i] < l[i - 1]:
            leg_out_dir = "D"
        else:
            continue  # closed inside the previous candle's range: not an impulse
        if small[i]:
            continue  # a basing candle can't be the impulse
        if cfg.legout_multiple is not None and rng[i] < cfg.legout_multiple * avg_prev_range[i]:
            continue

        n = 0
        while n < cfg.max_base_candles and small[i - 1 - n]:
            n += 1
        leg_in = i - n - 1
        if n < cfg.min_base_candles or small[leg_in]:
            continue  # no base, or the consolidation runs longer than max_base_candles

        base = slice(i - n, i)
        leg_in_dir = "R" if c[leg_in] > o[leg_in] else "D"
        # Far edge (stop side) is always the base's wick extreme; the near (entry) edge is the
        # wick for a single-candle base and the body for a multi-candle base.
        top, bottom = h[base].max(), l[base].min()
        if n > 1 and leg_out_dir == "R":
            top = max(o[base].max(), c[base].max())
        elif n > 1:
            bottom = min(o[base].min(), c[base].min())
        zones.append({
            "start_time": df["time"].iloc[i - n],
            "legout_time": df["time"].iloc[i],
            "type": f"{leg_in_dir}B{leg_out_dir}",
            "kind": "demand" if leg_out_dir == "R" else "supply",
            "top": top,
            "bottom": bottom,
            "base_candles": n,
            "impulse_pips": round(rng[i] / pip, 1),
            "impulse_ratio": round(rng[i] / avg_prev_range[i], 2),
            "impulse_atr": round(rng[i] / atr_vals[i - 1], 2),
            "atr_pips": round(atr_vals[i - 1] / pip, 1),
        })
    return pd.DataFrame(zones, columns=ZONE_COLUMNS)
