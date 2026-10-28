"""Supply/demand zone detection on saved candle data.

Pure analysis: reads OHLC dataframes, returns zones. No API access, no orders.

A zone is a base of 1-3 small candles followed by a leg-out candle whose range
is at least `legout_multiple` times the average range of the candles before it.
The candle before the base is the leg-in. Zone types:

    RBR  rally-base-rally  (demand)     DBR  drop-base-rally  (demand)
    DBD  drop-base-drop    (supply)     RBD  rally-base-drop  (supply)

Zone edges: a single-candle base uses its wick high/low; a multi-candle base
uses the highest and lowest candle body (open/close) across the base.
"""

from dataclasses import dataclass

import pandas as pd

ZONE_COLUMNS = [
    "start_time", "legout_time", "type", "kind", "top", "bottom", "base_candles",
    "impulse_pips", "impulse_ratio", "impulse_atr", "atr_pips",
]


@dataclass
class ZoneConfig:
    # Leg-out: range >= legout_multiple x mean range of the previous legout_lookback candles.
    legout_multiple: float = 2.5
    legout_lookback: int = 10
    # Leg-out must be a decisive candle: body at least this fraction of its range.
    legout_min_body_ratio: float = 0.5
    # Base length. A run of small candles longer than max_base_candles is not a base.
    min_base_candles: int = 1
    max_base_candles: int = 3
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
    """Return a dataframe of zones (one per qualifying leg-out), oldest first."""
    cfg = config or ZoneConfig()
    df = df.reset_index(drop=True)
    o, h, l, c = (df[col].to_numpy() for col in ("open", "high", "low", "close"))
    rng = h - l
    body = abs(c - o)
    body_ratio = pd.Series(body / rng).fillna(0).to_numpy()
    atr_vals = atr(df, cfg.atr_period).to_numpy()
    avg_prev_range = pd.Series(rng).rolling(cfg.legout_lookback).mean().shift().to_numpy()
    small = (body_ratio <= cfg.base_max_body_ratio) & (rng <= cfg.base_max_atr * atr_vals)

    zones = []
    first = max(cfg.legout_lookback, cfg.atr_period) + cfg.max_base_candles + 1
    for i in range(first, len(df)):
        if not (rng[i] >= cfg.legout_multiple * avg_prev_range[i]
                and body_ratio[i] >= cfg.legout_min_body_ratio):
            continue

        n = 0
        while n < cfg.max_base_candles and small[i - 1 - n]:
            n += 1
        leg_in = i - n - 1
        if n < cfg.min_base_candles or small[leg_in]:
            continue  # no base, or the consolidation runs longer than max_base_candles

        base = slice(i - n, i)
        if n == 1:
            top, bottom = h[i - 1], l[i - 1]
        else:
            top = max(o[base].max(), c[base].max())
            bottom = min(o[base].min(), c[base].min())

        leg_in_dir = "R" if c[leg_in] > o[leg_in] else "D"
        leg_out_dir = "R" if c[i] > o[i] else "D"
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
