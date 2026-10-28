"""Market structure: swing highs/lows, trend, and break of structure (BOS).

Pure analysis on an OHLC dataframe. Everything is computed without lookahead:
a swing at candle j is only known once the `swing_strength` candles after it
have closed, and each row describes what was known at that candle's close.

Trend, as the course defines its stages: compares the last two confirmed swing
highs and lows, and the slope of the 20- and 40-period moving averages.
    up        (Stage 2) higher high and higher low, both averages rising
    down      (Stage 4) lower high and lower low, both averages falling
    sideways  (Stage 1/3) anything else, or not enough data yet
An average is rising if it's higher than `ma_slope_lookback` candles ago.

BOS: a candle closes above the most recent swing high (bullish) or below the
most recent swing low (bearish). Each swing can be broken once. `last_bos` carries
the direction of the most recent BOS forward. With
`bos_invalidates_trend`, a BOS against the current trend turns it sideways
until the next swing is confirmed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StructureConfig:
    # A swing high is higher than the `swing_strength` candles on each side (a 5-candle fractal at 2).
    swing_strength: int = 2
    bos_invalidates_trend: bool = True
    # Stage filter: both moving averages must slope with the trend. None = swings only.
    stage_mas: tuple | None = (20, 40)
    ma_slope_lookback: int = 3


def find_swings(df, strength=2):
    """Flag swing highs/lows. Ties: strictly beyond the left side, at least equal to the right side."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)
    for j in range(strength, len(df) - strength):
        left, right = slice(j - strength, j), slice(j + 1, j + strength + 1)
        swing_high[j] = h[j] > h[left].max() and h[j] >= h[right].max()
        swing_low[j] = l[j] < l[left].min() and l[j] <= l[right].min()
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low}, index=df.index)


def trend_label(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return "sideways"
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "up"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "down"
    return "sideways"


def market_structure(df, config=None):
    """Per-candle trend, BOS events and the latest confirmed swings, as known at each close."""
    cfg = config or StructureConfig()
    df = df.reset_index(drop=True)
    n = cfg.swing_strength
    swings = find_swings(df, n)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    slopes = []
    for period in cfg.stage_mas or ():
        ma = df["close"].rolling(period).mean()
        slopes.append((ma - ma.shift(cfg.ma_slope_lookback)).to_numpy())

    highs, lows = [], []
    high_broken = low_broken = False
    against = None  # direction of a BOS that went against the trend since the last swing
    last_bos = ""
    rows = []
    for k in range(len(df)):
        j = k - n  # the candle whose swing status is confirmed by this close
        if j >= 0 and swings["swing_high"].iat[j]:
            highs.append(h[j])
            high_broken, against = False, None
        if j >= 0 and swings["swing_low"].iat[j]:
            lows.append(l[j])
            low_broken, against = False, None

        structure = trend_label(highs, lows)
        bos = ""
        if highs and not high_broken and c[k] > highs[-1]:
            bos, high_broken = "bullish", True
        elif lows and not low_broken and c[k] < lows[-1]:
            bos, low_broken = "bearish", True
        last_bos = bos or last_bos
        if (bos == "bearish" and structure == "up") or (bos == "bullish" and structure == "down"):
            against = bos

        trend = "sideways" if cfg.bos_invalidates_trend and against else structure
        if trend == "up" and not all(s[k] > 0 for s in slopes):
            trend = "sideways"
        elif trend == "down" and not all(s[k] < 0 for s in slopes):
            trend = "sideways"
        rows.append({
            "time": df["time"].iat[k],
            "trend": trend,
            "bos": bos,
            "last_bos": last_bos,
            "last_swing_high": highs[-1] if highs else np.nan,
            "last_swing_low": lows[-1] if lows else np.nan,
        })
    out = pd.DataFrame(rows)
    out["swing_high"] = swings["swing_high"].to_numpy()
    out["swing_low"] = swings["swing_low"].to_numpy()
    return out
