"""Supply/demand zone detection on saved candle data, following the TWP course
("Supply and Demand - Steps to Drawing Zones").

Pure analysis: reads OHLC dataframes, returns zones. No API access, no orders.

A zone is a move-in candle, 1-5 basing candles, and a move-out candle:

  - Basing candle: body <= 50% of its range (wick to wick), or a close inside
    the previous candle's range.
  - Rally candle (move-in or move-out): any other candle that closes above the
    previous candle's high. A drop candle closes below the previous candle's low.
    (The course's optional "close in the top 25% of the range" check is
    `close_zone`, off by default.)
  - Move-out: also at least 2x the range of the biggest basing candle and taller
    than the move-in candle.
  - More than 5 basing candles disqualifies the zone.
  - Breakout: the impulse must make a new swing high (demand) or swing low
    (supply). The reference is the most recent confirmed H1 swing high above the
    zone that existed before the base formed; the zone becomes valid when a
    candle closes above it. If price comes back to the zone's entry edge before
    that, the zone is void. (Mirror image for supply.)
  - Move away: price must travel at least 2x the zone's height away from the
    zone (entry edge to the move's extreme) before coming back.
  - The zone is valid (`valid_time`) once both the breakout and the move away
    have happened. If price returns to the entry edge before then, it's void.

Zone types (the move-in and move-out directions):

    RBR  rally-base-rally  (demand)     DBR  drop-base-rally  (demand)
    DBD  drop-base-drop    (supply)     RBD  rally-base-drop  (supply)

The course trades RBR and DBD; the strategy decides which types to trade.

Zone edges: one basing candle is drawn wick to wick. With 2-5, a demand zone
runs from the highest candle body (entry) to the lowest wick (stop side); a
supply zone from the lowest candle body (entry) to the highest wick.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from structure import find_swings

ZONE_COLUMNS = [
    "start_time", "legout_time", "breakout_time", "valid_time", "breakout_level", "type", "kind", "top", "bottom",
    "base_candles", "impulse_pips", "impulse_ratio", "impulse_vs_base", "impulse_atr", "atr_pips",
]


@dataclass
class ZoneConfig:
    # Basing candle: body <= base_max_body_ratio of its range, or (if inside_close_is_base) a close
    # inside the previous candle's range.
    base_max_body_ratio: float = 0.5
    inside_close_is_base: bool = True
    min_base_candles: int = 1
    max_base_candles: int = 5
    # Move-in / move-out candles close in the top (rally) or bottom (drop) close_zone of their range
    # (None = off: closing beyond the previous candle is enough).
    close_zone: float | None = None
    # Move-out range >= min_impulse_vs_base x the biggest basing candle (None = off), and (if
    # move_out_taller_than_move_in) taller than the move-in candle.
    min_impulse_vs_base: float | None = 2.0
    move_out_taller_than_move_in: bool = True
    # Breakout: the impulse must close beyond the prior swing before price returns to the zone.
    require_breakout: bool = True
    # Move away: price must travel min_move_away x the zone height from the entry edge (None = off).
    min_move_away: float | None = 2.0
    swing_strength: int = 2
    # Optional extra strength filter: move-out range >= legout_multiple x mean range of the previous
    # legout_lookback candles. None = off.
    legout_multiple: float | None = None
    legout_lookback: int = 10
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


def candle_direction(h, l, c, i, close_zone):
    """'R' for a rally candle, 'D' for a drop candle, '' otherwise (see module docstring)."""
    rng = h[i] - l[i]
    if rng <= 0 or i == 0:
        return ""
    zone = rng if close_zone is None else close_zone * rng
    if c[i] >= h[i] - zone and c[i] > h[i - 1]:
        return "R"
    if c[i] <= l[i] + zone and c[i] < l[i - 1]:
        return "D"
    return ""


def reference_swing(levels, swing_idx, base_start, strength, beyond, demand):
    """Most recent swing (confirmed before the base) beyond the zone: above it for demand, below for supply."""
    confirmed = swing_idx[swing_idx + strength < base_start]
    for j in confirmed[::-1]:
        if (levels[j] > beyond) if demand else (levels[j] < beyond):
            return levels[j]
    return None


def detect_zones(df, config=None, pip=0.0001):
    """Return a dataframe of valid zones (one per move-out candle), oldest first.

    `breakout_time` is the open time of the candle whose close made the breakout, and `valid_time`
    the candle by which both the breakout and the move away had happened; NaT if not yet in this
    data (the zone isn't tradable yet).
    """
    cfg = config or ZoneConfig()
    df = df.reset_index(drop=True)
    o, h, l, c = (df[col].to_numpy() for col in ("open", "high", "low", "close"))
    rng = h - l
    body_ratio = np.divide(abs(c - o), rng, out=np.zeros_like(rng), where=rng > 0)
    basing = body_ratio <= cfg.base_max_body_ratio
    if cfg.inside_close_is_base:
        prev_h, prev_l = np.roll(h, 1), np.roll(l, 1)
        basing |= (c <= prev_h) & (c >= prev_l) & (np.arange(len(df)) > 0)
    atr_vals = atr(df, cfg.atr_period).to_numpy()
    avg_prev_range = pd.Series(rng).rolling(cfg.legout_lookback).mean().shift().to_numpy()
    swings = find_swings(df, cfg.swing_strength)
    swing_highs = np.flatnonzero(swings["swing_high"].to_numpy())
    swing_lows = np.flatnonzero(swings["swing_low"].to_numpy())

    zones = []
    first = max(cfg.legout_lookback, cfg.atr_period) + cfg.max_base_candles + 2
    for i in range(first, len(df)):
        out_dir = "" if basing[i] else candle_direction(h, l, c, i, cfg.close_zone)
        if not out_dir:
            continue

        n = 0
        while n < cfg.max_base_candles and basing[i - 1 - n]:
            n += 1
        leg_in = i - n - 1
        if n < cfg.min_base_candles or basing[leg_in]:
            continue  # no base, or more than max_base_candles basing candles
        in_dir = candle_direction(h, l, c, leg_in, cfg.close_zone)
        if not in_dir:
            continue  # the candle before the base isn't a move-in candle

        base = slice(i - n, i)
        biggest_base = rng[base].max()
        if cfg.min_impulse_vs_base is not None and rng[i] < cfg.min_impulse_vs_base * biggest_base:
            continue
        if cfg.move_out_taller_than_move_in and rng[i] <= rng[leg_in]:
            continue  # the move-out must be taller than the move-in
        if cfg.legout_multiple is not None and rng[i] < cfg.legout_multiple * avg_prev_range[i]:
            continue

        demand = out_dir == "R"
        top, bottom = h[base].max(), l[base].min()
        if n > 1 and demand:
            top = max(o[base].max(), c[base].max())
        elif n > 1:
            bottom = min(o[base].min(), c[base].min())

        level, k_break = np.nan, i
        if cfg.require_breakout:
            ref = reference_swing(h if demand else l, swing_highs if demand else swing_lows, i - n,
                                  cfg.swing_strength, top if demand else bottom, demand)
            if ref is None:
                continue  # nothing to break out of
            level = ref
            beyond = np.flatnonzero(c[i:] > ref if demand else c[i:] < ref)
            k_break = i + beyond[0] if len(beyond) else len(df)
        k_move = i
        if cfg.min_move_away is not None:
            travel = np.maximum.accumulate(h[i:]) - top if demand else bottom - np.minimum.accumulate(l[i:])
            far = np.flatnonzero(travel >= cfg.min_move_away * (top - bottom))
            k_move = i + far[0] if len(far) else len(df)
        v = max(k_break, k_move)
        back = l[i + 1:v + 1] <= top if demand else h[i + 1:v + 1] >= bottom
        if back.any():
            continue  # price returned to the zone before it became valid: void
        breakout_time = df["time"].iloc[k_break] if k_break < len(df) else pd.NaT
        valid_time = df["time"].iloc[v] if v < len(df) else pd.NaT

        zones.append({
            "start_time": df["time"].iloc[i - n],
            "legout_time": df["time"].iloc[i],
            "breakout_time": breakout_time,
            "valid_time": valid_time,
            "breakout_level": level,
            "type": f"{in_dir}B{out_dir}",
            "kind": "demand" if demand else "supply",
            "top": top,
            "bottom": bottom,
            "base_candles": n,
            "impulse_pips": round(rng[i] / pip, 1),
            "impulse_ratio": round(rng[i] / avg_prev_range[i], 2),
            "impulse_vs_base": round(rng[i] / biggest_base, 2) if biggest_base > 0 else float("inf"),
            "impulse_atr": round(rng[i] / atr_vals[i - 1], 2),
            "atr_pips": round(atr_vals[i - 1] / pip, 1),
        })
    out = pd.DataFrame(zones, columns=ZONE_COLUMNS)
    out["breakout_time"] = pd.to_datetime(out["breakout_time"], utc=True)
    out["valid_time"] = pd.to_datetime(out["valid_time"], utc=True)
    return out
