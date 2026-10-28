"""Strategy rules, shared by the backtest and the live bot. They follow the TWP
Market Structure course where it's specific, and the strategy spec elsewhere.

Timeframes: trend (stage) on D1, momentum on H4, zones and entries on H1.

For a demand zone (supply is the mirror image):
  1. Zones: valid course zones (zones.py), all four types by default.
  2. The limit order goes in when the zone becomes valid (breakout and move away
     done):
     - entry at the zone's entry edge (top of a demand zone)
     - stop below the zone's far edge by `atr_buffer` x the daily ATR
       (course: 10% of the daily ATR for trades on 30-minute charts and above)
  3. When price comes back to the zone:
     - target: the nearer of the prior pivot high (most recent confirmed H1
       swing high above the entry) and the bottom of the nearest active
       supply zone above (course: "target at prior pivot high/low or opposing
       zone"). Neither, no trade.
     - reward:risk at least `min_reward_risk` (2:1)
     - direction: D1 trend up (Stage 2) and H4 momentum up when the zone became
       valid; all timeframes must align (countertrend off by default)
     - no high-impact news for either currency inside the news window
     - market not erratic: fewer than `max_big_wicks` of the last `wick_lookback`
       H1 candles have wicks of at least `wick_atr` x ATR
  4. Sizing: first entry `initial_fraction` of the per-trade risk; up to `max_adds`
     more entries of `add_fraction` each, at market, after H1 breaks of structure
     in the trade's favour. Same stop and target for every entry; stops are not moved.
  5. Level-on-level: a zone stacked on another same-side zone (H1 or H4) is flagged
     as the highest-priority setup.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NEVER = pd.Timestamp.max.tz_localize("UTC")


@dataclass
class StrategyConfig:
    trade_types: tuple = ("RBR", "DBR", "DBD", "RBD")
    atr_buffer: float = 0.1          # stop buffer as a fraction of the daily ATR
    min_reward_risk: float = 2.0
    wick_lookback: int = 20
    wick_atr: float = 1.0
    max_big_wicks: int = 5
    news_before_min: int = 30
    news_after_min: int = 30
    allow_countertrend_with_bos: bool = False
    initial_fraction: float = 1 / 3
    add_fraction: float = 1 / 3
    max_adds: int = 2
    lol_tolerance_atr: float = 0.5


def entry_stop(zone, daily_atr, cfg):
    """Entry at the zone's entry edge, stop beyond the far edge by `atr_buffer` x the daily ATR."""
    buffer = cfg.atr_buffer * daily_atr
    if zone.kind == "demand":
        return zone.top, zone.bottom - buffer
    return zone.bottom, zone.top + buffer


def prior_pivot(long, entry, t, pivots):
    """Most recent swing high above the entry (long) / swing low below it (short) confirmed by t."""
    known = pivots[pivots["confirmed_time"] <= t]
    if long:
        levels = known.loc[(known["kind"] == "high") & (known["level"] > entry), "level"]
    else:
        levels = known.loc[(known["kind"] == "low") & (known["level"] < entry), "level"]
    return levels.iloc[-1] if len(levels) else None


def trade_target(long, entry, t, all_zones, pivots):
    """(target, source): the nearer of the prior pivot and the opposing zone's edge, or (None, '')."""
    candidates = [(p, "pivot") for p in [prior_pivot(long, entry, t, pivots)] if p is not None]
    candidates += [(e, "zone") for e in [opposing_edge(long, entry, t, all_zones)] if e is not None]
    if not candidates:
        return None, ""
    return min(candidates) if long else max(candidates)


def h1_pivots(structure, tf_delta, strength):
    """Swing levels with the time they were confirmed (close of the `strength`-th candle after)."""
    times = structure["time"]
    rows = []
    for kind, flag, col in (("high", "swing_high", "last_swing_high"), ("low", "swing_low", "last_swing_low")):
        for j in structure.index[structure[flag]]:
            if j + strength < len(structure):
                rows.append({"kind": kind, "level": structure[col].iat[j + strength],
                             "confirmed_time": times.iat[j + strength] + tf_delta})
    out = pd.DataFrame(rows, columns=["kind", "level", "confirmed_time"])
    return out.sort_values("confirmed_time").reset_index(drop=True)


def opposing_edge(long, entry, t, all_zones):
    """Near edge of the nearest active opposing zone beyond the entry at time t, or None.

    Long: the bottom of the lowest supply zone above the entry.
    Short: the top of the highest demand zone below the entry.
    """
    opp = active_at(all_zones, t)
    if long:
        edges = opp.loc[(opp["kind"] == "supply") & (opp["bottom"] > entry), "bottom"]
        return edges.min() if len(edges) else None
    edges = opp.loc[(opp["kind"] == "demand") & (opp["top"] < entry), "top"]
    return edges.max() if len(edges) else None


def direction_check(long, d1_trend, d1_last_bos, h4_at_formation, cfg):
    """(allowed, reason) given the D1 trend now and the H4 momentum when the zone formed."""
    want, bos_want = ("up", "bullish") if long else ("down", "bearish")
    if h4_at_formation != want:
        return False, f"H4 momentum {h4_at_formation} when zone formed"
    if d1_trend == want:
        return True, "with D1 trend"
    if cfg.allow_countertrend_with_bos and d1_trend == "sideways" and d1_last_bos == bos_want:
        return True, f"countertrend after {bos_want} D1 BOS"
    return False, f"D1 trend {d1_trend}" + (f", last D1 BOS {d1_last_bos}" if d1_last_bos else "")


def zone_lifecycle(zones, candles, tf_delta):
    """Add available_time (the zone became valid; NaT if not yet) and broken_time (first close
    beyond the far edge after that)."""
    out = zones.copy()
    out["available_time"] = out["valid_time"] + tf_delta
    closes = candles["close"].to_numpy()
    broken = []
    for z in out.itertuples(index=False):
        if pd.isna(z.available_time):
            broken.append(NEVER)
            continue
        start = int(candles["time"].searchsorted(z.available_time))
        beyond = closes[start:] < z.bottom if z.kind == "demand" else closes[start:] > z.top
        hit = np.flatnonzero(beyond)
        broken.append(candles["time"].iat[start + hit[0]] + tf_delta if len(hit) else NEVER)
    out["broken_time"] = pd.to_datetime(broken, utc=True) if broken else pd.Series(dtype="datetime64[ns, UTC]")
    return out


def active_at(zones, t):
    return zones[(zones["available_time"] <= t) & (zones["broken_time"] > t)]


def big_wicks(candles, atr_values, cfg):
    """1 for each candle whose total wick is at least `wick_atr` x ATR, else 0."""
    wick = (candles["high"] - candles["low"]) - (candles["close"] - candles["open"]).abs()
    return (wick >= cfg.wick_atr * pd.Series(atr_values, index=candles.index)).astype(int)


def big_wick_counts(candles, atr_values, cfg):
    """For each candle index i: how many of the `wick_lookback` candles before i had big wicks."""
    big = big_wicks(candles, atr_values, cfg)
    return big.rolling(cfg.wick_lookback, min_periods=1).sum().shift(fill_value=0).to_numpy()


def level_on_level(zone, all_zones, atr_price, cfg):
    """True if another same-side zone that was active when this one formed is stacked on it."""
    tol = cfg.lol_tolerance_atr * atr_price
    others = active_at(all_zones, zone.legout_time)
    others = others[(others["kind"] == zone.kind) & (others["start_time"] != zone.start_time)]
    stacked = (others["bottom"] <= zone.top + tol) & (others["top"] >= zone.bottom - tol)
    return bool(stacked.any())


def load_news(path):
    """High-impact events from a CSV with columns time (UTC), currency, impact. None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    events = pd.read_csv(path, parse_dates=["time"])
    events["time"] = pd.to_datetime(events["time"], utc=True)
    return events[events["impact"].str.lower() == "high"].reset_index(drop=True)


def news_conflict(events, instrument, start, end, cfg):
    """First high-impact event for either currency within the window around [start, end], or None."""
    if events is None or events.empty:
        return None
    currencies = instrument.split("_")
    lo = start - pd.Timedelta(minutes=cfg.news_after_min)
    hi = end + pd.Timedelta(minutes=cfg.news_before_min)
    hits = events[events["currency"].isin(currencies) & (events["time"] >= lo) & (events["time"] <= hi)]
    return hits.iloc[0] if len(hits) else None
