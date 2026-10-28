"""Strategy rules from the spec, shared by the backtest and the live bot.

Timeframes: trend on D1, momentum on H4, zones and entries on H1.

For a demand zone (supply is the mirror image):
  1. Impulse: the move out of the zone is measured from its origin (the zone's
     far edge) to the highest high it reaches. That high is only treated as the
     extreme once an H1 swing high confirms it; the limit order goes in then.
  2. Plan: entry at the zone's near edge, stop below the far edge minus
     `atr_buffer` x ATR, target at `target_fraction` of the impulse from the origin.
  3. Filters, checked when price comes back to the zone:
     - reward:risk at least `min_reward_risk`
     - direction: H4 momentum was up when the zone formed (the leg-out candle
       closed), and the D1 trend is up when price returns; or, countertrend,
       the D1 trend has been neutralised by a bullish D1 break of structure.
       Momentum is checked at formation, not at the return, because price must
       pull back against the trade to reach the zone, which turns momentum.
     - no high-impact news for either currency inside the news window
     - market not erratic: fewer than `max_big_wicks` of the last `wick_lookback`
       H1 candles have wicks of at least `wick_atr` x ATR
     - room: the nearest active opposing (supply) zone above entry is at least
       `min_room_rr` x stop distance away
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
    atr_buffer: float = 0.1
    target_fraction: float = 0.8
    min_reward_risk: float = 2.0
    min_room_rr: float = 2.0
    impulse_swing_strength: int = 2
    wick_lookback: int = 20
    wick_atr: float = 1.0
    max_big_wicks: int = 5
    news_before_min: int = 30
    news_after_min: int = 30
    allow_countertrend_with_bos: bool = True
    initial_fraction: float = 1 / 3
    add_fraction: float = 1 / 3
    max_adds: int = 2
    lol_tolerance_atr: float = 0.5


def plan_trade(zone, extreme, cfg, pip):
    """Entry, stop and target for a zone whose impulse reached `extreme`."""
    buffer = cfg.atr_buffer * zone.atr_pips * pip
    if zone.kind == "demand":
        origin = zone.bottom
        return zone.top, zone.bottom - buffer, origin + cfg.target_fraction * (extreme - origin)
    origin = zone.top
    return zone.bottom, zone.top + buffer, origin - cfg.target_fraction * (origin - extreme)


def impulse_extreme(kind, highs, lows, swing_high, swing_low, legout_idx, strength):
    """(index of the candle whose close confirms the extreme, extreme price), or None if not yet confirmed."""
    flags = swing_high if kind == "demand" else swing_low
    hits = np.flatnonzero(flags[legout_idx:])
    if not len(hits):
        return None
    j = legout_idx + hits[0]
    if kind == "demand":
        return j + strength, highs[legout_idx:j + 1].max()
    return j + strength, lows[legout_idx:j + 1].min()


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
    """Add available_time (leg-out closed) and broken_time (first close beyond the far edge)."""
    out = zones.copy()
    out["available_time"] = out["legout_time"] + tf_delta
    closes = candles["close"].to_numpy()
    broken = []
    for z in out.itertuples(index=False):
        start = int(candles["time"].searchsorted(z.available_time))
        beyond = closes[start:] < z.bottom if z.kind == "demand" else closes[start:] > z.top
        hit = np.flatnonzero(beyond)
        broken.append(candles["time"].iat[start + hit[0]] + tf_delta if len(hit) else NEVER)
    out["broken_time"] = pd.to_datetime(broken, utc=True) if broken else pd.Series(dtype="datetime64[ns, UTC]")
    return out


def active_at(zones, t):
    return zones[(zones["available_time"] <= t) & (zones["broken_time"] > t)]


def room_to_opposing(long, entry, t, all_zones):
    """Distance from entry to the nearest active opposing zone in the trade's path (inf if none)."""
    opp = active_at(all_zones, t)
    if long:
        opp = opp[(opp["kind"] == "supply") & (opp["top"] > entry)]
        gaps = (opp["bottom"] - entry).clip(lower=0)
    else:
        opp = opp[(opp["kind"] == "demand") & (opp["bottom"] < entry)]
        gaps = (entry - opp["top"]).clip(lower=0)
    return gaps.min() if len(gaps) else np.inf


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
