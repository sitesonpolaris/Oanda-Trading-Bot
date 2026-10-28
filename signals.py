"""Turn the latest candles into trade signals, using the same rules as the backtest.

Each H1 zone gets a verdict with a reason, so the bot can log why it was or
wasn't traded:

    waiting_breakout  the impulse hasn't broken the prior swing yet
    touched           price already came back to the zone (not fresh)
    no_opposing_zone  no active opposing zone to aim for when the zone formed
    skipped_rr        target < min_reward_risk x stop distance
    skipped_trend     D1 trend now / H4 momentum at zone formation don't allow this direction
    skipped_news      high-impact news within the window
    skipped_erratic   too many big wicks recently
    signal            place a limit order

Scale-ins: an open trade gets a market add signal after an H1 break of
structure in its favour that happened after its latest entry.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_manager import Signal
from strategy import big_wicks, direction_check, level_on_level, news_conflict, opposing_edge, plan_trade
from zones import atr

H1 = np.timedelta64(1, "h")


@dataclass
class ZoneVerdict:
    setup_id: str
    zone: object
    status: str
    detail: str
    signal: Signal | None = None
    level_on_level: bool = False


def setup_id(instrument, zone):
    return f"{instrument}-H1-{zone.start_time:%Y%m%d%H%M}"


def latest(structure, now):
    known = structure[structure["available_time"] <= now]
    return known.iloc[-1] if len(known) else None


def recent_big_wicks(candles, cfg):
    """Big-wick candles among the last `wick_lookback` closed candles (same measure as the backtest)."""
    return int(big_wicks(candles, atr(candles).to_numpy(), cfg).iloc[-cfg.wick_lookback:].sum())


def evaluate_zones(zones, market, cfg, pip, now):
    """A ZoneVerdict per H1 zone as of `now` (market candles must all have closed by then)."""
    bars = market.bars
    h, l = bars["high"].to_numpy(), bars["low"].to_numpy()
    d1 = latest(market.d1, now)
    d1_trend, d1_bos = (d1["trend"], d1["last_bos"]) if d1 is not None else ("sideways", "")
    h1_bos = {"long": _last_bos_time(market.h1, "bullish", now), "short": _last_bos_time(market.h1, "bearish", now)}
    big_wicks = recent_big_wicks(bars, cfg)
    news = news_conflict(market.news, market.instrument, now, now + H1, cfg)

    verdicts = []
    for z in zones.itertuples(index=False):
        if z.type not in cfg.trade_types:
            continue
        sid = setup_id(market.instrument, z)
        long = z.kind == "demand"
        lol = level_on_level(z, market.all_zones, z.atr_pips * pip, cfg)
        tag = " [level-on-level]" if lol else ""
        legout = int(bars["time"].searchsorted(z.legout_time))
        near = z.top if long else z.bottom
        if (l[legout + 1:] <= near).any() if long else (h[legout + 1:] >= near).any():
            verdicts.append(ZoneVerdict(sid, z, "touched", f"{z.type}{tag}: price already returned to the zone",
                                        level_on_level=lol))
            continue
        if pd.isna(z.breakout_time):
            verdicts.append(ZoneVerdict(sid, z, "waiting_breakout",
                                        f"{z.type}{tag}: waiting for a close beyond {z.breakout_level:.5f}",
                                        level_on_level=lol))
            continue
        valid_from = z.breakout_time + H1
        edge = opposing_edge(long, near, valid_from, market.all_zones)  # as of the breakout
        if edge is None:
            verdicts.append(ZoneVerdict(sid, z, "no_opposing_zone", f"{z.type}{tag}: no opposing zone to aim for",
                                        level_on_level=lol))
            continue

        entry, stop, target = plan_trade(z, edge, cfg, pip)
        risk, reward = abs(entry - stop), (target - entry) * (1 if long else -1)
        plan = f"{z.type}{tag} entry {entry:.5f} SL {stop:.5f} TP {target:.5f} RR {reward / risk:.2f}"
        h4 = latest(market.h4, valid_from)  # momentum when the zone became valid
        ok, why = direction_check(long, d1_trend, d1_bos, h4["trend"] if h4 is not None else "sideways", cfg)
        if reward < cfg.min_reward_risk * risk:
            status, detail = "skipped_rr", f"{plan}: below {cfg.min_reward_risk}R"
        elif not ok:
            status, detail = "skipped_trend", f"{plan}: {why}"
        elif news is not None:
            status, detail = "skipped_news", f"{plan}: {news['currency']} news at {news['time']:%Y-%m-%d %H:%M}"
        elif big_wicks >= cfg.max_big_wicks:
            status, detail = "skipped_erratic", f"{plan}: {big_wicks} big wicks in last {cfg.wick_lookback} candles"
        else:
            side = "long" if long else "short"
            sig = Signal(market.instrument, side, entry, stop, target, sid, now, bos_time=h1_bos[side])
            verdicts.append(ZoneVerdict(sid, z, "signal", f"{plan}, {why}", sig, lol))
            continue
        verdicts.append(ZoneVerdict(sid, z, status, detail, level_on_level=lol))
    # Level-on-level setups first: they get the open-position slots if there aren't enough.
    return sorted(verdicts, key=lambda v: not v.level_on_level)


def add_signals(trades, market, cfg, now):
    """Market scale-in signals for open bot trades with a favourable H1 BOS since their latest entry."""
    out = []
    by_setup = {}
    for t in trades:
        if t.setup_id != "manual" and t.target is not None:
            by_setup.setdefault(t.setup_id, []).append(t)
    last_close = market.bars["close"].iat[-1]
    for sid, entries in by_setup.items():
        first = min(entries, key=lambda t: t.opened)
        long = first.units > 0
        bos = _last_bos_time(market.h1, "bullish" if long else "bearish", now)
        if len(entries) > cfg.max_adds or bos is None or bos <= max(t.opened for t in entries):
            continue
        out.append(Signal(market.instrument, "long" if long else "short", last_close, first.stop, first.target,
                          sid, now, bos_time=bos, order_type="MARKET"))
    return out


def _last_bos_time(structure, direction, now):
    hits = structure[(structure["bos"] == direction) & (structure["available_time"] <= now)]
    return hits["available_time"].iat[-1].to_pydatetime() if len(hits) else None
