"""Turn the latest candles into trade signals, using the same rules as the backtest.

A zone produces a signal when it is fresh (price has not come back to its entry
since the leg-out), the H4 trend agrees with it, and the target is at least
`min_reward_risk` times the stop distance. Every zone's outcome is returned
with a reason, so the bot can log why each one was or wasn't traded.
"""

from dataclasses import dataclass

from backtest import TIMEFRAMES, plan_trade
from risk_manager import Signal


@dataclass
class ZoneVerdict:
    setup_id: str
    zone: object
    status: str        # "signal", "touched", "skipped_trend", "skipped_rr"
    detail: str
    signal: Signal | None = None


def setup_id(instrument, zone_tf, zone):
    return f"{instrument}-{zone_tf}-{zone.start_time:%Y%m%d%H%M}"


def latest_bos_time(structure, direction):
    hits = structure.loc[structure["bos"] == direction, "available_time"]
    return hits.iloc[-1].to_pydatetime() if len(hits) else None


def evaluate_zones(zones, candles, structure, cfg, instrument, zone_tf, pip, now):
    """Return a ZoneVerdict per zone. `structure` needs trend, bos and available_time columns."""
    known = structure[structure["available_time"] <= now]
    trend = known["trend"].iloc[-1] if len(known) else "sideways"
    bos = {"long": latest_bos_time(known, "bullish"), "short": latest_bos_time(known, "bearish")}
    verdicts = []
    for z in zones.itertuples(index=False):
        sid = setup_id(instrument, zone_tf, z)
        entry, stop, target = plan_trade(z, cfg, pip)
        long = z.kind == "demand"
        after = candles[candles["time"] >= z.legout_time + TIMEFRAMES[zone_tf]]
        touched = (after["low"] <= entry).any() if long else (after["high"] >= entry).any()
        risk, reward = abs(entry - stop), abs(target - entry)
        plan = f"{z.type} entry {entry:.5f} SL {stop:.5f} TP {target:.5f} RR {reward / risk:.2f}"
        if touched:
            verdicts.append(ZoneVerdict(sid, z, "touched", f"{plan}: price already returned to the zone"))
        elif reward < cfg.min_reward_risk * risk:
            verdicts.append(ZoneVerdict(sid, z, "skipped_rr", f"{plan}: below {cfg.min_reward_risk}R"))
        elif trend != ("up" if long else "down"):
            verdicts.append(ZoneVerdict(sid, z, "skipped_trend", f"{plan}: H4 trend is {trend}"))
        else:
            side = "long" if long else "short"
            sig = Signal(instrument, side, entry, stop, target, sid, now, bos_time=bos[side])
            verdicts.append(ZoneVerdict(sid, z, "signal", plan, sig))
    return verdicts

