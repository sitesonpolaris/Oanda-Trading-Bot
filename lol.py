"""Level on Level Entry: a reversal zone that breaks structure, with a continuation
zone stacked on it in the same move.

Long (short is the mirror image):
  1. Downtrend: when the DBR's rally starts, the last two H1 swing highs and lows
     are lower highs and lower lows.
  2. Reversal: a DBR demand zone. It doesn't need the 2x move-out (the break of
     structure is the strength test). Its rally closes above the most recent
     lower high (break of structure) before price comes back to the zone.
  3. Level on level (drop-base-rally-base-rally): the first RBR demand zone that
     completes after the break of structure. The DBR's rally candle can be the
     RBR's move-in. "Same rally": until the RBR's base starts, no candle closes
     back below the broken lower high (the pause can chop sideways above it).
  4. Entry: limit order at the top of the RBR once the RBR is valid (2x move away).
  5. Stop: bottom of the RBR minus `atr_buffer` x the daily ATR.
  6. Trend: not required. The break of structure and the stacked zone are the
     momentum signal, so the setup is traded even against the D1/H4 trend.
     Each setup records whether D1 and H4 agreed (`trend_aligned`), so aligned
     and countertrend setups can be compared.
  7. Target (when price comes back to the RBR): `target_fraction` (80%) of the
     way to the nearest unbroken H1 supply zone above; if there is none, the
     RBR's move-out distance (top of the RBR to the high before the return)
     projected from the entry. Under `min_reward_risk`, no trade.

Usage:
    python lol.py                                   # count setups and simulate them, four pairs
    python lol.py --instruments EUR_USD
"""

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import (HOLDOUT_FROM, OUT_DIR, SPREAD_PIPS, build_market, exit_after_fill, first_true, load,
                      state_at_bars)
from strategy import active_at
from structure import StructureConfig, market_structure
from zones import ZoneConfig, detect_zones, pip_size

SWINGS_ONLY = StructureConfig(stage_mas=None, bos_invalidates_trend=False)


@dataclass
class LoLConfig:
    target_fraction: float = 0.8
    atr_buffer: float = 0.1          # fraction of the daily ATR
    min_reward_risk: float = 2.0
    spread_pips: float = 1.2
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    # The reversal zone doesn't need the move-out to be 2x the base.
    reversal_impulse_vs_base: float | None = None


def find_setups(h1, cfg=None, pip=0.0001):
    """One row per reversal zone that has a stacked continuation zone (see module docstring)."""
    cfg = cfg or LoLConfig()
    h1 = h1.reset_index(drop=True)
    times = h1["time"]
    no_breakout = {**cfg.zones.__dict__, "require_breakout": False}
    reversals = detect_zones(h1, ZoneConfig(**{**no_breakout, "min_impulse_vs_base": cfg.reversal_impulse_vs_base}), pip=pip)
    continuations = detect_zones(h1, ZoneConfig(**no_breakout), pip=pip)
    ms = market_structure(h1, SWINGS_ONLY)
    trend, bos_dir = ms["trend"].to_numpy(), ms["bos"].to_numpy()
    lo, hi, c = h1["low"].to_numpy(), h1["high"].to_numpy(), h1["close"].to_numpy()
    rows = []
    for rev in reversals[reversals["type"].isin(["DBR", "RBD"])].itertuples(index=False):
        demand = rev.kind == "demand"
        legout = int(times.searchsorted(rev.legout_time))
        if trend[legout - 1] != ("down" if demand else "up"):
            continue  # no lower highs and lower lows (or higher highs and lows) for the rally to reverse
        breaks = np.flatnonzero(bos_dir[legout:] == ("bullish" if demand else "bearish"))
        if not len(breaks):
            continue
        bos = legout + breaks[0]
        back = lo[legout + 1:bos + 1] <= rev.top if demand else hi[legout + 1:bos + 1] >= rev.bottom
        if back.any():
            continue  # price came back to the reversal zone before breaking structure
        level = ms["last_swing_high" if demand else "last_swing_low"].iat[bos - 1]  # the level the BOS broke
        for cont in continuations[continuations["type"] == ("RBR" if demand else "DBD")].itertuples(index=False):
            move_in = int(times.searchsorted(cont.start_time)) - 1
            cont_out = int(times.searchsorted(cont.legout_time))
            if move_in < legout or cont_out <= bos:
                continue  # the continuation must complete after the break of structure
            pause = slice(bos + 1, move_in + 1)
            failed = (c[pause] < level) if demand else (c[pause] > level)
            if failed.any():
                break  # the broken level gave way before this pause: not the same rally
            rows.append({
                "side": "long" if demand else "short",
                "rev_type": rev.type, "rev_start": rev.start_time, "rev_legout": rev.legout_time,
                "rev_top": rev.top, "rev_bottom": rev.bottom,
                "bos_time": times.iat[bos], "bos_level": level,
                "cont_type": cont.type, "cont_start": cont.start_time, "cont_legout": cont.legout_time,
                "cont_top": cont.top, "cont_bottom": cont.bottom, "cont_valid": cont.valid_time,
            })
            break
    return pd.DataFrame(rows)


def simulate(setups, market, cfg, pip):
    """Trade each setup once: fill, target at the return, exit. R is in units of the full risk."""
    bars, times = market.bars, market.bars["time"]
    half = cfg.spread_pips * pip / 2
    o, h, l, c = (bars[col].to_numpy() for col in ("open", "high", "low", "close"))
    bid = {"open": o - half, "high": h - half, "low": l - half, "close": c - half}
    ask = {"open": o + half, "high": h + half, "low": l + half, "close": c + half}
    daily_atr = pd.merge_asof(bars[["time"]], market.d1_atr.sort_values("available_time"),
                              left_on="time", right_on="available_time")["atr"].to_numpy()
    h1_zones = market.all_zones[market.all_zones["tf"] == "H1"]
    d1, h4 = state_at_bars(bars, market.d1, ["trend"])["trend"], state_at_bars(bars, market.h4, ["trend"])["trend"]
    out = []
    for s in setups.itertuples(index=False):
        long = s.side == "long"
        row = s._asdict() | {"status": ""}
        out.append(row)
        if pd.isna(s.cont_valid):
            row["status"] = "not_valid"
            continue
        order = int(times.searchsorted(s.cont_valid)) + 1
        if order >= len(bars) or np.isnan(daily_atr[order]):
            row["status"] = "untouched"
            continue
        want = "up" if long else "down"
        row.update(d1_trend=d1[order], h4_trend=h4[order], trend_aligned=d1[order] == want and h4[order] == want)
        buffer = cfg.atr_buffer * daily_atr[order]
        entry, stop = (s.cont_top, s.cont_bottom - buffer) if long else (s.cont_bottom, s.cont_top + buffer)
        risk = abs(entry - stop)
        touch = ask["low"][order:] <= entry if long else bid["high"][order:] >= entry
        f = first_true(touch)
        row.update(order_time=times.iat[order], entry=entry, stop=round(stop, 5), risk_pips=round(risk / pip, 1))
        if f is None:
            row["status"] = "untouched"
            continue
        f += order
        t = times.iat[f]
        opp = active_at(h1_zones, t)
        if long:
            edges = opp.loc[(opp["kind"] == "supply") & (opp["bottom"] > entry), "bottom"]
        else:
            edges = opp.loc[(opp["kind"] == "demand") & (opp["top"] < entry), "top"]
        legout = int(times.searchsorted(s.cont_legout))
        if len(edges):
            edge = edges.min() if long else edges.max()
            target, source = entry + cfg.target_fraction * (edge - entry), "80% to zone"
            row["target_zone_edge"] = edge
        else:
            move = (h[legout:f].max() - entry) if long else (entry - l[legout:f].min())
            target, source = (entry + move if long else entry - move), "move-out distance"
        reward = abs(target - entry)
        row.update(entry_time=t, target=round(target, 5), target_source=source, planned_rr=round(reward / risk, 2))
        if reward < cfg.min_reward_risk * risk:
            row["status"] = "skipped_rr"
            continue
        fill = min(entry, ask["open"][f]) if long else max(entry, bid["open"][f])
        e, result, exit_price = exit_after_fill(f, long, stop, target, bid, ask)
        move = exit_price - fill if long else fill - exit_price
        row.update(status="filled", fill_price=round(fill, 5), exit_time=times.iat[e], exit_price=round(exit_price, 5),
                   result=result, r_multiple=round(move / abs(fill - stop), 3))
    return pd.DataFrame(out)


def run(instrument, holdout, cfg):
    pip = pip_size(instrument)
    h1, h4, d1 = (load(instrument, g, holdout) for g in ("H1", "H4", "D"))
    market, _ = build_market(h1, h4, d1, cfg.zones, instrument, pip)
    setups = find_setups(h1, cfg, pip)
    trades = simulate(setups, market, cfg, pip) if len(setups) else setups
    return market, trades.assign(instrument=instrument)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY"])
    args = parser.parse_args()
    holdout = pd.Timestamp(HOLDOUT_FROM, tz="UTC")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for instrument in args.instruments:
        _, trades = run(instrument, holdout, LoLConfig(spread_pips=SPREAD_PIPS.get(instrument, 1.5)))
        trades.to_csv(OUT_DIR / f"lol_{instrument}.csv", index=False)
        counts = trades["status"].value_counts().to_dict() if len(trades) else {}
        print(f"{instrument}: {len(trades)} Level on Level setups | {counts}")


if __name__ == "__main__":
    main()
