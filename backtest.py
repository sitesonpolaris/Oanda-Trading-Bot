"""Backtest the supply/demand zone strategy on the in-sample period only.

Rules:
  - Zones come from zones.py (H1 or H4). A zone becomes active once its leg-out candle closes.
  - Only trade with the H4 trend: demand zones in an uptrend, supply zones in a downtrend.
  - Limit entry at the zone's near edge (top of demand, bottom of supply) on the first
    return to the zone. That first touch uses up the zone, traded or not.
  - Stop beyond the far edge plus `atr_buffer` x ATR. Target at `target_fraction` of the
    impulse (leg-out range) from entry. Skip if target < `min_reward_risk` x stop distance.

Execution is simulated on H1 mid candles with a fixed spread: buys fill and exit
on the ask/bid side of the spread, so every trade pays it. When one candle hits
both stop and target, the stop is assumed to come first.

The last `validation_months` of data are cut off before anything is computed.

Usage:
    python backtest.py                      # baseline + parameter grid, H1 and H4 zones
    python backtest.py --spread 1.0 --zones-tf H1
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from structure import StructureConfig, market_structure
from zones import ZoneConfig, detect_zones, pip_size

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "backtest"

TIMEFRAMES = {"H1": pd.Timedelta("1h"), "H4": pd.Timedelta("4h")}
BUFFERS = [0.1, 0.2, 0.25]
MULTIPLES = [2.0, 2.5, 3.0]


@dataclass
class BacktestConfig:
    atr_buffer: float = 0.1
    target_fraction: float = 0.8
    min_reward_risk: float = 2.0
    spread_pips: float = 1.2
    risk_per_trade: float = 0.01
    starting_equity: float = 10_000.0


def first_true(mask):
    return int(np.argmax(mask)) if mask.any() else None


def simulate(zones, bars, trend, cfg, zone_tf, pip):
    """One row per zone: trade details and outcome, or why it wasn't traded."""
    half = cfg.spread_pips * pip / 2
    o, h, l, c = (bars[col].to_numpy() for col in ("open", "high", "low", "close"))
    bid = {"open": o - half, "high": h - half, "low": l - half, "close": c - half}
    ask = {"open": o + half, "high": h + half, "low": l + half}
    trend_at = pd.merge_asof(bars[["time"]], trend.sort_values("available_time"),
                             left_on="time", right_on="available_time")["trend"].to_numpy()

    out = []
    for z in zones.itertuples(index=False):
        long = z.kind == "demand"
        buffer = cfg.atr_buffer * z.atr_pips * pip
        impulse = cfg.target_fraction * z.impulse_pips * pip
        if long:
            entry, stop, target = z.top, z.bottom - buffer, z.top + impulse
        else:
            entry, stop, target = z.bottom, z.top + buffer, z.bottom - impulse
        risk, reward = abs(entry - stop), abs(target - entry)
        row = {
            "zone_start": z.start_time, "zone_type": z.type, "side": "long" if long else "short",
            "zone_top": z.top, "zone_bottom": z.bottom, "entry": entry, "stop": stop,
            "target": round(target, 5), "risk_pips": round(risk / pip, 1),
            "planned_rr": round(reward / risk, 2), "spread_r": round(cfg.spread_pips * pip / risk, 3),
            "status": "", "trend_at_touch": "", "entry_time": pd.NaT, "fill_price": np.nan,
            "exit_time": pd.NaT, "exit_price": np.nan, "result": "", "pips": np.nan, "r_multiple": np.nan,
        }
        out.append(row)
        if reward < cfg.min_reward_risk * risk:
            row["status"] = "skipped_rr"
            continue

        start = int(bars["time"].searchsorted(z.legout_time + zone_tf))
        touch = ask["low"][start:] <= entry if long else bid["high"][start:] >= entry
        f = first_true(touch)
        if f is None:
            row["status"] = "untouched"
            continue
        f += start
        row["entry_time"], row["trend_at_touch"] = bars["time"].iat[f], trend_at[f]
        if trend_at[f] != ("up" if long else "down"):
            row["status"] = "skipped_trend"
            continue

        # Limit order: fills at the entry price, or better if the candle opens through it.
        fill = min(entry, ask["open"][f]) if long else max(entry, bid["open"][f])
        side = bid if long else ask
        if long:
            stop_hit, target_hit = side["low"][f:] <= stop, side["high"][f:] >= target
        else:
            stop_hit, target_hit = side["high"][f:] >= stop, side["low"][f:] <= target
        target_hit[0] = False  # can't tell whether the target came after the fill within the candle
        s, t = first_true(stop_hit), first_true(target_hit)

        if s is None and t is None:
            e, result, exit_price = len(bars) - 1, "open_at_end", side["close"][-1] if long else c[-1] + half
        else:
            e = min(x for x in (s, t) if x is not None) + f
            gap_open = side["open"][e]
            opened_at_target = e > f and (gap_open >= target if long else gap_open <= target)
            if t is not None and t + f == e and (s is None or s + f != e or opened_at_target):
                result, exit_price = "win", max(target, gap_open) if long else min(target, gap_open)
            else:
                result = "loss"
                gapped = e > f and (gap_open <= stop if long else gap_open >= stop)
                exit_price = gap_open if gapped else stop

        move = exit_price - fill if long else fill - exit_price
        row.update(status="filled", fill_price=round(fill, 5), exit_time=bars["time"].iat[e],
                   exit_price=round(exit_price, 5), result=result, pips=round(move / pip, 1),
                   r_multiple=round(move / abs(fill - stop), 3))
    return pd.DataFrame(out)


def summarize(trades, cfg):
    filled = trades[trades["status"] == "filled"].sort_values("exit_time")
    r = filled["r_multiple"].to_numpy()
    equity = cfg.starting_equity * np.cumprod(1 + cfg.risk_per_trade * r)
    curve = np.concatenate([[cfg.starting_equity], equity])
    drawdown = 1 - curve / np.maximum.accumulate(curve)
    wins, losses = r[r > 0], r[r <= 0]
    counts = trades["status"].value_counts()
    return {
        "trades": len(r),
        "win_rate_%": round(100 * len(wins) / len(r), 1) if len(r) else 0.0,
        "avg_planned_rr": round(filled["planned_rr"].mean(), 2) if len(r) else 0.0,
        "avg_win_r": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss_r": round(losses.mean(), 2) if len(losses) else 0.0,
        "expectancy_r": round(r.mean(), 3) if len(r) else 0.0,
        "total_return_%": round(100 * (curve[-1] / cfg.starting_equity - 1), 2),
        "max_drawdown_%": round(100 * drawdown.max(), 2),
        "zones": len(trades),
        "skipped_rr": int(counts.get("skipped_rr", 0)),
        "skipped_trend": int(counts.get("skipped_trend", 0)),
        "untouched": int(counts.get("untouched", 0)),
    }


def load(instrument, granularity, before):
    path = DATA_DIR / f"{instrument}_{granularity}.csv"
    if not path.exists():
        sys.exit(f"Error: {path} not found. Run fetch_candles.py first.")
    df = pd.read_csv(path, parse_dates=["time"])
    return df[df["time"] < before].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--zones-tf", nargs="+", default=["H1", "H4"], choices=list(TIMEFRAMES))
    parser.add_argument("--spread", type=float, default=BacktestConfig.spread_pips, help="pips")
    parser.add_argument("--risk", type=float, default=BacktestConfig.risk_per_trade,
                        help="fraction of equity risked per trade")
    parser.add_argument("--in-sample-months", type=int, default=18)
    args = parser.parse_args()

    pip = pip_size(args.instrument)
    first = pd.read_csv(DATA_DIR / f"{args.instrument}_H1.csv", nrows=1, parse_dates=["time"])["time"].iat[0]
    split = (first + pd.DateOffset(months=args.in_sample_months)).normalize()
    bars = load(args.instrument, "H1", split)
    h4 = load(args.instrument, "H4", split)
    print(f"In-sample: {bars['time'].iat[0]:%Y-%m-%d} -> {bars['time'].iat[-1]:%Y-%m-%d %H:%M} UTC "
          f"({len(bars):,} H1 bars). Held out for validation: everything from {split:%Y-%m-%d}.")
    print(f"Spread {args.spread} pips, risk {args.risk:.1%} per trade, compounding.\n")

    structure = market_structure(h4, StructureConfig())
    structure["available_time"] = structure["time"] + TIMEFRAMES["H4"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    structure.to_csv(OUT_DIR / f"{args.instrument}_H4_structure.csv", index=False)
    share = structure["trend"].value_counts(normalize=True).mul(100).round(1)
    print(f"H4 trend share: {share.to_dict()} | BOS events: "
          f"{(structure['bos'] == 'bullish').sum()} bullish, {(structure['bos'] == 'bearish').sum()} bearish\n")

    base_cfg = BacktestConfig(spread_pips=args.spread, risk_per_trade=args.risk)
    grid = []
    for tf in args.zones_tf:
        candles = h4 if tf == "H4" else bars
        for multiple, buffer in itertools.product(MULTIPLES, BUFFERS):
            zones = detect_zones(candles, ZoneConfig(legout_multiple=multiple), pip=pip)
            cfg = replace(base_cfg, atr_buffer=buffer)
            trades = simulate(zones, bars, structure[["available_time", "trend"]], cfg, TIMEFRAMES[tf], pip)
            trades.to_csv(OUT_DIR / f"trades_{args.instrument}_{tf}_m{multiple}_b{buffer}.csv", index=False)
            grid.append({"zones_tf": tf, "multiple": multiple, "atr_buffer": buffer, **summarize(trades, cfg)})

    results = pd.DataFrame(grid)
    results.to_csv(OUT_DIR / "grid_results.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["zones_tf", "multiple", "atr_buffer", "trades", "win_rate_%", "avg_planned_rr", "avg_win_r",
            "avg_loss_r", "expectancy_r", "total_return_%", "max_drawdown_%"]
    for tf in args.zones_tf:
        base = results[(results["zones_tf"] == tf) & (results["multiple"] == 2.5) & (results["atr_buffer"] == 0.1)]
        print(f"== Baseline, {tf} zones (multiple 2.5, buffer 0.1 ATR)")
        print(base.drop(columns=["zones_tf", "multiple", "atr_buffer"]).T.to_string(header=False), "\n")
    print("== Parameter grid")
    print(results[cols].to_string(index=False))
    print(f"\nTrade logs and grid_results.csv in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
