"""Backtest the Daily Anchored VWAP strategy on OANDA 1-minute bid/ask data.

Uses only data before the holdout date (the last six months stay untouched).
Results are per 1 unit in price points (pips for FX pairs), after the spread;
for SPX500_USD, 1 unit = $1 per point.

Usage:
    python -m vwap.backtest                                  # SPX500_USD and EUR_USD
    python -m vwap.backtest --instruments SPX500_USD --no-costs
"""

import argparse
from dataclasses import fields

import numpy as np
import pandas as pd

from backtest import HOLDOUT_FROM, OUT_DIR
from vwap.fetch import path_for
from vwap.strategy import VwapParams, resample, run

PIP = {"JPY": 0.01}


def point_size(instrument):
    if instrument.endswith("_USD") and instrument[:3] in ("SPX", "NAS", "US3", "DE3", "UK1", "JP2"):
        return 1.0
    return PIP["JPY"] if instrument.endswith("JPY") else 0.0001


def load_bars(instrument, before, minutes, costs=True):
    m1 = pd.read_csv(path_for(instrument))
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1 = m1[m1["time"] < before]
    if not costs:  # trade at mid prices, no spread
        for k in ("open", "high", "low", "close"):
            mid = (m1[f"bid_{k}"] + m1[f"ask_{k}"]) / 2
            m1[f"bid_{k}"] = m1[f"ask_{k}"] = mid
    return resample(m1, minutes)


def summarize(trades, unit, label):
    if trades.empty:
        return f"{label}: no trades"
    pnl = trades["pnl"] / unit
    eq = pnl.cumsum()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    days = trades["entry_time"].dt.tz_convert("America/New_York").dt.date.nunique()
    return (f"{label}: {len(pnl)} trades on {days} days | win rate {len(wins) / len(pnl):.0%} | "
            f"total {pnl.sum():+.1f} | avg {pnl.mean():+.2f} | avg win {wins.mean():+.2f} / "
            f"avg loss {losses.mean():+.2f} | profit factor {pf:.2f} | max drawdown {(eq - eq.cummax()).min():.1f}")


def report(instrument, trades, unit, units_name):
    print(f"\n=== {instrument} (per 1 unit, in {units_name}, after spread) ===")
    print(summarize(trades, unit, "All"))
    if trades.empty:
        return
    year = trades["entry_time"].dt.year
    for y, g in trades.groupby(year):
        print("  " + summarize(g, unit, str(y)))
    for side, g in trades.groupby("side"):
        print("  " + summarize(g, unit, "long" if side > 0 else "short"))
    for reason, g in trades.groupby("entry_reason"):
        print("  " + summarize(g, unit, f"entry {reason}"))
    for reason, g in trades.groupby("exit_reason"):
        print("  " + summarize(g, unit, f"exit {reason}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruments", nargs="+", default=["SPX500_USD", "EUR_USD"])
    parser.add_argument("--holdout-from", default=HOLDOUT_FROM)
    parser.add_argument("--no-costs", action="store_true", help="trade at mid prices (no spread)")
    for f in fields(VwapParams):  # every strategy input can be overridden, e.g. --ema-length 20
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type in (bool, "bool") else type(f.default)
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
    args = parser.parse_args()
    p = VwapParams(**{f.name: getattr(args, f.name) for f in fields(VwapParams)})
    before = pd.Timestamp(args.holdout_from, tz="UTC")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Inputs: {p}")
    for instrument in args.instruments:
        bars = load_bars(instrument, before, p.bar_minutes, costs=not args.no_costs)
        print(f"\n{instrument}: {len(bars):,} {p.bar_minutes}-minute bars, "
              f"{bars['time'].min():%Y-%m-%d} -> {bars['time'].max():%Y-%m-%d}")
        trades, _ = run(bars, p)
        unit = point_size(instrument)
        report(instrument, trades, unit, "points" if unit == 1.0 else "pips")
        if not trades.empty:
            path = OUT_DIR / f"vwap_{instrument}{'_nocost' if args.no_costs else ''}.csv"
            trades.assign(pnl_units=np.round(trades["pnl"] / unit, 2)).to_csv(path, index=False)
            print(f"  trade log: {path}")


if __name__ == "__main__":
    main()
