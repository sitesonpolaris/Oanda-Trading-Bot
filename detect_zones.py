"""Detect supply/demand zones in saved candle CSVs, save them, and chart a sample.

Usage:
    python detect_zones.py                              # EUR_USD H1 + H4, default settings
    python detect_zones.py --multiple 3 --max-base 2    # tweak detection settings
    python detect_zones.py --plot-at "2026-03-12 14:00" # chart the zone(s) nearest a time
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from zones import ZoneConfig, detect_zones, pip_size

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
CHART_DIR = OUT_DIR / "charts"

ZONE_COLORS = {"demand": "#2e7d32", "supply": "#c62828"}
CONTEXT_BEFORE, CONTEXT_AFTER = 40, 60


def load_candles(instrument, granularity):
    path = DATA_DIR / f"{instrument}_{granularity}.csv"
    if not path.exists():
        sys.exit(f"Error: {path} not found. Run fetch_candles.py first.")
    return pd.read_csv(path, parse_dates=["time"])


def per_year_counts(zones):
    counts = zones.groupby([zones["start_time"].dt.year.rename("year"), "type"]).size()
    table = counts.unstack(fill_value=0).reindex(columns=["RBR", "DBR", "DBD", "RBD"], fill_value=0)
    table["total"] = table.sum(axis=1)
    return table


def plot_zone(candles, zone, title, path):
    """Candlestick chart around one zone; the zone box extends right from its base."""
    start = candles.index[candles["time"] == zone.start_time][0]
    legout = candles.index[candles["time"] == zone.legout_time][0]
    window = candles.iloc[max(0, start - CONTEXT_BEFORE):legout + CONTEXT_AFTER + 1]
    x0 = window.index[0]

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, row in window.iterrows():
        x = idx - x0
        up = row.close >= row.open
        color = "#26a69a" if up else "#ef5350"
        ax.vlines(x, row.low, row.high, color=color, linewidth=0.8)
        ax.add_patch(Rectangle((x - 0.35, min(row.open, row.close)), 0.7,
                               max(abs(row.close - row.open), 1e-6), color=color))

    color = ZONE_COLORS[zone.kind]
    ax.add_patch(Rectangle((start - x0 - 0.5, zone.bottom), window.index[-1] - start + 1,
                           zone.top - zone.bottom, color=color, alpha=0.18, linewidth=0))
    ax.hlines([zone.top, zone.bottom], start - x0 - 0.5, window.index[-1] - x0 + 0.5,
              color=color, linewidth=0.8, linestyles="--")
    ax.axvspan(start - x0 - 0.5, legout - x0 - 0.5, color="#ffb300", alpha=0.15)  # base candles

    ticks = range(0, len(window), max(1, len(window) // 10))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([window["time"].iloc[t].strftime("%Y-%m-%d\n%H:%M") for t in ticks], fontsize=8)
    ax.set_xlim(-1, len(window))
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def zone_title(instrument, gran, z):
    return (f"{instrument} {gran} {z.type} ({z.kind})  base {z.start_time:%Y-%m-%d %H:%M} UTC, "
            f"{z.base_candles} candle(s)  zone {z.bottom:.5f}-{z.top:.5f}  "
            f"impulse {z.impulse_pips} pips = {z.impulse_ratio}x avg range")


def pick_samples(zones, per_type):
    """Most recent zones of each type, so they're easy to find on your own charts."""
    return pd.concat([zones[zones["type"] == t].tail(per_type) for t in ["RBR", "DBR", "DBD", "RBD"]])


def main():
    defaults = ZoneConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", nargs="+", default=["H1", "H4"])
    parser.add_argument("--multiple", type=float, default=defaults.legout_multiple,
                        help="optional impulse strength filter: range vs average prior range (default: off)")
    parser.add_argument("--lookback", type=int, default=defaults.legout_lookback,
                        help="candles averaged for the strength filter (default %(default)s)")
    parser.add_argument("--min-base", type=int, default=defaults.min_base_candles)
    parser.add_argument("--max-base", type=int, default=defaults.max_base_candles)
    parser.add_argument("--plots-per-type", type=int, default=1,
                        help="sample charts per zone type per timeframe (default %(default)s)")
    parser.add_argument("--plot-at", help="also chart the zone with the base nearest this UTC time")
    args = parser.parse_args()

    cfg = ZoneConfig(legout_multiple=args.multiple, legout_lookback=args.lookback,
                     min_base_candles=args.min_base, max_base_candles=args.max_base)
    OUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    print(f"Settings: {cfg}\n")

    for gran in args.granularity:
        candles = load_candles(args.instrument, gran)
        zones = detect_zones(candles, cfg, pip=pip_size(args.instrument))
        csv_path = OUT_DIR / f"{args.instrument}_{gran}_zones.csv"
        zones.to_csv(csv_path, index=False)

        print(f"== {args.instrument} {gran}: {len(zones)} zones from {len(candles):,} candles "
              f"-> {csv_path.relative_to(ROOT)}")
        if zones.empty:
            continue
        print(per_year_counts(zones).to_string(), "\n")

        samples = pick_samples(zones, args.plots_per_type)
        if args.plot_at:
            target = pd.Timestamp(args.plot_at, tz="UTC")
            nearest = (zones["start_time"] - target).abs().idxmin()
            samples = pd.concat([samples, zones.loc[[nearest]]])
        for _, z in samples.iterrows():
            path = CHART_DIR / f"{args.instrument}_{gran}_{z.type}_{z.start_time:%Y%m%d_%H%M}.png"
            plot_zone(candles, z, zone_title(args.instrument, gran, z), path)
            print(f"  chart: {path.relative_to(ROOT)}")
        print()


if __name__ == "__main__":
    main()
