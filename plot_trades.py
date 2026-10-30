"""Chart backtest trades: zones, entry, stop loss, take profit, fill and exit.

Reads a trade log written by backtest.py, rebuilds that pair's market the same
way, and saves one PNG per trade to output/charts/trades/.

Usage:
    python plot_trades.py                                   # 3 winners + 3 losers, all four pairs
    python plot_trades.py --instruments EUR_USD --winners 5 --losers 0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from backtest import HOLDOUT_FROM, OUT_DIR, build_market, load
from strategy import active_at
from zones import ZoneConfig, pip_size

ROOT = Path(__file__).resolve().parent
CHART_DIR = ROOT / "output" / "charts" / "trades"
BEFORE, AFTER = 40, 15
COLORS = {"demand": "#2e7d32", "supply": "#c62828"}


def trade_log(instrument, buffer=0.1):
    t = pd.read_csv(OUT_DIR / f"trades_{instrument}_H1_moff_b{buffer}.csv")
    t = t[t["status"] == "filled"].copy()
    for col in ("zone_start", "order_time", "entry_time", "exit_time"):
        t[col] = pd.to_datetime(t[col], utc=True)
    return t.assign(instrument=instrument)


def pick(trades, n, result):
    """n trades with this result, spread across pairs, oldest first within each pair."""
    pool = trades[trades["result"] == result].sort_values("entry_time")
    picked = []
    while len(picked) < n and len(pool):
        for _, group in pool.groupby("instrument"):
            if len(picked) < n:
                picked.append(group.iloc[0])
        pool = pool.drop([p.name for p in picked if p.name in pool.index])
    return picked


def plot_trade(trade, market, zones, pip, path):
    bars = market.bars
    z = zones[zones["start_time"] == trade.zone_start].iloc[0]
    start = int(bars["time"].searchsorted(z.start_time)) - BEFORE
    end = int(bars["time"].searchsorted(trade.exit_time)) + AFTER
    start = max(0, start)
    w = bars.iloc[start:end + 1].reset_index(drop=True)
    x = lambda t: int(w["time"].searchsorted(t))  # noqa: E731
    n = len(w)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for k, r in w.iterrows():
        color = "#26a69a" if r.close >= r.open else "#ef5350"
        ax.vlines(k, r.low, r.high, color=color, linewidth=0.8)
        ax.add_patch(Rectangle((k - 0.35, min(r.open, r.close)), 0.7, max(abs(r.close - r.open), 1e-9), color=color))

    lo, hi = w["low"].min(), w["high"].max()
    others = active_at(market.all_zones, trade.entry_time)
    others = others[(others["top"] >= lo) & (others["bottom"] <= hi) & (others["start_time"] != z.start_time)]
    for o in others.itertuples():
        x0 = max(0, x(o.start_time))
        ax.add_patch(Rectangle((x0 - 0.5, o.bottom), n - x0, o.top - o.bottom, color=COLORS[o.kind], alpha=0.10))
        ax.text(x0, o.top, f" {o.type} {o.kind} ({o.tf})", color=COLORS[o.kind], fontsize=7, va="bottom", alpha=0.8)

    zs, zl = x(z.start_time), x(z.legout_time)
    color = COLORS[z.kind]
    ax.add_patch(Rectangle((zs - 0.5, z.bottom), n - zs, z.top - z.bottom, color=color, alpha=0.28))
    ax.text(zs, z.top if z.kind == "demand" else z.bottom, f" TRADED: {z.type} {z.kind} zone",
            color=color, fontsize=9, fontweight="bold", va="bottom" if z.kind == "demand" else "top")
    ax.axvspan(zs - 0.5, zl - 0.5, color="#ffb300", alpha=0.18)
    for k, label, dx in ((zs - 1, "move-in", -30), (zl, "move-out", 30)):
        r = w.iloc[k]
        y = r.low if z.kind == "demand" else r.high
        ax.annotate(label, (k, y), xytext=(dx, -28 if z.kind == "demand" else 22), textcoords="offset points",
                    ha="center", fontsize=8, arrowprops={"arrowstyle": "->", "color": "#555"})

    xe, xx, xo = x(trade.entry_time), x(trade.exit_time), x(trade.order_time)
    lines = [(trade.entry, f"ENTRY {trade.entry:.5f}", "#1565c0"),
             (trade.stop, f"STOP LOSS {trade.stop:.5f}", "#c62828"),
             (trade.target, f"TAKE PROFIT {trade.target:.5f} ({trade.target_source})", "#2e7d32")]
    for y, label, c in lines:
        ax.hlines(y, xo - 0.5, n - 0.5, colors=c, linestyles="--", linewidth=1.3)
        ax.text(n - 1, y, label, color=c, fontsize=9, fontweight="bold", ha="right",
                va="bottom", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
    ax.axvline(xo, color="#1565c0", linewidth=0.6, alpha=0.4)
    ax.text(xo, hi, " order placed", color="#1565c0", fontsize=7, va="top")
    ax.plot([xe], [trade.fill_price], marker="^" if z.kind == "demand" else "v", color="#1565c0", markersize=12)
    ax.plot([xx], [trade.exit_price], marker="X", color="#2e7d32" if trade.result == "win" else "#c62828",
            markersize=12)

    ticks = range(0, n, max(1, n // 12))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([w["time"].iat[t].strftime("%Y-%m-%d\n%H:%M") for t in ticks], fontsize=7)
    ax.set_xlim(-1, n + 1)
    side = "LONG" if z.kind == "demand" else "SHORT"
    adds = f", {int(trade.adds)} add(s)" if trade.adds else ""
    ax.set_title(f"{trade.instrument} H1 {side} from {z.type} zone | risk {trade.risk_pips} pips, "
                 f"planned {trade.planned_rr}R | {trade.result.upper()} {trade.r_multiple:+.2f}R of full risk{adds} | "
                 f"{trade.reason}", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY"])
    parser.add_argument("--winners", type=int, default=3)
    parser.add_argument("--losers", type=int, default=3)
    args = parser.parse_args()

    trades = pd.concat([trade_log(i) for i in args.instruments], ignore_index=True)
    chosen = pick(trades, args.winners, "win") + pick(trades, args.losers, "loss")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    holdout = pd.Timestamp(HOLDOUT_FROM, tz="UTC")
    markets = {}
    for t in chosen:
        if t.instrument not in markets:
            pip = pip_size(t.instrument)
            h1, h4, d1 = (load(t.instrument, g, holdout) for g in ("H1", "H4", "D"))
            markets[t.instrument] = (*build_market(h1, h4, d1, ZoneConfig(), t.instrument, pip), pip)
        market, zones, pip = markets[t.instrument]
        path = CHART_DIR / f"{t.result}_{t.instrument}_{t.entry_time:%Y%m%d_%H%M}.png"
        plot_trade(t, market, zones, pip, path)
        print(f"{t.result:4s} {t.instrument} {t.zone_type} {t.side:5s} entry {t.entry_time:%Y-%m-%d %H:%M} "
              f"{t.r_multiple:+.2f}R -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


def plot_lol(trade, market, pip, path):
    """Chart one Level on Level Entry: downtrend swing broken, reversal zone, stacked entry zone, levels."""
    bars = market.bars
    long = trade.side == "long"
    bos_idx = int(bars["time"].searchsorted(trade.bos_time))
    rev_idx = int(bars["time"].searchsorted(trade.rev_start))
    levels = bars["high"] if long else bars["low"]
    swing_idx = max(i for i in range(rev_idx) if abs(levels.iat[i] - trade.bos_level) < 1e-9)
    end = int(bars["time"].searchsorted(trade.exit_time if pd.notna(trade.exit_time) else trade.entry_time)) + AFTER
    start = max(0, min(swing_idx, rev_idx) - 25)
    w = bars.iloc[start:end + 1].reset_index(drop=True)
    x = lambda t: int(w["time"].searchsorted(t))  # noqa: E731
    n = len(w)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for k, r in w.iterrows():
        color = "#26a69a" if r.close >= r.open else "#ef5350"
        ax.vlines(k, r.low, r.high, color=color, linewidth=0.8)
        ax.add_patch(Rectangle((k - 0.35, min(r.open, r.close)), 0.7, max(abs(r.close - r.open), 1e-9), color=color))

    kind, opposite = ("demand", "supply") if long else ("supply", "demand")
    if trade.target_source == "80% to zone":
        zones = active_at(market.all_zones[market.all_zones["tf"] == "H1"], trade.entry_time)
        col = "bottom" if long else "top"
        tz = zones[(zones["kind"] == opposite) & (zones[col] == trade.target_zone_edge)
                   & (zones["start_time"] == trade.target_zone_start)].iloc[0]
        x0 = max(0, x(tz.start_time))
        ax.add_patch(Rectangle((x0 - 0.5, tz.bottom), n - x0, tz.top - tz.bottom, color=COLORS[opposite], alpha=0.22))
        ax.text(x0, tz.top if long else tz.bottom, f" TARGET ZONE: {tz.type} {opposite}", color=COLORS[opposite],
                fontsize=9, fontweight="bold", va="bottom" if long else "top")

    xs, xb = swing_idx - start, bos_idx - start
    ax.hlines(trade.bos_level, xs, xb + 0.5, colors="#6a1b9a", linewidth=1.5)
    ax.plot([xs], [trade.bos_level], "o", color="#6a1b9a", markersize=8, fillstyle="none")
    ax.text(xs, trade.bos_level, (" last lower high" if long else " last higher low"), color="#6a1b9a", fontsize=8,
            va="bottom" if long else "top")
    ax.annotate("BOS", (xb, trade.bos_level), xytext=(8, 14 if long else -18), textcoords="offset points",
                color="#6a1b9a", fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "->", "color": "#6a1b9a"})

    for top, bottom, start_t, label, alpha in ((trade.rev_top, trade.rev_bottom, trade.rev_start,
                                                f"{trade.rev_type} (reversal)", 0.15),
                                               (trade.cont_top, trade.cont_bottom, trade.cont_start,
                                                f"{trade.cont_type} ENTRY ZONE (level on level)", 0.35)):
        x0 = x(start_t)
        ax.add_patch(Rectangle((x0 - 0.5, bottom), n - x0, top - bottom, color=COLORS[kind], alpha=alpha))
        ax.text(x0, bottom if long else top, f" {label}", color=COLORS[kind], fontsize=9, fontweight="bold",
                va="top" if long else "bottom")

    xo = x(trade.order_time)
    for y, label, c in ((trade.entry, f"ENTRY {trade.entry:.5f}", "#1565c0"),
                        (trade.stop, f"STOP LOSS {trade.stop:.5f}", "#c62828"),
                        (trade.target, f"TAKE PROFIT {trade.target:.5f} ({trade.target_source})", "#2e7d32")):
        ax.hlines(y, xo - 0.5, n - 0.5, colors=c, linestyles="--", linewidth=1.3)
        ax.text(n - 1, y, label, color=c, fontsize=9, fontweight="bold", ha="right", va="bottom",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
    if trade.status == "filled":
        ax.plot([x(trade.entry_time)], [trade.fill_price], marker="^" if long else "v", color="#1565c0", markersize=12)
        ax.plot([x(trade.exit_time)], [trade.exit_price], marker="X",
                color="#2e7d32" if trade.result == "win" else "#c62828", markersize=12)

    ticks = range(0, n, max(1, n // 12))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([w["time"].iat[t].strftime("%Y-%m-%d\n%H:%M") for t in ticks], fontsize=7)
    ax.set_xlim(-1, n + 1)
    outcome = f"{trade.result.upper()} {trade.r_multiple:+.2f}R" if trade.status == "filled" else trade.status
    ax.set_title(f"{trade.instrument} H1 Level on Level Entry, {trade.side.upper()}: {trade.rev_type} breaks structure, "
                 f"{trade.cont_type} stacked | risk {trade.risk_pips} pips, planned {trade.planned_rr}R | {outcome}",
                 fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
