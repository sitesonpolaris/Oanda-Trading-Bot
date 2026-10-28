"""Backtest the strategy spec on the in-sample period only.

Rules (see strategy.py): H1 zones, D1 trend + H4 momentum at zone formation, limit entry at the
zone's near edge once the impulse extreme is confirmed, stop beyond the far edge
+ ATR buffer, target at 80% of the impulse from its origin, playing-field filters
(room to opposing zone, erratic wicks, news), a one-third first entry with up to
two one-third adds on H1 breaks of structure in the trade's favour.

Execution is simulated on H1 mid candles with a fixed spread: buys fill and exit
on the ask/bid side, so every entry, add and exit pays it. When one candle hits
both stop and target, the stop is assumed to come first. A setup's result is in R
of the full per-trade risk: a stop-out with all three entries is -1R, a stop-out
with only the first entry is -1/3R.

Everything from HOLDOUT_FROM (2026-04-28) on is cut off before anything is
computed, for every instrument.

Usage:
    python backtest.py                                        # EUR_USD, all data before the hold-out
    python backtest.py --start 2024-10-28                     # the original 18-month window
    python backtest.py --instruments GBP_USD AUD_USD USD_JPY  # other pairs
"""

import argparse
import itertools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import (StrategyConfig, big_wick_counts, direction_check, impulse_extreme, level_on_level,
                      load_news, news_conflict, plan_trade, room_to_opposing, zone_lifecycle)
from structure import StructureConfig, find_swings, market_structure
from zones import ZoneConfig, atr, detect_zones, pip_size

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "backtest"
NEWS_PATH = DATA_DIR / "news_events.csv"

TIMEFRAMES = {"H1": pd.Timedelta("1h"), "H4": pd.Timedelta("4h"), "D": pd.Timedelta("1D")}
BUFFERS = [0.1, 0.2, 0.25]
HOLDOUT_FROM = "2026-04-28"  # last ~6 months, kept for final validation
SPREAD_PIPS = {"EUR_USD": 1.2, "GBP_USD": 1.6, "AUD_USD": 1.4, "USD_JPY": 1.4}  # typical practice spreads
MULTIPLES = [2.0, 2.5, 3.0]


@dataclass
class BacktestConfig(StrategyConfig):
    spread_pips: float = 1.2
    risk_per_trade: float = 0.01   # full per-trade risk; entries are fractions of it
    starting_equity: float = 10_000.0


@dataclass
class Market:
    """Everything the strategy looks at, built once per zone-detection setting."""
    bars: pd.DataFrame        # H1 candles: zones, entries and execution
    d1: pd.DataFrame          # D1 structure (trend, last_bos, available_time)
    h4: pd.DataFrame          # H4 structure (momentum)
    h1: pd.DataFrame          # H1 structure (BOS events for adds), aligned with bars
    all_zones: pd.DataFrame   # H1 + H4 zones with available/broken times (room, level-on-level)
    news: pd.DataFrame | None
    instrument: str


def with_available_time(structure, granularity):
    return structure.assign(available_time=structure["time"] + TIMEFRAMES[granularity])


def build_market(h1, h4, d1, zone_cfg, instrument, pip, news=None, structure_cfg=None):
    """Return (Market, H1 zones with lifecycle columns)."""
    scfg = structure_cfg or StructureConfig()
    h1_zones = zone_lifecycle(detect_zones(h1, zone_cfg, pip=pip), h1, TIMEFRAMES["H1"]).assign(tf="H1")
    h4_zones = zone_lifecycle(detect_zones(h4, zone_cfg, pip=pip), h4, TIMEFRAMES["H4"]).assign(tf="H4")
    market = Market(
        bars=h1,
        d1=with_available_time(market_structure(d1, scfg), "D"),
        h4=with_available_time(market_structure(h4, scfg), "H4"),
        h1=with_available_time(market_structure(h1, scfg), "H1"),
        all_zones=pd.concat([h1_zones, h4_zones], ignore_index=True),
        news=news,
        instrument=instrument,
    )
    return market, h1_zones


def state_at_bars(bars, structure, columns):
    """Structure columns as known at each bar's open (latest row whose candle had closed)."""
    merged = pd.merge_asof(bars[["time"]], structure.sort_values("available_time")[["available_time", *columns]],
                           left_on="time", right_on="available_time")
    return {c: merged[c].fillna("sideways" if c == "trend" else "").to_numpy() for c in columns}


def first_true(mask):
    return int(np.argmax(mask)) if mask.any() else None


def exit_after_fill(f, long, stop, target, bid, ask):
    """(exit bar, result, exit price) for a position filled during bar f."""
    side = bid if long else ask
    if long:
        stop_hit, target_hit = side["low"][f:] <= stop, side["high"][f:] >= target
    else:
        stop_hit, target_hit = side["high"][f:] >= stop, side["low"][f:] <= target
    target_hit[0] = False  # can't tell whether the target came after the fill within the candle
    s, t = first_true(stop_hit), first_true(target_hit)
    if s is None and t is None:
        return len(side["close"]) - 1, "open_at_end", side["close"][-1]
    e = min(x for x in (s, t) if x is not None) + f
    gap_open = side["open"][e]
    opened_at_target = e > f and (gap_open >= target if long else gap_open <= target)
    if t is not None and t + f == e and (s is None or s + f != e or opened_at_target):
        return e, "win", max(target, gap_open) if long else min(target, gap_open)
    gapped = e > f and (gap_open <= stop if long else gap_open >= stop)
    return e, "loss", gap_open if gapped else stop


def simulate(zones, market, cfg, pip):
    """One row per zone: the trade and its outcome, or why it wasn't traded."""
    bars = market.bars
    times = bars["time"]
    half = cfg.spread_pips * pip / 2
    o, h, l, c = (bars[col].to_numpy() for col in ("open", "high", "low", "close"))
    bid = {"open": o - half, "high": h - half, "low": l - half, "close": c - half}
    ask = {"open": o + half, "high": h + half, "low": l + half, "close": c + half}
    swings = find_swings(bars, cfg.impulse_swing_strength)
    swing_high, swing_low = swings["swing_high"].to_numpy(), swings["swing_low"].to_numpy()
    atr_values = atr(bars).to_numpy()
    wicks = big_wick_counts(bars, atr_values, cfg)
    d1 = state_at_bars(bars, market.d1, ["trend", "last_bos"])
    h4 = state_at_bars(bars, market.h4, ["trend"])
    h1_bos = market.h1["bos"].to_numpy()

    out = []
    for z in zones.itertuples(index=False):
        long = z.kind == "demand"
        row = {"zone_start": z.start_time, "zone_type": z.type, "side": "long" if long else "short",
               "level_on_level": level_on_level(z, market.all_zones, z.atr_pips * pip, cfg),
               "zone_top": z.top, "zone_bottom": z.bottom, "status": "", "reason": ""}
        out.append(row)
        legout = int(times.searchsorted(z.legout_time))
        confirmed = impulse_extreme(z.kind, h, l, swing_high, swing_low, legout, cfg.impulse_swing_strength)
        if confirmed is None or confirmed[0] + 1 >= len(bars):
            row["status"] = "no_impulse_extreme"
            continue
        ready, extreme = confirmed
        entry, stop, target = plan_trade(z, extreme, cfg, pip)
        risk, reward = abs(entry - stop), (target - entry) * (1 if long else -1)
        row.update(extreme=extreme, order_time=times.iat[ready + 1], entry=entry, stop=round(stop, 5),
                   target=round(target, 5), risk_pips=round(risk / pip, 1), planned_rr=round(reward / risk, 2),
                   spread_r=round(cfg.spread_pips * pip / risk, 3))

        early = ask["low"][legout + 1:ready + 1] <= entry if long else bid["high"][legout + 1:ready + 1] >= entry
        if early.any():
            row["status"] = "touched_before_order"
            continue
        if reward < cfg.min_reward_risk * risk:
            row["status"], row["reason"] = "skipped_rr", f"{reward / risk:.2f}R"
            continue
        touch = ask["low"][ready + 1:] <= entry if long else bid["high"][ready + 1:] >= entry
        f = first_true(touch)
        if f is None:
            row["status"] = "untouched"
            continue
        f += ready + 1
        t = times.iat[f]
        row["entry_time"] = t

        formed = min(legout + 1, len(bars) - 1)  # first bar after the leg-out closed
        ok, why = direction_check(long, d1["trend"][f], d1["last_bos"][f], h4["trend"][formed], cfg)
        row["reason"] = why
        if not ok:
            row["status"] = "skipped_trend"
            continue
        if (event := news_conflict(market.news, market.instrument, t, t + TIMEFRAMES["H1"], cfg)) is not None:
            row["status"], row["reason"] = "skipped_news", f"{event['currency']} news at {event['time']}"
            continue
        if wicks[f] >= cfg.max_big_wicks:
            row["status"], row["reason"] = "skipped_erratic", f"{int(wicks[f])} big wicks in {cfg.wick_lookback}"
            continue
        room = room_to_opposing(long, entry, t, market.all_zones)
        row["room_r"] = round(room / risk, 2) if np.isfinite(room) else np.inf
        if room < cfg.min_room_rr * risk:
            row["status"] = "skipped_room"
            continue

        fill = min(entry, ask["open"][f]) if long else max(entry, bid["open"][f])
        e, result, exit_price = exit_after_fill(f, long, stop, target, bid, ask)
        fills = [(fill, cfg.initial_fraction)]
        add_times = []
        want = "bullish" if long else "bearish"
        for k in np.flatnonzero(h1_bos[f:e] == want) + f:  # BOS known at close of k, add at open of k + 1
            if len(add_times) >= cfg.max_adds or k + 1 > e:
                break
            price = ask["open"][k + 1] if long else bid["open"][k + 1]
            if (stop < price < target) if long else (target < price < stop):
                fills.append((price, cfg.add_fraction))
                add_times.append(times.iat[k + 1])

        sign = 1 if long else -1
        r_setup = sum(frac * sign * (exit_price - p) / abs(p - stop) for p, frac in fills)
        row.update(status="filled", fill_price=round(fill, 5), adds=len(add_times),
                   add_times=";".join(f"{x:%Y-%m-%d %H:%M}" for x in add_times), exit_time=times.iat[e],
                   exit_price=round(exit_price, 5), result=result,
                   pips=round(sign * (exit_price - fill) / pip, 1), r_multiple=round(r_setup, 3))
    return pd.DataFrame(out)


def summarize(trades, cfg):
    filled = trades[trades["status"] == "filled"].sort_values("exit_time") if "status" in trades else trades
    r = filled["r_multiple"].to_numpy(dtype=float)
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
        "avg_adds": round(filled["adds"].mean(), 2) if len(r) and "adds" in filled else 0.0,
        "zones": len(trades),
        **{s: int(counts.get(s, 0)) for s in ("touched_before_order", "skipped_rr", "skipped_trend",
                                              "skipped_news", "skipped_erratic", "skipped_room", "untouched")},
    }


def load(instrument, granularity, before, start=None):
    path = DATA_DIR / f"{instrument}_{granularity}.csv"
    if not path.exists():
        sys.exit(f"Error: {path} not found. Run: python fetch_candles.py --instrument {instrument} "
                 f"--granularity H1 H4 D --years 8")
    df = pd.read_csv(path, parse_dates=["time"])
    # Keep only candles that had closed before the hold-out cutoff.
    keep = df["time"] + TIMEFRAMES[granularity] <= before
    if start is not None:
        keep &= df["time"] >= start
    return df[keep].reset_index(drop=True)


def run_grid(instrument, start, holdout, base_cfg, news, write_logs=True):
    pip = pip_size(instrument)
    h1, h4, d1 = (load(instrument, g, holdout, start) for g in ("H1", "H4", "D"))
    print(f"== {instrument}: {h1['time'].iat[0]:%Y-%m-%d} -> {h1['time'].iat[-1]:%Y-%m-%d %H:%M} UTC "
          f"({len(h1):,} H1 bars), spread {base_cfg.spread_pips} pips")
    grid = []
    for multiple in MULTIPLES:
        market, zones = build_market(h1, h4, d1, ZoneConfig(legout_multiple=multiple), instrument, pip, news)
        for buffer in BUFFERS:
            cfg = replace(base_cfg, atr_buffer=buffer)
            trades = simulate(zones, market, cfg, pip)
            if write_logs:
                tag = f"{instrument}_H1_m{multiple}_b{buffer}_{start:%Y%m%d}" if start is not None else \
                    f"{instrument}_H1_m{multiple}_b{buffer}"
                trades.to_csv(OUT_DIR / f"trades_{tag}.csv", index=False)
            filled = trades[trades["status"] == "filled"]
            lol = filled[filled["level_on_level"]]
            grid.append({"instrument": instrument, "multiple": multiple, "atr_buffer": buffer,
                         **summarize(trades, cfg), "lol_trades": len(lol),
                         "lol_expectancy_r": round(lol["r_multiple"].mean(), 3) if len(lol) else np.nan})
    return pd.DataFrame(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD"])
    parser.add_argument("--start", help="first candle date (UTC), default: all data in the file")
    parser.add_argument("--holdout-from", default=HOLDOUT_FROM,
                        help="validation data from this date is never used (default %(default)s)")
    parser.add_argument("--spread", type=float, help="pips; default per instrument from SPREAD_PIPS")
    parser.add_argument("--risk", type=float, default=BacktestConfig.risk_per_trade,
                        help="full per-trade risk as a fraction of equity")
    args = parser.parse_args()

    holdout = pd.Timestamp(args.holdout_from, tz="UTC")
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    news = load_news(NEWS_PATH)
    print(f"Held out for validation: everything from {holdout:%Y-%m-%d}. Full risk {args.risk:.1%} per setup "
          f"(entries of 1/3), compounding.")
    print(f"News filter: {f'{len(news)} high-impact events' if news is not None else 'off (no data/news_events.csv)'}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for instrument in args.instruments:
        spread = args.spread if args.spread is not None else SPREAD_PIPS.get(instrument, 1.5)
        cfg = BacktestConfig(spread_pips=spread, risk_per_trade=args.risk)
        res = run_grid(instrument, start, holdout, cfg, news)
        results.append(res)
        base = res[(res["multiple"] == 2.5) & (res["atr_buffer"] == 0.1)]
        print(base.drop(columns=["instrument", "multiple", "atr_buffer"]).T.to_string(header=False), "\n")

    results = pd.concat(results, ignore_index=True)
    results.to_csv(OUT_DIR / "grid_results.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["instrument", "multiple", "atr_buffer", "trades", "win_rate_%", "avg_planned_rr", "avg_win_r",
            "avg_loss_r", "expectancy_r", "total_return_%", "max_drawdown_%", "avg_adds", "lol_trades"]
    print("== Parameter grid")
    print(results[cols].to_string(index=False))
    print(f"\nTrade logs and grid_results.csv in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
