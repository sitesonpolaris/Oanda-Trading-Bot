"""Backtest the daily trend-following strategy across FX pairs and check the pass mark.

Development data: everything before the holdout date. `--holdout` runs on the
holdout period instead (only once the development run has passed).

Usage:
    python -m trend.backtest
    python -m trend.backtest --holdout
"""

import argparse

import pandas as pd

from backtest import HOLDOUT_FROM, OUT_DIR
from fetch_candles import DATA_DIR
from trend.strategy import PAIRS, PASS_MAX_DRAWDOWN, PASS_PROFIT_FACTOR, TrendParams, metrics, simulate


def load_daily(pair):
    """Daily candles keyed by trading date (OANDA days run 17:00-17:00 New York).

    Weekend stub candles in older data are merged into the following Monday.
    """
    df = pd.read_csv(DATA_DIR / f"{pair}_D.csv", parse_dates=["time"])
    day = (df["time"].dt.tz_convert("America/New_York") + pd.Timedelta(hours=7)).dt.normalize().dt.tz_localize(None)
    weekend = day.dt.dayofweek >= 5
    day = day.where(~weekend, day + pd.to_timedelta(7 - day.dt.dayofweek, unit="D"))
    g = df.assign(date=day).groupby("date", sort=True)
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last()}).reset_index()


def report(trades, equity, p, label):
    m = metrics(trades, equity, p.starting_balance)
    eq = equity.set_index("date")["equity"]
    print(f"\n=== {label}: {pd.Timestamp(eq.index[0]):%Y-%m-%d} -> {pd.Timestamp(eq.index[-1]):%Y-%m-%d} "
          f"({m['years']:.1f} years), ${p.starting_balance:,.0f} start, {p.risk_pct}% risk per trade ===")
    print(f"Trades {m['trades']} ({m['trades_per_year']:.0f}/year) | win rate {m['win_rate']:.0%} | "
          f"avg win {m['avg_win_r']:+.2f}R, avg loss {m['avg_loss_r']:+.2f}R")
    print(f"Profit factor {m['profit_factor']:.2f} | net ${m['net_profit']:+,.0f} | final ${m['final_equity']:,.0f} | "
          f"{m['cagr']:+.1%} a year | worst drawdown {m['max_drawdown']:.1%} (${m['max_drawdown_usd']:,.0f})")
    passed = m["profit_factor"] >= PASS_PROFIT_FACTOR and m["max_drawdown"] < PASS_MAX_DRAWDOWN
    print(f"Pass mark (profit factor >= {PASS_PROFIT_FACTOR} and drawdown < {PASS_MAX_DRAWDOWN:.0%}): "
          f"{'PASS' if passed else 'FAIL'}")
    yearly = eq.groupby(pd.DatetimeIndex(eq.index).year).agg(["first", "last"])
    yearly["return"] = yearly["last"] / pd.concat([pd.Series([p.starting_balance], index=[yearly.index[0]]),
                                                   yearly["last"].iloc[:-1]]).set_axis(yearly.index) - 1
    print("By year: " + ", ".join(f"{y} {r:+.1%}" for y, r in yearly["return"].items()))
    by_pair = trades.groupby("pair")["pnl"].agg(["count", "sum"]).sort_values("sum")
    print("By pair ($): " + ", ".join(f"{k} {v['sum']:+,.0f} ({v['count']:.0f})" for k, v in by_pair.iterrows()))
    print("Exits: " + ", ".join(f"{k} {v}" for k, v in trades["exit_reason"].value_counts().items()))
    return m, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", action="store_true", help="run on the holdout period instead")
    args = parser.parse_args()
    p = TrendParams()
    holdout = pd.Timestamp(HOLDOUT_FROM)
    candles = {pair: load_daily(pair) for pair in PAIRS}
    if args.holdout:  # a fresh account from the holdout date; indicators warm up on earlier data
        trades, equity = simulate(candles, p, start=holdout)
        label = "HOLDOUT"
    else:
        trades, equity = simulate(candles, p, end=holdout)
        label = "Development"
    report(trades, equity, p, label)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_holdout" if args.holdout else ""
    trades.to_csv(OUT_DIR / f"trend_trades{suffix}.csv", index=False)
    equity.to_csv(OUT_DIR / f"trend_equity{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
