"""Daily trend following across many FX pairs: a 55-day breakout with a 20-day exit.

Written down, with the pass mark, before the first run (Oct 31, 2026). The rules are
not to be tuned on the development data.

Rules (long; short is the mirror image), on daily mid-price candles:
  1. Entry signal: the day closes above the highest high of the previous
     `entry_days` (55) days.
  2. Fill: the next day's open, paying half the spread plus `slippage_pips`.
  3. Initial stop: `stop_atr` (2) x ATR(`atr_days`, 20) below the fill, held as a
     stop order: it fills at the stop, or at the open if the day gaps through it.
  4. Exit: the day closes below the lowest low of the previous `exit_days` (20)
     days; fill at the next open. No profit target.
  5. Size: risk `risk_pct` (0.25%) of the account balance between the fill and the
     stop, converted to USD at the day's rate.
  6. Portfolio limits: at most `max_positions` (5) open, and at most
     `max_per_currency` (2) positions leaning the same way on any one currency
     (long EUR/USD and long EUR/JPY both count as long EUR). When more signals
     arrive than fit, the strongest breakouts (close beyond the channel, in ATRs)
     go first.
  7. Costs: the spread (fixed per pair, `SPREAD_PIPS`), slippage on every fill,
     and financing of `financing_pct_year` (1.5%) a year of the position's USD value
     for every night it's held. OANDA's real financing can be a cost or a credit;
     a flat cost is the cautious choice.

Pass mark, decided in advance, on the development data (everything before the
holdout date): profit factor >= 1.3 AND worst drawdown of the daily
marked-to-market equity < 8%. If it passes, it gets one run on the holdout.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PASS_PROFIT_FACTOR = 1.3
PASS_MAX_DRAWDOWN = 0.08

PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD", "USD_CHF",
         "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY",
         "EUR_GBP", "EUR_AUD", "EUR_CHF", "EUR_CAD", "GBP_CHF", "GBP_AUD", "AUD_NZD", "AUD_CAD"]
# Typical OANDA spreads in pips, on the cautious side.
SPREAD_PIPS = {"EUR_USD": 1.4, "GBP_USD": 2.0, "AUD_USD": 1.6, "NZD_USD": 2.2, "USD_JPY": 1.6, "USD_CAD": 2.2,
               "USD_CHF": 2.0, "EUR_JPY": 2.2, "GBP_JPY": 3.5, "AUD_JPY": 2.4, "CAD_JPY": 3.0, "NZD_JPY": 3.0,
               "CHF_JPY": 3.5, "EUR_GBP": 1.8, "EUR_AUD": 3.0, "EUR_CHF": 2.2, "EUR_CAD": 3.0, "GBP_CHF": 4.0,
               "GBP_AUD": 4.0, "AUD_NZD": 3.0, "AUD_CAD": 3.0}
# The USD pair that converts each quote currency to USD: (pair, True if USD is that pair's quote).
TO_USD = {"USD": None, "EUR": ("EUR_USD", True), "GBP": ("GBP_USD", True), "AUD": ("AUD_USD", True),
          "NZD": ("NZD_USD", True), "JPY": ("USD_JPY", False), "CAD": ("USD_CAD", False), "CHF": ("USD_CHF", False)}


@dataclass
class TrendParams:
    entry_days: int = 55
    exit_days: int = 20
    atr_days: int = 20
    stop_atr: float = 2.0
    risk_pct: float = 0.25
    max_positions: int = 5
    max_per_currency: int = 2
    slippage_pips: float = 0.5
    financing_pct_year: float = 1.5
    starting_balance: float = 10_000.0


def pip(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def atr(df, days):
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / days, adjust=False, min_periods=days).mean()


def prepare(df, p):
    """Add the channels and ATR, each known at that day's close (previous days only)."""
    out = df.copy()
    out["entry_high"] = out["high"].rolling(p.entry_days).max().shift()
    out["entry_low"] = out["low"].rolling(p.entry_days).min().shift()
    out["exit_high"] = out["high"].rolling(p.exit_days).max().shift()
    out["exit_low"] = out["low"].rolling(p.exit_days).min().shift()
    out["atr"] = atr(out, p.atr_days)
    return out


def usd_rates(candles):
    """Daily close of 1 unit of each quote currency in USD, by date (forward-filled)."""
    dates = sorted(set().union(*(set(c["date"]) for c in candles.values())))
    rates = pd.DataFrame(index=pd.Index(dates, name="date"))
    for ccy, conv in TO_USD.items():
        if conv is None:
            rates[ccy] = 1.0
        elif conv[0] in candles:
            pair, usd_quote = conv
            close = candles[pair].set_index("date")["close"]
            rates[ccy] = (close if usd_quote else 1 / close).reindex(rates.index).ffill()
    return rates


def simulate(candles, p=None, end=None, start=None):
    """Run the portfolio day by day. `candles`: {pair: DataFrame(date, open, high, low, close)}.

    Trades from `start` (indicators use the earlier data) up to, not including, `end`.

    Returns (trades, equity): closed trades in USD, and daily marked-to-market equity.
    """
    p = p or TrendParams()
    data = {k: prepare(v, p).set_index("date") for k, v in candles.items()}
    rates = usd_rates(candles)
    dates = rates.index
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates < end]
    balance = p.starting_balance
    open_pos = {}          # pair -> position dict
    orders = []            # signals from the previous close: (pair, side, strength, atr)
    exits = set()          # pairs to close at the next open
    trades, equity = [], []

    def fx(pair, d):
        return rates.at[d, pair[4:]]

    def close(pair, d, price, reason):
        nonlocal balance
        pos = open_pos.pop(pair)
        half = SPREAD_PIPS[pair] * pip(pair) / 2 + p.slippage_pips * pip(pair)
        fill = price - half * pos["side"]
        pnl = (fill - pos["entry"]) * pos["side"] * pos["units"] * fx(pair, d) - pos["financing"]
        balance += pnl
        trades.append({**pos, "pair": pair, "exit_date": d, "exit": fill, "exit_reason": reason, "pnl": pnl,
                       "r": pnl / pos["risk_usd"]})

    def exposure(extra=None):
        net = {}
        for pair, pos in list(open_pos.items()) + ([extra] if extra else []):
            net[pair[:3]] = net.get(pair[:3], 0) + pos["side"]
            net[pair[4:]] = net.get(pair[4:], 0) - pos["side"]
        return net

    for d in dates:
        bars = {k: v.loc[d] for k, v in data.items() if d in v.index}
        # 1. Exits ordered at the last close fill at today's open.
        for pair in sorted(exits):
            if pair in open_pos and pair in bars:
                close(pair, d, bars[pair]["open"], "exit_channel")
        exits = set()
        # 2. Entries ordered at the last close, strongest first, within the limits.
        for pair, side, strength, a in sorted(orders, key=lambda o: -o[2]):
            if pair in open_pos or pair not in bars or len(open_pos) >= p.max_positions:
                continue
            if any(abs(v) > p.max_per_currency for v in exposure((pair, {"side": side})).values()):
                continue
            half = SPREAD_PIPS[pair] * pip(pair) / 2 + p.slippage_pips * pip(pair)
            entry = bars[pair]["open"] + half * side
            stop = entry - side * p.stop_atr * a
            risk_usd = balance * p.risk_pct / 100
            units = risk_usd / (p.stop_atr * a * fx(pair, d))
            open_pos[pair] = {"side": side, "entry_date": d, "entry": entry, "stop": stop, "units": units,
                              "risk_usd": risk_usd, "financing": 0.0}
        orders = []
        # 3. Stops during the day (a stop on the entry day counts too).
        for pair in list(open_pos):
            if pair not in bars:
                continue
            pos, b = open_pos[pair], bars[pair]
            if pos["side"] > 0 and b["low"] <= pos["stop"]:
                close(pair, d, min(pos["stop"], b["open"]), "stop")
            elif pos["side"] < 0 and b["high"] >= pos["stop"]:
                close(pair, d, max(pos["stop"], b["open"]), "stop")
        # 4. At the close: financing, marked equity, then signals for tomorrow's open.
        marked = balance
        for pair, pos in open_pos.items():
            if pair not in bars:
                continue
            value = abs(pos["units"]) * bars[pair]["close"] * fx(pair, d)
            pos["financing"] += value * p.financing_pct_year / 100 / 365
            marked += (bars[pair]["close"] - pos["entry"]) * pos["side"] * pos["units"] * fx(pair, d) - pos["financing"]
        equity.append({"date": d, "equity": marked, "balance": balance, "open": len(open_pos)})
        for pair, b in bars.items():
            if np.isnan(b["entry_high"]) or np.isnan(b["atr"]):
                continue
            if pair in open_pos:
                side = open_pos[pair]["side"]
                if (side > 0 and b["close"] < b["exit_low"]) or (side < 0 and b["close"] > b["exit_high"]):
                    exits.add(pair)
            elif b["close"] > b["entry_high"]:
                orders.append((pair, 1, (b["close"] - b["entry_high"]) / b["atr"], b["atr"]))
            elif b["close"] < b["entry_low"]:
                orders.append((pair, -1, (b["entry_low"] - b["close"]) / b["atr"], b["atr"]))
    return pd.DataFrame(trades), pd.DataFrame(equity)


def metrics(trades, equity, starting_balance):
    eq = equity.set_index("date")["equity"]
    dd = eq / eq.cummax() - 1
    wins, losses = trades.loc[trades["pnl"] > 0, "pnl"], trades.loc[trades["pnl"] <= 0, "pnl"]
    years = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25
    return {
        "trades": len(trades), "trades_per_year": len(trades) / years if years else np.nan,
        "win_rate": len(wins) / len(trades) if len(trades) else np.nan,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "avg_win_r": trades.loc[trades["pnl"] > 0, "r"].mean(), "avg_loss_r": trades.loc[trades["pnl"] <= 0, "r"].mean(),
        "net_profit": eq.iloc[-1] - starting_balance, "final_equity": eq.iloc[-1],
        "cagr": (eq.iloc[-1] / starting_balance) ** (1 / years) - 1 if years else np.nan,
        "max_drawdown": -dd.min(), "max_drawdown_usd": -(eq - eq.cummax()).min(),
        "years": years,
    }
