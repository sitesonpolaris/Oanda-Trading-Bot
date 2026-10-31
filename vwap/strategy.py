"""Daily Anchored VWAP Signal Strategy: a port of the RoboQuant strategy
(daily_anchored_vwap_strategy_1.rq) with the MES_3min.set inputs, for backtesting
on OANDA bid/ask candles.

Indicators (on mid prices, 3-minute bars):
  - VWAP of the typical price (high + low + close) / 3, restarting at midnight ET.
    OANDA volume is a tick count, so this VWAP approximates the exchange VWAP.
  - EMA of the close (`ema_length`), ATR (`atr_period`) and a baseline ATR
    (`baseline_atr_period`), all running continuously across days.

Signals, only on bars inside the session (`session_start` to `session_end` ET,
inclusive), checked in this order each bar, exactly as the RoboQuant code does:
  1. Waiting for a red triangle and the close is above the EMA: red triangle (closes a short).
  2. Waiting for a green triangle and the close is below the EMA: green triangle (closes a long).
  3. After a red triangle, the EMA crosses above VWAP: arm the pullback buy.
  4. After a green triangle, the EMA crosses below VWAP: arm the pullback sell.
  5. Pullback buy armed and the close crosses below VWAP: green dot (long).
  6. Pullback sell armed and the close crosses above VWAP: red dot (short).
  7. Otherwise, with nothing in progress: EMA below VWAP and the close below the EMA
     is a red dot (short, then wait for the red triangle); EMA above VWAP and the
     close above the EMA is a green dot (long, then wait for the green triangle).
  A day starts with everything reset.

Trading:
  - Enter on a dot when flat, when the day's attempts allow it, and when the ATR
    is at most `max_relative_atr` x the baseline ATR (if the filter is on).
  - Attempts: if the day's first trade wins, no more trades that day; if it loses,
    up to `additional_attempts` more.
  - Exits: the matching triangle, the optional ATR stop, and (added here) every
    position closes at the session end. No entry that would fill on the
    session-end bar. A pullback dot (steps 5-6) arms no triangle, so those trades
    close at the session end unless a later dot arms one.
  - Orders are sent at a bar's close and fill at the next bar's open (buys at the
    ask, sells at the bid), so the spread is the trading cost. The session-end
    exit fills at the session-end bar's close. An ATR stop fills at the stop, or
    at the open if the bar gaps through it.

Variants (not in the RoboQuant strategy): `anchor="ema"` replaces VWAP with a slow
EMA of the close (`slow_ema_length`) everywhere above; `signal_mode="crossover"`
replaces the dots and triangles with the EMA crossing the anchor line (see VwapParams).

Bars are stamped with their close time: the 15:57-16:00 bar is the 16:00 bar.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"


@dataclass
class VwapParams:
    # Defaults are the MES_3min.set inputs.
    use_typical_price: bool = True
    session_start_hhmm: int = 930
    session_end_hhmm: int = 1600
    ema_length: int = 11
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    use_atr_stop: bool = False
    use_relative_volatility_filter: bool = True
    baseline_atr_period: int = 50
    max_relative_atr: float = 1.1
    additional_attempts: int = 2
    # Not in the RoboQuant strategy:
    close_at_session_end: bool = True
    bar_minutes: int = 3
    # The line the EMA is compared with: "vwap" (the original) or "ema" (a slow EMA of the close).
    anchor: str = "vwap"
    slow_ema_length: int = 50
    # "dots": the RoboQuant signals. "crossover": the EMA crossing the anchor line is the signal;
    # a cross up is a green dot and closes a short, a cross down a red dot and closes a long, and an
    # opposite cross reverses the position (same attempts and filters).
    signal_mode: str = "dots"


def resample(m1, minutes=3):
    """1-minute bid/ask candles -> `minutes` bars: mid OHLC, bid/ask open and close, volume.

    Bars are stamped with their close time and only contain minutes that traded.
    """
    df = m1.copy()
    for k in ("open", "high", "low", "close"):
        df[k] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2
    df["bucket"] = df["time"].dt.floor(f"{minutes}min")
    g = df.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "bid_open": g["bid_open"].first(), "ask_open": g["ask_open"].first(),
        "bid_close": g["bid_close"].last(), "ask_close": g["ask_close"].last(),
        "bid_low": g["bid_low"].min(), "ask_high": g["ask_high"].max(),
        "volume": g["volume"].sum(),
    })
    out.index = out.index + pd.Timedelta(minutes=minutes)
    return out.rename_axis("time").reset_index()


def wilder_atr(df, period):
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def indicators(bars, p):
    """Add ET date/time, VWAP, EMA, ATR and baseline ATR columns."""
    df = bars.copy()
    et = df["time"].dt.tz_convert(ET)
    df["et_date"] = et.dt.date
    df["et_hhmm"] = et.dt.hour * 100 + et.dt.minute
    price = (df["high"] + df["low"] + df["close"]) / 3 if p.use_typical_price else df["close"]
    vol = df["volume"].clip(lower=0)
    cum_pv = (price * vol).groupby(df["et_date"]).cumsum()
    cum_v = vol.groupby(df["et_date"]).cumsum()
    df["vwap"] = (cum_pv / cum_v).where(cum_v > 0)
    df["ema"] = ema(df["close"], p.ema_length)
    if p.anchor == "ema":  # the anchor line replaces VWAP everywhere ("vwap" below means the anchor)
        df["vwap"] = ema(df["close"], p.slow_ema_length)
    elif p.anchor != "vwap":
        raise ValueError(f"anchor must be 'vwap' or 'ema', not {p.anchor!r}")
    df["atr"] = wilder_atr(df, max(p.atr_period, 1))
    df["baseline_atr"] = wilder_atr(df, max(p.baseline_atr_period, 2))
    return df


def ema(series, length):
    length = max(length, 1)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def in_session(hhmm, p):
    a, b = min(max(p.session_start_hhmm, 0), 2359), min(max(p.session_end_hhmm, 0), 2359)
    return a <= hhmm <= b if a <= b else hhmm >= a or hhmm <= b


def run(bars, p=None):
    """Simulate the strategy on 3-minute bars from `resample`. Returns (trades, signals)."""
    p = p or VwapParams()
    df = indicators(bars, p)
    cols = {c: df[c].to_numpy() for c in ("open", "high", "low", "close", "bid_open", "ask_open", "bid_close",
                                          "ask_close", "bid_low", "ask_high", "vwap", "ema", "atr", "baseline_atr",
                                          "et_hhmm")}
    times, dates = df["time"].to_numpy(), df["et_date"].to_numpy()
    n = len(df)

    day = None
    regime = 0
    waiting_red = waiting_green = has_prev = False
    prev_close = prev_vwap = prev_ema = np.nan
    if p.signal_mode not in ("dots", "crossover"):
        raise ValueError(f"signal_mode must be 'dots' or 'crossover', not {p.signal_mode!r}")
    first_closed = first_won = False
    attempts = 0
    position = 0            # +1 long, -1 short (filled or pending fill at the next open)
    entry = None            # dict for the open trade
    pending = None          # ("enter", side, reason, stop) or ("exit", reason, then_enter) for the next open
    trades, signals = [], []

    def close_trade(k, price, reason):
        nonlocal position, entry, first_closed, first_won, attempts
        pnl = (price - entry["entry_price"]) * entry["side"]
        trades.append({**entry, "exit_time": times[k], "exit_price": price, "exit_reason": reason, "pnl": pnl})
        if not first_closed:
            first_closed, first_won = True, pnl > 0
        else:
            attempts += 1
        position, entry = 0, None

    for k in range(n):
        # 1. Fill whatever was ordered at the previous bar's close, at this bar's open. A reversal
        #    closes first, then enters if the day's attempts still allow it.
        if pending is not None and entry is not None and pending[0] == "exit":
            close_trade(k, cols["bid_open"][k] if entry["side"] > 0 else cols["ask_open"][k], pending[1])
            allowed = not first_closed or (not first_won and attempts < p.additional_attempts)
            pending = pending[2] if allowed else None
            position = pending[1] if pending is not None else 0
        if pending is not None and entry is None and pending[0] == "enter":
            _, side, reason, stop_dist = pending
            price = cols["ask_open"][k] if side > 0 else cols["bid_open"][k]
            entry = {"side": side, "entry_time": times[k], "entry_price": price, "entry_reason": reason,
                     "stop": price - side * stop_dist if stop_dist else np.nan}
        pending = None

        # 2. ATR stop during the bar.
        if entry is not None and not np.isnan(entry["stop"]):
            s = entry["stop"]
            if entry["side"] > 0 and cols["bid_low"][k] <= s:
                close_trade(k, min(s, cols["bid_open"][k]), "atr_stop")
            elif entry["side"] < 0 and cols["ask_high"][k] >= s:
                close_trade(k, max(s, cols["ask_open"][k]), "atr_stop")

        # 3. New day: reset everything, as on_bar does.
        if dates[k] != day:
            day = dates[k]
            regime, waiting_red, waiting_green, has_prev = 0, False, False, False
            first_closed = first_won = False
            attempts = 0

        close, ema, vwap = cols["close"][k], cols["ema"][k], cols["vwap"][k]
        atr, base = cols["atr"][k], cols["baseline_atr"][k]
        if np.isnan(ema) or np.isnan(vwap) or np.isnan(atr) or np.isnan(base) or atr <= 0 or base <= 0:
            continue
        vol_ok = not p.use_relative_volatility_filter or atr / base <= max(p.max_relative_atr, 0.1)
        hhmm = cols["et_hhmm"][k]
        red_dot = green_dot = red_tri = green_tri = pullback = False
        if in_session(hhmm, p) and p.signal_mode == "crossover":
            if has_prev and prev_ema <= prev_vwap and ema > vwap:
                green_dot = red_tri = True
            elif has_prev and prev_ema >= prev_vwap and ema < vwap:
                red_dot = green_tri = True
        elif in_session(hhmm, p):
            if waiting_red and close > ema:
                waiting_red, regime, red_tri = False, -2, True
            elif waiting_green and close < ema:
                waiting_green, regime, green_tri = False, 2, True
            elif regime == -2 and ema > vwap:
                regime = -3
            elif regime == 2 and ema < vwap:
                regime = 3
            elif regime == -3 and has_prev and prev_close >= prev_vwap and close < vwap:
                regime, green_dot, pullback = 0, True, True
            elif regime == 3 and has_prev and prev_close <= prev_vwap and close > vwap:
                regime, red_dot, pullback = 0, True, True
            elif regime == 0:
                if ema < vwap and close < ema:
                    waiting_red, regime, red_dot = True, -1, True
                elif ema > vwap and close > ema:
                    waiting_green, regime, green_dot = True, 1, True
        for name, flag in (("red_dot", red_dot), ("green_dot", green_dot), ("red_triangle", red_tri),
                           ("green_triangle", green_tri)):
            if flag:
                signals.append({"time": times[k], "signal": name, "close": close, "vwap": vwap, "ema": ema})

        allowed = not first_closed or (not first_won and attempts < p.additional_attempts)
        closes_at_end = p.close_at_session_end and p.session_start_hhmm <= p.session_end_hhmm
        end_minutes = p.session_end_hhmm // 100 * 60 + p.session_end_hhmm % 100
        after_end = closes_at_end and hhmm >= p.session_end_hhmm
        last_entry_bar = closes_at_end and hhmm // 100 * 60 + hhmm % 100 >= end_minutes - p.bar_minutes
        order = None
        if (green_dot or red_dot) and vol_ok and not last_entry_bar and k + 1 < n:
            side = 1 if green_dot else -1
            stop_dist = atr * max(p.atr_stop_multiplier, 0.1) if p.use_atr_stop else 0.0
            reason = ("green_dot" if green_dot else "red_dot") + ("_pullback" if pullback else "")
            order = ("enter", side, reason, stop_dist)
        if order is not None and allowed and position == 0:
            pending, position = order, order[1]
        elif green_tri and position > 0 and entry is not None and k + 1 < n:
            pending = ("exit", "green_triangle", order)
        elif red_tri and position < 0 and entry is not None and k + 1 < n:
            pending = ("exit", "red_triangle", order)

        # Session end: flat at this bar's close, and any unfilled order is dropped.
        if after_end:
            if entry is not None:
                close_trade(k, cols["bid_close"][k] if entry["side"] > 0 else cols["ask_close"][k], "session_end")
            pending, position = None, 0

        prev_close, prev_vwap, prev_ema, has_prev = close, vwap, ema, True

    trade_df = pd.DataFrame(trades)
    if len(trade_df):
        for c in ("entry_time", "exit_time"):
            trade_df[c] = pd.to_datetime(trade_df[c], utc=True)
    return trade_df, pd.DataFrame(signals)
