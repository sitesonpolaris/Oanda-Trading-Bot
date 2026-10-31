import numpy as np
import pandas as pd
import pytest

from vwap.strategy import VwapParams, indicators, resample, run

ET = "America/New_York"
FAST = VwapParams(ema_length=3, atr_period=2, baseline_atr_period=3, use_relative_volatility_filter=False)


def bars_from_closes(closes, start="2025-03-03 10:00", spread=0.2, volume=10):
    """3-minute bars stamped with their close time (ET), each opening at the previous close."""
    times = pd.date_range(pd.Timestamp(start, tz=ET), periods=len(closes), freq="3min").tz_convert("UTC")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    half = spread / 2
    return pd.DataFrame({"time": times, "open": o, "high": h, "low": l, "close": c,
                         "bid_open": o - half, "ask_open": o + half, "bid_close": c - half, "ask_close": c + half,
                         "bid_low": l - half, "ask_high": h + half, "volume": volume})


def test_resample_stamps_close_time_and_uses_mid_prices():
    t = pd.date_range("2025-03-03 15:00", periods=6, freq="1min", tz="UTC")
    m1 = pd.DataFrame({"time": t, "bid_open": 1.0, "bid_high": 2.0, "bid_low": 0.5, "bid_close": [1, 1, 1.5, 1, 1, 1.2],
                       "ask_open": 1.2, "ask_high": 2.2, "ask_low": 0.7, "ask_close": [1.2, 1.2, 1.7, 1.2, 1.2, 1.4],
                       "volume": [1, 2, 3, 4, 5, 6]})
    b = resample(m1, 3)
    assert list(b["time"]) == [t[0] + pd.Timedelta("3min"), t[0] + pd.Timedelta("6min")]
    assert b["close"].iat[0] == pytest.approx(1.6) and b["volume"].tolist() == [6, 15]
    assert b["ask_close"].iat[1] == pytest.approx(1.4) and b["high"].iat[0] == pytest.approx(2.1)


def test_vwap_restarts_at_midnight_et():
    b = bars_from_closes([100.0] * 3 + [200.0] * 3, start="2025-03-03 23:54")
    df = indicators(b, FAST)
    assert df["vwap"].iat[1] == pytest.approx(100.0)   # 23:57, still Monday
    tue = df[df["et_date"] == df["et_date"].iat[-1]]  # bars stamped 00:00 onwards (close time)
    typical = (tue["high"] + tue["low"] + tue["close"]) / 3
    assert len(tue) == 4 and df["vwap"].iat[-1] == pytest.approx(typical.mean())  # Monday's bars don't count


def test_red_dot_short_closed_by_red_triangle_at_next_opens():
    closes = [100.0] * 12 + [99.0, 98.0, 97.0, 96.5, 98.0, 99.5]
    b = bars_from_closes(closes)
    trades, signals = run(b, FAST)
    first_red = signals[signals["signal"] == "red_dot"]["time"].iat[0]
    tri = signals[signals["signal"] == "red_triangle"]["time"].iat[0]
    t = trades.iloc[0]
    k_dot, k_tri = int(b["time"].searchsorted(first_red)), int(b["time"].searchsorted(tri))
    assert t.side == -1 and t.entry_reason == "red_dot" and t.exit_reason == "red_triangle"
    assert t.entry_time == b["time"].iat[k_dot + 1] and t.entry_price == b["bid_open"].iat[k_dot + 1]
    assert t.exit_time == b["time"].iat[k_tri + 1] and t.exit_price == b["ask_open"].iat[k_tri + 1]


def test_position_closed_at_session_end():
    closes = [100.0] * 12 + list(np.linspace(99, 90, 40))  # keeps falling: no red triangle
    b = bars_from_closes(closes, start="2025-03-03 15:00")
    trades, _ = run(b, FAST)
    t = trades.iloc[-1]
    assert t.exit_reason == "session_end" and t.exit_time.tz_convert(ET).strftime("%H:%M") == "16:00"
    assert t.exit_price == b.loc[b["time"] == t.exit_time, "ask_close"].iat[0]


def test_daily_rules_hold_on_random_prices():
    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 480 * 5))  # 5 days of 3-minute bars, round the clock
    b = bars_from_closes(closes, start="2025-03-03 00:03")
    trades, _ = run(b, FAST)
    assert len(trades) > 5
    et_in, et_out = trades["entry_time"].dt.tz_convert(ET), trades["exit_time"].dt.tz_convert(ET)
    hhmm = lambda s: s.dt.hour * 100 + s.dt.minute  # noqa: E731
    assert (hhmm(et_in) > 930).all() and (hhmm(et_out) <= 1600).all()
    for _, day in trades.groupby(et_in.dt.date):
        assert len(day) <= 1 + FAST.additional_attempts
        if day["pnl"].iat[0] > 0:
            assert len(day) == 1  # first trade won: done for the day


def test_crossover_mode_reverses_after_a_loss_and_stops_after_a_win():
    from dataclasses import replace
    p = replace(FAST, anchor="ema", ema_length=2, slow_ema_length=6, signal_mode="crossover")
    losing = bars_from_closes([100.0] * 12 + [101, 101.5, 100.5, 98, 96, 95, 98, 101, 103, 104])  # long, cross down, cross up
    trades, _ = run(losing, p)
    first, second = trades.iloc[0], trades.iloc[1]
    assert (first.side, second.side) == (1, -1) and first.pnl < 0
    assert first.exit_reason == "green_triangle" and second.entry_time == first.exit_time  # reversed at one open
    assert first.exit_price == second.entry_price == losing.loc[losing["time"] == first.exit_time, "bid_open"].iat[0]

    winning = bars_from_closes([100.0] * 12 + [102, 104, 106, 106, 106, 103, 100, 97, 97])
    trades, signals = run(winning, p)
    assert len(trades) == 1 and trades["pnl"].iat[0] > 0  # first trade won: no reversal, done for the day
    assert (signals["signal"] == "red_dot").any()
