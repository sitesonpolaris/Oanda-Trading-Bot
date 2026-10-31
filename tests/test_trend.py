import numpy as np
import pandas as pd
import pytest

from trend.backtest import load_daily
from trend.strategy import SPREAD_PIPS, TrendParams, simulate

P = TrendParams(entry_days=5, exit_days=3, atr_days=3, slippage_pips=0.0, financing_pct_year=0.0)
FLAT = (1.1000, 1.1010, 1.0990, 1.1000)


def daily(rows, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(date=dates)


def breakout_then(rest):
    """Flat days, then a close above the 5-day high (signal), then `rest`."""
    return [FLAT] * 8 + [(1.1000, 1.1040, 1.0995, 1.1035)] + rest


def test_entry_next_open_sized_to_quarter_percent_risk():
    rows = breakout_then([(1.1036, 1.1050, 1.1030, 1.1045)] * 3)
    candles = {"EUR_USD": daily(rows)}
    trades, equity = simulate(candles, P)
    assert trades.empty and equity["open"].iat[-1] == 1
    # The open position: fill = next open + half spread, stop 2 ATR below, $25 at risk.
    half = SPREAD_PIPS["EUR_USD"] * 0.0001 / 2
    rows_df = daily(rows)
    from trend.strategy import prepare
    a = prepare(rows_df, P)["atr"].iat[8]
    units = 25 / (2 * a)  # $25 (0.25% of $10,000) lost if the 2-ATR stop is hit
    assert equity["equity"].iat[9] == pytest.approx(10_000 + (1.1045 - (1.1036 + half)) * units)


def test_stop_fills_at_stop_or_gap_open():
    gap_down = (1.0900, 1.0910, 1.0880, 1.0890)  # opens far below any 2-ATR stop
    trades, _ = simulate({"EUR_USD": daily(breakout_then([(1.1036, 1.1040, 1.1030, 1.1035), gap_down]))}, P)
    t = trades.iloc[0]
    half = SPREAD_PIPS["EUR_USD"] * 0.0001 / 2
    assert t.exit_reason == "stop" and t.exit == pytest.approx(1.0900 - half)
    assert t.r < -1  # a gap through the stop loses more than 1R


def test_exit_channel_closes_at_next_open():
    rise = [(1.1036 + i * 0.001, 1.1046 + i * 0.001, 1.1030 + i * 0.001, 1.1044 + i * 0.001) for i in range(6)]
    slip = [(1.1090, 1.1092, 1.1060, 1.1062), (1.1050, 1.1055, 1.1040, 1.1045)]  # 2nd closes below the 3-day low
    after = [(1.1048, 1.1052, 1.1044, 1.1050)]  # exit fills at this open
    trades, _ = simulate({"EUR_USD": daily(breakout_then(rise + slip + after))}, P)
    t = trades.iloc[0]
    half = SPREAD_PIPS["EUR_USD"] * 0.0001 / 2
    assert t.exit_reason == "exit_channel" and t.exit == pytest.approx(1.1048 - half) and t.pnl > 0


def test_no_more_than_two_positions_leaning_on_one_currency():
    rows = breakout_then([(1.1036, 1.1050, 1.1030, 1.1045)] * 2)
    jpy = [(150.0, 150.1, 149.9, 150.0)] * len(rows)
    candles = {"EUR_USD": daily(rows), "EUR_GBP": daily(rows), "EUR_CHF": daily(rows),
               "GBP_USD": daily([(1.3, 1.301, 1.299, 1.3)] * len(rows)),
               "USD_CHF": daily([(0.9, 0.901, 0.899, 0.9)] * len(rows)), "USD_JPY": daily(jpy)}
    _, equity = simulate(candles, P)
    assert equity["open"].max() == 2  # three long-EUR breakouts, only two allowed


def test_weekend_stub_candles_merge_into_monday(tmp_path, monkeypatch):
    import trend.backtest as tb
    times = pd.to_datetime(["2008-01-03 22:00", "2008-01-04 22:00", "2008-01-05 22:00", "2008-01-06 22:00"], utc=True)
    pd.DataFrame({"time": times, "open": [1, 2, 3, 4], "high": [5, 6, 9, 7], "low": [1, 1, 0.5, 2],
                  "close": [2, 3, 4, 5], "volume": 1}).to_csv(tmp_path / "EUR_USD_D.csv", index=False)
    monkeypatch.setattr(tb, "DATA_DIR", tmp_path)
    d = load_daily("EUR_USD")
    # Thu 17:00 NY -> Friday; Fri 17:00 and Sat 17:00 stubs -> Monday, merged with Sun 17:00 -> Monday.
    assert list(d["date"].dt.day_name()) == ["Friday", "Monday"]
    assert d.iloc[1][["open", "high", "low", "close"]].tolist() == [2, 9, 0.5, 5]
