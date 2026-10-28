import pandas as pd
import pytest

from backtest import BacktestConfig, simulate, summarize

PIP = 0.0001


def bars(rows):
    times = pd.date_range("2025-01-06", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(time=times)


def trend(times, label):
    return pd.DataFrame({"available_time": [times.iloc[0]], "trend": [label]})


def demand_zone(t0, top=1.1000, bottom=1.0990, impulse_pips=60, atr_pips=10):
    return pd.DataFrame([{
        "start_time": t0, "legout_time": t0, "type": "RBR", "kind": "demand",
        "top": top, "bottom": bottom, "base_candles": 1,
        "impulse_pips": impulse_pips, "impulse_ratio": 3.0, "impulse_atr": 3.0, "atr_pips": atr_pips,
    }])


AWAY = (1.1030, 1.1040, 1.1025, 1.1035)
TOUCH = (1.1010, 1.1012, 1.0995, 1.1005)


def run(rows, cfg=BacktestConfig(spread_pips=0), label="up", zone=None):
    b = bars(rows)
    z = demand_zone(b["time"].iloc[0]) if zone is None else zone
    return simulate(z, b, trend(b["time"], label), cfg, zone_tf=pd.Timedelta("1h"), pip=PIP)


def test_long_hits_target():
    win = (1.1010, 1.1060, 1.1005, 1.1055)  # target = 1.1000 + 0.8 * 60 pips = 1.1048
    t = run([AWAY, AWAY, TOUCH, win]).iloc[0]
    assert t.status == "filled" and t.result == "win"
    assert (t.entry, t.target) == (1.1000, pytest.approx(1.1048))
    assert t.stop == pytest.approx(1.0990 - 0.1 * 10 * PIP)
    assert t.r_multiple == pytest.approx(48 / 11, abs=1e-3)


def test_long_hits_stop():
    loss = (1.1000, 1.1002, 1.0980, 1.0985)
    t = run([AWAY, AWAY, TOUCH, loss]).iloc[0]
    assert t.result == "loss" and t.r_multiple == pytest.approx(-1)


def test_stop_wins_tie_when_bar_hits_both():
    both = (1.1010, 1.1060, 1.0980, 1.1000)
    assert run([AWAY, AWAY, TOUCH, both]).iloc[0].result == "loss"


def test_spread_can_prevent_fill():
    graze = (1.1010, 1.1012, 1.1000, 1.1005)  # mid low exactly at entry; ask stays above it
    assert run([AWAY, graze], BacktestConfig(spread_pips=0)).iloc[0].status == "filled"
    assert run([AWAY, graze], BacktestConfig(spread_pips=1.0)).iloc[0].status == "untouched"


def test_spread_is_paid_through_bid_ask_fills():
    win = (1.1010, 1.1060, 1.1005, 1.1055)
    with_spread = run([AWAY, AWAY, TOUCH, win], BacktestConfig(spread_pips=1.0)).iloc[0]
    assert with_spread.result == "win"
    assert with_spread.r_multiple == pytest.approx(48 / 11, abs=1e-3)  # limit fills at quoted prices
    assert with_spread.spread_r == pytest.approx(1 / 11, abs=1e-3)


def test_counter_trend_touch_is_skipped_and_burns_zone():
    t = run([AWAY, TOUCH, (1.1010, 1.1060, 1.1005, 1.1055)], label="down").iloc[0]
    assert t.status == "skipped_trend"


def test_low_reward_to_risk_is_skipped():
    z = demand_zone(bars([AWAY])["time"].iloc[0], impulse_pips=20)  # 16-pip target vs 11-pip stop
    assert run([AWAY, TOUCH], zone=z).iloc[0].status == "skipped_rr"


def test_summary_metrics():
    times = pd.date_range("2025-01-06", periods=4, freq="h", tz="UTC")
    trades = pd.DataFrame({
        "status": ["filled"] * 4, "exit_time": times,
        "r_multiple": [2.0, -1.0, -1.0, 3.0], "planned_rr": [2.0, 2.0, 2.0, 3.0],
    })
    s = summarize(trades, BacktestConfig(risk_per_trade=0.01))
    assert s["trades"] == 4 and s["win_rate_%"] == 50.0
    assert s["avg_planned_rr"] == 2.25
    assert s["total_return_%"] == pytest.approx((1.02 * 0.99 * 0.99 * 1.03 - 1) * 100, abs=0.01)
    assert s["max_drawdown_%"] == pytest.approx(1.99, abs=0.01)
