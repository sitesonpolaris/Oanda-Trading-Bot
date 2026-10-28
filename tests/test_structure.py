import pandas as pd

from structure import StructureConfig, find_swings, market_structure


def frame(closes):
    """Candles with a 2-pip wick either side of a 1-pip body ending at each close."""
    times = pd.date_range("2025-01-06", periods=len(closes), freq="4h", tz="UTC")
    rows = [(c - 0.0001, c + 0.0002, c - 0.0003, c) for c in closes]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(time=times)


def zigzag(points, steps=4):
    """Linear legs between turning points, e.g. [1.10, 1.12, 1.11, 1.13]."""
    out = []
    for a, b in zip(points, points[1:]):
        out += [a + (b - a) * k / steps for k in range(steps)]
    return out + [points[-1]]


def test_swings_are_local_extremes():
    df = frame(zigzag([1.10, 1.12, 1.11, 1.13]))
    swings = find_swings(df, strength=2)
    assert list(swings.index[swings["swing_high"]]) == [4]
    assert list(swings.index[swings["swing_low"]]) == [8]


def test_rising_highs_and_lows_is_uptrend():
    s = market_structure(frame(zigzag([1.10, 1.12, 1.11, 1.13, 1.12, 1.14, 1.13, 1.15])))
    assert s["trend"].iloc[-1] == "up"


def test_falling_highs_and_lows_is_downtrend():
    s = market_structure(frame(zigzag([1.15, 1.13, 1.14, 1.12, 1.13, 1.11, 1.12, 1.10])))
    assert s["trend"].iloc[-1] == "down"


def test_mixed_swings_is_sideways():
    s = market_structure(frame(zigzag([1.10, 1.13, 1.11, 1.12, 1.12, 1.14, 1.10, 1.115])))
    assert s["trend"].iloc[-1] == "sideways"


def test_bullish_bos_when_close_breaks_last_swing_high():
    df = frame(zigzag([1.10, 1.12, 1.11, 1.13]))
    s = market_structure(df)
    bos = s.index[s["bos"] == "bullish"]
    assert len(bos) == 1
    assert df["close"].iloc[bos[0]] > df["high"].iloc[4]
    assert df["close"].iloc[bos[0] - 1] <= df["high"].iloc[4]


def test_opposite_bos_neutralises_trend():
    # Higher highs and lows, then a close below the last swing low (1.125) before a new swing forms.
    up_then_break = zigzag([1.10, 1.12, 1.11, 1.13, 1.12, 1.14, 1.125, 1.145, 1.12])
    assert market_structure(frame(up_then_break))["trend"].iloc[-1] == "sideways"
    kept = market_structure(frame(up_then_break), StructureConfig(bos_invalidates_trend=False))
    assert kept["trend"].iloc[-1] == "up"


def test_swing_not_known_before_confirmation():
    df = frame(zigzag([1.10, 1.12, 1.11, 1.13]))
    s = market_structure(df)
    assert pd.isna(s["last_swing_high"].iloc[5])  # swing at 4 needs candles 5 and 6
    assert s["last_swing_high"].iloc[6] == df["high"].iloc[4]
