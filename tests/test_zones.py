import pandas as pd
import pytest

from zones import ZoneConfig, atr, detect_zones

FLAT = (1.1000, 1.1010, 1.0990, 1.1002)  # open, high, low, close: 20-pip range


def frame(candles):
    times = pd.date_range("2025-01-06", periods=len(candles), freq="h", tz="UTC")
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"]).assign(time=times)


def history(n=20):
    return [FLAT] * n


# Leg-in / base / leg-out building blocks around price ~1.1000.
RALLY_IN = (1.0990, 1.1014, 1.0988, 1.1012)  # closes above the previous high (1.1010): move-in rally
DROP_IN = (1.1010, 1.1012, 1.0986, 1.0988)   # closes below the previous low (1.0990): move-in drop
BASE_A = (1.1000, 1.1006, 1.0995, 1.1002)  # body 1.1000-1.1002, wick 1.0995-1.1006
BASE_B = (1.1003, 1.1008, 1.0997, 1.1001)  # body 1.1001-1.1003, wick 1.0997-1.1008
RALLY_OUT = (1.1002, 1.1062, 1.1000, 1.1060)  # 62-pip range, strong body
DROP_OUT = (1.1002, 1.1004, 1.0942, 1.0944)


@pytest.mark.parametrize("leg_in,leg_out,expected", [
    (RALLY_IN, RALLY_OUT, "RBR"),
    (DROP_IN, RALLY_OUT, "DBR"),
    (DROP_IN, DROP_OUT, "DBD"),
    (RALLY_IN, DROP_OUT, "RBD"),
])
def test_classifies_all_four_zone_types(leg_in, leg_out, expected):
    zones = detect_zones(frame(history() + [leg_in, BASE_A, leg_out]))
    assert list(zones["type"]) == [expected]
    assert zones["kind"].iloc[0] == ("demand" if expected.endswith("R") else "supply")


def test_single_candle_base_uses_wick_edges():
    z = detect_zones(frame(history() + [RALLY_IN, BASE_A, RALLY_OUT])).iloc[0]
    assert (z.top, z.bottom) == (1.1006, 1.0995)
    assert z.base_candles == 1


def test_multi_candle_base_entry_at_body_stop_side_at_wick():
    df = frame(history() + [RALLY_IN, BASE_A, BASE_B, RALLY_OUT])
    z = detect_zones(df).iloc[0]
    assert (z.top, z.bottom) == (1.1003, 1.0995)  # demand: entry = body top, far edge = lowest wick
    supply = detect_zones(frame(history() + [DROP_IN, BASE_A, BASE_B, DROP_OUT])).iloc[0]
    assert (supply.top, supply.bottom) == (1.1008, 1.1000)  # supply: entry = body bottom, far = highest wick
    assert z.base_candles == 2
    assert z.start_time == df["time"].iloc[-3]  # first base candle


def test_impulse_size_reported_in_pips_and_multiples():
    z = detect_zones(frame(history() + [RALLY_IN, BASE_A, RALLY_OUT])).iloc[0]
    assert z.impulse_pips == pytest.approx(62.0)
    assert z.impulse_ratio >= 2.5


def test_any_close_outside_previous_range_is_an_impulse_by_default():
    weak = (1.1002, 1.1040, 1.1000, 1.1038)  # 40 pips, only ~2x the 20-pip average
    assert len(detect_zones(frame(history() + [RALLY_IN, BASE_A, weak]))) == 1


def test_optional_strength_multiple():
    weak = (1.1002, 1.1040, 1.1000, 1.1038)
    assert detect_zones(frame(history() + [RALLY_IN, BASE_A, weak]), ZoneConfig(legout_multiple=2.5)).empty
    assert len(detect_zones(frame(history() + [RALLY_IN, BASE_A, weak]), ZoneConfig(legout_multiple=1.5))) == 1


def test_close_inside_previous_range_is_not_an_impulse():
    # Big candle, strong body, but closes 1 pip below the base candle's high (1.1006).
    inside = (1.0995, 1.1050, 1.0994, 1.1005)
    assert detect_zones(frame(history() + [RALLY_IN, BASE_A, inside])).empty


def test_direction_comes_from_the_break_not_the_candle_colour():
    # Gaps up, closes red but still above the base high: bullish impulse.
    red_break = (1.1060, 1.1065, 1.1020, 1.1025)
    z = detect_zones(frame(history() + [RALLY_IN, BASE_A, red_break])).iloc[0]
    assert z.kind == "demand" and z.type == "RBR"


def test_basing_candle_cannot_be_the_impulse():
    drift = (1.1004, 1.1009, 1.1001, 1.1007)  # small doji-ish candle closing just above BASE_A's high
    assert detect_zones(frame(history() + [RALLY_IN, BASE_A, drift])).empty


def test_base_of_up_to_five_candles():
    df = frame(history() + [RALLY_IN] + [BASE_A, BASE_B, BASE_A, BASE_B, BASE_A] + [RALLY_OUT])
    assert detect_zones(df).iloc[0].base_candles == 5


def test_base_longer_than_max_is_rejected():
    df = frame(history() + [RALLY_IN] + [BASE_A, BASE_B] * 3 + [RALLY_OUT])
    assert detect_zones(df).empty
    assert detect_zones(df, ZoneConfig(max_base_candles=6)).iloc[0].base_candles == 6


def test_leg_in_must_be_an_impulse():
    no_break = (1.0992, 1.1009, 1.0991, 1.1008)  # strong body but closes inside FLAT's range
    assert detect_zones(frame(history() + [no_break, BASE_A, RALLY_OUT])).empty


def test_leg_in_direction_comes_from_the_break_not_the_candle_colour():
    green_drop = (1.0975, 1.0995, 1.0970, 1.0985)  # green, but closes below FLAT's low: a drop in
    z = detect_zones(frame(history() + [green_drop, BASE_A, RALLY_OUT])).iloc[0]
    assert z.type == "DBR"


def test_no_base_means_no_zone():
    assert detect_zones(frame(history() + [RALLY_IN, RALLY_OUT])).empty


def test_atr_matches_wilder_smoothing():
    df = frame([FLAT] * 20)
    assert atr(df, 14).iloc[-1] == pytest.approx(0.0020)
    assert atr(df, 14).iloc[:13].isna().all()
