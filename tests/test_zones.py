import pandas as pd
import pytest

from zones import ZoneConfig, atr, detect_zones

FLAT = (1.1000, 1.1010, 1.0990, 1.1002)  # open, high, low, close: 20-pip range, small body (a basing candle)
NO_BREAKOUT = ZoneConfig(require_breakout=False)


def frame(candles):
    times = pd.date_range("2025-01-06", periods=len(candles), freq="h", tz="UTC")
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"]).assign(time=times)


def history(n=20):
    return [FLAT] * n


# Move-in / base / move-out building blocks around price ~1.1000.
RALLY_IN = (1.0990, 1.1014, 1.0988, 1.1012)  # green, closes in its top 25%, above the previous high (1.1010)
DROP_IN = (1.1010, 1.1012, 1.0986, 1.0988)   # red, closes in its bottom 25%, below the previous low (1.0990)
BASE_A = (1.1000, 1.1006, 1.0995, 1.1002)    # body 1.1000-1.1002, wick 1.0995-1.1006
BASE_B = (1.1003, 1.1008, 1.0997, 1.1001)    # body 1.1001-1.1003, wick 1.0997-1.1008
RALLY_OUT = (1.1002, 1.1062, 1.1000, 1.1060)  # 62 pips: 5.6x the base, taller than the move-in
DROP_OUT = (1.1002, 1.1004, 1.0942, 1.0944)

# History with a swing high at 1.1030 and a swing low at 1.0970 before the base, for breakout tests.
PEAK = (1.1005, 1.1030, 1.1000, 1.1008)
TROUGH = (1.0995, 1.1000, 1.0970, 1.0992)
SWINGS = history(14) + [PEAK, FLAT, FLAT, TROUGH, FLAT, FLAT]
SHORT_OUT = (1.1002, 1.1029, 1.1000, 1.1027)  # strong rally, but closes below the 1.1030 swing high


def zones(candles, cfg=NO_BREAKOUT):
    return detect_zones(frame(candles), cfg)


# --- candle rules (breakout off) -------------------------------------------------

@pytest.mark.parametrize("leg_in,leg_out,expected", [
    (RALLY_IN, RALLY_OUT, "RBR"),
    (DROP_IN, RALLY_OUT, "DBR"),
    (DROP_IN, DROP_OUT, "DBD"),
    (RALLY_IN, DROP_OUT, "RBD"),
])
def test_classifies_all_four_zone_types(leg_in, leg_out, expected):
    z = zones(history() + [leg_in, BASE_A, leg_out])
    assert list(z["type"]) == [expected]
    assert z["kind"].iloc[0] == ("demand" if expected.endswith("R") else "supply")


def test_single_candle_base_is_wick_to_wick():
    z = zones(history() + [RALLY_IN, BASE_A, RALLY_OUT]).iloc[0]
    assert (z.top, z.bottom) == (1.1006, 1.0995)
    assert z.base_candles == 1


def test_multi_candle_base_body_to_wick():
    df = history() + [RALLY_IN, BASE_A, BASE_B, RALLY_OUT]
    z = zones(df).iloc[0]
    assert (z.top, z.bottom) == (1.1003, 1.0995)  # demand: highest body to lowest wick
    supply = zones(history() + [DROP_IN, BASE_A, BASE_B, DROP_OUT]).iloc[0]
    assert (supply.top, supply.bottom) == (1.1008, 1.1000)  # supply: highest wick to lowest body
    assert z.base_candles == 2 and z.start_time == frame(df)["time"].iloc[-3]


def test_rally_candle_must_close_in_top_quarter():
    weak_close = (1.1002, 1.1062, 1.1000, 1.1040)  # closes 35% below its high
    assert zones(history() + [RALLY_IN, BASE_A, weak_close]).empty
    loose = ZoneConfig(require_breakout=False, close_zone=0.4)
    assert len(zones(history() + [RALLY_IN, BASE_A, weak_close], loose)) == 1


def test_red_candle_closing_near_its_high_is_a_basing_candle_not_a_rally():
    red = (1.1066, 1.1068, 1.1006, 1.1064)  # closes in its top 25%, but its body is tiny
    assert zones(history() + [RALLY_IN, BASE_A, red]).empty


def test_rally_candle_must_close_above_previous_high():
    inside = (1.0995, 1.1006, 1.0994, 1.1005)  # strong green close, but not above the base high (1.1006)
    assert zones(history() + [RALLY_IN, BASE_A, inside]).empty


def test_move_in_must_be_a_rally_or_drop_candle():
    no_break = (1.0992, 1.1009, 1.0991, 1.1008)  # strong green body but closes inside FLAT's range
    assert zones(history() + [no_break, BASE_A, RALLY_OUT]).empty
    green_below = (1.0975, 1.0995, 1.0970, 1.0985)  # closes below FLAT's low but green: not a drop
    assert zones(history() + [green_below, BASE_A, RALLY_OUT]).empty


def test_move_out_at_least_twice_the_biggest_base_candle():
    soft = (1.1002, 1.1021, 1.1001, 1.1020)  # 20 pips vs 11-pip base: 1.8x
    assert zones(history() + [RALLY_IN, BASE_A, soft]).empty
    strong = (1.1002, 1.1030, 1.1001, 1.1029)  # 29 pips: 2.6x, taller than the 26-pip move-in
    assert zones(history() + [RALLY_IN, BASE_A, strong]).iloc[0].impulse_vs_base == pytest.approx(2.64, abs=0.01)


def test_move_out_must_be_taller_than_move_in():
    big_in = (1.0950, 1.1014, 1.0948, 1.1012)  # 66-pip move-in, taller than the 62-pip move-out
    assert zones(history() + [big_in, BASE_A, RALLY_OUT]).empty


def test_basing_candles_have_small_bodies_and_number_one_to_five():
    assert zones(history() + [RALLY_IN] + [BASE_A, BASE_B] * 2 + [BASE_A, RALLY_OUT]).iloc[0].base_candles == 5
    assert zones(history() + [RALLY_IN] + [BASE_A, BASE_B] * 3 + [RALLY_OUT]).empty  # 6: disqualified
    assert zones(history() + [RALLY_IN, RALLY_OUT]).empty  # no base


def test_zone_valid_only_after_moving_away_twice_its_height():
    # Base 1.0995-1.1006 (11 pips): price must reach 1.1006 + 22 pips = 1.1028 before the zone is valid.
    short_move = (1.1000, 1.1025, 1.0998, 1.1024)  # strong move-out, but only 19 pips above the zone
    further = (1.1024, 1.1030, 1.1020, 1.1028)
    df = frame(history() + [RALLY_IN, BASE_A, short_move])
    assert pd.isna(detect_zones(df, NO_BREAKOUT).iloc[0].valid_time)
    df = frame(history() + [RALLY_IN, BASE_A, short_move, further])
    assert detect_zones(df, NO_BREAKOUT).iloc[0].valid_time == df["time"].iloc[-1]
    back = (1.1024, 1.1026, 1.1005, 1.1010)  # returns to the zone before the move away is complete
    assert detect_zones(frame(history() + [RALLY_IN, BASE_A, short_move, back, further]), NO_BREAKOUT).empty


def test_optional_average_range_multiple():
    cfg = ZoneConfig(require_breakout=False, legout_multiple=4.0)
    assert zones(history() + [RALLY_IN, BASE_A, RALLY_OUT], cfg).empty


# --- breakout ------------------------------------------------------------------------

def test_zone_valid_once_impulse_closes_above_prior_swing_high():
    z = detect_zones(frame(SWINGS + [RALLY_IN, BASE_A, RALLY_OUT])).iloc[0]
    assert z.breakout_level == 1.1030
    assert z.breakout_time == z.legout_time  # the move-out itself closed above 1.1030


def test_breakout_can_come_after_the_move_out():
    breakout = (1.1027, 1.1040, 1.1024, 1.1036)
    df = frame(SWINGS + [RALLY_IN, BASE_A, SHORT_OUT, breakout])
    assert detect_zones(df).iloc[0].breakout_time == df["time"].iloc[-1]


def test_no_breakout_yet_means_not_tradable():
    assert pd.isna(detect_zones(frame(SWINGS + [RALLY_IN, BASE_A, SHORT_OUT])).iloc[0].breakout_time)


def test_return_to_zone_before_breakout_voids_it():
    dip = (1.1027, 1.1028, 1.1004, 1.1010)  # back into the zone (top 1.1006) before any breakout
    breakout = (1.1010, 1.1040, 1.1008, 1.1036)
    assert detect_zones(frame(SWINGS + [RALLY_IN, BASE_A, SHORT_OUT, dip, breakout])).empty


def test_supply_zone_needs_close_below_prior_swing_low():
    z = detect_zones(frame(SWINGS + [DROP_IN, BASE_A, DROP_OUT])).iloc[0]
    assert z.type == "DBD" and z.breakout_level == 1.0970 and z.breakout_time == z.legout_time


def test_no_swing_to_break_means_no_zone():
    assert detect_zones(frame(history() + [RALLY_IN, BASE_A, RALLY_OUT])).empty


def test_atr_matches_wilder_smoothing():
    df = frame([FLAT] * 20)
    assert atr(df, 14).iloc[-1] == pytest.approx(0.0020)
    assert atr(df, 14).iloc[:13].isna().all()
