import pandas as pd

from helpers import T0, bars, lifecycle_zone
from lol import first_touch


def test_first_touch_is_when_price_returns_to_the_entry_edge():
    b = bars([(1.1000, 1.1010, 1.0995, 1.1005), (1.1005, 1.1052, 1.1000, 1.1040), (1.1040, 1.1045, 1.0985, 1.0990)])
    zones = pd.DataFrame([lifecycle_zone("supply", 1.1060, 1.1050, available=T0),
                          lifecycle_zone("demand", 1.0990, 1.0980, available=T0),
                          lifecycle_zone("supply", 1.1080, 1.1070, available=T0)])
    touched = first_touch(zones, b)
    assert touched.iat[0] == T0 + pd.Timedelta(hours=1)   # high 1.1052 reached the supply's bottom
    assert touched.iat[1] == T0 + pd.Timedelta(hours=2)   # low 1.0985 reached the demand's top
    assert pd.isna(touched.iat[2])                        # never reached: still fresh


def test_touches_before_the_zone_is_available_dont_count():
    b = bars([(1.1000, 1.1055, 1.0995, 1.1005), (1.1005, 1.1010, 1.1000, 1.1008)])
    zones = pd.DataFrame([lifecycle_zone("supply", 1.1060, 1.1050, available=T0 + pd.Timedelta(hours=1))])
    assert pd.isna(first_touch(zones, b).iat[0])
