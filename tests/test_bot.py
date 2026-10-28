from datetime import datetime, timedelta, timezone

import pandas as pd

from bot import Bot, BotConfig, closed_by
from executioner import PendingOrder
from signals import ZoneVerdict

NOW = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)


class StubExecutioner:
    def __init__(self):
        self.cancelled = []

    def cancel(self, order, reason):
        self.cancelled.append((order.setup_id, reason))
        return True


def bot():
    b = Bot.__new__(Bot)
    b.cfg, b.ex = BotConfig(max_order_age_candles=24), StubExecutioner()
    return b


def pending(setup, hours_old):
    return PendingOrder("1", f"zb-{setup}-1", setup, "EUR_USD", 1000, 1.1, 1.099, 1.105,
                        NOW - timedelta(hours=hours_old))


class Book:
    def __init__(self, orders):
        self.pending = orders


def test_cancels_invalidated_missing_and_expired_orders_only():
    b = bot()
    verdicts = {
        "valid": ZoneVerdict("valid", None, "signal", ""),
        "touched": ZoneVerdict("touched", None, "touched", ""),
        "trend": ZoneVerdict("trend", None, "skipped_trend", ""),
        "old": ZoneVerdict("old", None, "signal", ""),
    }
    orders = [pending("valid", 3), pending("touched", 3), pending("trend", 3), pending("gone", 3), pending("old", 24)]
    b.manage_pending(Book(orders), verdicts, NOW)
    reasons = dict(b.ex.cancelled)
    assert set(reasons) == {"touched", "trend", "gone", "old"}
    assert "skipped_trend" in reasons["trend"] and "no longer detected" in reasons["gone"]
    assert "unfilled after 24" in reasons["old"]


def test_closed_by_drops_candles_still_forming():
    t = pd.date_range("2026-05-05 00:00", periods=4, freq="4h", tz="UTC")
    df = pd.DataFrame({"time": t})
    assert list(closed_by(df, "H4", pd.Timestamp("2026-05-05 13:00", tz="UTC"))["time"]) == list(t[:3])
