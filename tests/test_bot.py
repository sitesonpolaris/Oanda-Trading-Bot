from datetime import datetime, timedelta, timezone

import pandas as pd

from bot import Bot, BotConfig, closed_by
from executioner import PendingOrder
from signals import ZoneVerdict

NOW = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)


class StubExecutioner:
    def __init__(self):
        self.cancelled = []
        self.take_profits = []

    def cancel(self, order, reason):
        self.cancelled.append((order.setup_id, reason))
        return True

    def set_take_profit(self, trade, price, reason):
        self.take_profits.append((trade.trade_id, price))
        return True


def bot():
    b = Bot.__new__(Bot)
    b.cfg, b.ex, b.pip = BotConfig(max_order_age_candles=24), StubExecutioner(), 0.0001
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


def test_every_entry_on_a_setup_gets_the_target_as_of_the_first_fill(monkeypatch):
    import bot as bot_module
    from executioner import OpenTrade
    b = bot()
    first = OpenTrade("1", "s", "EUR_USD", 30_000, 1.1000, 1.0983, NOW - timedelta(hours=5), 0.0, 1.1050)
    add = OpenTrade("2", "s", "EUR_USD", 30_000, 1.1020, 1.0983, NOW - timedelta(hours=1), 0.0, 1.1050)
    seen = []
    monkeypatch.setattr(bot_module, "target_for_trade", lambda t, m: seen.append(t.trade_id) or 1.1045)
    book = Book([])
    book.trades = [add, first]
    b.update_targets(book, market=None)
    assert seen == ["1"] and b.ex.take_profits == [("2", 1.1045), ("1", 1.1045)]
