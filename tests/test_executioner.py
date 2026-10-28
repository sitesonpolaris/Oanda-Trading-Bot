from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from oandapyV20.exceptions import V20Error
from requests.exceptions import ConnectionError as RequestsConnectionError

import executioner
from backtest import BacktestConfig
from executioner import Executioner, PendingOrder, RefuseToRun, client_order_id, resolve_environment
from risk_manager import Order
from signals import evaluate_zones

ORDER = Order("EUR_USD", "short", 250_000, 1.10004, 1.10204, 1.09404, "EUR_USD-H1-202605040900", 1, 500.0, 0.5)
CID = client_order_id(ORDER.setup_id, 1)


class FakeOanda:
    """Minimal stand-in for oandapyV20.API: stores orders by client ID, can inject failures."""

    def __init__(self, create_failures=(), create_lands_anyway=False):
        self.orders = {}
        self.create_calls = 0
        self.create_failures = list(create_failures)
        self.create_lands_anyway = create_lands_anyway
        self.cancelled = []

    def request(self, endpoint):
        path, method = str(endpoint), endpoint.method
        if method == "POST" and path.endswith("/orders"):
            self.create_calls += 1
            if self.create_failures:
                err = self.create_failures.pop(0)
                if self.create_lands_anyway:  # request reached OANDA, response was lost
                    self._store(endpoint.data)
                raise err
            return {"orderCreateTransaction": self._store(endpoint.data)}
        if method == "GET" and "/orders/@" in path:
            cid = path.split("/orders/@")[1]
            if cid not in self.orders:
                raise V20Error(404, '{"errorMessage":"not found"}')
            return {"order": self.orders[cid]}
        if method == "PUT" and path.endswith("/cancel"):
            self.cancelled.append(path.split("/")[-2])
            return {}
        raise AssertionError(f"unexpected request {method} {path}")

    def _store(self, data):
        o = dict(data["order"], id=str(100 + len(self.orders)), state="PENDING")
        self.orders[o["clientExtensions"]["id"]] = o
        return o


def ex(client, dry_run=False):
    return Executioner(client, "101-001-1-001", dry_run=dry_run, sleep=lambda s: None)


# --- environment guard ---------------------------------------------------------

def test_practice_is_default():
    assert resolve_environment({}) == "practice"
    assert resolve_environment({"OANDA_ENVIRONMENT": "Practice"}) == "practice"


@pytest.mark.parametrize("env", [
    {"OANDA_ENVIRONMENT": "live"},
    {"OANDA_ENVIRONMENT": "live", "OANDA_LIVE_CONFIRM": executioner.LIVE_CONFIRM_PHRASE},
    {"OANDA_ENVIRONMENT": "fxtrade"},
])
def test_refuses_anything_but_practice_while_locked(env):
    with pytest.raises(RefuseToRun):
        resolve_environment(env)


def test_live_needs_code_unlock_and_phrase(monkeypatch):
    monkeypatch.setattr(executioner, "LIVE_TRADING_UNLOCKED", True)
    with pytest.raises(RefuseToRun):
        resolve_environment({"OANDA_ENVIRONMENT": "live", "OANDA_LIVE_CONFIRM": "yes"})
    assert resolve_environment({"OANDA_ENVIRONMENT": "live",
                                "OANDA_LIVE_CONFIRM": executioner.LIVE_CONFIRM_PHRASE}) == "live"


# --- placing -----------------------------------------------------------------

def test_limit_order_has_stop_and_target_attached():
    req = ex(FakeOanda()).order_request(ORDER, CID)["order"]
    assert req["type"] == "LIMIT" and req["units"] == "-250000" and req["price"] == "1.10004"
    assert req["stopLossOnFill"]["price"] == "1.10204"
    assert req["takeProfitOnFill"]["price"] == "1.09404"
    assert req["clientExtensions"]["id"] == CID and req["tradeClientExtensions"]["tag"] == ORDER.setup_id


def test_places_order_once():
    oanda = FakeOanda()
    assert ex(oanda).place(ORDER)["id"] == "100"
    assert ex(oanda).place(ORDER)["id"] == "100"  # second call finds it, doesn't resend
    assert oanda.create_calls == 1


def test_dry_run_sends_nothing():
    oanda = FakeOanda()
    assert ex(oanda, dry_run=True).place(ORDER) is None
    assert oanda.create_calls == 0 and oanda.orders == {}


def test_retries_after_connection_error():
    oanda = FakeOanda(create_failures=[RequestsConnectionError("reset")])
    assert ex(oanda).place(ORDER) is not None
    assert oanda.create_calls == 2 and len(oanda.orders) == 1


def test_no_double_place_when_lost_response_actually_succeeded():
    oanda = FakeOanda(create_failures=[RequestsConnectionError("timeout")], create_lands_anyway=True)
    placed = ex(oanda).place(ORDER)
    assert placed["id"] == "100"
    assert oanda.create_calls == 1 and len(oanda.orders) == 1


def test_retries_server_errors_but_not_rejections():
    oanda = FakeOanda(create_failures=[V20Error(503, "busy")])
    assert ex(oanda).place(ORDER) is not None and oanda.create_calls == 2
    oanda = FakeOanda(create_failures=[V20Error(400, "PRICE_PRECISION_EXCEEDED")])
    assert ex(oanda).place(ORDER) is None and oanda.create_calls == 1


def test_gives_up_after_max_retries():
    oanda = FakeOanda(create_failures=[RequestsConnectionError("down")] * 10)
    assert ex(oanda).place(ORDER) is None
    assert oanda.create_calls == executioner.MAX_RETRIES


# --- cancelling --------------------------------------------------------------

def pending(oid="42"):
    return PendingOrder(oid, CID, ORDER.setup_id, "EUR_USD", -250_000, 1.10004, 1.10204, 1.09404,
                        datetime(2026, 5, 4, 10, tzinfo=timezone.utc))


def test_cancel_and_dry_run_cancel():
    oanda = FakeOanda()
    assert ex(oanda).cancel(pending(), "expired") and oanda.cancelled == ["42"]
    oanda = FakeOanda()
    assert ex(oanda, dry_run=True).cancel(pending(), "expired") and oanda.cancelled == []


# --- signals -------------------------------------------------------------------

def test_signals_only_for_fresh_trend_aligned_zones():
    t = pd.Timestamp("2026-05-04 09:00", tz="UTC")
    zones = pd.DataFrame([
        {"start_time": t, "legout_time": t + timedelta(hours=1), "type": "RBR", "kind": "demand",
         "top": 1.1000, "bottom": 1.0990, "base_candles": 1, "impulse_pips": 60, "impulse_ratio": 3,
         "impulse_atr": 3, "atr_pips": 10},
        {"start_time": t + timedelta(hours=3), "legout_time": t + timedelta(hours=4), "type": "DBR",
         "kind": "demand", "top": 1.1040, "bottom": 1.1030, "base_candles": 1, "impulse_pips": 60,
         "impulse_ratio": 3, "impulse_atr": 3, "atr_pips": 10},
    ])
    times = pd.date_range(t, periods=8, freq="h")
    # Zone 1 is never revisited after its leg-out; zone 2 (entry 1.1040) is touched at 1.1039.
    lows = [1.0995, 1.1000, 1.1020, 1.1035, 1.1050, 1.1039, 1.1050, 1.1060]
    candles = pd.DataFrame({"time": times, "low": lows, "high": [x + 0.001 for x in lows]})
    up = pd.DataFrame({"available_time": [t], "trend": ["up"], "bos": ["bullish"]})
    v = evaluate_zones(zones, candles, up, BacktestConfig(), "EUR_USD", "H1", 0.0001, times[-1])
    assert [x.status for x in v] == ["signal", "touched"]
    assert v[0].signal.entry == 1.1000 and v[0].signal.setup_id == "EUR_USD-H1-202605040900"
    down = up.assign(trend="down")
    assert evaluate_zones(zones, candles, down, BacktestConfig(), "EUR_USD", "H1", 0.0001,
                          times[-1])[0].status == "skipped_trend"
