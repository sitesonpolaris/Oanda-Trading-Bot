from datetime import datetime, timezone

import pytest
from oandapyV20.exceptions import V20Error
from requests.exceptions import ConnectionError as RequestsConnectionError

import executioner
from executioner import Executioner, PendingOrder, RefuseToRun, client_order_id, resolve_environment
from risk_manager import Order

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
        self.take_profits = []

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
        if method == "PUT" and "/trades/" in path and path.endswith("/orders"):
            self.take_profits.append((path.split("/")[-2], endpoint.data["takeProfit"]["price"]))
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


def test_scale_in_is_a_market_order_with_stop_and_target():
    add = Order("EUR_USD", "long", 30_000, 1.1030, 1.0980, 1.1090, "z", 2, 150.0, 0.15, "MARKET")
    req = ex(FakeOanda()).order_request(add, client_order_id("z", 2))["order"]
    assert req["type"] == "MARKET" and req["timeInForce"] == "FOK" and "price" not in req
    assert req["stopLossOnFill"]["price"] == "1.09800" and req["takeProfitOnFill"]["price"] == "1.10900"


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


def test_set_take_profit_on_open_trade():
    from executioner import OpenTrade
    trade = OpenTrade("77", "z", "EUR_USD", 30_000, 1.1000, 1.0983, datetime(2026, 5, 4, tzinfo=timezone.utc), 0.0, 1.1050)
    oanda = FakeOanda()
    assert ex(oanda).set_take_profit(trade, 1.10453, "test") and oanda.take_profits == [("77", "1.10453")]
    oanda = FakeOanda()
    assert ex(oanda, dry_run=True).set_take_profit(trade, 1.10453, "test") and oanda.take_profits == []
