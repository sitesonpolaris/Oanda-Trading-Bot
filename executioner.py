"""Executioner: places risk-approved orders on the OANDA practice account.

- Refuses to run against anything but the practice environment unless live
  trading has been deliberately unlocked (see LIVE_TRADING_UNLOCKED below).
- Places LIMIT orders with the stop loss and take profit attached to the order
  (stopLossOnFill / takeProfitOnFill), so OANDA manages them, not this bot.
- Never double-places: every order carries a deterministic client ID. Before
  each attempt, including retries after connection errors, OANDA is asked
  whether that ID already exists. OANDA also rejects a reused client ID.
- Tracks pending orders and open trades, and cancels pending orders on request.
- Dry run: reads from OANDA as normal but only logs the orders and
  cancellations it would send.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
from requests.exceptions import RequestException

from risk_manager import Position

# Live trading needs BOTH of these, on purpose:
#   1. change LIVE_TRADING_UNLOCKED to True here (a code change that shows up in git), and
#   2. export OANDA_ENVIRONMENT=live and OANDA_LIVE_CONFIRM="<LIVE_CONFIRM_PHRASE>".
# Anything else runs on practice or refuses to start.
LIVE_TRADING_UNLOCKED = False
LIVE_CONFIRM_PHRASE = "I understand this trades real money"

CLIENT_ID_PREFIX = "zb-"
MAX_RETRIES = 5
REQUEST_TIMEOUT = 15

log = logging.getLogger("executioner")


class RefuseToRun(RuntimeError):
    pass


def resolve_environment(env=None):
    """Return the OANDA environment to use, or raise RefuseToRun."""
    env = os.environ if env is None else env
    name = env.get("OANDA_ENVIRONMENT", "practice").strip().lower()
    if name == "practice":
        return "practice"
    if name == "live":
        if not LIVE_TRADING_UNLOCKED:
            raise RefuseToRun("OANDA_ENVIRONMENT=live but LIVE_TRADING_UNLOCKED is False in executioner.py.")
        if env.get("OANDA_LIVE_CONFIRM") != LIVE_CONFIRM_PHRASE:
            raise RefuseToRun("Live trading needs OANDA_LIVE_CONFIRM set to the exact confirmation phrase.")
        return "live"
    raise RefuseToRun(f"Unknown OANDA_ENVIRONMENT {name!r}; expected 'practice'.")


def make_client(token, environment):
    return API(access_token=token, environment=environment, request_params={"timeout": REQUEST_TIMEOUT})


def client_order_id(setup_id, entry_number):
    return f"{CLIENT_ID_PREFIX}{setup_id}-{entry_number}"


def price_str(instrument, price):
    return f"{price:.{3 if instrument.endswith('JPY') else 5}f}"


def parse_time(s):
    return pd.Timestamp(s).tz_convert("UTC").to_pydatetime()


@dataclass
class PendingOrder:
    order_id: str
    client_id: str
    setup_id: str
    instrument: str
    units: int
    price: float
    stop: float
    target: float
    created: datetime

    def as_position(self):
        """Risk if this order fills and then stops out, for the risk manager's caps."""
        return Position(self.instrument, "long" if self.units > 0 else "short", abs(self.units),
                        self.price, self.stop, self.setup_id, self.created)


@dataclass
class OpenTrade:
    trade_id: str
    setup_id: str
    instrument: str
    units: int
    price: float
    stop: float
    opened: datetime
    unrealized_pl: float

    def as_position(self):
        return Position(self.instrument, "long" if self.units > 0 else "short", abs(self.units),
                        self.price, self.stop, self.setup_id, self.opened)


@dataclass
class Book:
    balance: float
    equity: float
    currency: str
    pending: list
    trades: list


class Executioner:
    def __init__(self, client, account_id, dry_run=True, sleep=time.sleep):
        self.client = client
        self.account_id = account_id
        self.dry_run = dry_run
        self.sleep = sleep
        self.mode = "DRY RUN" if dry_run else "LIVE ORDERS (practice)"

    # --- requests -------------------------------------------------------------

    def _request(self, endpoint):
        """Retry with backoff on rate limits, 5xx and network errors. Only for requests safe to repeat."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.client.request(endpoint)
            except V20Error as e:
                if e.code != 429 and e.code < 500:
                    raise
                reason = f"HTTP {e.code}"
            except RequestException as e:
                reason = f"network error: {e}"
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"{endpoint} failed after {MAX_RETRIES} attempts ({reason})")
            log.warning("Request %s failed (%s); retry %d in %ds", endpoint, reason, attempt, 2 ** attempt)
            self.sleep(2 ** attempt)

    def find_order(self, client_id):
        """The order with this client ID in any state (pending, filled, cancelled), or None."""
        try:
            return self._request(orders.OrderDetails(self.account_id, f"@{client_id}"))["order"]
        except V20Error as e:
            if e.code == 404:
                return None
            raise

    # --- state ----------------------------------------------------------------

    def sync(self):
        summary = self._request(accounts.AccountSummary(self.account_id))["account"]
        pending = [self._pending(o) for o in self._request(orders.OrdersPending(self.account_id))["orders"]
                   if o.get("type") == "LIMIT"
                   and o.get("clientExtensions", {}).get("id", "").startswith(CLIENT_ID_PREFIX)]
        open_trades = [self._trade(t) for t in self._request(trades.OpenTrades(self.account_id))["trades"]]
        book = Book(float(summary["balance"]), float(summary["NAV"]), summary["currency"], pending, open_trades)
        log.info("Account: balance %.2f %s, equity %.2f, %d pending bot order(s), %d open trade(s)",
                 book.balance, book.currency, book.equity, len(pending), len(open_trades))
        for o in pending:
            log.info("  pending %s %s %+d @ %s SL %s TP %s since %s", o.client_id, o.instrument, o.units,
                     o.price, o.stop, o.target, f"{o.created:%Y-%m-%d %H:%M}")
        for t in open_trades:
            log.info("  open trade %s %s %+d @ %s SL %s P/L %.2f", t.trade_id, t.instrument, t.units,
                     t.price, t.stop, t.unrealized_pl)
        return book

    @staticmethod
    def _pending(o):
        ext = o["clientExtensions"]
        return PendingOrder(o["id"], ext["id"], ext.get("tag", ""), o["instrument"], int(float(o["units"])),
                            float(o["price"]), float(o["stopLossOnFill"]["price"]),
                            float(o["takeProfitOnFill"]["price"]), parse_time(o["createTime"]))

    @staticmethod
    def _trade(t):
        stop = t.get("stopLossOrder", {}).get("price")
        units = int(float(t["currentUnits"]))
        # No stop on the trade (e.g. opened by hand): treat the whole entry price as at risk.
        stop = float(stop) if stop else (0.0 if units > 0 else float(t["price"]) * 2)
        return OpenTrade(t["id"], t.get("clientExtensions", {}).get("tag", "manual"), t["instrument"], units,
                         float(t["price"]), stop, parse_time(t["openTime"]), float(t.get("unrealizedPL", 0)))

    # --- actions --------------------------------------------------------------

    def order_request(self, order, client_id):
        ext = {"id": client_id, "tag": order.setup_id, "comment": f"entry {order.entry_number}"}
        return {"order": {
            "type": "LIMIT",
            "instrument": order.instrument,
            "units": str(order.oanda_units),
            "price": price_str(order.instrument, order.entry),
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": price_str(order.instrument, order.stop), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": price_str(order.instrument, order.target), "timeInForce": "GTC"},
            "clientExtensions": ext,
            "tradeClientExtensions": ext,
        }}

    def place(self, order):
        """Place an approved order once. Returns the OANDA order dict, or None if not placed."""
        client_id = client_order_id(order.setup_id, order.entry_number)
        request = self.order_request(order, client_id)
        summary = (f"{client_id} LIMIT {order.instrument} {order.oanda_units:+d} @ {request['order']['price']} "
                   f"SL {request['order']['stopLossOnFill']['price']} "
                   f"TP {request['order']['takeProfitOnFill']['price']} (risk {order.risk_amount:.2f})")
        if self.dry_run:
            log.info("[DRY RUN] would place %s", summary)
            return None

        for attempt in range(1, MAX_RETRIES + 1):
            existing = self.find_order(client_id)
            if existing:
                log.warning("Not placing %s: order already exists (state %s, id %s)",
                            client_id, existing.get("state"), existing.get("id"))
                return existing
            try:
                resp = self.client.request(orders.OrderCreate(self.account_id, data=request))
            except V20Error as e:
                if e.code != 429 and e.code < 500:
                    log.error("OANDA rejected %s: HTTP %s %s", client_id, e.code, e.msg)
                    return None
                reason = f"HTTP {e.code}"
            except RequestException as e:
                reason = f"network error: {e}"
            else:
                if "orderCancelTransaction" in resp:
                    log.error("OANDA cancelled %s on creation: %s", client_id,
                              resp["orderCancelTransaction"].get("reason"))
                    return None
                created = resp["orderCreateTransaction"]
                log.info("Placed %s -> order id %s", summary, created["id"])
                return created
            if attempt == MAX_RETRIES:
                log.error("Giving up on %s after %d attempts (%s)", client_id, MAX_RETRIES, reason)
                return None
            # The order may have gone through even though we saw an error; the next loop checks first.
            log.warning("Placing %s failed (%s); checking and retrying in %ds", client_id, reason, 2 ** attempt)
            self.sleep(2 ** attempt)

    def cancel(self, pending, reason):
        if self.dry_run:
            log.info("[DRY RUN] would cancel %s (%s)", pending.client_id, reason)
            return True
        try:
            self._request(orders.OrderCancel(self.account_id, pending.order_id))
        except V20Error as e:
            if e.code == 404:
                log.info("Cancel %s: order no longer pending (filled or already cancelled)", pending.client_id)
                return False
            log.error("Cancel %s failed: HTTP %s %s", pending.client_id, e.code, e.msg)
            return False
        log.info("Cancelled %s (%s)", pending.client_id, reason)
        return True
