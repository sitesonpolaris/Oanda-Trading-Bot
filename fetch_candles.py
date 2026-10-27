"""Download historical OANDA candles (practice environment only) and save as CSV.

Reads credentials from environment variables:
    OANDA_API_TOKEN   - personal access token for the practice account (required)
    OANDA_ACCOUNT_ID  - practice account ID (optional for candle data; used to
                        verify the token belongs to a practice account)

Usage:
    python fetch_candles.py                       # EUR_USD, H1 + H4, ~2 years
    python fetch_candles.py --instrument GBP_USD --years 1 --granularity H1
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.accounts as accounts
from requests.exceptions import RequestException

# Hard-wired to the practice environment. Never change this to "live".
OANDA_ENVIRONMENT = "practice"

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_RETRIES = 5

log = logging.getLogger("fetch_candles")


def load_credentials():
    token = os.environ.get("OANDA_API_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not token:
        sys.exit("Error: OANDA_API_TOKEN environment variable is not set.")
    return token, account_id


def make_client(token):
    return API(access_token=token, environment=OANDA_ENVIRONMENT)


def verify_account(client, account_id):
    """Fail fast on a bad token or an account the token can't see."""
    resp = request_with_retry(client, accounts.AccountList())
    ids = [a["id"] for a in resp.get("accounts", [])]
    if account_id and account_id not in ids:
        sys.exit(f"Error: OANDA_ACCOUNT_ID {account_id} is not accessible with this "
                 f"token on the practice environment. Accounts found: {ids}")
    log.info("Authenticated against practice API (%d account(s) visible).", len(ids))


def request_with_retry(client, endpoint):
    """Run a request, retrying with backoff on rate limits, 5xx and network errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.request(endpoint)
        except V20Error as e:
            # 4xx other than 429 means our request is wrong; retrying won't help.
            if e.code != 429 and e.code < 500:
                raise
            reason = f"HTTP {e.code}: {e.msg}"
        except RequestException as e:
            reason = f"network error: {e}"
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"Giving up after {MAX_RETRIES} attempts ({reason})")
        wait = 2 ** attempt
        log.warning("Request failed (%s); retrying in %ds...", reason, wait)
        time.sleep(wait)


def fetch_candles(client, instrument, granularity, start, end):
    """Fetch mid-price candles between start and end, paging in 5000-candle chunks."""
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "granularity": granularity,
        "price": "M",
        "count": 5000,
    }
    rows = []
    for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
        resp = request_with_retry(client, req)
        for c in resp.get("candles", []):
            mid = c["mid"]
            rows.append({
                "time": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
                "complete": c["complete"],
            })
    return to_dataframe(rows)


def to_dataframe(rows):
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "complete"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Drop the still-forming last candle and any overlap between pages.
    df = df[df["complete"]].drop(columns="complete")
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


def save_csv(df, instrument, granularity):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{instrument}_{granularity}.csv"
    df.to_csv(path, index=False)
    return path


def summarize(df, instrument, granularity, path):
    if df.empty:
        return f"{instrument} {granularity}: no candles returned"
    return (
        f"{instrument} {granularity}: {len(df):,} candles | "
        f"{df['time'].iloc[0]:%Y-%m-%d %H:%M} -> {df['time'].iloc[-1]:%Y-%m-%d %H:%M} UTC | "
        f"close range {df['close'].min():.5f}-{df['close'].max():.5f} | "
        f"last close {df['close'].iloc[-1]:.5f} | {path.relative_to(DATA_DIR.parent)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", nargs="+", default=["H1", "H4"])
    parser.add_argument("--years", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token, account_id = load_credentials()
    client = make_client(token)

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=round(365 * args.years))

    try:
        verify_account(client, account_id)
        summaries = []
        for gran in args.granularity:
            log.info("Fetching %s %s from %s to %s...", args.instrument, gran, start, end)
            df = fetch_candles(client, args.instrument, gran, start, end)
            path = save_csv(df, args.instrument, gran)
            summaries.append(summarize(df, args.instrument, gran, path))
    except V20Error as e:
        sys.exit(f"OANDA API error (HTTP {e.code}): {e.msg}")
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    print("\nDownload summary")
    for line in summaries:
        print("  " + line)


if __name__ == "__main__":
    main()
