"""Download 1-minute OANDA bid/ask candles (practice environment) for the VWAP backtest.

Saves data/<instrument>_M1_BA.csv with bid and ask OHLC and tick volume. Re-running
only fetches what's missing after the last saved candle.

Usage:
    python -m vwap.fetch --instruments SPX500_USD EUR_USD --years 3
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

from fetch_candles import DATA_DIR, load_credentials, make_client, request_with_retry, verify_account

COLUMNS = ["time", "bid_open", "bid_high", "bid_low", "bid_close",
           "ask_open", "ask_high", "ask_low", "ask_close", "volume"]
log = logging.getLogger("vwap.fetch")


def path_for(instrument):
    return DATA_DIR / f"{instrument}_M1_BA.csv"


def fetch(client, instrument, start, end):
    params = {"from": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "granularity": "M1", "price": "BA", "count": 5000}
    rows = []
    for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
        for c in request_with_retry(client, req).get("candles", []):
            if not c["complete"]:
                continue
            b, a = c["bid"], c["ask"]
            rows.append([c["time"], *(float(b[k]) for k in "ohlc"), *(float(a[k]) for k in "ohlc"), int(c["volume"])])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruments", nargs="+", default=["SPX500_USD"])
    parser.add_argument("--years", type=float, default=3.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token, account_id = load_credentials()
    client = make_client(token)
    verify_account(client, account_id)
    end = datetime.now(timezone.utc)
    for instrument in args.instruments:
        path = path_for(instrument)
        old = pd.read_csv(path, parse_dates=["time"]) if path.exists() else pd.DataFrame(columns=COLUMNS)
        start = old["time"].max() + timedelta(minutes=1) if len(old) else end - timedelta(days=365.25 * args.years)
        log.info("%s: fetching M1 bid/ask from %s", instrument, start)
        new = fetch(client, instrument, start, end)
        df = pd.concat([old, new]).drop_duplicates("time").sort_values("time")
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        log.info("%s: %d candles, %s -> %s, saved %s", instrument, len(df), df["time"].min(), df["time"].max(), path.name)


if __name__ == "__main__":
    main()
