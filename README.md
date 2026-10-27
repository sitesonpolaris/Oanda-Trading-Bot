# OANDA Trading Bot

Automated Forex trading bot for the **OANDA practice (fxTrade Practice) account only**.

Current stage: **data feed only** — no strategy logic yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your credentials as environment variables (never commit them):

```bash
export OANDA_API_TOKEN="your-practice-token"
export OANDA_ACCOUNT_ID="101-001-XXXXXXX-001"
```

Get both from the OANDA practice hub under *Manage API Access*. The script is
hard-wired to the practice environment (`api-fxpractice.oanda.com`).

## Download historical candles

```bash
python fetch_candles.py
```

Defaults: EUR/USD, H1 and H4 candles, about 2 years back. Mid prices, completed
candles only. Output goes to `data/`:

- `data/EUR_USD_H1.csv`
- `data/EUR_USD_H4.csv`

Columns: `time` (UTC), `open`, `high`, `low`, `close`, `volume` (tick volume).

Options:

```bash
python fetch_candles.py --instrument GBP_USD --granularity H1 D --years 1
```

## Error handling

- Missing token → exits with a clear message.
- Token/account is checked against the practice API before downloading.
- Rate limits (HTTP 429), server errors and network failures are retried with
  exponential backoff (up to 5 attempts); other API errors stop with the
  OANDA error message.
