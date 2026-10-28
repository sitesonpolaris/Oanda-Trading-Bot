# OANDA Trading Bot

Automated Forex trading bot for the **OANDA practice (fxTrade Practice) account only**.

Current stage: **data feed + zone detection** — analysis only, no trading or order code yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the install fails building `oandapyV20` with `AttributeError: install_layout`
(seen on Debian/Ubuntu system Python), use `pip install --use-pep517 -r requirements.txt`.

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

## Zone detection

Finds supply/demand zones in the saved CSVs (no API calls, no orders):

```bash
python detect_zones.py
```

How a zone is found (`zones.py`):

1. ATR (14, Wilder) is calculated on each timeframe.
2. A **leg-out** is a candle whose range is at least **2.5×** the average range of
   the 10 candles before it, with a body of at least 50% of its range.
3. The **base** is the 1–3 small candles right before the leg-out (body ≤ 50% of
   range, range ≤ 1× ATR). If the run of small candles is longer than the max,
   it's rejected as a consolidation, not a base.
4. The candle before the base is the **leg-in**. Leg-in and leg-out direction
   give the type: RBR / DBR (demand), DBD / RBD (supply).
5. Edges: a single-candle base uses its wick high/low; a multi-candle base uses
   the highest/lowest candle body.

Output:

- `output/EUR_USD_H1_zones.csv`, `output/EUR_USD_H4_zones.csv` — `start_time`
  (first base candle), `legout_time`, `type`, `kind`, `top`, `bottom`,
  `base_candles`, `impulse_pips` (leg-out range), `impulse_ratio` (× average
  range), `impulse_atr` (× ATR), `atr_pips`
- `output/charts/*.png` — the most recent zone of each type per timeframe
- Zones-per-year table printed to the terminal

Change settings on the command line, or edit the defaults in `ZoneConfig` in `zones.py`:

```bash
python detect_zones.py --multiple 3 --max-base 2 --lookback 20
python detect_zones.py --plot-at "2026-03-12 14:00"   # chart the zone nearest a time
```

Tests: `pip install -r requirements-dev.txt && python -m pytest`
