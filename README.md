# OANDA Trading Bot

Automated Forex trading bot for the **OANDA practice (fxTrade Practice) account only**.

Current stage: **full pipeline on the practice account, dry run by default.** The strategy still loses in backtest, so don't send orders expecting profit.

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

## Market structure (`structure.py`)

On H4: swing highs/lows are 5-candle fractals (2 candles each side), only known
once those candles close. Trend is **up** (higher high + higher low), **down**
(lower high + lower low) or **sideways** (mixed). A **break of structure** is a
close beyond the most recent swing high (bullish) or swing low (bearish); a BOS
against the trend turns it sideways until the next swing forms.

## Backtest

```bash
python backtest.py
```

- Uses only the **first 18 months** of data; the rest is cut off before anything
  is computed and kept for validation.
- Demand zones are traded only in an H4 uptrend, supply zones only in a downtrend
  (trend as of the last closed H4 candle).
- Limit entry at the zone's near edge on the first return to a fresh zone. That
  first touch uses up the zone, traded or not.
- Stop: far edge + 0.1 × ATR. Target: 80% of the leg-out range from entry.
  Skipped if target < 2 × stop distance.
- Simulated on H1 candles with a 1.2-pip spread (buys fill on the ask, sells on
  the bid). If a candle hits stop and target, the stop counts first.
- 1% risk per trade, compounding, for return and drawdown.
- Runs the baseline plus a grid of ATR buffer 0.1 / 0.2 / 0.25 × leg-out multiple
  2 / 2.5 / 3, for H1 and H4 zones.

Output in `output/backtest/`: `trades_<instrument>_<tf>_m<multiple>_b<buffer>.csv`
(every zone with entry, stop, target, fill, exit, result, R), `grid_results.csv`,
and `<instrument>_H4_structure.csv` (trend and BOS per H4 candle).

Options: `--spread 1.0`, `--risk 0.005`, `--zones-tf H1`, `--in-sample-months 18`.

## Risk manager (`risk_manager.py`)

Every signal must pass `RiskManager.evaluate(signal, account)` before an order
can be placed. It returns an approved order with a position size, or a
rejection with a reason. It never talks to the broker.

- **Sizing:** a stop-out loses `risk_per_trade_pct` of current balance (default 0.5%).
- **Limits** (all in `RiskLimits`): max daily loss, max total drawdown from the
  starting balance, max open positions, cap on total open risk.
- **Safety margin:** the bot halts `safety_margin_pct` points *before* the firm's
  limits (5% daily → halts at 4%). It also rejects any trade whose stop, plus
  every open stop, would cross a halt line.
- **Halt:** when a daily or total limit is hit, every new signal is rejected.
  The halt is saved to `state/risk_state.json` and survives restarts until
  reset by hand:

```bash
python risk_manager.py status
```

```bash
python risk_manager.py reset
```

- **Scaling in:** up to 3 entries per setup (`setup_id`). Each extra entry needs
  a favourable break of structure after the previous entry, and all entries on
  the setup together may not risk more than `risk_per_trade_pct`. A scale-in is
  sized down to whatever risk room is left (usually created by moving earlier
  stops to breakeven).

**The prop firm values in `RiskLimits` are placeholders (5% daily, 10% total)
until the real challenge rules are filled in.**

## Executioner (`executioner.py`)

Places risk-approved orders on the **practice** account.

- **Practice only.** Starts only with `OANDA_ENVIRONMENT` unset or `practice`.
  Live needs *both* `LIVE_TRADING_UNLOCKED = True` in `executioner.py` (a code
  change) *and* `OANDA_ENVIRONMENT=live` plus `OANDA_LIVE_CONFIRM` set to the
  exact phrase in that file.
- **LIMIT orders** at the zone edge with `stopLossOnFill` and `takeProfitOnFill`,
  so OANDA holds the stop and target.
- **No double orders.** Each order has a fixed client ID
  (`zb-<setup>-<entry number>`). Before every attempt, including retries after a
  connection error, it asks OANDA whether that ID already exists.
- **Retries** with exponential backoff on network errors, rate limits and 5xx.
  Rejections (4xx) aren't retried.

## Main loop (`bot.py`)

Runs once per H1 candle close. Each cycle it syncs the account, checks the risk
limits (a halt cancels pending orders), pulls 90 days of H1/H4 candles, updates
the H4 trend and H1 zones, and cancels pending orders whose zone was touched or
no longer matches the trend, or that are older than 24 candles. Then it passes
each new signal through the risk manager and sends approved orders to the
executioner. A setup is only ever traded once.

```bash
python bot.py --once                        # one dry-run cycle now
python bot.py                               # dry run, every candle close
python bot.py --as-of "2025-11-19 03:00"    # dry-run replay of a past candle close
python bot.py --send-orders                 # place orders on the practice account
```

Everything is logged with UTC timestamps to `logs/bot.log` (per-zone detail at
DEBUG level). State such as the starting balance and day-start balance is kept
in `state/`. Settings live in `BotConfig` in `bot.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
