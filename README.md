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

How a zone is found (`zones.py`), following the TWP course "Supply and Demand –
Steps to Drawing Zones":

1. **Basing candle:** body ≤ 50% of its range, or a close inside the previous
   candle's range. 1–5 of them form the base; more than 5 disqualifies the zone.
2. **Move-in and move-out candles:** a rally candle closes above the previous
   candle's high (a drop candle closes below its low); colour isn't checked.
   The move-out must also be at least 2× the biggest basing candle and taller
   than the move-in. (The course's "close in the top 25%" check is available as
   `close_zone=0.25`, off by default.)
3. **Breakout:** the impulse must close beyond the most recent swing high
   (demand) or swing low (supply) that existed before the base.
4. **Move away:** price must travel at least 2× the zone's height away from it.
5. **Valid** once the breakout and the move away have both happened
   (`valid_time`); if price comes back to the zone first, it's void.
6. **Types:** RBR / DBR (demand), DBD / RBD (supply), all traded.
7. **Drawing:** one basing candle is wick to wick; with 2–5, a demand zone runs
   from the highest body (entry) to the lowest wick, a supply zone from the
   lowest body (entry) to the highest wick.

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

On any timeframe (D1 stage, H4 momentum, H1 adds): swing highs/lows are 5-candle
fractals (2 candles each side), only known once those candles close. Trend uses
the course's stages: **up** (Stage 2) = higher high + higher low with rising 20-
and 40-period averages; **down** (Stage 4) = the mirror image; **sideways**
otherwise. A **break of structure** is a close beyond the most recent swing high
(bullish) or swing low (bearish); a BOS against the trend turns it sideways
until the next swing forms.

## Strategy (`strategy.py`)

Follows the TWP Market Structure course where it's specific, and the strategy
spec (Oct 10, 2026) elsewhere. The same rules drive the backtest and the live bot.

- **Timeframes:** stage on D1, momentum on H4, zones and entries on H1.
- **Direction:** all timeframes must align: D1 in Stage 2 for longs (Stage 4 for
  shorts) when price returns, and H4 momentum the same way when the zone became
  valid. Countertrend trades are off (`allow_countertrend_with_bos`).
- **Entry:** limit order at the zone's entry edge as soon as the zone is valid.
- **Stop:** beyond the zone's far edge by 10% of the daily ATR (course: swing
  trades on 30-minute charts and above).
- **Target:** when price returns, the nearer of the prior pivot (most recent
  confirmed H1 swing high above a long's entry / low below a short's) and the
  edge of the nearest active opposing H1 or H4 zone. Neither, or under 2:1, no
  trade. Live, the bot sets this target on the trade once the order fills.
- **Playing field:** skip if 5+ of the last 20 H1 candles have wicks ≥ 1× ATR;
  skip within 30 min of high-impact news for either currency.
  The news filter needs `data/news_events.csv` (`time,currency,impact`, UTC)
  and is off without it.
- **Sizing:** the first entry risks ⅓ of the per-trade risk. Up to two more ⅓
  entries are added at market after H1 breaks of structure in the trade's
  favour, with the same stop and target. Stops are never tightened.
- **Level-on-level:** zones stacked on another same-side zone are flagged and
  get priority.

All settings live in `StrategyConfig`.

## Backtest

```bash
python fetch_candles.py --instrument EUR_USD --granularity H1 H4 D --years 8
python backtest.py                                        # all EUR_USD data before the hold-out
python backtest.py --start 2024-10-28                     # the original 18-month window
python backtest.py --holdout-from 2024-10-28              # EUR_USD 2018-2024 only
python backtest.py --instruments GBP_USD AUD_USD USD_JPY  # other pairs (per-pair spreads)
```

- Everything from **2026-04-28** on is held out for final validation and cut off
  before anything is computed, for every pair (`--holdout-from` can move the cutoff earlier).
- Simulated on H1 candles with a 1.2-pip spread (buys fill on the ask, sells on
  the bid, adds included). If a candle hits stop and target, the stop counts
  first. Weekend gaps fill at the gap price.
- Results are in R of the full per-trade risk (1% by default): a stop-out with
  only the first entry is −⅓R.
- Grid: stop buffer 0.1 / 0.2 / 0.25 of the daily ATR × impulse strength filter off / 2× / 2.5× / 3×.

Chart trades from the logs (zones, entry, stop loss, take profit, fill, exit):

```bash
python plot_trades.py                      # 3 winners + 3 losers across the four pairs
```

Charts go to `output/charts/trades/`.

Output in `output/backtest/`: `trades_<instrument>_H1_m<multiple>_b<buffer>.csv`
(every zone: plan, filter verdict, fill, adds, exit, R) and `grid_results.csv`.

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
limits (a halt cancels pending orders), pulls D1/H4/H1 candles, applies the
strategy rules, and cancels pending orders whose zone no longer passes, or that are older than
24 candles. It then passes each new signal (level-on-level first) and each
scale-in through the risk manager and sends approved orders to the executioner. A setup is only ever traded once.

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

## Daily Anchored VWAP strategy (`vwap/`)

A separate strategy, ported from the RoboQuant "Daily Anchored VWAP Signal
Strategy" with the `MES_3min.set` inputs. It's backtest only and shares nothing
with the zone bot. See `vwap/strategy.py` for the rules.

OANDA has no MES futures, so the test uses OANDA's S&P 500 CFD (`SPX500_USD`)
as a stand-in, plus FX pairs. OANDA volume is a tick count, so the VWAP is an
approximation of the exchange VWAP.

```bash
python -m vwap.fetch --instruments SPX500_USD EUR_USD --years 3   # 1-minute bid/ask candles
python -m vwap.backtest                                           # SPX500_USD and EUR_USD
python -m vwap.backtest --instruments SPX500_USD --ema-length 20  # any input can be overridden
```

Changes from the RoboQuant code, as agreed:
- Every position closes at the 16:00 ET bar's close, and no new entries are
  taken on or after it. (The original has no stop and no session-end exit.)
- Orders fill at the next 3-minute bar's open, buying at the ask and selling at
  the bid, so OANDA's spread is the trading cost.
- The VWAP restarts at midnight ET, as in the original. Bars are stamped with
  their close time.

Variants: `--anchor ema --slow-ema-length 34` compares the EMA with a slow EMA
instead of VWAP; `--signal-mode crossover` trades the EMA crossing the anchor
line (with reversals) instead of the dots and triangles.

Results are per 1 unit, in points for the S&P 500 ($1 per point per unit) and
pips for FX pairs. Data from the holdout date onwards is excluded.
