# finagent

An autonomous **paper-trading** agent in pure Python: market data providers (bundled sample data or
free live daily bars from Yahoo Finance, cached on disk), technical indicators, pluggable strategies,
a deterministic risk engine, a simulated broker with persisted state, a backtester with benchmark
comparison and per-trade analytics, parameter sweeps and walk-forward testing, decision
notifications, and an optional Claude analyst.

> [!WARNING]
> **Paper trading only. Not financial advice.** finagent has no brokerage integration and
> no code path that places a real order. The bundled market data is **synthetic**; live data from
> Yahoo is unofficial and may be delayed, adjusted or wrong. Backtest
> results are not a prediction of future returns. Do not use this to make investment decisions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core has zero runtime dependencies
pytest -q

finagent backtest                # all sample symbols, combined strategy, vs SYN_INDEX
finagent run --once              # one autonomous tick against the paper account
finagent portfolio               # positions, cash, P&L
finagent report                  # HTML report of the paper account

# real market data (free, no API key), benchmarked against SPY
finagent backtest --provider yahoo --symbols SPY AAPL MSFT --start 2021-01-01 --trailing-stop 0.1
finagent sweep       --provider yahoo --symbols SPY AAPL MSFT --grid entry=0.2,0.3,0.4 --grid trailing_stop=0,0.1
finagent walkforward --provider yahoo --symbols SPY AAPL MSFT --grid entry=0.2,0.3,0.4 --test-days 252
finagent run --once  --provider yahoo --symbols SPY AAPL MSFT --webhook https://hooks.slack.com/...
```

Outputs go to `reports/`, paper state to `state/finagent.db` and cached market data to `data/cache/`
(all git-ignored).

## How it works

```
observe ──> analyze ──> decide ─────────────> risk-check ──> execute (paper) ──> journal
 data       indicators   rule policy, or        RiskEngine     PaperBroker         sqlite
 providers  + strategy   Claude proposals       (every order)  slippage + fees     journal table
            scores       (fallback: rules)
```

| Module | What it does |
|---|---|
| `data.py` | `DataProvider` protocol, `CSVProvider`, `YahooProvider` (chart JSON via `urllib`), `StooqProvider`, `CachedProvider` (disk cache), `FallbackProvider`; explicit bot-challenge detection |
| `indicators.py` | SMA, EMA, RSI (Wilder), MACD, ATR (Wilder), Bollinger bands |
| `strategies.py` | `momentum`, `mean_reversion`, `combined`, `breakout` (Donchian: enter on a close above the prior 55-day high, exit below the prior 20-day low); each returns a score in [-1, 1] with a reason. Add one by implementing `signal()` and registering it in `STRATEGIES` |
| `risk.py` | Position sizing and pre-trade checks (below) |
| `broker.py` | Paper fills with slippage (5 bps) and commission ($0.005/share, $1 min); cash, positions, fills, equity and journal in sqlite |
| `backtest.py` | Daily event loop: decide at close, fill at next open. CAGR, Sharpe, Sortino, max drawdown, win rate, turnover; buy-and-hold benchmark (excess return, beta, alpha, correlation); round-trip trade analytics |
| `optimize.py` | Parameter sweep (in-sample vs out-of-sample) and rolling walk-forward evaluation with overfitting checks |
| `report.py` | Self-contained HTML report: SVG equity + drawdown chart with benchmark overlay, metrics, per-symbol and per-trade tables |
| `notify.py` | One-line decision summaries on stdout, optionally POSTed to a webhook |
| `agent.py` | The autonomous loop and decision journal |
| `llm.py` | Optional Claude analyst |

### Risk engine

Every order, whether it comes from the rule-based policy or from Claude, passes through
`RiskEngine.check`. `PaperBroker.execute` refuses any order that was not approved.

| Rule | Default |
|---|---|
| Sizing | ATR-based: risk 1% of equity per 2×ATR move (or fixed-fractional 10% with `--sizing fixed`) |
| Max single position | 20% of equity (buys are clipped) |
| Max gross exposure | 95% of equity |
| Sector exposure | 40% of equity per sector (built-in map of common US tickers, index ETFs share one bucket; extend with `--sectors map.json`, tune with `--max-sector-pct`) |
| Correlated exposure | 40% of equity across the order symbol plus held names whose last 60 daily returns correlate ≥ 0.70 with it |
| Trailing stop | off by default; `--trailing-stop 0.1` exits a long once it closes 10% below its highest close since entry. High-water marks persist across restarts; `--stop-basis high` tracks the highest intraday high instead of the highest close |
| Stop-loss / take-profit | off by default; `--stop-loss 0.08` / `--take-profit 0.25` exit a long once it closes that fraction below / above its average entry price. Checked before the trailing stop; one exit per symbol |
| Cash check | no margin; 0.5% headroom for costs |
| Daily loss limit | no new buys after a 3% down day |
| Max drawdown kill switch | at 20% from peak: latch, block buys, flatten all positions. The latch persists across restarts |
| Shorting | disabled; sells are clipped to the held quantity |

Sells always pass (they reduce risk), including while halted.

### Claude analyst (optional)

When `ANTHROPIC_API_KEY` is set and the SDK is installed (`pip install -e ".[llm]"`),
`finagent run` asks Claude (`claude-opus-5-5`) for per-symbol target weights, each with a rationale
and confidence, returned as schema-constrained JSON. finagent then:

1. validates the output (drops unknown symbols and invalid actions, clamps weights to [0, 1]),
2. converts target weights into share deltas,
3. runs every order through the same risk engine,
4. journals Claude's market view and every rationale, including rejected and hold decisions.

If there is no key, or the call fails, is refused, or returns invalid JSON, the tick falls back to
the rule-based policy and journals why. `--no-llm` forces rules.

### Data

`data/sample/` holds five **synthetic** tickers (`SYN_*`), about five years of business days, generated
deterministically by `scripts/gen_sample_data.py` (seed 42, GBM with regime switching). CI checks that
regenerating them gives byte-identical files.

`--provider` works on `backtest`, `sweep`, `walkforward`, `run` and `portfolio`:

| Provider | Source | Notes |
|---|---|---|
| `csv` (default) | `--data` directory of `SYMBOL.csv` | Offline. Used by tests and CI |
| `yahoo` | Yahoo Finance v8 chart JSON, keyless | ~10 years of daily bars, OHLC scaled by adjusted close (dividends + splits). Sent with a descriptive `finagent/…` User-Agent: spoofed browser UAs get HTTP 429 |
| `stooq` | Stooq CSV | Currently answers non-browser clients with a JavaScript bot-challenge page. finagent detects this (`BotChallenge`) and falls back instead of parsing HTML |

Live providers need `--symbols`. Fetched bars are cached as CSV in `--cache-dir` (default `data/cache`) and
reused for `--cache-hours` (default 12). If a refresh fails, a stale cached copy is used and labelled as
such; if there is no cache either, finagent falls back to `--data` CSVs, and otherwise fails with both errors.
Every command prints which source served each symbol (`live`, `cache`, `STALE cache`, or fallback).
You can also point `--data` at your own directory of `SYMBOL.csv` files (`date,open,high,low,close,volume`).

### Benchmark and trade analytics

Backtests compare the strategy with buy-and-hold of `--benchmark` (default `auto`: SPY for live data,
SYN_INDEX for the sample; `none` disables) over the same dates: benchmark return/CAGR/Sharpe/drawdown,
excess return, beta, correlation and alpha. Fills are grouped into round trips (flat → long → flat) with
commission-inclusive P&L, return, holding period and exit reason, summarised as win rate, average win/loss,
profit factor and expectancy. Both appear in the HTML report and `round_trips.csv`.

### Monte Carlo trade resampling

`finagent backtest --monte-carlo 2000 [--seed 0]` bootstraps the closed round trips: it redraws the same number of
trades with replacement, in random order, 2000 times and reports the 5th/50th/95th percentile total return, the
probability of ending with a loss and the median / 95th-percentile trade-to-trade drawdown. It answers "how much of
this result was the lucky order of a few trades?". It assumes trades are independent (streaks are broken up) and
ignores drawdown inside open trades, so treat the drawdown numbers as a floor.

### Overfitting guards: sweep and walk-forward

`finagent sweep` backtests every combination in `--grid` (strategy `entry`/`exit` or any numeric
`RiskConfig` field, e.g. `trailing_stop`, `max_position_pct`) on an in-sample window and on the held-out
period after `--split` (default 70% through the data). Rows are ranked by **in-sample** Sharpe, the only
information you would have had. It reports where the in-sample winner ranks out-of-sample and the IS/OOS
rank correlation, and warns when OOS Sharpe falls below half of IS or the ranking does not carry over.

`finagent walkforward` repeats that on rolling windows: choose parameters on `--train-days` (504), trade
them on the next unseen `--test-days` (126), roll forward, and stitch the test windows into one
out-of-sample curve, compared with the benchmark over the same span. Only the stitched OOS numbers are a
fair estimate; the in-sample Sharpe column is there to show how much of it was fitting.

### Notifications

`finagent run` prints one line per tick with the mode, equity, cash and each order's requested/approved
quantity and outcome. `--webhook URL` (or `FINAGENT_WEBHOOK_URL`) also POSTs it as JSON
`{"text", "content", "summary"}`, which Slack and Discord incoming webhooks accept. Ticks with no orders are
not posted unless `--notify-all`. Webhook errors are printed and never stop the loop. `--json` prints the full
tick summary.

## CLI

```
common:   [--provider csv|yahoo|stooq] [--data DIR] [--cache-dir data/cache] [--cache-hours 12] [--include-partial]
          [--symbols ...] [--strategy momentum|mean_reversion|combined|breakout] [--sizing atr|fixed] [--cash N]
          [--trailing-stop F] [--stop-basis close|high] [--stop-loss F] [--take-profit F]
          [--max-sector-pct F] [--sectors map.json]

finagent backtest    [--start D] [--end D] [--benchmark auto|SYMBOL|none] [--monte-carlo RUNS] [--seed N]
                     [--out reports/backtest]
finagent sweep       [--grid NAME=V1,V2 ...] [--split D] [--start D] [--end D] [--out reports/sweep.csv]
finagent walkforward [--grid NAME=V1,V2 ...] [--train-days 504] [--test-days 126] [--benchmark ...]
                     [--out reports/walkforward.csv]
finagent run         [--once | --interval SECONDS] [--db state/finagent.db] [--no-llm] [--model claude-opus-5-5]
                     [--webhook URL] [--notify-all] [--json]
finagent portfolio   [--db ...] [--provider ...]          # positions with stop/target/trailing exit levels
finagent report      [--db ...] [--provider ...] [--out reports/paper_report.html]
```

`backtest` writes `equity.csv`, `trades.csv`, `round_trips.csv`, `journal.csv` and `report.html`.

## Example output (synthetic data)

`finagent backtest` with the defaults:

```
Backtest combined | SYN_BANK, SYN_ENERGY, SYN_INDEX, SYN_TECH, SYN_UTIL
  total_return     -19.69%
  cagr             -4.29%
  sharpe           -0.95
  sortino          -1.27
  max_drawdown     20.12%
  win_rate         28.89%
  turnover         3.43
  buy_hold_return  -7.31%
  benchmark_total_return   -10.67%   (SYN_INDEX buy-and-hold)
```

The strategies do not make money on the synthetic random-walk data, and they should not be expected to.
This run shows the kill switch working: drawdown reached 20%, the engine flattened the book, and it stopped
trading. The strategies are simple examples for the framework, not a trading edge.

## Example output (real data, fetched 2026-09-25)

Observed output, abridged. It is a record of one run, not a forecast.

```
$ finagent backtest --provider yahoo --symbols SPY AAPL MSFT --start 2021-01-01 --trailing-stop 0.1 --cache-hours 0
data SPY: yahoo (live, cached 2513 bars)
Backtest combined | SPY, AAPL, MSFT
  total_return             10.80%      benchmark                SPY buy-and-hold
  cagr                     1.81%       benchmark_total_return   120.69%
  sharpe                   0.31        benchmark_sharpe         0.92
  max_drawdown             14.39%      benchmark_max_drawdown   24.50%
  round_trips              61          beta                     0.22
  profit_factor            1.27        alpha                    -1.30%

$ finagent walkforward --provider yahoo --symbols SPY AAPL MSFT --grid entry=0.2,0.3,0.4 --grid trailing_stop=0,0.1 --test-days 252
  folds 8 | oos_period 2018-09-26 .. 2026-09-24
  mean_is_sharpe 1.24 | mean_oos_sharpe 0.84 | oos_folds_profitable 75.00%
  oos_total_return 50.93% | oos_sharpe 0.82 | oos_max_drawdown 11.72%
  benchmark SPY: total_return 197.25% | sharpe 0.80

$ finagent run --once --provider yahoo --symbols SPY AAPL MSFT --trailing-stop 0.1
[finagent paper] 2026-09-24 mode=rules equity=99,978.12 cash=60,212.16 | BUY SPY 26/75: filled buy 26 @ 767.56 (paper); BUY AAPL 59/71: filled buy 59 @ 336.09 (paper)
```

On real large-cap data, the example strategies have lower drawdowns than SPY but trail buy-and-hold by a wide
margin. In-sample Sharpe overstates what the walk-forward delivers out-of-sample.

## Example output (v0.3.0 features, real data, 2026-09-25)

Observed output, abridged. Breakout strategy with a stop-loss, trailing stop on intraday highs and a Monte Carlo
pass over its trades (Yahoo bars fetched the same morning, served from the cache):

```
$ finagent backtest --provider yahoo --symbols SPY AAPL MSFT --start 2021-01-01 --strategy breakout \
    --stop-loss 0.08 --trailing-stop 0.1 --stop-basis high --monte-carlo 2000
Backtest breakout | SPY, AAPL, MSFT
  total_return             8.90%       benchmark_total_return   120.69%   (SPY buy-and-hold)
  sharpe                   0.31        benchmark_sharpe         0.92
  max_drawdown             10.57%      benchmark_max_drawdown   24.50%
  round_trips              43          profit_factor            1.40
  avg_days_held            65.72       beta                     0.15
  mc_return_p5             -7.09%      mc_prob_loss             18.50%
  mc_return_p50            8.53%       mc_max_drawdown_p50      5.83%
  mc_return_p95            23.97%      mc_max_drawdown_p95      12.43%
```

Resampling 43 trades gives roughly a one-in-five chance that the same trades, in another order and mix, would
have lost money. Tuning the stops in-sample does not carry over cleanly:

```
$ finagent sweep --provider yahoo --symbols SPY AAPL MSFT --strategy breakout \
    --grid stop_loss=0,0.05,0.1 --grid trailing_stop=0,0.1
in-sample best: IS Sharpe 0.82 -> OOS Sharpe 0.74 (ranks 1/6 out-of-sample; IS/OOS rank correlation -0.09)
WARNING: in-sample ranking does not carry over out-of-sample (rank correlation -0.09)

$ finagent walkforward ... (same grid) --test-days 252
  mean_is_sharpe 0.82 | mean_oos_sharpe 0.42 | oos_folds_profitable 62.50%
  oos_total_return 24.73% | oos_sharpe 0.51 | benchmark SPY: total_return 197.25%, sharpe 0.80
```

A live paper tick, then a second tick on the same bar, then the book with its exit levels:

```
$ finagent run --once --provider yahoo --symbols SPY AAPL MSFT --stop-loss 0.08 --take-profit 0.3 --trailing-stop 0.1 --cache-hours 0
[finagent paper] 2026-09-24 mode=rules equity=99,978.12 cash=60,212.16 | BUY SPY 26/75: filled buy 26 @ 767.56 (paper); BUY AAPL 59/71: filled buy 59 @ 336.09 (paper)
$ finagent run --once ... (same flags)
[finagent paper] 2026-09-24 mode=no new bar equity=99,978.12 cash=60,212.16 | no orders
$ finagent portfolio --provider yahoo
  AAPL  59 @ 336.09  last 335.92  value 19,819.28  P&L -9.91  weight 19.8%  exits: stop loss 309.20, take profit 436.91, trailing stop 302.33
  SPY   26 @ 767.56  last 767.18  value 19,946.68  P&L -9.97  weight 20.0%  exits: stop loss 706.16, take profit 997.83, trailing stop 690.46
```

## Limitations

- Long-only, daily bars, a single currency, no corporate actions, no intraday risk. No pairs / stat-arb strategy:
  it needs short legs, which the risk engine deliberately does not allow.
- Monte Carlo resampling treats trades as independent and only sees trade-to-trade drawdown.
- Fills are simulated at the next open (backtest) or the last close (live loop) plus fixed slippage. There is no order book or partial-fill model.
- A tick whose latest bar and prices match the previous tick is a no-op (`mode=no new bar`), so re-running
  `run --once` on static data no longer re-trades the same bar. It is journalled as a skip.
- During market hours Yahoo's last bar is today's unfinished session. It is dropped by default (detected from Yahoo's
  own session clock, `currentTradingPeriod` vs `regularMarketTime`); `--include-partial` keeps it for intraday ticks,
  treating the latest trade as the close, and refetches on every tick instead of reusing the cache.
- Stops trigger on the daily close, never intrabar. `--stop-basis high` ratchets the trailing stop's high-water mark
  on intraday highs instead of closes, but the exit is still decided at the close. Correlations use daily closes.
- Yahoo's endpoint is unofficial and rate limited; the cache keeps repeated runs to one request per symbol.

## License

MIT
