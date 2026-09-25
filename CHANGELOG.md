# Changelog

All notable changes to finagent. Paper trading only: no release adds real order placement.

## [0.3.0] - 2026-09-25

### Added
- `breakout` strategy: Donchian channel trend following (enter above the prior 55-day high, exit below the prior
  20-day low); `features()` exposes `high_55` / `low_20`.
- Position-level exits: `--stop-loss` and `--take-profit` against the average entry price, sharing one exit pass
  with the trailing stop (at most one exit per symbol). Both can be tuned in `sweep` / `walkforward` grids.
- `--stop-basis high`: ratchet the trailing stop on intraday highs instead of closes.
- Exit levels in reports: backtest and paper-account reports and `finagent portfolio` list open positions with
  their stop-loss, take-profit and trailing-stop prices; positions without a current price are flagged.
- `backtest --monte-carlo RUNS [--seed N]`: bootstrap of closed round trips with return and drawdown percentiles
  and the probability of a loss.
- `--include-partial`: keep Yahoo's in-progress bar for intraday ticks (always refetches).
- Tests for strategies, agent degraded-data paths and run loop, CLI sweep/walkforward/validation, report escaping.

### Changed
- Yahoo's in-progress daily bar is dropped by default during market hours (detected from Yahoo's session clock).
- A tick that sees the same bar and prices as the previous one is a journalled no-op (`mode=no new bar`).
- `RiskEngine.trailing_stop_orders` is now `RiskEngine.exit_orders`; `PortfolioState` carries `costs` and `highs`.
- `report` accepts the data-provider options (to price open positions).
- The version is single-sourced from `finagent.__version__` and sent in the data User-Agent.

### Fixed
- Sweep IS/OOS rank correlation averages tied ranks; all-tied grids no longer report a spurious -1.00 and warning.

## [0.2.0] - 2026-09-25

### Added
- `YahooProvider`: free, keyless daily bars from Yahoo Finance's chart JSON endpoint, adjusted for dividends
  and splits (`--provider yahoo`).
- `CachedProvider`: disk cache for live bars (`--cache-dir`, `--cache-hours`), with atomic writes and a
  clearly labelled stale-cache fallback when a refresh fails.
- `--provider csv|yahoo|stooq` on `backtest`, `sweep`, `walkforward`, `run` and `portfolio`; each command
  reports which source served every symbol.
- Benchmark comparison: buy-and-hold of `--benchmark` (SPY for live data) with excess return, beta,
  correlation and alpha; drawn as an overlay on the report's equity chart.
- Per-trade analytics: round trips with P&L, return, holding period and exit reason; win rate, profit factor,
  expectancy; per-symbol summary and trade table in the HTML report; `round_trips.csv`.
- `finagent sweep`: parameter grid with in-sample vs out-of-sample results, ranked by in-sample Sharpe, with
  overfitting warnings (OOS Sharpe collapse, IS/OOS rank correlation).
- `finagent walkforward`: rolling train/test evaluation with a stitched out-of-sample equity curve.
- Risk: trailing stops (`--trailing-stop`), sector exposure cap (`--max-sector-pct`, `--sectors`) and a cap on
  exposure to clusters of correlated holdings.
- Decision notifications: one line per tick on stdout, optional JSON webhook (`--webhook` /
  `FINAGENT_WEBHOOK_URL`) compatible with Slack and Discord.

### Changed
- All network fetches go through one helper that raises `BotChallenge` for HTML responses (Stooq now serves a
  JavaScript challenge page) and reports HTTP 429 as rate limiting.
- `finagent run` prints a human-readable decision line; the raw JSON summary moved behind `--json`.
- `FallbackProvider` reports both errors when neither provider can serve a symbol.
- Backtests pass only the indicator lookback window to `features()` (same results, less copying).

## [0.1.0] - 2026-09-25

- Initial release: indicators, momentum / mean-reversion / combined strategies, risk engine with kill switch,
  sqlite paper broker, backtester with HTML report, autonomous loop, optional Claude analyst, CI.
