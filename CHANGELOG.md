# Changelog

All notable changes to finagent. Paper trading only: no release adds real order placement.

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
