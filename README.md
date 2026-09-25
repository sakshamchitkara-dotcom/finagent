# finagent

An autonomous **paper-trading** agent in pure Python: market data providers, technical
indicators, pluggable strategies, a deterministic risk engine, a simulated broker with
persisted state, a backtester with an HTML report, and an optional Claude analyst.

> [!WARNING]
> **Paper trading only. Not financial advice.** finagent has no brokerage integration and
> no code path that places a real order. The bundled market data is **synthetic**. Backtest
> results are not a prediction of future returns. Do not use this to make investment decisions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core has zero runtime dependencies
pytest -q

finagent backtest                # all sample symbols, combined strategy
finagent run --once              # one autonomous tick against the paper account
finagent portfolio               # positions, cash, P&L
finagent report                  # HTML report of the paper account
```

Outputs go to `reports/` and paper state to `state/finagent.db` (both git-ignored).

## How it works

```
observe ──> analyze ──> decide ─────────────> risk-check ──> execute (paper) ──> journal
 data       indicators   rule policy, or        RiskEngine     PaperBroker         sqlite
 providers  + strategy   Claude proposals       (every order)  slippage + fees     journal table
            scores       (fallback: rules)
```

| Module | What it does |
|---|---|
| `data.py` | `DataProvider` protocol, `CSVProvider`, `StooqProvider` (live daily CSV via `urllib`), `FallbackProvider` |
| `indicators.py` | SMA, EMA, RSI (Wilder), MACD, ATR (Wilder), Bollinger bands |
| `strategies.py` | `momentum`, `mean_reversion`, `combined`; each returns a score in [-1, 1] with a reason. Add one by implementing `signal()` and registering it in `STRATEGIES` |
| `risk.py` | Position sizing and pre-trade checks (below) |
| `broker.py` | Paper fills with slippage (5 bps) and commission ($0.005/share, $1 min); cash, positions, fills, equity and journal in sqlite |
| `backtest.py` | Daily event loop: decide at close, fill at next open. CAGR, Sharpe, Sortino, max drawdown, win rate, turnover |
| `report.py` | Self-contained HTML report with an inline SVG equity + drawdown chart |
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

`--provider stooq` fetches free daily data from Stooq and falls back to `--data` CSVs when the fetch fails.
Stooq sometimes serves a browser challenge page instead of CSV; finagent detects that and falls back.
You can also point `--data` at your own directory of `SYMBOL.csv` files (`date,open,high,low,close,volume`).

## CLI

```
finagent backtest [--strategy momentum|mean_reversion|combined] [--symbols ...] [--start D] [--end D]
                  [--sizing atr|fixed] [--cash N] [--data DIR] [--out reports/backtest]
finagent run      [--once | --interval SECONDS] [--db state/finagent.db] [--provider csv|stooq]
                  [--no-llm] [--model claude-opus-5-5] [--strategy ...] [--symbols ...]
finagent portfolio [--db ...]
finagent report    [--db ...] [--out reports/paper_report.html]
```

`backtest` writes `equity.csv`, `trades.csv`, `journal.csv` and `report.html`.

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
```

The strategies do not make money on the synthetic random-walk data, and they should not be expected to.
This run shows the kill switch working: drawdown reached 20%, the engine flattened the book, and it stopped
trading. The strategies are simple examples for the framework, not a trading edge.

## Limitations

- Long-only, daily bars, a single currency, no corporate actions, no intraday risk.
- Fills are simulated at the next open (backtest) or the last close (live loop) plus fixed slippage. There is no order book or partial-fill model.
- Running `run --once` again on the same static CSV data re-evaluates the same bar.

## License

MIT
