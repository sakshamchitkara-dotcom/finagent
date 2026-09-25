"""finagent command line. PAPER TRADING ONLY - nothing here can place a real order."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from . import llm, optimize
from .agent import Agent
from .backtest import compute_metrics, monte_carlo, round_trips, run_backtest, trade_stats, write_csv
from .broker import PaperBroker
from .data import (SAMPLE_DIR, CachedProvider, CSVProvider, DataUnavailable, FallbackProvider, StooqProvider,
                   YahooProvider)
from .notify import Notifier
from .report import PCT, write_report
from .risk import PortfolioState, RiskConfig, RiskEngine
from .strategies import STRATEGIES, get_strategy

DISCLAIMER = "PAPER TRADING ONLY. Not financial advice. finagent never places real orders."
DEFAULT_DB = "state/finagent.db"
DEFAULT_CASH = 100_000.0


LIVE = {"yahoo": YahooProvider, "stooq": StooqProvider}


def _fraction(text: str) -> float:
    """argparse type: a fraction in [0, 1], e.g. 0.1 for 10%."""
    v = float(text)
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not a fraction in [0, 1], e.g. 0.1 for 10%")
    return v


def _non_negative(text: str) -> float:
    v = float(text)
    if not v >= 0:
        raise argparse.ArgumentTypeError(f"{text} must be 0 or more")
    return v


def _positive(text: str) -> float:
    v = float(text)
    if not v > 0:
        raise argparse.ArgumentTypeError(f"{text} must be greater than 0")
    return v


def _count(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"{text} must be 0 or more")
    return v


def _days(text: str) -> int:
    v = int(text)
    if v < 2:
        raise argparse.ArgumentTypeError(f"{text} must be at least 2")
    return v


def _date(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date like 2021-01-31") from None


def _provider(args):
    """csv = --data directory; yahoo/stooq = live, disk-cached, falling back to --data CSVs if a fetch fails."""
    csv = CSVProvider(args.data)
    if args.provider == "csv":
        return csv
    if args.provider == "yahoo":
        live = YahooProvider(include_partial=args.include_partial)
    else:
        live = LIVE[args.provider]()
    # An in-progress bar changes every minute, so --include-partial always refetches (the cache stays a fallback).
    hours = 0.0 if args.include_partial else args.cache_hours
    return FallbackProvider(CachedProvider(live, args.cache_dir, hours), csv)


def _risk_config(args) -> RiskConfig:
    cfg = RiskConfig(sizing=args.sizing, trailing_stop=args.trailing_stop, stop_basis=args.stop_basis,
                     stop_loss=args.stop_loss,
                     take_profit=args.take_profit, max_sector_pct=args.max_sector_pct,
                     regime_symbol=(args.regime_filter or "").upper(), regime_sma=args.regime_sma)
    if args.sectors:
        try:
            extra = json.loads(Path(args.sectors).read_text())
        except (OSError, ValueError) as e:
            sys.exit(f"--sectors: cannot read {args.sectors}: {e}")
        if not isinstance(extra, dict) or not all(isinstance(v, str) for v in extra.values()):
            sys.exit('--sectors must be a JSON object like {"AAPL": "tech"}')
        cfg.sectors.update({k.upper(): v for k, v in extra.items()})
    return cfg


def _print_sources(provider) -> None:
    for s, src in getattr(provider, "served_by", {}).items():
        print(f"data {s}: {src}")


def _symbols(args) -> list[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    if args.provider != "csv":
        sys.exit(f"--provider {args.provider} needs --symbols (e.g. --symbols SPY AAPL MSFT)")
    syms = CSVProvider(args.data).symbols()
    if not syms:
        sys.exit(f"no CSV files in {args.data}; pass --symbols")
    return syms


def _benchmark(args) -> str | None:
    """auto = SPY for live providers, SYN_INDEX for the bundled sample data, else none."""
    if args.benchmark.lower() == "none":
        return None
    if args.benchmark.lower() != "auto":
        return args.benchmark.upper()
    if args.provider != "csv":
        return "SPY"
    return "SYN_INDEX" if (Path(args.data) / "SYN_INDEX.csv").exists() else None


def _print_metrics(m: dict) -> None:
    for k, v in m.items():
        if isinstance(v, float):
            v = f"{v:.2%}" if k in PCT or k.removeprefix("oos_") in PCT or k == "oos_folds_profitable" else f"{v:,.2f}"
        print(f"  {k:<24} {v}")


def _data_note(args) -> str:
    if args.provider != "csv":
        return f"Data: {args.provider} (live, cached in {args.cache_dir})."
    return f"Data: {args.data}" + (" (SYNTHETIC sample data)." if Path(args.data).resolve() == SAMPLE_DIR else ".")


def cmd_backtest(args) -> int:
    provider = _provider(args)
    symbols = _symbols(args)
    try:
        res = run_backtest(provider, symbols, get_strategy(args.strategy), _risk_config(args),
                           cash=args.cash, start=args.start, end=args.end, benchmark=_benchmark(args))
    except (DataUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print_sources(provider)
    res.metrics.update(monte_carlo(round_trips(res.fills), args.cash, args.monte_carlo, args.seed))
    out = Path(args.out)
    write_csv(res.equity, out / "equity.csv")
    write_csv(res.fills, out / "trades.csv")
    write_csv(res.journal, out / "journal.csv")
    write_csv(round_trips(res.fills), out / "round_trips.csv")
    report = write_report(out / "report.html", f"Backtest: {args.strategy} on {', '.join(symbols)}",
                          res.equity, res.metrics, res.fills, note=_data_note(args), benchmark=res.benchmark,
                          positions=res.positions)
    print(f"Backtest {args.strategy} | {', '.join(symbols)}")
    _print_metrics(res.metrics)
    print(f"wrote {out}/{{equity,trades,round_trips,journal}}.csv and {report}")
    return 0


def _print_table(rows: list[dict], cols: list[str]) -> None:
    def cell(k, v):
        if isinstance(v, float):
            return f"{v:.2%}" if k.split("_", 1)[-1] in PCT else f"{v:.2f}"
        return str(v)

    table = [[cell(c, r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for t in table:
        print("  ".join(v.rjust(w) for v, w in zip(t, widths)))


DEFAULT_GRID = ["entry=0.2,0.3,0.4", "exit=-0.2,-0.1"]


def _grid(args) -> dict[str, list[float]]:
    try:
        return optimize.parse_grid(args.grid or DEFAULT_GRID)
    except ValueError as e:
        sys.exit(f"--grid: {e}")


def _print_overfit(check: dict) -> None:
    print(f"in-sample best: IS Sharpe {check['best_is_sharpe']:.2f} -> OOS Sharpe {check['best_oos_sharpe']:.2f}"
          f" (ranks {check['oos_rank_of_is_best']} out-of-sample; IS/OOS rank correlation "
          f"{check['is_oos_rank_correlation']:+.2f})")
    for w in check["warnings"]:
        print(f"WARNING: {w}")
    if not check["warnings"]:
        print("no overfitting red flags (which is not proof of an edge)")


def cmd_sweep(args) -> int:
    provider, symbols, grid = _provider(args), _symbols(args), _grid(args)
    print(f"Sweep {args.strategy} | {', '.join(symbols)} | {len(optimize.combos(grid))} combinations")
    try:
        rows = optimize.sweep(provider, symbols, args.strategy, grid, args.split, args.start, args.end,
                              _risk_config(args), args.cash)
    except (DataUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print_sources(provider)
    print(f"in-sample < {rows[0]['split']} <= out-of-sample; sorted by in-sample Sharpe (selection uses IS only)")
    _print_table(rows, list(grid) + [f"{p}_{k}" for p in ("is", "oos") for k in optimize.REPORTED])
    _print_overfit(optimize.overfit_check(rows))
    write_csv(rows, Path(args.out))
    print(f"wrote {args.out}")
    return 0


def cmd_walkforward(args) -> int:
    provider, symbols, grid = _provider(args), _symbols(args), _grid(args)
    print(f"Walk-forward {args.strategy} | {', '.join(symbols)} | {len(optimize.combos(grid))} combinations | "
          f"train {args.train_days}d, test {args.test_days}d")
    try:
        folds, summary = optimize.walk_forward(provider, symbols, args.strategy, grid, args.train_days,
                                               args.test_days, args.start, args.end, _risk_config(args), args.cash,
                                               _benchmark(args))
    except (DataUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print_sources(provider)
    _print_table(folds, ["fold", "test", *grid, "is_sharpe", "oos_sharpe", "oos_total_return", "oos_max_drawdown",
                         "oos_trades"])
    print("stitched out-of-sample result (parameters re-chosen on each training window only):")
    _print_metrics(summary)
    if summary["mean_is_sharpe"] > 0 and summary["mean_oos_sharpe"] < 0.5 * summary["mean_is_sharpe"]:
        print("WARNING: mean out-of-sample Sharpe is less than half the in-sample Sharpe: likely overfit")
    write_csv(folds, Path(args.out))
    print(f"wrote {args.out}")
    return 0


def cmd_run(args) -> int:
    provider = _provider(args)
    symbols = _symbols(args)
    analyst = None
    if not args.no_llm and llm.available():
        analyst = llm.ClaudeAnalyst(model=args.model)
        print(f"analyst: Claude ({args.model})")
    else:
        print("analyst: rule-based" + ("" if args.no_llm else " (no ANTHROPIC_API_KEY or anthropic SDK)"))
    try:
        notify = Notifier(args.webhook or os.environ.get("FINAGENT_WEBHOOK_URL"), only_orders=not args.notify_all)
    except ValueError as e:
        sys.exit(f"--webhook: {e}")

    def log(summary: dict) -> None:
        if args.json:
            print(json.dumps(summary, default=str))
        notify(summary)

    broker = PaperBroker(args.db, starting_cash=args.cash)
    if args.cash_given and broker.starting_cash != args.cash:
        print(f"WARNING: --cash {args.cash:,.2f} ignored: {args.db} is an existing paper account that started with "
              f"{broker.starting_cash:,.2f} (use a new --db for a fresh account)", file=sys.stderr)
    agent = Agent(provider, broker, symbols, get_strategy(args.strategy),
                  RiskEngine(_risk_config(args)), analyst, log=log)
    try:
        if args.once:
            agent.tick()
        else:
            agent.run(args.interval)
    except DataUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    _print_sources(provider)
    return 0


def _paper_book(b: PaperBroker, args) -> tuple[dict[str, float], list[dict], list[str]]:
    """Latest prices, open positions with exit levels, and the symbols that had to be marked at cost."""
    positions, provider, prices, at_cost = b.positions(), _provider(args), {}, []
    for s in positions:
        try:
            prices[s] = provider.history(s)[-1].close
        except DataUnavailable:
            prices[s] = positions[s][1]
            at_cost.append(s)
    risk = RiskEngine(RiskConfig(**(b.get_meta("exit_config") or {})))
    risk.stop_highs = b.get_meta("stop_highs") or {}
    state = PortfolioState(b.cash, {s: q for s, (q, _) in positions.items()}, prices, 0.0, 0.0,
                           costs={s: a for s, (_, a) in positions.items()})  # read-only: no begin_day side effects
    return prices, risk.levels(state), at_cost


def cmd_portfolio(args) -> int:
    b = PaperBroker(args.db)
    prices, rows, at_cost = _paper_book(b, args)
    eq = b.equity(prices)
    print(f"cash      {b.cash:>14,.2f}")
    print(f"equity    {eq:>14,.2f}   (start {b.starting_cash:,.2f}, {eq / b.starting_cash - 1:+.2%})")
    print(f"kill switch: {'ENGAGED' if b.get_meta('killed') else 'off'}")
    for r in rows:
        exits = ", ".join(f"{k.replace('_', ' ')} {r[k]:.2f}" for k in ("stop_loss", "take_profit", "trailing_stop")
                          if r[k] is not None)
        print(f"  {r['symbol']:<12} {r['qty']:>8} @ {r['avg_entry']:>10.2f}  last {r['last']:>10.2f}  "
              f"value {r['value']:>12,.2f}  P&L {r['unrealized_pnl']:>+12,.2f}  weight {r['weight']:.1%}"
              + ("  (NO PRICE: marked at cost)" if r["symbol"] in at_cost else "")
              + (f"  exits: {exits}" if exits else ""))
    if not rows:
        print("  (no positions)")
    return 0


def cmd_report(args) -> int:
    b = PaperBroker(args.db)
    eq, fills = b.rows("equity"), b.rows("fills")
    metrics = compute_metrics([r["equity"] for r in eq], fills, b.starting_cash)
    metrics.update(trade_stats(round_trips(fills)))
    _, positions, at_cost = _paper_book(b, args)
    note = f"State from {args.db}." + (f" No current price for {', '.join(at_cost)}: marked at cost." if at_cost else "")
    path = write_report(Path(args.out), "finagent paper account", eq, metrics, fills, b.rows("journal"),
                        note=note, positions=positions)
    print(f"wrote {path} ({len(eq)} equity points, {len(fills)} fills, {len(positions)} open positions)")
    return 0


def main(argv: list[str] | None = None) -> int:
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--provider", choices=["csv", *LIVE], default="csv",
                      help="csv = --data directory (default); yahoo/stooq = live daily bars, disk-cached, "
                           "falling back to --data CSVs when unavailable")
    data.add_argument("--data", default=str(SAMPLE_DIR), help="directory of <SYMBOL>.csv files")
    data.add_argument("--cache-dir", default="data/cache", help="where live bars are cached")
    data.add_argument("--cache-hours", type=float, default=12.0, help="refetch cached bars older than this")
    data.add_argument("--include-partial", action="store_true",
                      help="yahoo: keep today's in-progress bar during market hours (its close is the latest "
                           "trade, not a daily close); disables cache reuse. Default: drop it")

    common = argparse.ArgumentParser(add_help=False, parents=[data])
    common.add_argument("--symbols", nargs="+", help="default: every CSV in --data")
    common.add_argument("--strategy", default="combined", choices=sorted(STRATEGIES))
    common.add_argument("--sizing", default="atr", choices=["atr", "fixed"])
    common.add_argument("--cash", type=_positive, default=None,
                        help=f"starting cash (default {DEFAULT_CASH:,.0f}); run: only used when --db is a new account")
    common.add_argument("--trailing-stop", type=_fraction, default=0.0, metavar="FRACTION",
                        help="exit a long after it falls this fraction from its highest close (e.g. 0.1); 0 = off")
    common.add_argument("--stop-basis", choices=["close", "high"], default="close",
                        help="trailing-stop high-water mark: highest close (default) or highest intraday high")
    common.add_argument("--stop-loss", type=_fraction, default=0.0, metavar="FRACTION",
                        help="exit a long once it closes this fraction below its average entry (e.g. 0.08); 0 = off")
    common.add_argument("--take-profit", type=_non_negative, default=0.0, metavar="FRACTION",
                        help="exit a long once it closes this fraction above its average entry (e.g. 0.25); 0 = off")
    common.add_argument("--max-sector-pct", type=_fraction, default=RiskConfig.max_sector_pct, metavar="FRACTION",
                        help="cap on total long exposure per sector (default %(default)s)")
    common.add_argument("--regime-filter", metavar="SYMBOL",
                        help="no new buys while SYMBOL (e.g. SPY) closes below its --regime-sma-day SMA; sells and "
                             "exits still run. Default off")
    common.add_argument("--regime-sma", type=_days, default=RiskConfig.regime_sma, metavar="DAYS",
                        help="moving-average length for --regime-filter (default %(default)s)")
    common.add_argument("--sectors", metavar="JSON", help='JSON file {"SYMBOL": "sector"} extending the built-in map')

    p = argparse.ArgumentParser(prog="finagent", description=DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", parents=[common], help="run a backtest and write CSV + HTML report")
    bt.add_argument("--start", type=_date)
    bt.add_argument("--end", type=_date)
    bt.add_argument("--out", default="reports/backtest")
    bt.add_argument("--benchmark", default="auto",
                    help="buy-and-hold benchmark symbol; auto = SPY (live) / SYN_INDEX (sample); none disables")
    bt.add_argument("--monte-carlo", type=_count, default=0, metavar="RUNS",
                    help="bootstrap the closed trades RUNS times and report return/drawdown percentiles")
    bt.add_argument("--seed", type=int, default=0, help="random seed for --monte-carlo (default 0)")
    bt.set_defaults(fn=cmd_backtest)

    sw = sub.add_parser("sweep", parents=[common], help="parameter grid: in-sample vs out-of-sample results table")
    sw.add_argument("--grid", action="append", metavar="NAME=V1,V2",
                    help=f"repeatable; strategy entry/exit or any numeric RiskConfig field (default {DEFAULT_GRID})")
    sw.add_argument("--split", type=_date, help="first out-of-sample date (default: 70%% through the data)")
    sw.add_argument("--start", type=_date)
    sw.add_argument("--end", type=_date)
    sw.add_argument("--out", default="reports/sweep.csv")
    sw.set_defaults(fn=cmd_sweep)

    wf = sub.add_parser("walkforward", parents=[common], help="rolling walk-forward out-of-sample evaluation")
    wf.add_argument("--grid", action="append", metavar="NAME=V1,V2", help="as for sweep")
    wf.add_argument("--train-days", type=int, default=504, help="in-sample window in trading days (default 504)")
    wf.add_argument("--test-days", type=int, default=126, help="out-of-sample window in trading days (default 126)")
    wf.add_argument("--start", type=_date)
    wf.add_argument("--end", type=_date)
    wf.add_argument("--benchmark", default="auto", help="as for backtest")
    wf.add_argument("--out", default="reports/walkforward.csv")
    wf.set_defaults(fn=cmd_walkforward)

    run = sub.add_parser("run", parents=[common], help="run the autonomous paper-trading loop")
    run.add_argument("--once", action="store_true", help="single tick, then exit")
    run.add_argument("--interval", type=_positive, default=86_400, help="seconds between ticks")
    run.add_argument("--db", default=DEFAULT_DB)
    run.add_argument("--no-llm", action="store_true", help="force the rule-based policy")
    run.add_argument("--webhook", metavar="URL",
                     help="POST each decision as JSON (Slack/Discord compatible); default $FINAGENT_WEBHOOK_URL")
    run.add_argument("--notify-all", action="store_true", help="also POST ticks with no orders")
    run.add_argument("--json", action="store_true", help="also print the full tick summary as JSON")
    run.add_argument("--model", default=llm.MODEL)
    run.set_defaults(fn=cmd_run)

    pf = sub.add_parser("portfolio", parents=[data], help="show the paper portfolio")
    pf.add_argument("--db", default=DEFAULT_DB)
    pf.set_defaults(fn=cmd_portfolio)

    rp = sub.add_parser("report", parents=[data], help="HTML report of the paper account")
    rp.add_argument("--db", default=DEFAULT_DB)
    rp.add_argument("--out", default="reports/paper_report.html")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    if hasattr(args, "cash"):
        args.cash_given = args.cash is not None
        args.cash = DEFAULT_CASH if args.cash is None else args.cash
    print(DISCLAIMER)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
