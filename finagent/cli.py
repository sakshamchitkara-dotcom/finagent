"""finagent command line. PAPER TRADING ONLY - nothing here can place a real order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import llm
from .agent import Agent
from .backtest import compute_metrics, round_trips, run_backtest, trade_stats, write_csv
from .broker import PaperBroker
from .data import (SAMPLE_DIR, CachedProvider, CSVProvider, DataUnavailable, FallbackProvider, StooqProvider,
                   YahooProvider)
from .report import PCT, write_report
from .risk import RiskConfig, RiskEngine
from .strategies import STRATEGIES, get_strategy

DISCLAIMER = "PAPER TRADING ONLY. Not financial advice. finagent never places real orders."
DEFAULT_DB = "state/finagent.db"


LIVE = {"yahoo": YahooProvider, "stooq": StooqProvider}


def _provider(args):
    """csv = --data directory; yahoo/stooq = live, disk-cached, falling back to --data CSVs if a fetch fails."""
    csv = CSVProvider(args.data)
    if args.provider == "csv":
        return csv
    return FallbackProvider(CachedProvider(LIVE[args.provider](), args.cache_dir, args.cache_hours), csv)


def _risk_config(args) -> RiskConfig:
    cfg = RiskConfig(sizing=args.sizing, trailing_stop=args.trailing_stop, max_sector_pct=args.max_sector_pct)
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
            v = f"{v:.2%}" if k in PCT else f"{v:,.2f}"
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
    except DataUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print_sources(provider)
    out = Path(args.out)
    write_csv(res.equity, out / "equity.csv")
    write_csv(res.fills, out / "trades.csv")
    write_csv(res.journal, out / "journal.csv")
    write_csv(round_trips(res.fills), out / "round_trips.csv")
    report = write_report(out / "report.html", f"Backtest: {args.strategy} on {', '.join(symbols)}",
                          res.equity, res.metrics, res.fills, note=_data_note(args), benchmark=res.benchmark)
    print(f"Backtest {args.strategy} | {', '.join(symbols)}")
    _print_metrics(res.metrics)
    print(f"wrote {out}/{{equity,trades,round_trips,journal}}.csv and {report}")
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
    agent = Agent(provider, PaperBroker(args.db, starting_cash=args.cash), symbols, get_strategy(args.strategy),
                  RiskEngine(_risk_config(args)), analyst,
                  log=lambda s: print(json.dumps(s, default=str)))
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


def cmd_portfolio(args) -> int:
    b = PaperBroker(args.db)
    positions = b.positions()
    provider, prices = _provider(args), {}
    for s in positions:
        try:
            prices[s] = provider.history(s)[-1].close
        except DataUnavailable:
            prices[s] = positions[s][1]  # fall back to cost
    eq = b.equity(prices)
    print(f"cash      {b.cash:>14,.2f}")
    print(f"equity    {eq:>14,.2f}   (start {b.starting_cash:,.2f}, {eq / b.starting_cash - 1:+.2%})")
    print(f"kill switch: {'ENGAGED' if b.get_meta('killed') else 'off'}")
    for s, (q, avg) in positions.items():
        px = prices[s]
        print(f"  {s:<12} {q:>8} @ {avg:>10.2f}  last {px:>10.2f}  value {q * px:>12,.2f}  "
              f"P&L {(px - avg) * q:>+12,.2f}  weight {q * px / eq:.1%}")
    if not positions:
        print("  (no positions)")
    return 0


def cmd_report(args) -> int:
    b = PaperBroker(args.db)
    eq, fills = b.rows("equity"), b.rows("fills")
    metrics = compute_metrics([r["equity"] for r in eq], fills, b.starting_cash)
    metrics.update(trade_stats(round_trips(fills)))
    path = write_report(Path(args.out), "finagent paper account", eq, metrics, fills, b.rows("journal"),
                        note=f"State from {args.db}.")
    print(f"wrote {path} ({len(eq)} equity points, {len(fills)} fills)")
    return 0


def main(argv: list[str] | None = None) -> int:
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--provider", choices=["csv", *LIVE], default="csv",
                      help="csv = --data directory (default); yahoo/stooq = live daily bars, disk-cached, "
                           "falling back to --data CSVs when unavailable")
    data.add_argument("--data", default=str(SAMPLE_DIR), help="directory of <SYMBOL>.csv files")
    data.add_argument("--cache-dir", default="data/cache", help="where live bars are cached")
    data.add_argument("--cache-hours", type=float, default=12.0, help="refetch cached bars older than this")

    common = argparse.ArgumentParser(add_help=False, parents=[data])
    common.add_argument("--symbols", nargs="+", help="default: every CSV in --data")
    common.add_argument("--strategy", default="combined", choices=sorted(STRATEGIES))
    common.add_argument("--sizing", default="atr", choices=["atr", "fixed"])
    common.add_argument("--cash", type=float, default=100_000.0)
    common.add_argument("--trailing-stop", type=float, default=0.0, metavar="FRACTION",
                        help="exit a long after it falls this fraction from its highest close (e.g. 0.1); 0 = off")
    common.add_argument("--max-sector-pct", type=float, default=RiskConfig.max_sector_pct, metavar="FRACTION",
                        help="cap on total long exposure per sector (default %(default)s)")
    common.add_argument("--sectors", metavar="JSON", help='JSON file {"SYMBOL": "sector"} extending the built-in map')

    p = argparse.ArgumentParser(prog="finagent", description=DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", parents=[common], help="run a backtest and write CSV + HTML report")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--out", default="reports/backtest")
    bt.add_argument("--benchmark", default="auto",
                    help="buy-and-hold benchmark symbol; auto = SPY (live) / SYN_INDEX (sample); none disables")
    bt.set_defaults(fn=cmd_backtest)

    run = sub.add_parser("run", parents=[common], help="run the autonomous paper-trading loop")
    run.add_argument("--once", action="store_true", help="single tick, then exit")
    run.add_argument("--interval", type=float, default=86_400, help="seconds between ticks")
    run.add_argument("--db", default=DEFAULT_DB)
    run.add_argument("--no-llm", action="store_true", help="force the rule-based policy")
    run.add_argument("--model", default=llm.MODEL)
    run.set_defaults(fn=cmd_run)

    pf = sub.add_parser("portfolio", parents=[data], help="show the paper portfolio")
    pf.add_argument("--db", default=DEFAULT_DB)
    pf.set_defaults(fn=cmd_portfolio)

    rp = sub.add_parser("report", help="HTML report of the paper account")
    rp.add_argument("--db", default=DEFAULT_DB)
    rp.add_argument("--out", default="reports/paper_report.html")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    print(DISCLAIMER)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
