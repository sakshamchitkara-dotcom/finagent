"""finagent command line. PAPER TRADING ONLY - nothing here can place a real order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import llm
from .agent import Agent
from .backtest import compute_metrics, run_backtest, write_csv
from .broker import PaperBroker
from .data import SAMPLE_DIR, CSVProvider, DataUnavailable, FallbackProvider, StooqProvider
from .report import write_report
from .risk import RiskConfig, RiskEngine
from .strategies import STRATEGIES, get_strategy

DISCLAIMER = "PAPER TRADING ONLY. Not financial advice. finagent never places real orders."
DEFAULT_DB = "state/finagent.db"


def _provider(args):
    csv = CSVProvider(args.data)
    if getattr(args, "provider", "csv") == "stooq":
        return FallbackProvider(StooqProvider(), csv)
    return csv


def _symbols(args) -> list[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    syms = CSVProvider(args.data).symbols()
    if not syms:
        sys.exit(f"no CSV files in {args.data}; pass --symbols")
    return syms


def _print_metrics(m: dict) -> None:
    for k, v in m.items():
        if isinstance(v, float):
            v = f"{v:.2%}" if k in {"total_return", "cagr", "max_drawdown", "win_rate", "buy_hold_return"} else f"{v:,.2f}"
        print(f"  {k:<16} {v}")


def cmd_backtest(args) -> int:
    provider = _provider(args)
    symbols = _symbols(args)
    res = run_backtest(provider, symbols, get_strategy(args.strategy), RiskConfig(sizing=args.sizing),
                       cash=args.cash, start=args.start, end=args.end)
    out = Path(args.out)
    write_csv(res.equity, out / "equity.csv")
    write_csv(res.fills, out / "trades.csv")
    write_csv(res.journal, out / "journal.csv")
    report = write_report(out / "report.html", f"Backtest: {args.strategy} on {', '.join(symbols)}",
                          res.equity, res.metrics, res.fills, note="Backtest on " + str(args.data)
                          + (" (SYNTHETIC sample data)" if Path(args.data).resolve() == SAMPLE_DIR else ""))
    print(f"Backtest {args.strategy} | {', '.join(symbols)}")
    _print_metrics(res.metrics)
    print(f"wrote {out / 'equity.csv'}, {out / 'trades.csv'}, {out / 'journal.csv'}, {report}")
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
                  RiskEngine(RiskConfig(sizing=args.sizing)), analyst,
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
    if isinstance(provider, FallbackProvider):
        for s, src in provider.served_by.items():
            print(f"data {s}: {src}")
    return 0


def cmd_portfolio(args) -> int:
    b = PaperBroker(args.db)
    positions = b.positions()
    prices = {}
    for s in positions:
        try:
            prices[s] = CSVProvider(args.data).history(s)[-1].close
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
    path = write_report(Path(args.out), "finagent paper account", eq, metrics, fills, b.rows("journal"),
                        note=f"State from {args.db}.")
    print(f"wrote {path} ({len(eq)} equity points, {len(fills)} fills)")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(SAMPLE_DIR), help="directory of <SYMBOL>.csv files")
    common.add_argument("--symbols", nargs="+", help="default: every CSV in --data")
    common.add_argument("--strategy", default="combined", choices=sorted(STRATEGIES))
    common.add_argument("--sizing", default="atr", choices=["atr", "fixed"])
    common.add_argument("--cash", type=float, default=100_000.0)

    p = argparse.ArgumentParser(prog="finagent", description=DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", parents=[common], help="run a backtest and write CSV + HTML report")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--out", default="reports/backtest")
    bt.set_defaults(fn=cmd_backtest)

    run = sub.add_parser("run", parents=[common], help="run the autonomous paper-trading loop")
    run.add_argument("--once", action="store_true", help="single tick, then exit")
    run.add_argument("--interval", type=float, default=86_400, help="seconds between ticks")
    run.add_argument("--db", default=DEFAULT_DB)
    run.add_argument("--provider", choices=["csv", "stooq"], default="csv",
                     help="stooq = live daily data, falls back to --data CSVs when unavailable")
    run.add_argument("--no-llm", action="store_true", help="force the rule-based policy")
    run.add_argument("--model", default=llm.MODEL)
    run.set_defaults(fn=cmd_run)

    pf = sub.add_parser("portfolio", help="show the paper portfolio")
    pf.add_argument("--db", default=DEFAULT_DB)
    pf.add_argument("--data", default=str(SAMPLE_DIR))
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
