"""Event-driven daily backtester. Signals at close of day t, fills at open of day t+1 (no look-ahead)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .broker import PaperBroker
from .data import DataProvider
from .risk import RiskConfig, RiskEngine
from .strategies import Strategy, features, rule_based_order

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: list[dict]   # {"ts", "equity", "cash"}
    fills: list[dict]
    journal: list[dict]
    metrics: dict


def compute_metrics(equity: list[float], fills: list[dict], starting_cash: float) -> dict:
    """Performance metrics from a daily equity series (starting value = starting_cash) and fill records."""
    series = [starting_cash] + list(equity)
    rets = [series[i] / series[i - 1] - 1 for i in range(1, len(series)) if series[i - 1] > 0]
    n = len(rets)
    years = n / TRADING_DAYS if n else 0.0
    total = series[-1] / starting_cash - 1
    mean = sum(rets) / n if n else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1)) if n > 1 else 0.0
    dsd = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / n) if n else 0.0
    peak, mdd = series[0], 0.0
    for v in series:
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak if peak > 0 else 0.0)
    closes = [f for f in fills if f["side"] == "sell"]
    traded = sum(f["qty"] * f["price"] for f in fills)
    avg_eq = sum(series) / len(series)
    return {
        "days": n,
        "start_equity": round(starting_cash, 2),
        "end_equity": round(series[-1], 2),
        "total_return": total,
        "cagr": (series[-1] / starting_cash) ** (1 / years) - 1 if years > 0 and series[-1] > 0 else 0.0,
        "sharpe": mean / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0,
        "sortino": mean / dsd * math.sqrt(TRADING_DAYS) if dsd > 0 else 0.0,
        "max_drawdown": mdd,
        "trades": len(fills),
        "closed_trades": len(closes),
        "win_rate": sum(1 for f in closes if f["realized_pnl"] > 0) / len(closes) if closes else 0.0,
        "turnover": traded / avg_eq / years if years > 0 and avg_eq > 0 else 0.0,  # annualized, x equity
        "commissions": round(sum(f["commission"] for f in fills), 2),
    }


def run_backtest(provider: DataProvider, symbols: list[str], strategy: Strategy,
                 risk_config: RiskConfig | None = None, cash: float = 100_000.0,
                 start: str | None = None, end: str | None = None, **broker_kwargs) -> BacktestResult:
    history = {s: provider.history(s) for s in symbols}
    common = set.intersection(*(set(b.date for b in bars) for bars in history.values()))
    # Warm-up bars before `start` are kept so indicators are ready on day one.
    dates = sorted(d for d in common if not end or d <= end)
    bars = {s: [b for b in history[s] if b.date in common and (not end or b.date <= end)] for s in symbols}
    first = next((i for i, d in enumerate(dates) if not start or d >= start), len(dates))
    if first >= len(dates):
        raise ValueError("no data in the requested date range")

    broker = PaperBroker(":memory:", starting_cash=cash, **broker_kwargs)
    risk = RiskEngine(risk_config)
    pending = []
    for i in range(first, len(dates)):
        day = dates[i]
        opens = {s: bars[s][i].open for s in symbols}
        # 1) execute yesterday's decisions at today's open, sells first to free cash
        for order in sorted(pending, key=lambda o: o.side != "sell"):
            state = broker.begin_day(day, opens)
            d = risk.check(order, state)
            outcome = "rejected"
            if d.approved:
                f = broker.execute(d, opens[order.symbol], day)
                outcome = f"filled {f.qty} @ {f.price:.2f}"
            broker.journal(day, order.symbol, order.source, order.side, order.qty, d, order.reason, outcome)
        pending = []
        broker.begin_day(day, opens)
        closes = {s: bars[s][i].close for s in symbols}
        equity = broker.mark(day, closes)
        # 2) decide at the close for tomorrow
        if i + 1 < len(dates):
            state = broker.begin_day(day, closes)
            if risk.update(state):
                pending = risk.liquidation_orders(state)
                continue
            held = state.positions
            for s in symbols:
                f = features(bars[s][:i + 1])
                if f is None:
                    continue
                order = rule_based_order(s, strategy.signal(f), strategy, held.get(s, 0), equity, risk)
                if order:
                    pending.append(order)

    eq, fills = broker.rows("equity"), broker.rows("fills")
    metrics = compute_metrics([r["equity"] for r in eq], fills, cash)
    metrics["strategy"] = strategy.name
    metrics["period"] = f"{dates[first]} .. {dates[-1]}"
    p0 = {s: bars[s][first].open for s in symbols}
    metrics["buy_hold_return"] = sum(bars[s][-1].close / p0[s] for s in symbols) / len(symbols) - 1
    return BacktestResult(eq, fills, broker.rows("journal"), metrics)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
