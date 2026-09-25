"""Event-driven daily backtester. Signals at close of day t, fills at open of day t+1 (no look-ahead)."""

from __future__ import annotations

import csv
import math
import random
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .broker import PaperBroker
from .data import Bar, DataProvider, DataUnavailable
from .risk import RiskConfig, RiskEngine, regime
from .strategies import LOOKBACK, Strategy, features, rule_based_order

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: list[dict]   # {"ts", "equity", "cash"}
    fills: list[dict]
    journal: list[dict]
    metrics: dict
    benchmark: list[dict] | None = None  # {"ts", "equity"} buy-and-hold curve on the same dates
    positions: list[dict] | None = None  # open positions at the end, with their exit levels (RiskEngine.levels)


def daily_returns(values: list[float]) -> list[float]:
    return [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]


def buy_and_hold(bars: list[Bar], dates: list[str], cash: float) -> list[float]:
    """Equity of `cash` put into one symbol at its first open on/after dates[0], marked at each close.

    Days the benchmark did not trade carry the previous close forward.
    """
    by, entry, last, out = {b.date: b for b in bars}, None, None, []
    for d in dates:
        b = by.get(d)
        if b is not None:
            entry = entry or b.open
            last = b.close
        out.append(cash * last / entry if entry else cash)
    return out


def relative_metrics(strategy: list[float], bench: list[float], starting_cash: float) -> dict:
    """Beta, correlation, annualized alpha and excess return of a strategy curve vs a benchmark curve."""
    s = daily_returns([starting_cash] + strategy)
    b = daily_returns([starting_cash] + bench)
    n = min(len(s), len(b))
    s, b = s[:n], b[:n]
    ms, mb = (sum(s) / n, sum(b) / n) if n else (0.0, 0.0)
    cov = sum((x - ms) * (y - mb) for x, y in zip(s, b)) / (n - 1) if n > 1 else 0.0
    vs = sum((x - ms) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
    vb = sum((y - mb) ** 2 for y in b) / (n - 1) if n > 1 else 0.0
    beta = cov / vb if vb > 0 else 0.0
    return {
        "excess_return": strategy[-1] / starting_cash - bench[-1] / starting_cash,
        "beta": beta,
        "correlation": cov / math.sqrt(vs * vb) if vs > 0 and vb > 0 else 0.0,
        "alpha": (ms - beta * mb) * TRADING_DAYS,  # annualized, simple (no risk-free rate)
    }


def compute_metrics(equity: list[float], fills: list[dict], starting_cash: float) -> dict:
    """Performance metrics from a daily equity series (starting value = starting_cash) and fill records."""
    series = [starting_cash] + list(equity)
    rets = daily_returns(series)
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


def round_trips(fills: list[dict]) -> list[dict]:
    """Group fills into round trips: a trade opens when a position goes 0 -> long and closes when it is flat again.

    P&L includes commissions on both legs. Positions still open at the end are returned with exit "open".
    """
    open_: dict[str, dict] = {}
    trips: list[dict] = []
    for f in fills:
        s = f["symbol"]
        t = open_.setdefault(s, {"symbol": s, "entry": f["ts"], "qty": 0, "held": 0, "cost": 0.0, "proceeds": 0.0,
                                 "commission": 0.0, "exit": "open", "exit_reason": ""})
        t["commission"] += f["commission"]
        if f["side"] == "buy":
            t["held"] += f["qty"]
            t["qty"] += f["qty"]
            t["cost"] += f["qty"] * f["price"]
        else:
            t["held"] -= f["qty"]
            t["proceeds"] += f["qty"] * f["price"]
            if t["held"] <= 0:
                t["exit"], t["exit_reason"] = f["ts"], f.get("reason") or f.get("source") or ""
                trips.append(open_.pop(s))
    trips += open_.values()
    out = []
    for t in trips:
        pnl = t["proceeds"] - t["cost"] - t["commission"]
        closed = t["exit"] != "open"
        out.append({
            "symbol": t["symbol"], "entry": t["entry"], "exit": t["exit"], "qty": t["qty"],
            "avg_entry": t["cost"] / t["qty"] if t["qty"] else 0.0,
            "avg_exit": t["proceeds"] / (t["qty"] - t["held"]) if t["qty"] - t["held"] else 0.0,
            "pnl": pnl if closed else None,
            "return": pnl / t["cost"] if closed and t["cost"] else None,
            "days_held": (date.fromisoformat(t["exit"][:10]) - date.fromisoformat(t["entry"][:10])).days
            if closed else None,
            "exit_reason": t["exit_reason"][:80],
        })
    return out


def trade_stats(trips: list[dict]) -> dict:
    """Summary of closed round trips: win rate, average win/loss, profit factor, expectancy, holding time."""
    closed = [t for t in trips if t["pnl"] is not None]
    wins = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed if t["pnl"] <= 0]
    n = len(closed)
    return {
        "round_trips": n,
        "open_trades": len(trips) - n,
        "trade_win_rate": len(wins) / n if n else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) < 0 else float("inf") if wins else 0.0,
        "expectancy": sum(t["pnl"] for t in closed) / n if n else 0.0,
        "avg_trade_return": sum(t["return"] for t in closed) / n if n else 0.0,
        "avg_days_held": sum(t["days_held"] for t in closed) / n if n else 0.0,
        "best_trade": max((t["pnl"] for t in closed), default=0.0),
        "worst_trade": min((t["pnl"] for t in closed), default=0.0),
    }


def _pct(sorted_values: list[float], q: float) -> float:
    return sorted_values[round(q * (len(sorted_values) - 1))]


def monte_carlo(trips: list[dict], starting_cash: float, runs: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the closed round trips: draw the same number of trades with replacement, in random order, `runs`
    times, and report the spread of outcomes the realised sequence was one draw from.

    P&L is added in dollars to `starting_cash`, and drawdown is measured trade-to-trade (open-trade marks are not
    part of the resample), so it understates intra-trade drawdown. Trades are assumed independent: streaks and
    regime clustering in the real sequence are broken up.
    """
    pnls = [t["pnl"] for t in trips if t["pnl"] is not None]
    if not pnls or runs <= 0:
        return {}
    rng = random.Random(seed)
    finals, dds = [], []
    for _ in range(runs):
        eq = peak = starting_cash
        mdd = 0.0
        for x in rng.choices(pnls, k=len(pnls)):
            eq += x
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak if peak > 0 else 1.0)
        finals.append(eq / starting_cash - 1)
        dds.append(mdd)
    finals.sort()
    dds.sort()
    return {
        "mc_runs": runs,
        "mc_return_p5": _pct(finals, 0.05), "mc_return_p50": _pct(finals, 0.5), "mc_return_p95": _pct(finals, 0.95),
        "mc_prob_loss": sum(f < 0 for f in finals) / runs,
        "mc_max_drawdown_p50": _pct(dds, 0.5), "mc_max_drawdown_p95": _pct(dds, 0.95),
    }


def run_backtest(provider: DataProvider, symbols: list[str], strategy: Strategy,
                 risk_config: RiskConfig | None = None, cash: float = 100_000.0,
                 start: str | None = None, end: str | None = None, benchmark: str | None = None,
                 **broker_kwargs) -> BacktestResult:
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
    rc = risk.config
    gate = regime(history.get(rc.regime_symbol) or provider.history(rc.regime_symbol), rc.regime_sma) \
        if rc.regime_symbol else []
    gate_days = [d for d, _ in gate]
    pending = []
    for i in range(first, len(dates)):
        day = dates[i]
        opens = {s: bars[s][i].open for s in symbols}
        # 1) execute yesterday's decisions at today's open, sells first to free cash
        lb = risk.config.correlation_lookback  # correlations use closes up to yesterday only (no look-ahead)
        rets = {s: daily_returns([b.close for b in bars[s][max(0, i - lb - 1):i]]) for s in symbols} \
            if any(o.side == "buy" for o in pending) else {}
        # regime as of the last regime-symbol close strictly before today (the decision was made at that close)
        j = bisect_left(gate_days, day) - 1
        risk_off = (gate[j][1] if j >= 0 else "regime filter: no data before this day") if gate else ""
        for order in sorted(pending, key=lambda o: o.side != "sell"):
            state = broker.begin_day(day, opens)
            state.returns, state.risk_off = rets, risk_off
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
            state.highs = {s: bars[s][i].high for s in symbols}
            if risk.update(state):
                pending = risk.liquidation_orders(state)
                continue
            pending = risk.exit_orders(state)
            stopped = {o.symbol for o in pending}
            held = state.positions
            for s in symbols:
                if s in stopped:
                    continue
                f = features(bars[s][max(0, i + 1 - LOOKBACK):i + 1])
                if f is None:
                    continue
                order = rule_based_order(s, strategy.signal(f), strategy, held.get(s, 0), equity, risk)
                if order:
                    pending.append(order)

    positions = risk.levels(broker.begin_day(dates[-1], {s: bars[s][-1].close for s in symbols}))
    eq, fills = broker.rows("equity"), broker.rows("fills")
    metrics = compute_metrics([r["equity"] for r in eq], fills, cash)
    metrics.update(trade_stats(round_trips(fills)))
    metrics["strategy"] = strategy.name
    metrics["period"] = f"{dates[first]} .. {dates[-1]}"
    p0 = {s: bars[s][first].open for s in symbols}
    metrics["buy_hold_return"] = sum(bars[s][-1].close / p0[s] for s in symbols) / len(symbols) - 1
    bench_rows = None
    if benchmark:
        try:
            bench_bars = history.get(benchmark) or provider.history(benchmark)
        except DataUnavailable as e:
            metrics["benchmark"] = f"{benchmark} unavailable: {e}"
        else:
            days = [r["ts"] for r in eq]
            curve = buy_and_hold(bench_bars, days, cash)
            bm = compute_metrics(curve, [], cash)
            metrics["benchmark"] = f"{benchmark} buy-and-hold"
            for k in ("total_return", "cagr", "sharpe", "max_drawdown"):
                metrics[f"benchmark_{k}"] = bm[k]
            metrics.update(relative_metrics([r["equity"] for r in eq], curve, cash))
            bench_rows = [{"ts": d, "equity": v} for d, v in zip(days, curve)]
    return BacktestResult(eq, fills, broker.rows("journal"), metrics, bench_rows, positions)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
