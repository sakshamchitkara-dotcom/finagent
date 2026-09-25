"""Autonomous loop: observe -> analyze -> decide -> risk-check -> execute (paper) -> journal."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from .broker import PaperBroker
from .data import DataProvider, DataUnavailable
from .llm import AnalystError, Proposal
from .risk import Order, RiskEngine
from .strategies import Strategy, features, rule_based_order


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def proposal_to_order(p: Proposal, held: int, price: float, equity: float) -> Order | None:
    """Turn a target weight into a share delta. Direction must agree with the proposed action."""
    target = math.floor(p.target_weight * equity / price) if price > 0 else 0
    delta = target - held
    why = f"LLM {p.action} -> {p.target_weight:.0%} (conf {p.confidence:.2f}): {p.rationale}"
    if p.action == "buy" and delta > 0:
        return Order(p.symbol, "buy", delta, why, "llm")
    if p.action == "sell" and delta < 0:
        return Order(p.symbol, "sell", -delta, why, "llm")
    return None


class Agent:
    def __init__(self, provider: DataProvider, broker: PaperBroker, symbols: list[str], strategy: Strategy,
                 risk: RiskEngine | None = None, analyst=None, log=print):
        self.provider, self.broker, self.symbols, self.strategy = provider, broker, symbols, strategy
        self.risk = risk or RiskEngine()
        self.risk.killed = bool(broker.get_meta("killed"))  # the kill switch survives restarts
        self.risk.stop_highs = broker.get_meta("stop_highs") or {}  # so do trailing-stop high-water marks
        self.analyst, self.log = analyst, log

    def tick(self) -> dict:
        now, b = _now(), self.broker

        # observe
        feats: dict[str, dict] = {}
        rets: dict[str, list[float]] = {}
        lb = self.risk.config.correlation_lookback
        for s in self.symbols:
            try:
                hist = self.provider.history(s)
                closes = [bar.close for bar in hist[-lb - 1:]]
                rets[s] = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
                f = features(hist)
            except DataUnavailable as e:
                b.journal(now, s, "observe", "skip", 0, None, str(e), "no data")
                continue
            if f is None:
                b.journal(now, s, "observe", "skip", 0, None, "not enough history for indicators", "no data")
                continue
            feats[s] = f
        if not feats:
            raise DataUnavailable("no symbol returned usable data")
        as_of = max(f["date"] for f in feats.values())
        prices = {s: f["close"] for s, f in feats.items()}
        for s, (q, _) in b.positions().items():  # held names without fresh data still need a mark
            if s not in prices:
                raise DataUnavailable(f"no price for held position {s}; refusing to trade on a partial view")

        # Same bar and same prices as the last completed tick: nothing new to decide on. Re-running would
        # re-journal identical decisions and, for LLM mode, pay for the same answer twice.
        seen = {"as_of": as_of, "prices": prices}
        if b.get_meta("last_bar") == seen:
            b.journal(now, "*", "observe", "skip", 0, None, f"no new data since the last tick (bar {as_of})",
                      "no new bar")
            summary = {"ts": now, "as_of": as_of, "mode": "no new bar", "orders": [],
                       "equity": round(b.equity(prices), 2), "cash": round(b.cash, 2), "killed": self.risk.killed}
            self.log(summary)
            return summary

        # analyze
        signals = {s: self.strategy.signal(f) for s, f in feats.items()}
        state = b.begin_day(as_of, prices)
        equity = state.equity

        # decide
        mode, orders, notes = "rules", [], {}
        stops = self.risk.exit_orders(state)
        if self.risk.update(state):
            mode, orders = "kill-switch", self.risk.liquidation_orders(state)
        elif self.analyst is not None:
            try:
                proposals, view = self.analyst.propose(self._snapshot(as_of, feats, signals, state))
                mode = "llm"
                b.journal(now, "*", "llm", "market_view", 0, None, view, "noted")
                for p in proposals:
                    o = proposal_to_order(p, state.positions.get(p.symbol, 0), prices[p.symbol], equity)
                    if o:
                        orders.append(o)
                    else:
                        notes[p.symbol] = f"LLM {p.action} {p.target_weight:.0%} (conf {p.confidence:.2f}): {p.rationale}"
            except AnalystError as e:
                mode = "rules (llm fallback)"
                b.journal(now, "*", "llm", "error", 0, None, str(e), "fell back to rule-based policy")
        if mode.startswith("rules"):
            for s, sig in signals.items():
                o = rule_based_order(s, sig, self.strategy, state.positions.get(s, 0), equity, self.risk)
                if o:
                    orders.append(o)
                else:
                    notes[s] = f"{self.strategy.name} score {sig.score:+.2f} (entry {self.strategy.entry}, " \
                               f"exit {self.strategy.exit}): {sig.reason}"

        if mode != "kill-switch" and stops:  # a stop exit overrides whatever was decided for that symbol
            stopped = {o.symbol for o in stops}
            orders = stops + [o for o in orders if o.symbol not in stopped]

        # risk-check + execute, sells first to free cash
        results = []
        for o in sorted(orders, key=lambda o: o.side != "sell"):
            st = b.begin_day(as_of, prices)
            st.returns = rets
            d = self.risk.check(o, st)
            outcome = "rejected: " + d.checks[-1] if not d.approved else ""
            if d.approved:
                f = b.execute(d, prices[o.symbol], now)
                outcome = f"filled {f.side} {f.qty} @ {f.price:.2f} (paper)"
            b.journal(now, o.symbol, o.source, o.side, o.qty, d, o.reason, outcome)
            results.append((o.symbol, o.side, o.qty, d.qty if d.approved else 0, outcome))
        ordered = {o.symbol for o in orders}
        for s, why in notes.items():
            if s not in ordered:
                b.journal(now, s, "llm" if mode == "llm" else "rules", "hold", 0, None, why, "no order")

        # journal the book
        b.set_meta("killed", self.risk.killed)
        held = b.positions()
        b.set_meta("stop_highs", {s: h for s, h in self.risk.stop_highs.items() if s in held})
        b.set_meta("last_bar", seen)
        final_equity = b.mark(as_of, prices)
        summary = {"ts": now, "as_of": as_of, "mode": mode, "orders": results,
                   "equity": round(final_equity, 2), "cash": round(b.cash, 2), "killed": self.risk.killed}
        self.log(summary)
        return summary

    def _snapshot(self, as_of, feats, signals, state) -> dict:
        eq = state.equity
        return {
            "as_of": as_of,
            "note": "Paper trading. Prices may be synthetic sample data.",
            "portfolio": {
                "cash": round(state.cash, 2), "equity": round(eq, 2),
                "positions": {s: {"qty": q, "weight": round(q * state.prices[s] / eq, 4)}
                              for s, q in state.positions.items() if q},
            },
            "risk_limits": {k: v for k, v in vars(self.risk.config).items() if k != "sectors"},
            "symbols": {s: {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()},
                            "strategy": self.strategy.name, "strategy_score": round(signals[s].score, 3),
                            "sector": self.risk.config.sectors.get(s)}
                        for s, f in feats.items()},
        }

    def run(self, interval: float, max_ticks: int | None = None) -> None:
        n = 0
        while max_ticks is None or n < max_ticks:
            try:
                self.tick()
            except DataUnavailable as e:
                self.log({"ts": _now(), "error": str(e)})
            n += 1
            if max_ticks is None or n < max_ticks:
                time.sleep(interval)
