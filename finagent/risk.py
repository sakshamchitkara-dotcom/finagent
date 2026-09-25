"""Deterministic risk engine. Every order - rule-based or LLM-proposed - must pass `RiskEngine.check`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    reason: str = ""
    source: str = "rules"  # "rules" | "llm"


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, int]
    prices: dict[str, float]
    peak_equity: float
    day_start_equity: float

    @property
    def gross_exposure(self) -> float:
        return sum(abs(q) * self.prices[s] for s, q in self.positions.items() if q)

    @property
    def equity(self) -> float:
        return self.cash + sum(q * self.prices[s] for s, q in self.positions.items() if q)


@dataclass
class RiskConfig:
    sizing: Literal["atr", "fixed"] = "atr"
    risk_per_trade: float = 0.01      # atr sizing: lose at most 1% of equity if price moves atr_multiple*ATR
    atr_multiple: float = 2.0
    fixed_fraction: float = 0.10      # fixed sizing: 10% of equity per entry
    max_position_pct: float = 0.20    # single-name cap, fraction of equity
    max_gross_exposure: float = 0.95  # total long exposure cap, fraction of equity
    max_drawdown: float = 0.20        # peak-to-trough drawdown that trips the kill switch
    daily_loss_limit: float = 0.03    # stop opening risk after a 3% down day
    cost_buffer: float = 0.005        # headroom for slippage + commission when checking cash


@dataclass(frozen=True)
class RiskDecision:
    order: Order
    approved: bool
    qty: int  # approved quantity (may be reduced from the requested quantity)
    checks: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {"approved": self.approved, "requested": self.order.qty, "approved_qty": self.qty,
                "checks": list(self.checks)}


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.killed = False  # latched once max drawdown is breached

    def size(self, equity: float, price: float, atr: float | None) -> int:
        """Shares to buy for a new entry, before caps."""
        c = self.config
        if price <= 0 or equity <= 0:
            return 0
        if c.sizing == "atr" and atr and atr > 0:
            return int(equity * c.risk_per_trade / (c.atr_multiple * atr))
        return int(equity * c.fixed_fraction / price)

    def check(self, order: Order, state: PortfolioState) -> RiskDecision:
        c, checks = self.config, []
        price = state.prices.get(order.symbol)
        held = state.positions.get(order.symbol, 0)

        def reject(why: str) -> RiskDecision:
            return RiskDecision(order, False, 0, tuple(checks + [f"REJECT: {why}"]))

        if order.qty <= 0 or not isinstance(order.qty, int):
            return reject(f"quantity must be a positive integer, got {order.qty!r}")
        if price is None or not math.isfinite(price) or price <= 0:
            return reject(f"no valid price for {order.symbol}")

        if order.side == "sell":
            # Risk-reducing: always allowed, even when halted. No short selling.
            if held <= 0:
                return reject("no position to sell (shorting disabled)")
            qty = min(order.qty, held)
            if qty < order.qty:
                checks.append(f"clipped sell {order.qty}->{qty} to held quantity")
            checks.append("ok: sell reduces risk")
            return RiskDecision(order, True, qty, tuple(checks))

        if order.side != "buy":
            return reject(f"unknown side {order.side!r}")

        equity = state.equity
        dd = 1 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
        if dd >= c.max_drawdown:
            self.killed = True
        if self.killed:
            return reject(f"kill switch engaged (drawdown {dd:.1%}, limit {c.max_drawdown:.0%})")
        checks.append(f"ok: drawdown {dd:.1%} < {c.max_drawdown:.0%}")

        day = equity / state.day_start_equity - 1 if state.day_start_equity > 0 else 0.0
        if day <= -c.daily_loss_limit:
            return reject(f"daily loss limit hit ({day:.1%}, limit -{c.daily_loss_limit:.0%})")
        checks.append(f"ok: day P&L {day:+.1%}")

        qty = order.qty
        caps = {
            "max position": (c.max_position_pct * equity - held * price) / price,
            "gross exposure": (c.max_gross_exposure * equity - state.gross_exposure) / price,
            "available cash": state.cash / (price * (1 + c.cost_buffer)),
        }
        for name, cap in caps.items():
            cap = max(0, math.floor(cap))
            if qty > cap:
                checks.append(f"clipped buy {qty}->{cap} by {name}")
                qty = cap
        if qty <= 0:
            return reject("no capacity left after position/exposure/cash caps")
        return RiskDecision(order, True, qty, tuple(checks))
