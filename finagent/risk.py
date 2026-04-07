"""Deterministic risk engine. Every order - rule-based or LLM-proposed - must pass `RiskEngine.check`."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# Symbol -> sector for the sector exposure cap. Symbols not listed are not sector-capped.
# Broad index ETFs share one bucket so the agent cannot stack SPY + QQQ + VOO as "diversification".
DEFAULT_SECTORS = {
    "SPY": "index", "VOO": "index", "IVV": "index", "QQQ": "index", "DIA": "index", "IWM": "index", "VTI": "index",
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "GOOG": "tech", "META": "tech", "AVGO": "tech",
    "AMD": "tech", "ORCL": "tech", "CRM": "tech", "INTC": "tech", "XLK": "tech",
    "AMZN": "consumer", "TSLA": "consumer", "HD": "consumer", "MCD": "consumer", "NKE": "consumer",
    "WMT": "staples", "PG": "staples", "KO": "staples", "PEP": "staples", "COST": "staples",
    "JPM": "financials", "BAC": "financials", "WFC": "financials", "GS": "financials", "MS": "financials",
    "V": "financials", "MA": "financials", "BRK-B": "financials", "XLF": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "XLE": "energy",
    "JNJ": "health", "UNH": "health", "LLY": "health", "PFE": "health", "MRK": "health", "ABBV": "health",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities", "XLU": "utilities",
    "SYN_TECH": "tech", "SYN_BANK": "financials", "SYN_ENERGY": "energy", "SYN_UTIL": "utilities",
    "SYN_INDEX": "index",
}


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
    returns: dict[str, list[float]] = field(default_factory=dict)  # recent daily returns, for correlation checks

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
    trailing_stop: float = 0.0        # exit a long after it falls this fraction from its high close; 0 = off
    max_sector_pct: float = 0.40      # total long exposure per sector, fraction of equity
    max_correlated_pct: float = 0.40  # order symbol + held names correlated >= threshold, fraction of equity
    correlation_threshold: float = 0.70
    correlation_lookback: int = 60    # daily returns used for the correlation estimate
    sectors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SECTORS))


def correlation(a: list[float], b: list[float], min_obs: int = 20) -> float:
    """Pearson correlation of the most recent overlapping observations; 0.0 with too little data."""
    n = min(len(a), len(b))
    if n < min_obs:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0


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
        self.stop_highs: dict[str, float] = {}  # highest close seen per open position (trailing stops)

    def size(self, equity: float, price: float, atr: float | None) -> int:
        """Shares to buy for a new entry, before caps."""
        c = self.config
        if price <= 0 or equity <= 0:
            return 0
        if c.sizing == "atr" and atr and atr > 0:
            return int(equity * c.risk_per_trade / (c.atr_multiple * atr))
        return int(equity * c.fixed_fraction / price)

    def drawdown(self, state: PortfolioState) -> float:
        return 1 - state.equity / state.peak_equity if state.peak_equity > 0 else 0.0

    def update(self, state: PortfolioState) -> bool:
        """Latch the kill switch if drawdown breached the limit. Returns True when trading is halted."""
        if self.drawdown(state) >= self.config.max_drawdown:
            self.killed = True
        return self.killed

    def liquidation_orders(self, state: PortfolioState) -> list[Order]:
        """Once halted, the policy is to flatten everything."""
        return [Order(s, "sell", q, "kill switch: flatten position", "risk") for s, q in state.positions.items() if q > 0]

    def trailing_stop_orders(self, state: PortfolioState) -> list[Order]:
        """Ratchet each long's high-water close and return full exits for positions that fell `trailing_stop` from it.

        Call once per bar at the close. Highs of positions that are no longer held are forgotten.
        """
        # ponytail: tracks closes, not intraday highs/lows; an intraday stop model needs bar high/low per tick.
        held = {s: q for s, q in state.positions.items() if q > 0}
        self.stop_highs = {s: h for s, h in self.stop_highs.items() if s in held}
        pct = self.config.trailing_stop
        if pct <= 0:
            return []
        out = []
        for s, q in held.items():
            px = state.prices.get(s)
            if px is None:
                continue
            hi = self.stop_highs[s] = max(self.stop_highs.get(s, px), px)
            if px <= hi * (1 - pct):
                out.append(Order(s, "sell", q, f"trailing stop: close {px:.2f} is {1 - px / hi:.1%} below "
                                               f"high {hi:.2f} (limit {pct:.0%})", "risk"))
        return out

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
        dd = self.drawdown(state)
        if self.update(state):
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
        sector = c.sectors.get(order.symbol)
        if sector:
            in_sector = sum(q * state.prices[s] for s, q in state.positions.items()
                            if q > 0 and c.sectors.get(s) == sector)
            caps[f"sector '{sector}' exposure"] = (c.max_sector_pct * equity - in_sector) / price
        mine = state.returns.get(order.symbol)
        if mine:
            peers = {s: correlation(mine, state.returns[s]) for s, q in state.positions.items()
                     if q > 0 and s != order.symbol and s in state.returns}
            peers = {s: r for s, r in peers.items() if r >= c.correlation_threshold}
            if peers:
                checks.append("correlated with " + ", ".join(f"{s} ({r:.2f})" for s, r in sorted(peers.items())))
                cluster = held * price + sum(state.positions[s] * state.prices[s] for s in peers)
                caps["correlated exposure"] = (c.max_correlated_pct * equity - cluster) / price
        for name, cap in caps.items():
            cap = max(0, math.floor(cap))
            if qty > cap:
                checks.append(f"clipped buy {qty}->{cap} by {name}")
                qty = cap
        if qty <= 0:
            return reject("no capacity left after position/exposure/cash caps")
        return RiskDecision(order, True, qty, tuple(checks))
