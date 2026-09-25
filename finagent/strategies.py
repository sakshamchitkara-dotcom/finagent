"""Pluggable signal strategies. A strategy maps recent bars -> Signal(score in [-1, 1], reason)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from . import indicators as ind
from .data import Bar
from .risk import Order

LOOKBACK = 250  # bars of history fed to indicators; bounds per-step cost in backtests
MIN_BARS = 60


@dataclass
class Signal:
    score: float  # +1 strong buy ... -1 strong sell
    reason: str
    features: dict = field(default_factory=dict)


def features(bars: Sequence[Bar]) -> Optional[dict]:
    """Latest indicator snapshot for a symbol, or None if there is not enough history."""
    bars = bars[-LOOKBACK:]
    if len(bars) < MIN_BARS:
        return None
    c = [b.close for b in bars]
    h = [b.high for b in bars]
    lo = [b.low for b in bars]
    line, sig, hist = ind.macd(c)
    mid, up, dn = ind.bollinger(c, 20, 2.0)
    width = up[-1] - dn[-1]
    return {
        "date": bars[-1].date,
        "close": c[-1],
        "sma20": ind.sma(c, 20)[-1],
        "sma50": ind.sma(c, 50)[-1],
        "ema20": ind.ema(c, 20)[-1],
        "rsi14": ind.rsi(c, 14)[-1],
        "macd": line[-1],
        "macd_signal": sig[-1],
        "macd_hist": hist[-1],
        "atr14": ind.atr(h, lo, c, 14)[-1],
        "bb_mid": mid[-1],
        "bb_upper": up[-1],
        "bb_lower": dn[-1],
        "pct_b": (c[-1] - dn[-1]) / width if width > 0 else 0.5,
        "ret_20d": c[-1] / c[-21] - 1,
    }


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


class Strategy(Protocol):
    name: str
    entry: float  # score >= entry opens/holds a long
    exit: float   # score <= exit closes the long

    def signal(self, f: dict) -> Signal: ...


class Momentum:
    """Trend following: SMA20/50 regime + MACD histogram + 20-day return."""
    name, entry, exit = "momentum", 0.35, -0.1

    def signal(self, f: dict) -> Signal:
        trend = 0.5 if f["sma20"] > f["sma50"] else -0.5
        macd = 0.3 * math.tanh(f["macd_hist"] / (0.1 * f["atr14"])) if f["atr14"] else 0.0
        ret = 0.2 * math.tanh(f["ret_20d"] * 10)
        score = _clip(trend + macd + ret)
        return Signal(score, f"momentum: sma20{'>' if trend > 0 else '<'}sma50, macd_hist={f['macd_hist']:+.3f}, "
                             f"ret20={f['ret_20d']:+.1%}", f)


class MeanReversion:
    """Fade stretched moves: oversold RSI and price near the lower Bollinger band score positive."""
    name, entry, exit = "mean_reversion", 0.45, -0.2

    def signal(self, f: dict) -> Signal:
        rsi_part = (50 - f["rsi14"]) / 25        # RSI 25 -> +1, RSI 75 -> -1
        band_part = (0.5 - f["pct_b"]) * 2       # lower band -> +1, upper band -> -1
        score = _clip(0.5 * rsi_part + 0.5 * band_part)
        return Signal(score, f"mean_reversion: rsi={f['rsi14']:.1f}, %b={f['pct_b']:.2f}", f)


class Combined:
    """Weighted blend of momentum and mean reversion."""
    name, entry, exit = "combined", 0.3, -0.15

    def __init__(self, w_momentum: float = 0.6, w_reversion: float = 0.4):
        self.parts = [(Momentum(), w_momentum), (MeanReversion(), w_reversion)]

    def signal(self, f: dict) -> Signal:
        sigs = [(s.signal(f), w) for s, w in self.parts]
        score = _clip(sum(s.score * w for s, w in sigs) / sum(w for _, w in sigs))
        return Signal(score, "combined[" + "; ".join(f"{s.reason} ({s.score:+.2f})" for s, _ in sigs) + "]", f)


STRATEGIES = {"momentum": Momentum, "mean_reversion": MeanReversion, "combined": Combined}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None


def rule_based_order(symbol: str, sig: Signal, strategy: Strategy, held: int, equity: float, risk) -> Optional[Order]:
    """Long-only policy: enter on score >= entry, exit fully on score <= exit. Sizing comes from the risk engine."""
    f = sig.features
    if held <= 0 and sig.score >= strategy.entry:
        qty = risk.size(equity, f["close"], f["atr14"])
        if qty > 0:
            return Order(symbol, "buy", qty, f"score {sig.score:+.2f} >= entry {strategy.entry}: {sig.reason}")
    elif held > 0 and sig.score <= strategy.exit:
        return Order(symbol, "sell", held, f"score {sig.score:+.2f} <= exit {strategy.exit}: {sig.reason}")
    return None
