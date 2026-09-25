"""Technical indicators, pure Python. Each returns a list aligned with the input; None during warm-up."""

from __future__ import annotations

import math
from typing import Optional, Sequence

Series = list[Optional[float]]


def sma(values: Sequence[float], period: int) -> Series:
    out: Series = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def ema(values: Sequence[float], period: int) -> Series:
    """EMA seeded with the SMA of the first `period` values."""
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    k = 2 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def _wilder(values: Sequence[float], period: int, start: int) -> Series:
    """Wilder smoothing of values[start:], seeded by a simple mean of the first `period`."""
    out: Series = [None] * len(values)
    if len(values) - start < period:
        return out
    prev = sum(values[start:start + period]) / period
    out[start + period - 1] = prev
    for i in range(start + period, len(values)):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


def rsi(closes: Sequence[float], period: int = 14) -> Series:
    """Wilder's RSI in [0, 100]."""
    gains = [0.0] + [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [0.0] + [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    ag, al = _wilder(gains, period, 1), _wilder(losses, period, 1)
    out: Series = [None] * len(closes)
    for i in range(len(closes)):
        if ag[i] is None:
            continue
        if al[i] == 0:
            out[i] = 100.0 if ag[i] > 0 else 50.0
        else:
            out[i] = 100 - 100 / (1 + ag[i] / al[i])
    return out


def macd(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Series, Series, Series]:
    """(macd line, signal line, histogram)."""
    ef, es = ema(closes, fast), ema(closes, slow)
    line: Series = [f - s if f is not None and s is not None else None for f, s in zip(ef, es)]
    first = next((i for i, v in enumerate(line) if v is not None), len(line))
    sig_tail = ema([v for v in line[first:]], signal) if first < len(line) else []
    sig: Series = [None] * first + sig_tail
    hist: Series = [m - s if m is not None and s is not None else None for m, s in zip(line, sig)]
    return line, sig, hist


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    tr = [highs[0] - lows[0]] if highs else []
    for i in range(1, len(highs)):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    return tr


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> Series:
    """Wilder's Average True Range."""
    return _wilder(true_range(highs, lows, closes), period, 0)


def bollinger(closes: Sequence[float], period: int = 20, k: float = 2.0) -> tuple[Series, Series, Series]:
    """(middle, upper, lower) using population standard deviation."""
    mid = sma(closes, period)
    up: Series = [None] * len(closes)
    lo: Series = [None] * len(closes)
    for i, m in enumerate(mid):
        if m is None:
            continue
        window = closes[i - period + 1:i + 1]
        sd = math.sqrt(sum((x - m) ** 2 for x in window) / period)
        up[i], lo[i] = m + k * sd, m - k * sd
    return mid, up, lo
