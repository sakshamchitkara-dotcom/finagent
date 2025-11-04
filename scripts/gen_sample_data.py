"""Generate the bundled SYNTHETIC sample OHLCV data (deterministic, seeded).

These are NOT real market prices. Tickers are prefixed SYN_ to make that obvious.
Model: geometric Brownian motion with regime switching (trend / chop / selloff) so
both momentum and mean-reversion strategies have something to find.

    python scripts/gen_sample_data.py            # writes data/sample/*.csv
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
START = date(2019, 1, 1)
DAYS = 1260  # ~5 trading years
OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

# symbol: (start price, annual drift, annual vol)
UNIVERSE = {
    "SYN_TECH": (100.0, 0.14, 0.32),
    "SYN_BANK": (50.0, 0.06, 0.24),
    "SYN_ENERGY": (80.0, 0.03, 0.30),
    "SYN_UTIL": (40.0, 0.04, 0.14),
    "SYN_INDEX": (300.0, 0.08, 0.17),
}

REGIMES = {"trend": (1.5, 0.8), "chop": (0.0, 1.0), "selloff": (-3.0, 1.5)}  # (drift mult, vol mult)


def business_days(start: date, n: int):
    d = start
    while n:
        if d.weekday() < 5:
            yield d
            n -= 1
        d += timedelta(days=1)


def generate(symbol: str, p0: float, mu: float, sigma: float, rng: random.Random) -> list[str]:
    rows = ["date,open,high,low,close,volume"]
    price, regime, dt = p0, "trend", 1 / 252
    for d in business_days(START, DAYS):
        if rng.random() < 0.02:  # ~every 50 days the regime may change
            regime = rng.choices(list(REGIMES), weights=[0.55, 0.38, 0.07])[0]
        dm, vm = REGIMES[regime]
        s = sigma * vm
        ret = (mu * dm - 0.5 * s * s) * dt + s * math.sqrt(dt) * rng.gauss(0, 1)
        gap = s * math.sqrt(dt) * rng.gauss(0, 0.3)
        open_ = price * math.exp(gap)
        close = price * math.exp(ret)
        span = abs(s * math.sqrt(dt) * rng.gauss(0, 0.6))
        high = max(open_, close) * (1 + span)
        low = min(open_, close) * (1 - span)
        vol = int(1_000_000 * (1 + abs(ret) * 40) * rng.uniform(0.7, 1.3))
        rows.append(f"{d.isoformat()},{open_:.4f},{high:.4f},{low:.4f},{close:.4f},{vol}")
        price = close
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for sym, (p0, mu, sigma) in UNIVERSE.items():
        (OUT / f"{sym}.csv").write_text("\n".join(generate(sym, p0, mu, sigma, rng)) + "\n")
        print(f"wrote {OUT / (sym + '.csv')}")


if __name__ == "__main__":
    main()
