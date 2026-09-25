"""Parameter sweeps and walk-forward evaluation.

Parameters are always chosen on in-sample data only. Out-of-sample results are reported next to the
in-sample ones so that overfitting is visible rather than hidden in a single flattering number.
"""

from __future__ import annotations

import dataclasses
import itertools
from datetime import date, timedelta

from .backtest import buy_and_hold, compute_metrics, run_backtest
from .data import Bar, DataProvider, DataUnavailable
from .risk import RiskConfig, correlation
from .strategies import Strategy, get_strategy

STRATEGY_PARAMS = ("entry", "exit")
RISK_PARAMS = tuple(k for k, v in vars(RiskConfig()).items() if isinstance(v, (int, float)) and not isinstance(v, bool))
REPORTED = ("sharpe", "total_return", "max_drawdown", "trades")


class _Memo:
    """Fetch each symbol once per sweep; every backtest in the sweep reuses the bars."""

    def __init__(self, provider: DataProvider):
        self.provider, self.cache = provider, {}

    def history(self, symbol: str) -> list[Bar]:
        if symbol not in self.cache:
            self.cache[symbol] = self.provider.history(symbol)
        return self.cache[symbol]


def parse_grid(specs: list[str]) -> dict[str, list[float]]:
    """["entry=0.2,0.3", "trailing_stop=0,0.1"] -> {"entry": [0.2, 0.3], "trailing_stop": [0.0, 0.1]}."""
    grid = {}
    for spec in specs:
        name, _, values = spec.partition("=")
        name = name.strip().replace("-", "_")
        if name not in STRATEGY_PARAMS + RISK_PARAMS:
            raise ValueError(f"unknown parameter {name!r}; choose from {', '.join(STRATEGY_PARAMS + RISK_PARAMS)}")
        try:
            grid[name] = [float(v) for v in values.split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"bad grid spec {spec!r}; use name=v1,v2,...") from None
        if not grid[name]:
            raise ValueError(f"bad grid spec {spec!r}; use name=v1,v2,...")
    return grid


def combos(grid: dict[str, list[float]]) -> list[dict]:
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def configure(strategy: str, params: dict, base: RiskConfig) -> tuple[Strategy, RiskConfig]:
    strat, risk = get_strategy(strategy), {}
    for k, v in params.items():
        if k in STRATEGY_PARAMS:
            setattr(strat, k, v)
        else:
            risk[k] = int(v) if isinstance(getattr(base, k), int) else v
    return strat, dataclasses.replace(base, **risk)


def _pick(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": metrics[k] for k in REPORTED}


def _ranks(values: list[float]) -> list[float]:
    # ponytail: ties get arbitrary adjacent ranks rather than averaged ranks; fine for a diagnostic.
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = float(r)
    return ranks


def common_dates(provider: DataProvider, symbols: list[str], start: str | None = None,
                 end: str | None = None) -> list[str]:
    days = set.intersection(*(set(b.date for b in provider.history(s)) for s in symbols))
    return sorted(d for d in days if (not start or d >= start) and (not end or d <= end))


def sweep(provider: DataProvider, symbols: list[str], strategy: str, grid: dict[str, list[float]],
          split: str | None = None, start: str | None = None, end: str | None = None, base: RiskConfig | None = None,
          cash: float = 100_000.0) -> list[dict]:
    """Backtest every combination in-sample (start .. split-1) and out-of-sample (split .. end).

    `split` defaults to the date 70% of the way through the range. Rows are sorted by in-sample Sharpe,
    i.e. in the order you would have picked them without peeking at the out-of-sample period.
    """
    p, base = _Memo(provider), base or RiskConfig()
    if split is None:
        days = common_dates(p, symbols, start, end)
        if len(days) < 2:
            raise ValueError("not enough data to split in-sample / out-of-sample")
        split = days[int(len(days) * 0.7)]
    is_end = (date.fromisoformat(split) - timedelta(days=1)).isoformat()
    rows = []
    for params in combos(grid) or [{}]:
        strat, cfg = configure(strategy, params, base)
        ins = run_backtest(p, symbols, strat, cfg, cash, start=start, end=is_end).metrics
        oos = run_backtest(p, symbols, strat, cfg, cash, start=split, end=end).metrics
        rows.append({**params, "split": split, **_pick("is", ins), **_pick("oos", oos)})
    rows.sort(key=lambda r: r["is_sharpe"], reverse=True)
    return rows


def overfit_check(rows: list[dict]) -> dict:
    """Compare the in-sample winner with its out-of-sample result, and IS vs OOS rankings across the grid."""
    best = rows[0]
    oos_rank = 1 + sorted((r["oos_sharpe"] for r in rows), reverse=True).index(best["oos_sharpe"])
    rho = correlation(_ranks([r["is_sharpe"] for r in rows]), _ranks([r["oos_sharpe"] for r in rows]), min_obs=3)
    warnings = []
    if best["is_sharpe"] > 0 and best["oos_sharpe"] < 0.5 * best["is_sharpe"]:
        warnings.append(f"out-of-sample Sharpe {best['oos_sharpe']:.2f} is less than half the in-sample "
                        f"{best['is_sharpe']:.2f}: the chosen parameters are likely overfit")
    if len(rows) >= 3 and rho < 0:
        warnings.append(f"in-sample ranking does not carry over out-of-sample (rank correlation {rho:+.2f})")
    return {"best_is_sharpe": best["is_sharpe"], "best_oos_sharpe": best["oos_sharpe"],
            "oos_rank_of_is_best": f"{oos_rank}/{len(rows)}", "is_oos_rank_correlation": rho, "warnings": warnings}


def walk_forward(provider: DataProvider, symbols: list[str], strategy: str, grid: dict[str, list[float]],
                 train_days: int = 504, test_days: int = 126, start: str | None = None, end: str | None = None,
                 base: RiskConfig | None = None, cash: float = 100_000.0,
                 benchmark: str | None = None) -> tuple[list[dict], dict]:
    """Rolling walk-forward: pick the best in-sample combination on each `train_days` window, then trade it on the
    next `test_days` (never seen during selection). Test windows are stitched into one out-of-sample equity curve.

    Returns (per-fold rows, summary). Each test window starts flat with indicators warmed up on prior bars.
    """
    if train_days < 20 or test_days < 20:
        raise ValueError("train_days and test_days must be at least 20")
    p, base = _Memo(provider), base or RiskConfig()
    days = common_dates(p, symbols, start, end)
    folds, curve, oos_days = [], [], []
    for i in range(train_days, len(days) - 19, test_days):  # last window needs >= 20 days
        train, test = days[i - train_days:i], days[i:i + test_days]
        best, best_m = {}, None
        for params in combos(grid) or [{}]:
            strat, cfg = configure(strategy, params, base)
            m = run_backtest(p, symbols, strat, cfg, cash, start=train[0], end=train[-1]).metrics
            if best_m is None or m["sharpe"] > best_m["sharpe"]:
                best, best_m = params, m
        strat, cfg = configure(strategy, best, base)
        res = run_backtest(p, symbols, strat, cfg, cash, start=test[0], end=test[-1])
        scale = (curve[-1] if curve else cash) / cash  # compound the stitched curve across folds
        curve += [r["equity"] * scale for r in res.equity]
        oos_days += [r["ts"] for r in res.equity]
        folds.append({"fold": len(folds) + 1, "train": f"{train[0]}..{train[-1]}", "test": f"{test[0]}..{test[-1]}",
                      **best, **_pick("is", best_m), **_pick("oos", res.metrics)})
    if not folds:
        raise ValueError(f"need more than {train_days} + 20 common trading days for one walk-forward fold, "
                         f"have {len(days)}")
    oos = compute_metrics(curve, [], cash)
    n = len(folds)
    summary = {
        "folds": n,
        "oos_period": f"{oos_days[0]} .. {oos_days[-1]}",
        "mean_is_sharpe": sum(f["is_sharpe"] for f in folds) / n,
        "mean_oos_sharpe": sum(f["oos_sharpe"] for f in folds) / n,
        "oos_folds_profitable": sum(f["oos_total_return"] > 0 for f in folds) / n,
        **{f"oos_{k}": oos[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown")},
    }
    if benchmark:
        try:
            bench = buy_and_hold(p.history(benchmark), oos_days, cash)
            bm = compute_metrics(bench, [], cash)
            summary.update({"benchmark": benchmark, "benchmark_total_return": bm["total_return"],
                            "benchmark_sharpe": bm["sharpe"]})
        except DataUnavailable as e:
            summary["benchmark"] = f"{benchmark} unavailable: {e}"
    return folds, summary
