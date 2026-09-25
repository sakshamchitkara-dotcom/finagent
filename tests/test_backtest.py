import pytest

from finagent.backtest import compute_metrics, run_backtest
from finagent.data import CSVProvider
from finagent.report import render_html
from finagent.risk import RiskConfig
from finagent.strategies import get_strategy


def test_metrics_on_known_series():
    m = compute_metrics([110.0, 99.0, 121.0], [], 100.0)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["max_drawdown"] == pytest.approx(0.1)
    assert m["win_rate"] == 0.0 and m["turnover"] == 0.0


def test_metrics_win_rate_and_turnover():
    fills = [
        {"side": "buy", "qty": 10, "price": 10.0, "commission": 1.0, "realized_pnl": 0.0},
        {"side": "sell", "qty": 10, "price": 12.0, "commission": 1.0, "realized_pnl": 19.0},
        {"side": "sell", "qty": 5, "price": 8.0, "commission": 1.0, "realized_pnl": -11.0},
    ]
    m = compute_metrics([100.0] * 252, fills, 100.0)
    assert m["win_rate"] == 0.5 and m["closed_trades"] == 2
    assert m["turnover"] == pytest.approx((100 + 120 + 40) / 100)


def test_backtest_deterministic_and_fills_at_next_open():
    p = CSVProvider()
    syms = ["SYN_TECH", "SYN_UTIL"]
    a = run_backtest(p, syms, get_strategy("combined"), start="2020-01-01", end="2021-06-30")
    b = run_backtest(p, syms, get_strategy("combined"), start="2020-01-01", end="2021-06-30")
    assert a.metrics == b.metrics and a.metrics["trades"] > 0
    opens = {(s, bar.date): bar.open for s in syms for bar in p.history(s)}
    for f in a.fills:  # every fill is the fill-day OPEN +/- 5bps slippage -> decided the previous close
        assert f["price"] == pytest.approx(opens[(f["symbol"], f["ts"])], rel=6e-4)
    assert a.equity[0]["ts"] >= "2020-01-01" and a.equity[-1]["ts"] <= "2021-06-30"


def test_every_executed_order_was_risk_approved():
    p = CSVProvider()
    r = run_backtest(p, p.symbols(), get_strategy("mean_reversion"), end="2020-12-31")
    filled = [j for j in r.journal if j["outcome"].startswith("filled")]
    assert len(filled) == len(r.fills) and all(j["approved"] for j in filled)


def test_kill_switch_limits_drawdown_in_backtest():
    p = CSVProvider()
    cfg = RiskConfig(max_drawdown=0.05, sizing="fixed", fixed_fraction=0.2)
    r = run_backtest(p, p.symbols(), get_strategy("momentum"), risk_config=cfg)
    assert r.metrics["max_drawdown"] < 0.12  # halts and flattens soon after -5%
    assert any(f["source"] == "risk" for f in r.fills)


def test_html_report_is_self_contained():
    p = CSVProvider()
    r = run_backtest(p, ["SYN_INDEX"], get_strategy("momentum"), end="2020-06-30")
    page = render_html("t", r.equity, r.metrics, r.fills, r.journal)
    assert "<svg" in page and "Paper trading only" in page
    assert "http://" not in page and "https://" not in page
