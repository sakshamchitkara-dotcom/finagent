import pytest

from finagent.backtest import (buy_and_hold, compute_metrics, relative_metrics, round_trips, run_backtest,
                               trade_stats)
from finagent.data import Bar
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
    assert "<svg" in page and "Paper trading only" in page and "Round-trip trades" in page
    assert 'class="bm"' not in page
    assert "http://" not in page and "https://" not in page


def test_buy_and_hold_enters_at_first_open_and_carries_forward():
    bars = [Bar("2024-01-02", 10, 11, 9, 11, 0), Bar("2024-01-04", 12, 13, 11, 12.5, 0)]
    curve = buy_and_hold(bars, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], 100.0)
    assert curve == pytest.approx([100.0, 110.0, 110.0, 125.0])


def test_relative_metrics_against_itself_and_a_leveraged_copy():
    bench = [101.0, 99.0, 103.0, 102.0, 106.0]
    same = relative_metrics(bench, bench, 100.0)
    assert same["beta"] == pytest.approx(1) and same["correlation"] == pytest.approx(1)
    assert same["alpha"] == pytest.approx(0) and same["excess_return"] == pytest.approx(0)
    levered, v = [], 100.0
    for r in [bench[0] / 100 - 1] + [bench[i] / bench[i - 1] - 1 for i in range(1, len(bench))]:
        v *= 1 + 2 * r
        levered.append(v)
    assert relative_metrics(levered, bench, 100.0)["beta"] == pytest.approx(2)


def test_backtest_reports_benchmark():
    p = CSVProvider()
    r = run_backtest(p, ["SYN_TECH"], get_strategy("momentum"), end="2020-12-31", benchmark="SYN_INDEX")
    assert r.metrics["benchmark"] == "SYN_INDEX buy-and-hold"
    assert {"benchmark_total_return", "excess_return", "beta", "alpha"} <= set(r.metrics)
    assert [b["ts"] for b in r.benchmark] == [e["ts"] for e in r.equity]
    missing = run_backtest(p, ["SYN_TECH"], get_strategy("momentum"), end="2020-06-30", benchmark="NOPE")
    assert missing.benchmark is None and "unavailable" in missing.metrics["benchmark"]


def _fill(ts, sym, side, qty, price, reason=""):
    return {"ts": ts, "symbol": sym, "side": side, "qty": qty, "price": price, "commission": 1.0,
            "realized_pnl": 0.0, "source": "rules", "reason": reason}


def test_round_trips_scale_in_scale_out_and_open_positions():
    fills = [_fill("2024-01-02", "A", "buy", 10, 10.0), _fill("2024-01-03", "B", "buy", 5, 20.0),
             _fill("2024-01-05", "A", "buy", 10, 12.0), _fill("2024-01-08", "A", "sell", 5, 13.0),
             _fill("2024-01-10", "A", "sell", 15, 14.0, "trailing stop: hit"),
             _fill("2024-01-11", "A", "buy", 1, 15.0)]
    trips = round_trips(fills)
    a = trips[0]
    assert (a["symbol"], a["entry"], a["exit"], a["qty"], a["days_held"]) == ("A", "2024-01-02", "2024-01-10", 20, 8)
    assert a["avg_entry"] == pytest.approx(11.0) and a["avg_exit"] == pytest.approx(13.75)
    assert a["pnl"] == pytest.approx(275 - 220 - 4) and a["exit_reason"] == "trailing stop: hit"
    assert {(t["symbol"], t["exit"]) for t in trips[1:]} == {("B", "open"), ("A", "open")}
    st = trade_stats(trips)
    assert st["round_trips"] == 1 and st["open_trades"] == 2 and st["trade_win_rate"] == 1.0
    assert st["profit_factor"] == float("inf") and st["expectancy"] == pytest.approx(51)


def test_trade_stats_profit_factor():
    fills = [_fill("2024-01-02", "A", "buy", 10, 10.0), _fill("2024-01-03", "A", "sell", 10, 13.0),
             _fill("2024-01-04", "A", "buy", 10, 10.0), _fill("2024-01-05", "A", "sell", 10, 9.0)]
    st = trade_stats(round_trips(fills))
    assert st["avg_win"] == pytest.approx(28) and st["avg_loss"] == pytest.approx(-12)
    assert st["profit_factor"] == pytest.approx(28 / 12) and st["trade_win_rate"] == 0.5


def test_html_report_overlays_benchmark():
    p = CSVProvider()
    r = run_backtest(p, ["SYN_TECH"], get_strategy("momentum"), end="2020-06-30", benchmark="SYN_INDEX")
    page = render_html("t", r.equity, r.metrics, r.fills, benchmark=r.benchmark)
    assert page.count('class="bm"') == 1 and "SYN_INDEX buy-and-hold" in page and "Per-symbol trade summary" in page


def test_trailing_stop_exits_in_backtest():
    p = CSVProvider()
    r = run_backtest(p, p.symbols(), get_strategy("momentum"), RiskConfig(trailing_stop=0.05), end="2021-12-31")
    stops = [f for f in r.fills if f["reason"].startswith("trailing stop")]
    assert stops and all(f["side"] == "sell" and f["source"] == "risk" for f in stops)


def test_backtest_reports_open_positions_with_exit_levels():
    p = CSVProvider()
    r = run_backtest(p, ["SYN_TECH", "SYN_UTIL"], get_strategy("momentum"),
                     RiskConfig(stop_loss=0.1, trailing_stop=0.15), end="2021-06-30")
    for row in r.positions:
        assert row["stop_loss"] == pytest.approx(row["avg_entry"] * 0.9) and row["take_profit"] is None
        assert row["trailing_stop"] >= row["last"] * 0.85
    page = render_html("t", r.equity, r.metrics, r.fills, positions=r.positions)
    assert "Open positions and exit levels" in page


def test_monte_carlo_is_seeded_and_brackets_the_mean():
    from finagent.backtest import monte_carlo

    trips = [{"pnl": p} for p in [500.0, -200.0, 300.0, -100.0, 250.0, -400.0, 150.0, None]]  # None = open trade
    a = monte_carlo(trips, 10_000, runs=2000, seed=7)
    assert a == monte_carlo(trips, 10_000, runs=2000, seed=7) and a != monte_carlo(trips, 10_000, runs=2000, seed=8)
    assert a["mc_return_p5"] < 0.05 < a["mc_return_p95"]  # realised total: +500 on 10k = +5%
    assert a["mc_return_p50"] == pytest.approx(0.05, abs=0.03)
    assert 0 < a["mc_prob_loss"] < 0.5 and 0 < a["mc_max_drawdown_p50"] <= a["mc_max_drawdown_p95"]
    assert monte_carlo([{"pnl": None}], 10_000) == {} and monte_carlo(trips, 10_000, runs=0) == {}
    sure = monte_carlo([{"pnl": 10.0}] * 5, 1_000, runs=50)
    assert sure["mc_prob_loss"] == 0 and sure["mc_max_drawdown_p95"] == 0 and sure["mc_return_p5"] == pytest.approx(0.05)


def test_report_escapes_untrusted_text_and_handles_short_history():
    evil = '<script>alert(1)</script>'
    journal = [{"ts": "t", "symbol": "A", "source": "llm", "action": "buy", "requested_qty": 1, "approved_qty": 1,
                "outcome": "ok", "rationale": evil}]  # LLM rationale is model output: must never become markup
    fills = [_fill("2024-01-02", "A", "buy", 1, 10.0, evil), _fill("2024-01-03", "A", "sell", 1, 11.0, evil)]
    page = render_html(evil, [{"ts": "2024-01-02", "equity": 1.0}], {"note": evil}, fills, journal, note=evil)
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "Not enough equity history to chart yet" in page
