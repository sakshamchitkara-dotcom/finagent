import pytest

from finagent.risk import Order, PortfolioState, RiskConfig, RiskEngine


def state(cash=100_000.0, positions=None, prices=None, peak=None, day_start=None):
    s = PortfolioState(cash, positions or {}, prices or {"A": 100.0, "B": 50.0}, 0, 0)
    s.peak_equity = peak if peak is not None else s.equity
    s.day_start_equity = day_start if day_start is not None else s.equity
    return s


def test_atr_and_fixed_sizing():
    r = RiskEngine(RiskConfig(risk_per_trade=0.01, atr_multiple=2))
    assert r.size(100_000, 100, atr=2.5) == 200  # 1000 risk / 5 per share
    r = RiskEngine(RiskConfig(sizing="fixed", fixed_fraction=0.1))
    assert r.size(100_000, 100, atr=2.5) == 100


def test_position_cap_clips_buy():
    d = RiskEngine().check(Order("A", "buy", 1000), state())
    assert d.approved and d.qty == 200  # 20% of 100k at $100
    assert any("max position" in c for c in d.checks)


def test_existing_position_counts_toward_cap():
    s = state(cash=85_000, positions={"A": 150})  # equity 100k, 15k in A
    d = RiskEngine().check(Order("A", "buy", 1000), s)
    assert d.qty == 50


def test_gross_exposure_cap():
    s = state(cash=10_000, positions={"B": 1800})  # 90k of B, equity 100k
    d = RiskEngine().check(Order("A", "buy", 100), s)
    assert d.approved and d.qty == 50  # 95k cap - 90k = 5k -> 50 shares


def test_cash_cap():
    s = state(cash=1_000, positions={"B": 1980}, prices={"A": 100.0, "B": 50.0})
    d = RiskEngine(RiskConfig(max_gross_exposure=5, max_position_pct=5)).check(Order("A", "buy", 100), s)
    assert d.qty == 9  # 1000 / (100 * 1.005)


def test_kill_switch_latches_and_allows_sells():
    r = RiskEngine(RiskConfig(max_drawdown=0.2))
    s = state(positions={"A": 100}, cash=65_000, peak=100_000)  # equity 75k -> 25% DD
    assert not r.check(Order("A", "buy", 1), s).approved
    assert r.killed
    recovered = state(cash=100_000, peak=100_000)
    assert not r.check(Order("A", "buy", 1), recovered).approved  # latched
    assert r.check(Order("A", "sell", 50), s).approved


def test_daily_loss_limit():
    s = state(day_start=104_000)  # equity 100k, down 3.85% on the day
    d = RiskEngine().check(Order("A", "buy", 10), s)
    assert not d.approved and "daily loss" in d.checks[-1]


def test_sell_clipped_and_no_shorting():
    r = RiskEngine()
    d = r.check(Order("A", "sell", 500), state(positions={"A": 100}, cash=90_000))
    assert d.approved and d.qty == 100
    assert not r.check(Order("B", "sell", 1), state()).approved


def test_invalid_orders_rejected():
    r = RiskEngine()
    assert not r.check(Order("A", "buy", 0), state()).approved
    assert not r.check(Order("A", "buy", 1.5), state()).approved
    assert not r.check(Order("ZZZ", "buy", 1), state()).approved


def test_liquidation_orders_after_kill():
    r = RiskEngine(RiskConfig(max_drawdown=0.1))
    s = state(positions={"A": 100, "B": 10}, cash=70_000, peak=100_000)
    assert r.update(s)
    orders = r.liquidation_orders(s)
    assert {(o.symbol, o.side, o.qty) for o in orders} == {("A", "sell", 100), ("B", "sell", 10)}
    assert all(r.check(o, s).approved for o in orders)


def test_trailing_stop_ratchets_and_fires():
    r = RiskEngine(RiskConfig(trailing_stop=0.10))
    assert r.exit_orders(state(positions={"A": 10}, prices={"A": 100.0})) == []
    assert r.exit_orders(state(positions={"A": 10}, prices={"A": 120.0})) == []  # new high 120
    assert r.exit_orders(state(positions={"A": 10}, prices={"A": 109.0})) == []  # -9.2%
    [o] = r.exit_orders(state(positions={"A": 10}, prices={"A": 108.0}))        # -10%
    assert (o.symbol, o.side, o.qty, o.source) == ("A", "sell", 10, "risk") and "trailing stop" in o.reason
    r.exit_orders(state(positions={}, prices={"A": 108.0}))
    assert r.stop_highs == {}  # forgotten once flat, so a re-entry starts a fresh high


def test_trailing_stop_off_by_default():
    r = RiskEngine()
    r.exit_orders(state(positions={"A": 10}, prices={"A": 100.0}))
    assert r.exit_orders(state(positions={"A": 10}, prices={"A": 1.0})) == []


def test_sector_cap_counts_every_name_in_the_sector():
    cfg = RiskConfig(sectors={"A": "tech", "B": "tech", "C": "tech"}, max_sector_pct=0.4)
    s = state(cash=70_000, positions={"B": 400, "C": 200}, prices={"A": 100.0, "B": 50.0, "C": 50.0})  # 30k tech
    d = RiskEngine(cfg).check(Order("A", "buy", 1000), s)
    assert d.approved and d.qty == 100 and any("sector 'tech'" in c for c in d.checks)  # 40k - 30k = 10k
    full = state(cash=60_000, positions={"B": 800}, prices={"A": 100.0, "B": 50.0})
    assert not RiskEngine(cfg).check(Order("A", "buy", 10), full).approved
    other = RiskEngine(RiskConfig(sectors={"B": "tech"})).check(Order("A", "buy", 10), full)
    assert other.approved  # unmapped symbols are not sector-capped


def test_default_sectors_group_index_etfs():
    from finagent.risk import DEFAULT_SECTORS

    assert DEFAULT_SECTORS["SPY"] == DEFAULT_SECTORS["QQQ"] and DEFAULT_SECTORS["AAPL"] == DEFAULT_SECTORS["MSFT"]


def test_correlation():
    from finagent.risk import correlation

    a = [0.01, -0.02, 0.03, 0.0, -0.01] * 5
    assert correlation(a, [2 * x for x in a]) == pytest.approx(1)
    assert correlation(a, [-x for x in a]) == pytest.approx(-1)
    assert correlation(a[:10], a[:10]) == 0.0  # too few observations


def test_correlated_exposure_cap():
    a = [0.01, -0.02, 0.03, 0.0, -0.01] * 6
    noise = [0.02, 0.01, -0.03, 0.02, -0.02, 0.0] * 5
    cfg = RiskConfig(sectors={}, max_correlated_pct=0.3)
    s = state(cash=80_000, positions={"B": 400}, prices={"A": 100.0, "B": 50.0})  # 20k in B
    s.returns = {"A": a, "B": [x * 1.5 for x in a]}
    d = RiskEngine(cfg).check(Order("A", "buy", 1000), s)
    assert d.qty == 100 and any("correlated with B (1.00)" in c for c in d.checks)  # 30k - 20k
    s.returns["B"] = noise
    assert RiskEngine(cfg).check(Order("A", "buy", 1000), s).qty == 200  # uncorrelated: only the 20% cap


def test_stop_loss_and_take_profit_use_average_entry():
    r = RiskEngine(RiskConfig(stop_loss=0.08, take_profit=0.25))

    def at(px):
        s = state(positions={"A": 10}, prices={"A": px})
        s.costs = {"A": 100.0}
        return r.exit_orders(s)

    assert at(93.0) == [] and at(124.0) == []
    [o] = at(92.0)
    assert (o.symbol, o.side, o.qty, o.source) == ("A", "sell", 10, "risk") and o.reason.startswith("stop loss")
    [o] = at(125.0)
    assert o.reason.startswith("take profit") and "25.0% above entry 100.00" in o.reason
    no_cost = state(positions={"A": 10}, prices={"A": 1.0})
    assert r.exit_orders(no_cost) == []  # unknown entry price: no stop/target


def test_one_exit_per_symbol_when_stops_overlap():
    r = RiskEngine(RiskConfig(stop_loss=0.05, trailing_stop=0.05))
    s = state(positions={"A": 10}, prices={"A": 120.0})
    s.costs = {"A": 120.0}
    r.exit_orders(s)
    s.prices["A"] = 100.0  # both the stop-loss and the trailing stop are hit
    [o] = r.exit_orders(s)
    assert o.reason.startswith("stop loss")


def test_backtest_honours_stop_loss_and_take_profit():
    from finagent.backtest import run_backtest
    from finagent.data import CSVProvider
    from finagent.strategies import get_strategy

    p = CSVProvider()
    r = run_backtest(p, p.symbols(), get_strategy("momentum"), RiskConfig(stop_loss=0.03, take_profit=0.05),
                     end="2021-12-31")
    reasons = {f["reason"].split(":")[0] for f in r.fills if f["source"] == "risk"}
    assert {"stop loss", "take profit"} <= reasons


def test_trailing_stop_can_ratchet_on_intraday_highs():
    by_close, by_high = RiskEngine(RiskConfig(trailing_stop=0.10)), RiskEngine(RiskConfig(trailing_stop=0.10,
                                                                                         stop_basis="high"))
    s = state(positions={"A": 10}, prices={"A": 110.0})
    s.highs = {"A": 115.0}  # spiked intraday, closed at 110
    assert by_close.exit_orders(s) == [] and by_high.exit_orders(s) == []
    assert by_close.stop_highs == {"A": 110.0} and by_high.stop_highs == {"A": 115.0}
    s.prices, s.highs = {"A": 103.0}, {"A": 104.0}  # 10.4% below the intraday high, 6.4% below the best close
    assert by_close.exit_orders(s) == []
    [o] = by_high.exit_orders(s)
    assert "below high 115.00" in o.reason
    assert by_high.levels(s)[0]["trailing_stop"] == pytest.approx(103.5)
