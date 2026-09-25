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
