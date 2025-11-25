import pytest

from finagent.broker import PaperBroker
from finagent.risk import Order, RiskDecision


def approved(symbol, side, qty):
    return RiskDecision(Order(symbol, side, qty), True, qty)


def test_buy_sell_slippage_commission_and_pnl():
    b = PaperBroker(starting_cash=10_000, slippage_bps=10, commission_per_share=0.01, min_commission=1.0)
    f = b.execute(approved("A", "buy", 50), 100.0, "d1")
    assert f.price == pytest.approx(100.1) and f.commission == 1.0
    assert b.cash == pytest.approx(10_000 - 50 * 100.1 - 1)
    f2 = b.execute(approved("A", "sell", 50), 110.0, "d2")
    assert f2.price == pytest.approx(109.89)
    assert f2.realized_pnl == pytest.approx((109.89 - 100.1) * 50 - 1)
    assert b.positions() == {}


def test_refuses_unapproved_orders():
    b = PaperBroker()
    with pytest.raises(PermissionError):
        b.execute(RiskDecision(Order("A", "buy", 10), False, 0), 100.0, "d1")


def test_refuses_overspend_and_oversell():
    b = PaperBroker(starting_cash=1_000)
    with pytest.raises(ValueError):
        b.execute(approved("A", "buy", 100), 100.0, "d1")
    with pytest.raises(ValueError):
        b.execute(approved("A", "sell", 1), 100.0, "d1")


def test_state_persists_across_reopen(tmp_path):
    db = tmp_path / "s.db"
    b = PaperBroker(db, starting_cash=5_000)
    b.execute(approved("A", "buy", 10), 100.0, "d1")
    b.mark("d1", {"A": 101.0})
    b.journal("d1", "A", "rules", "buy", 10, approved("A", "buy", 10), "test", "filled")
    cash = b.cash
    b2 = PaperBroker(db, starting_cash=999)  # starting cash ignored for existing db
    assert b2.cash == cash and b2.positions()["A"][0] == 10
    assert len(b2.rows("equity")) == 1 and b2.rows("journal")[0]["rationale"] == "test"


def test_begin_day_tracks_peak_and_day_start():
    b = PaperBroker(starting_cash=1_000)
    s = b.begin_day("d1", {})
    assert s.peak_equity == 1_000 and s.day_start_equity == 1_000
    b.execute(approved("A", "buy", 5), 100.0, "d1")
    s = b.begin_day("d1", {"A": 50.0})  # same day: day start unchanged
    assert s.day_start_equity == 1_000 and s.peak_equity == 1_000
    s = b.begin_day("d2", {"A": 300.0})
    assert s.day_start_equity == pytest.approx(s.equity) and s.peak_equity == pytest.approx(s.equity)
