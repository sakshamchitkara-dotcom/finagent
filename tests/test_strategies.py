import pytest

from finagent.data import Bar, CSVProvider
from finagent.risk import RiskEngine
from finagent.strategies import (LOOKBACK, MIN_BARS, STRATEGIES, Combined, MeanReversion, Momentum, Signal, features,
                                 get_strategy, rule_based_order)


def _bars(closes):
    return [Bar(f"d{i:04d}", c, c * 1.01, c * 0.99, c, 1000) for i, c in enumerate(closes)]


def _f(**over):
    base = {"close": 100.0, "sma20": 101.0, "sma50": 100.0, "macd_hist": 0.0, "atr14": 2.0, "ret_20d": 0.0,
            "rsi14": 50.0, "pct_b": 0.5}
    return {**base, **over}


def test_features_need_min_bars_and_use_only_the_lookback_window():
    assert features(_bars([100.0] * (MIN_BARS - 1))) is None
    long = CSVProvider().history("SYN_TECH")
    assert features(long) == features(long[-LOOKBACK:])  # older bars never change the snapshot
    f = features(long)
    assert f["date"] == long[-1].date and f["close"] == long[-1].close and 0 <= f["rsi14"] <= 100


def test_momentum_follows_the_trend():
    up = Momentum().signal(_f(sma20=105, macd_hist=1.0, ret_20d=0.08))
    down = Momentum().signal(_f(sma20=95, macd_hist=-1.0, ret_20d=-0.08))
    assert up.score > Momentum.entry and down.score < Momentum.exit and -1 <= down.score <= up.score <= 1
    assert Momentum().signal(_f(atr14=0.0)).score == pytest.approx(0.5)  # no ATR: MACD term is dropped


def test_mean_reversion_fades_stretched_moves():
    assert MeanReversion().signal(_f(rsi14=25, pct_b=0.0)).score == pytest.approx(1.0)
    assert MeanReversion().signal(_f(rsi14=75, pct_b=1.0)).score == pytest.approx(-1.0)
    assert MeanReversion().signal(_f(rsi14=10, pct_b=-1.0)).score == 1.0  # clipped


def test_combined_is_the_weighted_blend():
    f = _f(sma20=105, rsi14=70, pct_b=0.9)
    m, r = Momentum().signal(f).score, MeanReversion().signal(f).score
    assert Combined(0.6, 0.4).signal(f).score == pytest.approx(0.6 * m + 0.4 * r)
    assert Combined(1.0, 0.0).signal(f).score == pytest.approx(m)


def test_rule_based_order_enters_holds_and_exits():
    s, risk, f = get_strategy("momentum"), RiskEngine(), _f()
    buy = rule_based_order("A", Signal(0.9, "x", f), s, 0, 100_000, risk)
    assert buy.side == "buy" and buy.qty == risk.size(100_000, 100.0, 2.0) > 0
    assert rule_based_order("A", Signal(0.9, "x", f), s, 10, 100_000, risk) is None  # already long: hold
    assert rule_based_order("A", Signal(0.0, "x", f), s, 10, 100_000, risk) is None  # between thresholds
    sell = rule_based_order("A", Signal(-0.5, "x", f), s, 10, 100_000, risk)
    assert (sell.side, sell.qty) == ("sell", 10)
    assert rule_based_order("A", Signal(-0.5, "x", f), s, 0, 100_000, risk) is None  # flat: no short


def test_get_strategy_returns_fresh_instances():
    assert set(STRATEGIES) >= {"momentum", "mean_reversion", "combined"}
    a = get_strategy("combined")
    a.entry = 0.99
    assert get_strategy("combined").entry == Combined.entry
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("nope")
