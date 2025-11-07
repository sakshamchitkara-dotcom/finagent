import pytest

from finagent import indicators as ind


def test_sma_simple():
    assert ind.sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_ema_seeded_by_sma():
    out = ind.ema([1, 2, 3, 4, 5], 3)
    # seed = mean(1,2,3)=2; k=0.5 -> 3.0, 4.0
    assert out == [None, None, 2.0, 3.0, 4.0]


def test_rsi_extremes_and_bounds():
    up = list(range(1, 40))
    assert ind.rsi(up, 14)[-1] == 100.0
    down = list(range(40, 1, -1))
    assert ind.rsi(down, 14)[-1] == pytest.approx(0.0)
    flat = [10.0] * 30
    assert ind.rsi(flat, 14)[-1] == 50.0
    zig = [10 + (i % 2) for i in range(60)]
    assert all(0 <= v <= 100 for v in ind.rsi(zig, 14) if v is not None)


def test_rsi_known_value():
    # Classic Wilder example (Wilder 1978 / StockCharts table), first RSI ~70.46
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
              45.89, 46.03, 45.61, 46.28, 46.28]
    assert ind.rsi(closes, 14)[-1] == pytest.approx(70.46, abs=0.05)


def test_macd_constant_series_is_zero():
    line, sig, hist = ind.macd([5.0] * 60)
    assert line[-1] == pytest.approx(0) and sig[-1] == pytest.approx(0) and hist[-1] == pytest.approx(0)
    assert sig[25 + 7] is None and sig[25 + 8] is not None  # signal warm-up after the slow EMA


def test_atr_constant_range():
    h, l, c = [11.0] * 20, [9.0] * 20, [10.0] * 20
    assert ind.atr(h, l, c, 14)[-1] == pytest.approx(2.0)
    assert ind.atr(h, l, c, 14)[12] is None


def test_bollinger_symmetry():
    closes = [1, 2, 3, 4, 5] * 6
    mid, up, lo = ind.bollinger(closes, 5, 2)
    assert mid[-1] == pytest.approx(3.0)
    assert up[-1] - mid[-1] == pytest.approx(mid[-1] - lo[-1])
    assert up[-1] == pytest.approx(3 + 2 * 2 ** 0.5)
