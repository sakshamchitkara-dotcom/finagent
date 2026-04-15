import pytest

from finagent import optimize
from finagent.data import CSVProvider
from finagent.risk import RiskConfig


def test_parse_grid_and_combos():
    grid = optimize.parse_grid(["entry=0.2,0.3", "trailing-stop=0,0.1", "correlation_lookback=40"])
    assert grid == {"entry": [0.2, 0.3], "trailing_stop": [0.0, 0.1], "correlation_lookback": [40.0]}
    assert len(optimize.combos(grid)) == 4
    for bad in ["nope=1", "entry=", "entry=a,b", "sizing=1"]:
        with pytest.raises(ValueError):
            optimize.parse_grid([bad])


def test_configure_routes_params_and_keeps_int_fields_int():
    strat, cfg = optimize.configure("momentum", {"entry": 0.5, "trailing_stop": 0.1, "correlation_lookback": 40.0},
                                    RiskConfig(sizing="fixed"))
    assert strat.entry == 0.5 and cfg.trailing_stop == 0.1 and cfg.sizing == "fixed"
    assert cfg.correlation_lookback == 40 and isinstance(cfg.correlation_lookback, int)
    assert optimize.get_strategy("momentum").entry == 0.35  # class default untouched


class _Counting(CSVProvider):
    calls = 0

    def history(self, symbol):
        type(self).calls += 1
        return super().history(symbol)


def test_sweep_separates_in_and_out_of_sample_and_sorts_by_in_sample():
    p = _Counting()
    rows = optimize.sweep(p, ["SYN_TECH"], "momentum", {"entry": [0.2, 0.5]}, start="2020-01-01",
                          end="2021-12-31")
    assert len(rows) == 2 and rows[0]["split"] == rows[1]["split"] > "2021-01-01"
    assert rows[0]["is_sharpe"] >= rows[1]["is_sharpe"]
    assert _Counting.calls == 1  # bars fetched once for the whole sweep
    check = optimize.overfit_check(rows)
    assert check["oos_rank_of_is_best"] in {"1/2", "2/2"}


def test_overfit_check_flags_degradation_and_rank_inversion():
    rows = [{"is_sharpe": 2.0, "oos_sharpe": 0.1}, {"is_sharpe": 1.0, "oos_sharpe": 0.5},
            {"is_sharpe": 0.5, "oos_sharpe": 0.9}]
    check = optimize.overfit_check(rows)
    assert check["oos_rank_of_is_best"] == "3/3" and check["is_oos_rank_correlation"] == pytest.approx(-1)
    assert len(check["warnings"]) == 2
    fine = optimize.overfit_check([{"is_sharpe": 1.0, "oos_sharpe": 0.9}, {"is_sharpe": 0.5, "oos_sharpe": 0.4},
                                   {"is_sharpe": 0.1, "oos_sharpe": 0.0}])
    assert fine["warnings"] == []


def test_walk_forward_folds_never_overlap_their_training_window():
    folds, summary = optimize.walk_forward(CSVProvider(), ["SYN_TECH", "SYN_UTIL"], "momentum",
                                           {"entry": [0.2, 0.4]}, train_days=252, test_days=252,
                                           benchmark="SYN_INDEX")
    assert summary["folds"] == len(folds) >= 3
    for f in folds:
        assert f["train"].split("..")[1] < f["test"].split("..")[0]
        assert f["entry"] in (0.2, 0.4)
    for a, b in zip(folds, folds[1:]):
        assert a["test"].split("..")[1] < b["test"].split("..")[0]
    assert summary["oos_period"].startswith(folds[0]["test"][:10]) and summary["benchmark"] == "SYN_INDEX"
    with pytest.raises(ValueError, match="walk-forward fold"):
        optimize.walk_forward(CSVProvider(), ["SYN_TECH"], "momentum", {}, train_days=2000)
