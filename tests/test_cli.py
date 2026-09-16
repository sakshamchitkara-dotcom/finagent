import pytest

from finagent import data
from finagent.cli import main


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out, db = tmp_path / "bt", str(tmp_path / "s.db")
    assert main(["backtest", "--symbols", "SYN_UTIL", "--end", "2020-12-31", "--out", str(out)]) == 0
    assert (out / "report.html").exists() and (out / "equity.csv").read_text().startswith("ts,equity,cash")
    import json

    saved = json.loads((out / "metrics.json").read_text())
    assert saved["config"]["symbols"] == ["SYN_UTIL"] and saved["config"]["end"] == "2020-12-31"
    assert saved["config"]["risk"]["max_drawdown"] == 0.2 and "sharpe" in saved["metrics"]
    assert main(["run", "--once", "--db", db]) == 0
    assert main(["portfolio", "--db", db]) == 0
    assert main(["report", "--db", db, "--out", str(tmp_path / "r.html")]) == 0
    text = capsys.readouterr().out
    assert "PAPER TRADING ONLY" in text and "analyst: rule-based" in text and "sharpe" in text
    assert "[finagent paper] 2023-10-30 mode=rules" in text


def test_cli_live_provider_needs_symbols_and_uses_cache(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit, match="needs --symbols"):
        main(["backtest", "--provider", "yahoo"])
    monkeypatch.setattr(data.YahooProvider, "history", lambda self, s: data.CSVProvider().history("SYN_INDEX"))
    cache = tmp_path / "cache"
    args = ["backtest", "--provider", "yahoo", "--symbols", "SPY", "--end", "2020-06-30",
            "--cache-dir", str(cache), "--out", str(tmp_path / "bt")]
    assert main(args) == 0 and (cache / "yahoo_SPY.csv").exists()
    assert "data SPY: yahoo (live, cached" in capsys.readouterr().out


def test_portfolio_and_report_show_exit_levels(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = str(tmp_path / "s.db")
    assert main(["run", "--once", "--db", db, "--stop-loss", "0.08", "--take-profit", "0.25",
                 "--trailing-stop", "0.1", "--stop-basis", "high"]) == 0
    capsys.readouterr()
    from finagent.broker import PaperBroker

    assert PaperBroker(db).get_meta("exit_config")["stop_basis"] == "high"
    assert main(["portfolio", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "SYN_BANK" in out and "exits: stop loss" in out and "take profit" in out
    assert main(["report", "--db", db, "--out", str(tmp_path / "r.html")]) == 0
    page = (tmp_path / "r.html").read_text()
    assert "Open positions and exit levels" in page and "SYN_BANK" in page


def test_positions_without_data_are_flagged_as_marked_at_cost(tmp_path, capsys):
    from finagent.broker import PaperBroker
    from finagent.risk import Order, RiskDecision

    db = tmp_path / "s.db"
    PaperBroker(db).execute(RiskDecision(Order("GONE", "buy", 3), True, 3), 10.0, "d1")
    assert main(["portfolio", "--db", str(db)]) == 0
    assert "GONE" in (out := capsys.readouterr().out) and "NO PRICE: marked at cost" in out


def test_cli_sweep_and_walkforward_write_tables(tmp_path, capsys):
    sw, wf = tmp_path / "sweep.csv", tmp_path / "wf.csv"
    assert main(["sweep", "--symbols", "SYN_TECH", "--grid", "entry=0.3,0.4", "--grid", "stop_loss=0,0.1",
                 "--end", "2021-12-31", "--out", str(sw)]) == 0
    out = capsys.readouterr().out
    assert "4 combinations" in out and "in-sample best: IS Sharpe" in out
    assert sw.read_text().splitlines()[0].startswith("entry,stop_loss,split,is_sharpe")
    assert len(sw.read_text().splitlines()) == 5
    assert main(["walkforward", "--symbols", "SYN_TECH", "--grid", "entry=0.3,0.4", "--train-days", "252",
                 "--test-days", "252", "--out", str(wf)]) == 0
    out = capsys.readouterr().out
    assert "stitched out-of-sample result" in out and "benchmark_total_return" in out  # auto -> SYN_INDEX
    assert len(wf.read_text().splitlines()) == 1 + int(out.split("folds")[1].split()[0])


def test_cli_rejects_bad_grid_and_sector_files(tmp_path):
    with pytest.raises(SystemExit, match="unknown parameter"):
        main(["sweep", "--grid", "nope=1"])
    bad = tmp_path / "sectors.json"
    bad.write_text('["AAPL"]')
    with pytest.raises(SystemExit, match="JSON object"):
        main(["backtest", "--sectors", str(bad)])
    with pytest.raises(SystemExit, match="cannot read"):
        main(["backtest", "--sectors", str(tmp_path / "missing.json")])


def test_cli_benchmark_none_and_custom_sectors(tmp_path, capsys):
    sectors = tmp_path / "sectors.json"
    sectors.write_text('{"syn_tech": "tech", "SYN_UTIL": "tech"}')
    assert main(["backtest", "--symbols", "SYN_TECH", "SYN_UTIL", "--end", "2020-12-31", "--benchmark", "none",
                 "--sectors", str(sectors), "--max-sector-pct", "0.2", "--out", str(tmp_path / "bt")]) == 0
    assert "benchmark_total_return" not in (out := capsys.readouterr().out) and "beta" not in out
    journal = (tmp_path / "bt" / "journal.csv").read_text()
    assert "sector 'tech' exposure" in journal  # the lowercase key was normalised and the 20% cap bit


@pytest.mark.parametrize("argv, match", [
    (["backtest", "--stop-loss", "1.5"], "not a fraction"),
    (["backtest", "--trailing-stop", "-0.1"], "not a fraction"),
    (["backtest", "--take-profit", "-1"], "0 or more"),
    (["backtest", "--cash", "0"], "greater than 0"),
    (["backtest", "--monte-carlo", "-5"], "0 or more"),
    (["backtest", "--start", "2021/01/31"], "not a date"),
    (["sweep", "--split", "soon"], "not a date"),
    (["run", "--interval", "0"], "greater than 0"),
])
def test_cli_rejects_out_of_range_arguments(argv, match, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    assert match in capsys.readouterr().err


def test_cli_backtest_reports_an_empty_date_range_without_a_traceback(tmp_path, capsys):
    assert main(["backtest", "--start", "2030-01-01", "--out", str(tmp_path / "bt")]) == 2
    assert "no data in the requested date range" in capsys.readouterr().err


def test_run_warns_when_cash_is_ignored_for_an_existing_account(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = str(tmp_path / "s.db")
    assert main(["run", "--once", "--db", db, "--cash", "50000"]) == 0
    assert "WARNING" not in capsys.readouterr().err  # new account: --cash is used
    assert main(["run", "--once", "--db", db]) == 0
    assert "WARNING" not in capsys.readouterr().err  # --cash not given: nothing to warn about
    assert main(["run", "--once", "--db", db, "--cash", "250000"]) == 0
    err = capsys.readouterr().err
    assert "--cash 250,000.00 ignored" in err and "started with 50,000.00" in err


def test_compare_two_saved_backtests(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["backtest", "--symbols", "SYN_TECH", "SYN_BANK", "--end", "2021-12-31"]
    assert main(common + ["--out", str(a)]) == 0
    assert main(common + ["--regime-filter", "SYN_INDEX", "--out", str(b)]) == 0
    capsys.readouterr()
    assert main(["compare", str(a), str(b / "metrics.json")]) == 0
    out = capsys.readouterr().out
    assert "risk.regime_symbol" in out and "'' -> 'SYN_INDEX'" in out
    assert " pp" in out and "total_return" in out and "strategy" not in out  # equal labels are hidden
    assert main(["compare", str(a), str(a), "--all"]) == 0
    assert "config: identical" in (out := capsys.readouterr().out) and "strategy" in out
    (tmp_path / "bad.json").write_text("[]")
    with pytest.raises(SystemExit, match="not a finagent backtest"):
        main(["compare", str(a), str(tmp_path / "bad.json")])
    with pytest.raises(SystemExit, match="cannot read"):
        main(["compare", str(a), str(tmp_path / "missing")])
