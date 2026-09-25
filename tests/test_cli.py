import pytest

from finagent import data
from finagent.cli import main


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out, db = tmp_path / "bt", str(tmp_path / "s.db")
    assert main(["backtest", "--symbols", "SYN_UTIL", "--end", "2020-12-31", "--out", str(out)]) == 0
    assert (out / "report.html").exists() and (out / "equity.csv").read_text().startswith("ts,equity,cash")
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
    assert main(["run", "--once", "--db", db, "--stop-loss", "0.08", "--take-profit", "0.25"]) == 0
    capsys.readouterr()
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
