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
