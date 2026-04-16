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
