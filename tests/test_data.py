import io
import urllib.error

import pytest

from finagent import data
from finagent.data import (BotChallenge, CSVProvider, DataUnavailable, FallbackProvider, StooqProvider, http_get,
                           parse_csv)

CSV = "Date,Open,High,Low,Close,Volume\n2024-01-03,2,3,1,2.5,10\n2024-01-02,1,2,0.5,1.5,\nbad,x,y,z,w,1\n"


def test_parse_csv_sorts_skips_bad_rows_and_blank_volume():
    bars = parse_csv(CSV)
    assert [b.date for b in bars] == ["2024-01-02", "2024-01-03"]
    assert bars[0].volume == 0.0 and bars[1].close == 2.5


def test_parse_csv_missing_columns():
    with pytest.raises(DataUnavailable):
        parse_csv("date,close\n2024-01-01,1\n")


def test_sample_data_is_labelled_and_complete():
    p = CSVProvider()
    assert p.symbols() and all(s.startswith("SYN_") for s in p.symbols())
    bars = p.history("SYN_INDEX")
    assert len(bars) == 1260 and all(b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) for b in bars)
    with pytest.raises(DataUnavailable):
        p.history("NOPE")


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, ctype: str = "text/plain"):
        super().__init__(body)
        self.headers = {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_stooq_parses_csv(monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout: _Resp(CSV.encode()))
    assert len(StooqProvider().history("AAPL")) == 2


@pytest.mark.parametrize("behaviour", ["offline", "html"])
def test_stooq_degrades_gracefully(monkeypatch, behaviour):
    def fake(req, timeout):
        if behaviour == "offline":
            raise urllib.error.URLError("no network")
        return _Resp(b"<!DOCTYPE html><html>challenge</html>")

    monkeypatch.setattr(data.urllib.request, "urlopen", fake)
    with pytest.raises(DataUnavailable):
        StooqProvider().history("AAPL")
    fb = FallbackProvider(StooqProvider(), CSVProvider())
    assert fb.history("SYN_TECH")[-1].date == "2023-10-30"
    assert fb.served_by["SYN_TECH"].startswith("CSVProvider")


@pytest.mark.parametrize("body,ctype", [
    (b"<!DOCTYPE html><html><noscript>verify your browser</noscript></html>", "text/plain"),
    (b"  <html><body>captcha</body></html>", "application/octet-stream"),
    (b'{"ok": true}', "text/html; charset=utf-8"),
])
def test_http_get_flags_html_as_bot_challenge(monkeypatch, body, ctype):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout: _Resp(body, ctype))
    with pytest.raises(BotChallenge):
        http_get("https://example.test", "t")


def test_http_get_reports_rate_limit(monkeypatch):
    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake)
    with pytest.raises(DataUnavailable, match="rate limited"):
        http_get("https://example.test", "t")
