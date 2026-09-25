import io
import os
import json
import urllib.error

import pytest

from finagent import data
from finagent.data import (BotChallenge, CachedProvider, CSVProvider, DataUnavailable, FallbackProvider, StooqProvider,
                           YahooProvider, http_get, parse_csv, parse_yahoo_chart)

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


def _chart(**over):
    res = {"meta": {"gmtoffset": -14400},
           "timestamp": [1704205800, 1704292200, 1704378600],  # 2024-01-02/03/04 09:30 ET
           "indicators": {"quote": [{"open": [10.0, None, 12.0], "high": [11.0, None, 13.0], "low": [9.0, None, 11.0],
                                      "close": [10.0, None, 12.0], "volume": [100, None, None]}],
                          "adjclose": [{"adjclose": [5.0, None, 6.0]}]}}
    res.update(over)
    return json.dumps({"chart": {"result": [res], "error": None}})


def test_parse_yahoo_chart_adjusts_and_skips_nulls():
    bars = parse_yahoo_chart(_chart(), "X")
    assert [b.date for b in bars] == ["2024-01-02", "2024-01-04"]
    assert bars[0].open == 5.0 and bars[0].high == 5.5 and bars[1].close == 6.0 and bars[1].volume == 0.0
    raw = parse_yahoo_chart(_chart(), "X", adjust=False)
    assert raw[0].close == 10.0


@pytest.mark.parametrize("text,match", [
    (json.dumps({"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}),
     "No data found"),
    ("not json", "not chart JSON"),
    (json.dumps({"chart": {"result": [{"timestamp": []}]}}), "unexpected"),
])
def test_parse_yahoo_chart_errors(text, match):
    with pytest.raises(DataUnavailable, match=match):
        parse_yahoo_chart(text, "X")


def test_yahoo_provider_fetches_and_detects_challenge(monkeypatch):
    seen = []

    def fake(req, timeout):
        seen.append(req)
        return _Resp(_chart().encode(), "application/json")

    monkeypatch.setattr(data.urllib.request, "urlopen", fake)
    assert len(YahooProvider(range="1y").history("spy")) == 2
    assert "/chart/SPY?range=1y" in seen[0].full_url and "finagent" in seen[0].get_header("User-agent")
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda req, timeout: _Resp(b"<html>consent</html>"))
    with pytest.raises(BotChallenge):
        YahooProvider().history("SPY")


class _Counting:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def history(self, symbol):
        self.calls += 1
        if self.fail:
            raise DataUnavailable("offline")
        return parse_csv(CSV)


def test_cache_serves_fresh_then_stale_on_failure(tmp_path):
    inner = _Counting()
    c = CachedProvider(inner, tmp_path, max_age_hours=1)
    assert c.history("BRK.B") == c.history("BRK.B") and inner.calls == 1  # second hit from disk
    assert c.path("BRK.B").name == "_counting_BRK.B.csv" and "cache" in c.served_by["BRK.B"]
    os.utime(c.path("BRK.B"), (0, 0))  # expire it
    inner.fail = True
    assert len(c.history("BRK.B")) == 2 and "STALE" in c.served_by["BRK.B"]
    with pytest.raises(DataUnavailable):
        c.history("NEVER_CACHED")


def test_fallback_reports_both_failures():
    fb = FallbackProvider(_Counting(fail=True), CSVProvider())
    with pytest.raises(DataUnavailable, match="primary failed.*fallback failed"):
        fb.history("SPY")


def _live_chart(regular_market_time):
    # session 2024-01-04 09:30-16:00 ET; the last bar belongs to it
    meta = {"gmtoffset": -18000, "regularMarketTime": regular_market_time,
            "currentTradingPeriod": {"regular": {"start": 1704378600, "end": 1704402000}}}
    return _chart(meta=meta)


def test_yahoo_drops_in_progress_bar_only_while_session_is_open():
    trading = _live_chart(1704390000)  # 12:40 ET, session still open
    assert [b.date for b in parse_yahoo_chart(trading, "X", include_partial=False)] == ["2024-01-02"]
    assert parse_yahoo_chart(trading, "X")[-1].date == "2024-01-04"  # explicitly kept
    closed = _live_chart(1704402000)  # regularMarketTime == session end: the bar is final
    assert parse_yahoo_chart(closed, "X", include_partial=False)[-1].date == "2024-01-04"
    assert parse_yahoo_chart(_chart(), "X", include_partial=False)[-1].date == "2024-01-04"  # no meta: keep


def test_yahoo_provider_drops_partial_by_default(monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen",
                        lambda req, timeout: _Resp(_live_chart(1704390000).encode(), "application/json"))
    assert YahooProvider().history("SPY")[-1].date == "2024-01-02"
    assert YahooProvider(include_partial=True).history("SPY")[-1].date == "2024-01-04"
