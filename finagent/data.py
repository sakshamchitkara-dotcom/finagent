"""Market data: a provider protocol, a CSV provider (bundled sample data) and an optional live provider."""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@dataclass(frozen=True)
class Bar:
    date: str  # ISO yyyy-mm-dd
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataUnavailable(RuntimeError):
    """Raised when a provider cannot return data (offline, blocked, unknown symbol)."""


class BotChallenge(DataUnavailable):
    """The server answered with an HTML page (captcha / JS bot challenge) instead of data."""


# A descriptive, non-browser UA. Yahoo rate-limits (429) spoofed browser UAs harder than honest clients.
USER_AGENT = "finagent/0.2 (+https://github.com/sakshamchitkara-dotcom/finagent)"


def looks_like_html(text: str) -> bool:
    head = text.lstrip()[:512].lower()
    return head.startswith("<") or "<html" in head or "<!doctype" in head


def http_get(url: str, source: str, timeout: float = 15.0) -> str:
    """GET a data URL. Raises BotChallenge for HTML responses and DataUnavailable for every other failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json,text/csv;q=0.9,*/*;q=0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (getattr(resp, "headers", None) or {}).get("Content-Type", "") or ""
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        hint = " (rate limited, try again later)" if e.code == 429 else ""
        raise DataUnavailable(f"{source}: HTTP {e.code}{hint}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DataUnavailable(f"{source}: fetch failed: {e}") from e
    if "html" in ctype.lower() or looks_like_html(text):
        raise BotChallenge(f"{source}: got an HTML page (bot challenge?) instead of data: {text.strip()[:60]!r}")
    return text


class DataProvider(Protocol):
    def history(self, symbol: str) -> list[Bar]:
        """Daily bars, oldest first."""
        ...


def parse_csv(text: str) -> list[Bar]:
    """Parse `date,open,high,low,close,volume` CSV (header names case-insensitive)."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DataUnavailable("empty CSV")
    fields = {f.strip().lower(): f for f in reader.fieldnames}
    need = ("date", "open", "high", "low", "close")
    if any(k not in fields for k in need):
        raise DataUnavailable(f"CSV missing columns; got {reader.fieldnames}")
    bars = []
    for row in reader:
        try:
            bars.append(Bar(
                date=row[fields["date"]].strip(),
                open=float(row[fields["open"]]),
                high=float(row[fields["high"]]),
                low=float(row[fields["low"]]),
                close=float(row[fields["close"]]),
                volume=float(row[fields["volume"]]) if "volume" in fields and row[fields["volume"]] else 0.0,
            ))
        except (ValueError, TypeError):
            continue  # skip malformed rows rather than poison the series
    bars.sort(key=lambda b: b.date)
    return bars


def bars_to_csv(bars: list[Bar]) -> str:
    rows = ["date,open,high,low,close,volume"]
    rows += [f"{b.date},{b.open!r},{b.high!r},{b.low!r},{b.close!r},{b.volume!r}" for b in bars]
    return "\n".join(rows) + "\n"


class CSVProvider:
    """Reads `<dir>/<SYMBOL>.csv`."""

    def __init__(self, directory: str | Path = SAMPLE_DIR):
        self.directory = Path(directory)

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.csv"))

    def history(self, symbol: str) -> list[Bar]:
        path = self.directory / f"{symbol.upper()}.csv"
        if not path.exists():
            raise DataUnavailable(f"no CSV for {symbol} in {self.directory}")
        return parse_csv(path.read_text())


class StooqProvider:
    """Free daily data from Stooq's CSV endpoint. Raises DataUnavailable when offline or blocked."""

    URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

    def __init__(self, suffix: str = ".us", timeout: float = 10.0):
        self.suffix = suffix
        self.timeout = timeout

    def history(self, symbol: str) -> list[Bar]:
        sym = symbol.lower() + ("" if "." in symbol else self.suffix)
        # Stooq now serves a JavaScript bot challenge to most non-browser clients; http_get flags it.
        text = http_get(self.URL.format(sym=urllib.parse.quote(sym)), f"stooq {symbol}", self.timeout)
        if not text.lower().startswith("date,"):
            raise DataUnavailable(f"stooq returned non-CSV for {symbol}: {text[:60]!r}")
        bars = parse_csv(text)
        if not bars:
            raise DataUnavailable(f"stooq returned no bars for {symbol}")
        return bars


def yahoo_partial_day(meta: dict, stamps: list[int]) -> bool:
    """True when the last bar belongs to a regular session that has not closed yet (an in-progress bar).

    Uses Yahoo's own clock: the bar started inside `currentTradingPeriod.regular` and the last trade
    (`regularMarketTime`) is before that session's end. No local wall clock involved.
    """
    try:
        reg = meta["currentTradingPeriod"]["regular"]
        return bool(stamps) and stamps[-1] >= reg["start"] and meta["regularMarketTime"] < reg["end"]
    except (KeyError, TypeError):
        return False


def parse_yahoo_chart(text: str, symbol: str, adjust: bool = True, include_partial: bool = True) -> list[Bar]:
    """Parse Yahoo's v8 chart JSON. With `adjust`, OHLC are scaled by adjclose/close (dividends + splits).

    With `include_partial=False`, today's bar is dropped while its session is still trading, so its
    "close" (really the latest trade) is never treated as a daily close.
    """
    try:
        chart = json.loads(text).get("chart") or {}
    except (ValueError, AttributeError) as e:
        raise DataUnavailable(f"yahoo {symbol}: response is not chart JSON") from e
    if chart.get("error"):
        err = chart["error"]
        raise DataUnavailable(f"yahoo {symbol}: {err.get('description') or err.get('code') or err}")
    try:
        res = chart["result"][0]
        stamps = res.get("timestamp") or []
        q = res["indicators"]["quote"][0]
        adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        offset = int(res.get("meta", {}).get("gmtoffset") or 0)
        if not include_partial and yahoo_partial_day(res.get("meta") or {}, stamps):
            stamps = stamps[:-1]
    except (KeyError, IndexError, TypeError) as e:
        raise DataUnavailable(f"yahoo {symbol}: unexpected chart JSON layout") from e
    by_date: dict[str, Bar] = {}
    for i, t in enumerate(stamps):
        o, h, lo, c = (q[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, lo, c) or c <= 0:
            continue  # Yahoo pads holidays/halts with nulls
        f = adj[i] / c if adjust and adj and adj[i] else 1.0
        day = datetime.fromtimestamp(t + offset, tz=timezone.utc).date().isoformat()
        by_date[day] = Bar(day, *(round(x * f, 6) for x in (o, h, lo, c)), float(q["volume"][i] or 0))  # last wins
    if not by_date:
        raise DataUnavailable(f"yahoo returned no bars for {symbol}")
    return [by_date[d] for d in sorted(by_date)]


class YahooProvider:
    """Free, keyless daily bars from Yahoo Finance's chart JSON endpoint (dividend/split adjusted)."""

    URL = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range={range}&interval=1d&events=div%2Csplit"

    def __init__(self, range: str = "10y", adjust: bool = True, timeout: float = 15.0, include_partial: bool = False):
        self.range, self.adjust, self.timeout, self.include_partial = range, adjust, timeout, include_partial

    def history(self, symbol: str) -> list[Bar]:
        url = self.URL.format(sym=urllib.parse.quote(symbol.upper()), range=self.range)
        return parse_yahoo_chart(http_get(url, f"yahoo {symbol}", self.timeout), symbol, self.adjust,
                                 self.include_partial)


class CachedProvider:
    """Disk cache in front of a network provider: `<dir>/<source>_<SYMBOL>.csv`.

    Fresh files (younger than `max_age_hours`) are served without a request. When the live fetch fails,
    a stale cached copy is served instead of failing, and `served_by` says so.
    """

    def __init__(self, inner: DataProvider, cache_dir: str | Path = "data/cache", max_age_hours: float = 12.0):
        self.inner, self.dir, self.max_age = inner, Path(cache_dir), max_age_hours * 3600
        self.source = type(inner).__name__.lower().removesuffix("provider")
        self.served_by: dict[str, str] = {}

    def path(self, symbol: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-._^=" else "_" for ch in symbol.upper())
        return self.dir / f"{self.source}_{safe}.csv"

    def history(self, symbol: str) -> list[Bar]:
        path = self.path(symbol)
        age = time.time() - path.stat().st_mtime if path.exists() else None
        if age is not None and age < self.max_age:
            self.served_by[symbol] = f"{self.source} (cache, {age / 3600:.1f}h old)"
            return parse_csv(path.read_text())
        try:
            bars = self.inner.history(symbol)
        except DataUnavailable as e:
            if age is None:
                raise
            self.served_by[symbol] = f"{self.source} (STALE cache, {age / 3600:.1f}h old; live failed: {e})"
            return parse_csv(path.read_text())
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(bars_to_csv(bars))
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file
        self.served_by[symbol] = f"{self.source} (live, cached {len(bars)} bars)"
        return bars


class FallbackProvider:
    """Try `primary`, fall back to `fallback` on DataUnavailable. Records which one served each symbol."""

    def __init__(self, primary: DataProvider, fallback: DataProvider):
        self.primary, self.fallback = primary, fallback
        self.served_by: dict[str, str] = {}

    def history(self, symbol: str) -> list[Bar]:
        try:
            bars = self.primary.history(symbol)
            detail = getattr(self.primary, "served_by", {}).get(symbol)
            self.served_by[symbol] = detail or type(self.primary).__name__
        except DataUnavailable as e:
            self.served_by[symbol] = f"unavailable (primary failed: {e})"
            try:
                bars = self.fallback.history(symbol)
            except DataUnavailable as e2:
                raise DataUnavailable(f"{symbol}: primary failed ({e}); fallback failed ({e2})") from e2
            self.served_by[symbol] = f"{type(self.fallback).__name__} (primary failed: {e})"
        return bars
