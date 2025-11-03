"""Market data: a provider protocol, a CSV provider (bundled sample data) and an optional live provider."""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass
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
        req = urllib.request.Request(self.URL.format(sym=sym), headers={"User-Agent": "finagent/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise DataUnavailable(f"stooq fetch failed for {symbol}: {e}") from e
        if not text.lower().startswith("date,"):
            # Stooq returns HTML (bot challenge) or "No data" instead of CSV in some cases.
            raise DataUnavailable(f"stooq returned non-CSV for {symbol}: {text[:60]!r}")
        bars = parse_csv(text)
        if not bars:
            raise DataUnavailable(f"stooq returned no bars for {symbol}")
        return bars


class FallbackProvider:
    """Try `primary`, fall back to `fallback` on DataUnavailable. Records which one served each symbol."""

    def __init__(self, primary: DataProvider, fallback: DataProvider):
        self.primary, self.fallback = primary, fallback
        self.served_by: dict[str, str] = {}

    def history(self, symbol: str) -> list[Bar]:
        try:
            bars = self.primary.history(symbol)
            self.served_by[symbol] = type(self.primary).__name__
        except DataUnavailable as e:
            bars = self.fallback.history(symbol)
            self.served_by[symbol] = f"{type(self.fallback).__name__} (primary failed: {e})"
        return bars
