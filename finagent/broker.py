"""Paper broker: simulated fills with slippage and commissions, persisted to sqlite.

This module never talks to a real brokerage. There is no code path that places a real order.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .risk import PortfolioState, RiskDecision

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, qty INTEGER NOT NULL, avg_cost REAL NOT NULL);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty INTEGER, price REAL,
    commission REAL, realized_pnl REAL, source TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, cash REAL);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, source TEXT, action TEXT, requested_qty INTEGER,
    approved INTEGER, approved_qty INTEGER, rationale TEXT, risk TEXT, outcome TEXT);
"""


@dataclass(frozen=True)
class Fill:
    ts: str
    symbol: str
    side: str
    qty: int
    price: float
    commission: float
    realized_pnl: float


class PaperBroker:
    def __init__(self, db: str | Path = ":memory:", starting_cash: float = 100_000.0,
                 slippage_bps: float = 5.0, commission_per_share: float = 0.005, min_commission: float = 1.0):
        if str(db) != ":memory:":
            Path(db).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db))
        self.conn.executescript(SCHEMA)
        self.slippage = slippage_bps / 10_000
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        if self.get_meta("cash") is None:
            self.set_meta("cash", starting_cash)
            self.set_meta("starting_cash", starting_cash)
            self.conn.commit()

    # --- meta helpers -------------------------------------------------------------------
    def get_meta(self, key: str):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    # --- portfolio ----------------------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self.get_meta("cash"))

    @property
    def starting_cash(self) -> float:
        return float(self.get_meta("starting_cash"))

    def positions(self) -> dict[str, tuple[int, float]]:
        rows = self.conn.execute("SELECT symbol, qty, avg_cost FROM positions WHERE qty != 0 ORDER BY symbol")
        return {s: (q, a) for s, q, a in rows}

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, (q, _) in self.positions().items())

    def begin_day(self, day: str, prices: dict[str, float]) -> PortfolioState:
        """Snapshot for risk checks; rolls day-start equity and the high-water mark."""
        eq = self.equity(prices)
        if self.get_meta("day") != day:
            self.set_meta("day", day)
            self.set_meta("day_start_equity", eq)
        peak = max(self.get_meta("peak_equity") or eq, eq)
        self.set_meta("peak_equity", peak)
        self.conn.commit()
        pos = self.positions()
        return PortfolioState(self.cash, {s: q for s, (q, _) in pos.items()}, dict(prices),
                              peak, self.get_meta("day_start_equity"), costs={s: a for s, (_, a) in pos.items()})

    # --- execution ----------------------------------------------------------------------
    def execute(self, decision: RiskDecision, ref_price: float, ts: str) -> Fill:
        """Simulate a fill. Refuses anything the risk engine did not approve."""
        if not decision.approved or decision.qty <= 0:
            raise PermissionError(f"order for {decision.order.symbol} was not approved by the risk engine")
        o, qty = decision.order, decision.qty
        px = ref_price * (1 + self.slippage if o.side == "buy" else 1 - self.slippage)
        comm = max(self.min_commission, qty * self.commission_per_share)
        held, avg = self.positions().get(o.symbol, (0, 0.0))
        cash, realized = self.cash, 0.0
        if o.side == "buy":
            cost = qty * px + comm
            if cost > cash + 1e-9:
                raise ValueError(f"insufficient cash for {o.symbol}: need {cost:.2f}, have {cash:.2f}")
            new_qty = held + qty
            avg = (held * avg + qty * px) / new_qty
            cash -= cost
        else:
            if qty > held:
                raise ValueError(f"cannot sell {qty} {o.symbol}, only {held} held")
            realized = (px - avg) * qty - comm
            new_qty = held - qty
            cash += qty * px - comm
        self.conn.execute("INSERT OR REPLACE INTO positions VALUES (?, ?, ?)", (o.symbol, new_qty, avg))
        self.set_meta("cash", cash)
        self.conn.execute(
            "INSERT INTO fills (ts, symbol, side, qty, price, commission, realized_pnl, source, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?)", (ts, o.symbol, o.side, qty, px, comm, realized, o.source, o.reason))
        self.conn.commit()
        return Fill(ts, o.symbol, o.side, qty, px, comm, realized)

    def mark(self, ts: str, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?, ?, ?)", (ts, eq, self.cash))
        self.conn.commit()
        return eq

    # --- journal & history --------------------------------------------------------------
    def journal(self, ts: str, symbol: str, source: str, action: str, requested_qty: int,
                decision: RiskDecision | None, rationale: str, outcome: str) -> None:
        self.conn.execute(
            "INSERT INTO journal (ts, symbol, source, action, requested_qty, approved, approved_qty, rationale,"
            " risk, outcome) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, source, action, requested_qty, int(bool(decision and decision.approved)),
             decision.qty if decision else 0, rationale, json.dumps(decision.to_dict() if decision else None),
             outcome))
        self.conn.commit()

    def rows(self, table: str, limit: int | None = None) -> list[dict]:
        if table not in {"fills", "equity", "journal"}:
            raise ValueError(table)
        order = "ts" if table == "equity" else "id"
        sql = f"SELECT * FROM {table} ORDER BY {order}"
        cur = self.conn.execute(sql)
        cols = [c[0] for c in cur.description]
        out = [dict(zip(cols, r)) for r in cur.fetchall()]
        return out[-limit:] if limit else out
