"""Self-contained HTML report with an inline SVG equity/drawdown chart. No external assets."""

from __future__ import annotations

import html
from pathlib import Path

from .backtest import round_trips

PCT = {"total_return", "cagr", "max_drawdown", "win_rate", "buy_hold_return", "benchmark_total_return",
       "benchmark_cagr", "benchmark_max_drawdown", "excess_return", "alpha", "trade_win_rate",
       "avg_trade_return", "return"}


def _fmt(k: str, v) -> str:
    if v is None:
        return "&ndash;"
    if isinstance(v, float):
        return f"{v:.2%}" if k in PCT else f"{v:,.2f}"
    return html.escape(str(v))


def equity_svg(dates: list[str], values: list[float], width: int = 860, height: int = 300,
               bench: list[float] | None = None, bench_label: str = "benchmark") -> str:
    if len(values) < 2:
        return "<p>Not enough equity history to chart yet.</p>"
    pad_l, pad_r, pad_t, pad_b = 70, 10, 10, 24
    eq_h = (height - pad_t - pad_b) * 0.72
    dd_top = pad_t + eq_h + 14
    dd_h = height - pad_b - dd_top
    lo, hi = min(values + (bench or [])), max(values + (bench or []))
    span = (hi - lo) or 1.0
    n = len(values)
    x = lambda i: pad_l + (width - pad_l - pad_r) * i / (n - 1)  # noqa: E731
    y = lambda v: pad_t + eq_h * (1 - (v - lo) / span)  # noqa: E731
    peak, dd = values[0], []
    for v in values:
        peak = max(peak, v)
        dd.append(1 - v / peak if peak else 0.0)
    max_dd = max(dd) or 1.0
    eq_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    dd_pts = " ".join(f"{x(i):.1f},{dd_top + dd_h * d / max_dd:.1f}" for i, d in enumerate(dd))
    bench_svg = ""
    if bench and len(bench) == n:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(bench))
        bench_svg = (f'<polyline points="{pts}" class="bm"/>'
                     f'<text x="{pad_l + 8}" y="{pad_t + 12}" class="lbl"><tspan class="k-eq">&#9632; strategy</tspan>'
                     f' <tspan class="k-bm">&#9632; {html.escape(bench_label)}</tspan></text>')
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve and drawdown">
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + eq_h}" class="axis"/>
  <text x="{pad_l - 6}" y="{pad_t + 10}" class="lbl" text-anchor="end">{hi:,.0f}</text>
  <text x="{pad_l - 6}" y="{pad_t + eq_h}" class="lbl" text-anchor="end">{lo:,.0f}</text>
  {bench_svg}
  <polyline points="{eq_pts}" class="eq"/>
  <text x="{pad_l - 6}" y="{dd_top + 10}" class="lbl" text-anchor="end">DD</text>
  <text x="{pad_l - 6}" y="{dd_top + dd_h}" class="lbl" text-anchor="end">-{max_dd:.0%}</text>
  <polyline points="{pad_l},{dd_top} {dd_pts} {x(n - 1):.1f},{dd_top}" class="dd"/>
  <text x="{pad_l}" y="{height - 6}" class="lbl">{html.escape(dates[0])}</text>
  <text x="{width - pad_r}" y="{height - 6}" class="lbl" text-anchor="end">{html.escape(dates[-1])}</text>
</svg>"""


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "<p class='muted'>None.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c, r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _by_symbol(trips: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for t in trips:
        if t["pnl"] is None:
            continue
        r = rows.setdefault(t["symbol"], {"symbol": t["symbol"], "trades": 0, "wins": 0, "pnl": 0.0, "days": 0})
        r["trades"] += 1
        r["wins"] += t["pnl"] > 0
        r["pnl"] += t["pnl"]
        r["days"] += t["days_held"]
    return [{"symbol": r["symbol"], "trades": r["trades"], "win_rate": r["wins"] / r["trades"],
             "total_pnl": r["pnl"], "avg_pnl": r["pnl"] / r["trades"], "avg_days_held": r["days"] / r["trades"]}
            for r in sorted(rows.values(), key=lambda r: -r["pnl"])]


def render_html(title: str, equity: list[dict], metrics: dict, fills: list[dict],
                journal: list[dict] | None = None, note: str = "", benchmark: list[dict] | None = None) -> str:
    metric_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_fmt(k, v)}</td></tr>" for k, v in metrics.items())
    trips = round_trips(fills)
    closed = [t for t in trips if t["pnl"] is not None]
    trips_html = (
        "<h2>Per-symbol trade summary</h2>" + _table(_by_symbol(trips), ["symbol", "trades", "win_rate", "total_pnl",
                                                                          "avg_pnl", "avg_days_held"])
        + f"<h2>Round-trip trades (latest 100 of {len(closed)} closed)</h2>"
        + _table(trips[-100:][::-1], ["symbol", "entry", "exit", "qty", "avg_entry", "avg_exit", "pnl", "return",
                                      "days_held", "exit_reason"]))
    journal_html = ""
    if journal is not None:
        journal_html = "<h2>Decision journal (latest 100)</h2>" + _table(
            journal[-100:][::-1], ["ts", "symbol", "source", "action", "requested_qty", "approved_qty", "outcome", "rationale"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1b1f24; --muted:#5b6470; --line:#d0d7de; --eq:#1f6feb; --dd:#cf222e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#30363d; --eq:#4493f8; --dd:#f85149; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,-apple-system,sans-serif; margin:0 auto; max-width:900px; padding:16px; }}
.warn {{ border:1px solid var(--dd); padding:8px 12px; border-radius:6px; }}
.muted {{ color:var(--muted); }}
svg {{ width:100%; height:auto; }} .axis {{ stroke:var(--line); }}
.eq {{ fill:none; stroke:var(--eq); stroke-width:1.5; }} .dd {{ fill:var(--dd); fill-opacity:.25; stroke:var(--dd); stroke-width:1; }}
.lbl {{ fill:var(--muted); font-size:11px; }}
.bm {{ fill:none; stroke:var(--muted); stroke-width:1.2; stroke-dasharray:4 3; }}
.k-eq {{ fill:var(--eq); }} .k-bm {{ fill:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }}
.scroll {{ overflow-x:auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="warn"><strong>Paper trading only. Not financial advice.</strong> {html.escape(note)}</p>
{equity_svg([r["ts"] for r in equity], [r["equity"] for r in equity],
            bench=[r["equity"] for r in benchmark] if benchmark else None,
            bench_label=str(metrics.get("benchmark", "benchmark")))}
<h2>Metrics</h2><table>{metric_rows}</table>
{trips_html}
<h2>Fills (latest 100)</h2>
{_table(fills[-100:][::-1], ["ts", "symbol", "side", "qty", "price", "commission", "realized_pnl", "source"])}
{journal_html}
</body></html>
"""


def write_report(path: Path, *args, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(*args, **kwargs))
    return path
