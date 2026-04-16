"""Decision notifications: a one-line summary on stdout, optionally POSTed as JSON to a webhook.

The webhook payload carries the line as both `text` (Slack-style) and `content` (Discord-style) plus the full
tick summary under `summary`. Webhook failures are reported on stderr and never interrupt trading.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request


def format_summary(summary: dict) -> str:
    if "error" in summary:
        return f"[finagent paper] {summary.get('ts', '')} ERROR: {summary['error']}"
    head = (f"[finagent paper] {summary['as_of']} mode={summary['mode']} equity={summary['equity']:,.2f} "
            f"cash={summary['cash']:,.2f}" + (" KILL SWITCH ENGAGED" if summary.get("killed") else ""))
    orders = summary.get("orders") or []
    if not orders:
        return head + " | no orders"
    parts = [f"{side.upper()} {sym} {got}/{req}: {outcome}" for sym, side, req, got, outcome in orders]
    return head + " | " + "; ".join(parts)


class Notifier:
    def __init__(self, webhook_url: str | None = None, stream=None, timeout: float = 5.0, only_orders: bool = True):
        if webhook_url and urllib.parse.urlsplit(webhook_url).scheme not in ("http", "https"):
            raise ValueError("webhook URL must be http(s)")
        self.webhook_url, self.stream, self.timeout, self.only_orders = webhook_url, stream, timeout, only_orders

    def __call__(self, summary: dict) -> None:
        line = format_summary(summary)
        print(line, file=self.stream or sys.stdout)
        if not self.webhook_url:
            return
        if self.only_orders and not (summary.get("orders") or summary.get("killed") or "error" in summary):
            return  # quiet ticks stay off the webhook
        body = json.dumps({"text": line, "content": line, "summary": summary}, default=str).encode()
        req = urllib.request.Request(self.webhook_url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "finagent"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"notify: webhook failed: {e}", file=sys.stderr)
