import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from finagent.notify import Notifier, format_summary

TICK = {"ts": "t", "as_of": "2026-09-24", "mode": "rules", "equity": 100_000.0, "cash": 80_000.0, "killed": False,
        "orders": [("SPY", "buy", 30, 25, "filled buy 25 @ 767.56 (paper)")]}


def test_format_summary():
    line = format_summary(TICK)
    assert line.startswith("[finagent paper] 2026-09-24 mode=rules equity=100,000.00")
    assert "BUY SPY 25/30: filled buy 25 @ 767.56 (paper)" in line
    assert format_summary({**TICK, "orders": [], "killed": True}).endswith("KILL SWITCH ENGAGED | no orders")
    assert "ERROR: boom" in format_summary({"ts": "t", "error": "boom"})


@pytest.fixture
def hook():
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/hook", received
    srv.shutdown()


def test_webhook_posts_decisions_but_not_quiet_ticks(hook):
    url, received = hook
    out = io.StringIO()
    n = Notifier(url, stream=out)
    n(TICK)
    n({**TICK, "orders": []})
    assert len(received) == 1 and received[0]["text"] == received[0]["content"] == format_summary(TICK)
    assert received[0]["summary"]["orders"][0][0] == "SPY" and out.getvalue().count("\n") == 2
    Notifier(url, stream=out, only_orders=False)({**TICK, "orders": []})
    assert len(received) == 2


def test_webhook_failure_does_not_raise(capsys):
    Notifier("http://127.0.0.1:9/unreachable", timeout=1)(TICK)
    assert "webhook failed" in capsys.readouterr().err


def test_webhook_rejects_non_http_schemes():
    with pytest.raises(ValueError):
        Notifier("file:///etc/passwd")
