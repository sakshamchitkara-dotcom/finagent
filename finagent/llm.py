"""Optional Claude analyst. Proposes target weights with rationale as structured JSON.

Proposals are advisory: the agent converts them to orders that must still pass the deterministic
risk engine. If the SDK or ANTHROPIC_API_KEY is missing, or the call fails, the agent falls back
to the rule-based policy.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass

MODEL = "claude-opus-5-5"

SYSTEM = """You are the research analyst inside a PAPER-TRADING agent. No real money is at risk and \
your output never reaches a real brokerage. You receive a JSON snapshot: the portfolio, the risk \
limits, and per-symbol indicators plus the output of a rule-based strategy.

Propose a target portfolio. For each symbol choose an action (buy, sell or hold) and a target_weight \
(fraction of total equity, 0 to 1; 0 with sell means exit). The book is long-only. A deterministic \
risk engine will clip or reject anything that breaks the limits, so do not try to work around them. \
Prefer hold when evidence is weak; turnover costs slippage and commission. Give a short, specific \
rationale per symbol that cites the indicators you relied on, a confidence from 0 to 1, and a \
one-paragraph market_view. Only use symbols present in the snapshot."""

SCHEMA = {
    "type": "object",
    "properties": {
        "market_view": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                    "target_weight": {"type": "number"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["symbol", "action", "target_weight", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market_view", "proposals"],
    "additionalProperties": False,
}


class AnalystError(RuntimeError):
    pass


@dataclass
class Proposal:
    symbol: str
    action: str
    target_weight: float
    confidence: float
    rationale: str


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def _clamp(x, lo=0.0, hi=1.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    return lo if x != x else max(lo, min(hi, x))  # NaN -> lo


def parse_proposals(data: dict, universe: set[str]) -> list[Proposal]:
    """Validate model output. Unknown symbols, bad actions and duplicates are dropped; numbers clamped."""
    out, seen = [], set()
    for p in data.get("proposals") or []:
        if not isinstance(p, dict):
            continue
        sym, action = str(p.get("symbol", "")).upper(), p.get("action")
        if sym not in universe or sym in seen or action not in ("buy", "sell", "hold"):
            continue
        seen.add(sym)
        out.append(Proposal(sym, action, _clamp(p.get("target_weight")), _clamp(p.get("confidence")),
                            str(p.get("rationale", ""))[:2000]))
    return out


class ClaudeAnalyst:
    def __init__(self, client=None, model: str = MODEL, effort: str = "medium"):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort

    def propose(self, snapshot: dict) -> tuple[list[Proposal], str]:
        try:
            import anthropic
            errors = (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError)
        except ImportError:  # injected fake client in tests
            errors = ()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(snapshot, default=str)}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except errors as e:
            raise AnalystError(f"{type(e).__name__}: {e}") from e
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise AnalystError(f"model stopped with stop_reason={resp.stop_reason}")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise AnalystError("no text block in response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise AnalystError(f"invalid JSON from model: {e}") from e
        universe = set(snapshot.get("symbols", {}))
        return parse_proposals(data, universe), str(data.get("market_view", ""))
