import json
from types import SimpleNamespace

import pytest

from finagent.agent import Agent, proposal_to_order
from finagent.broker import PaperBroker
from finagent.data import CSVProvider
from finagent.llm import ClaudeAnalyst, Proposal, parse_proposals
from finagent.strategies import get_strategy

SYMS = ["SYN_TECH", "SYN_BANK", "SYN_ENERGY", "SYN_UTIL", "SYN_INDEX"]


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


def fake_analyst(payload, stop_reason="end_turn"):
    return ClaudeAnalyst(client=SimpleNamespace(messages=FakeMessages(payload, stop_reason)))


def test_rule_based_tick_persists_and_journals(tmp_path):
    db = tmp_path / "a.db"
    agent = Agent(CSVProvider(), PaperBroker(db), SYMS, get_strategy("mean_reversion"), log=lambda _: None)
    s = agent.tick()
    assert s["mode"] == "rules" and s["as_of"] == "2023-10-30"
    b = PaperBroker(db)
    assert len(b.rows("journal")) >= len(SYMS)  # every symbol gets a decision record
    assert len(b.rows("fills")) == sum(1 for o in s["orders"] if o[3] > 0)
    assert b.rows("equity")[-1]["ts"] == "2023-10-30"


def test_llm_proposals_are_risk_checked(tmp_path):
    payload = {"market_view": "test view", "proposals": [
        {"symbol": "SYN_TECH", "action": "buy", "target_weight": 0.9, "confidence": 0.8, "rationale": "oversold"},
        {"symbol": "FAKE", "action": "buy", "target_weight": 0.5, "confidence": 1, "rationale": "hallucinated"},
        {"symbol": "SYN_UTIL", "action": "hold", "target_weight": 0, "confidence": 0.5, "rationale": "flat"},
    ]}
    analyst = fake_analyst(payload)
    agent = Agent(CSVProvider(), PaperBroker(tmp_path / "b.db"), SYMS, get_strategy("combined"), analyst=analyst,
                  log=lambda _: None)
    s = agent.tick()
    assert s["mode"] == "llm"
    [(sym, side, req, got, outcome)] = s["orders"]
    assert sym == "SYN_TECH" and side == "buy" and 0 < got < req  # 90% asked, clipped to the 20% cap
    call = analyst.client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"]["format"]["type"] == "json_schema"
    journal = agent.broker.rows("journal")
    assert any(j["action"] == "market_view" and j["rationale"] == "test view" for j in journal)
    assert not any(j["symbol"] == "FAKE" for j in journal)


@pytest.mark.parametrize("payload,stop", [("not json", "end_turn"), ({"market_view": "", "proposals": []}, "refusal")])
def test_llm_failure_falls_back_to_rules(tmp_path, payload, stop):
    agent = Agent(CSVProvider(), PaperBroker(tmp_path / "c.db"), SYMS, get_strategy("combined"),
                  analyst=fake_analyst(payload, stop), log=lambda _: None)
    assert agent.tick()["mode"] == "rules (llm fallback)"
    assert any(j["action"] == "error" for j in agent.broker.rows("journal"))


def test_kill_switch_persists_across_restarts(tmp_path):
    db = tmp_path / "d.db"
    b = PaperBroker(db)
    b.set_meta("killed", True)
    b.conn.commit()
    agent = Agent(CSVProvider(), PaperBroker(db), SYMS, get_strategy("combined"), log=lambda _: None)
    assert agent.tick()["mode"] == "kill-switch"


def test_parse_and_convert_proposals():
    ps = parse_proposals({"proposals": [
        {"symbol": "a", "action": "buy", "target_weight": 5, "confidence": -1, "rationale": "x"},
        {"symbol": "A", "action": "sell", "target_weight": 0, "confidence": 1, "rationale": "dup"},
        {"symbol": "B", "action": "short", "target_weight": 0.1, "confidence": 1, "rationale": "bad action"},
    ]}, {"A", "B"})
    assert len(ps) == 1 and ps[0].target_weight == 1.0 and ps[0].confidence == 0.0
    assert proposal_to_order(Proposal("A", "sell", 0.0, 1, "exit"), 10, 100.0, 10_000).qty == 10
    assert proposal_to_order(Proposal("A", "buy", 0.05, 1, "already there"), 10, 100.0, 10_000) is None
