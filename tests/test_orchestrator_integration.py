"""
End-to-end integration tests for the Orchestrator.

We build an Orchestrator with real module wiring, then monkey-patch its LLM
clients and DB paths to:
    - never hit OpenAI
    - use isolated tmp databases per test
    - let us script responses from the small/large LLM

This exercises:
    - ML router cascade (high-conf skip, low-conf fallback to LLM classifier)
    - Big-route assembly (intro + core + closing)
    - Small-route direct answer
    - ConversationBuffer recording + history injection
    - ClassificationLogger writing
"""

import os
import pytest

from orchestrator import Orchestrator


@pytest.fixture
def orch(monkeypatch, tmp_path, fake_small_llm, fake_large_llm):
    """Fully wired Orchestrator with fake LLMs + tmp DBs."""
    # Isolate databases
    monkeypatch.setenv("DB_PATH", str(tmp_path / "profiles.db"))
    monkeypatch.setenv("ROUTING_LOG_PATH", str(tmp_path / "routing.db"))

    # Force config reload (it's read at import time, but the orchestrator
    # re-reads constants by module import — we patch them directly below)
    import config, importlib
    importlib.reload(config)

    o = Orchestrator()

    # Swap in fakes — we re-init the modules that depended on the real clients
    o.small_llm = fake_small_llm
    o.large_llm = fake_large_llm

    # Rebuild dependents that captured the original clients
    from modules.classifier import Classifier
    from modules.answerer import SimpleAnswerer
    from modules.reasoner import Reasoner
    from modules.personalizer import Personalizer
    from modules.profile_updater import ProfileUpdater

    o.classifier      = Classifier(fake_small_llm)
    o.answerer        = SimpleAnswerer(fake_small_llm)
    o.reasoner        = Reasoner(fake_large_llm)
    o.context_adder   = Personalizer(fake_small_llm)
    o.profile_updater = ProfileUpdater(fake_small_llm)

    # Re-init buffers dict so they also use the fake small LLM
    o.buffers.clear()

    # Re-init logger to tmp path
    from router.classification_logger import ClassificationLogger
    o.routing_log = ClassificationLogger(str(tmp_path / "routing.db"))

    return o


# ── ML router cascade ────────────────────────────────────────────────────────

def test_high_confidence_small_skips_llm_classifier(orch, fake_small_llm):
    """
    Classic small query (greeting). The ML router should confidently route
    'small' and we should never call the LLM classifier (i.e. the small LLM
    is only called once — for the actual answer).
    """
    fake_small_llm.default_response = "Hi there!"
    resp = orch.process("Hi there!", user_id="u_test")

    assert resp.routing_decision == "small"
    assert resp.routed_by == "ml"
    assert resp.ml_prediction["decision"] == "small"
    assert resp.classification == {}           # LLM classifier not called
    assert resp.error is None
    # Exactly one small-LLM call = the answerer (no classifier, no summarizer)
    assert len(fake_small_llm.calls) == 1


def test_high_confidence_big_routes_to_large(orch, fake_small_llm, fake_large_llm):
    """
    Classic big query. ML router should confidently route 'big', Large LLM
    produces the core, Small LLM summarizes + personalizes.
    """
    fake_large_llm.default_response = "A proof of irrationality of sqrt 2..."
    # Small LLM responds differently for summarize vs. personalize
    fake_small_llm.script(
        lambda s, _u: "one sentence" in s.lower() or "compress" in s.lower(),
        "The answer proves sqrt 2 is irrational.",
    )
    fake_small_llm.script(
        lambda s, _u: "personalisation" in s.lower() or "personalization" in s.lower(),
        "INTRO: As a math student, you'll enjoy this.\nCLOSING: Classic result!",
    )

    resp = orch.process(
        "Prove that the square root of 2 is irrational.",
        user_id="u_test",
    )

    assert resp.routing_decision == "big"
    assert resp.routed_by == "ml"
    assert resp.model_used == "fake-large"
    # Large LLM called exactly once
    assert len(fake_large_llm.calls) == 1
    # Final answer includes all three parts
    assert "As a math student" in resp.answer
    assert "proof of irrationality" in resp.answer
    assert "Classic result" in resp.answer


def test_low_confidence_cascades_to_llm_classifier(orch, fake_small_llm, monkeypatch):
    """
    Force the ML router's confidence below threshold → LLM classifier must
    be called and its decision must be used.
    """
    from router.ml_router import RouterPrediction

    # Force low-confidence ML prediction
    def fake_predict(_query):
        return RouterPrediction(
            decision="small",
            confidence=0.40,
            probs={"small": 0.40, "big": 0.60},
        )
    monkeypatch.setattr(orch.ml_router, "predict", fake_predict)

    # LLM classifier returns JSON claiming 'big'
    fake_small_llm.script(
        lambda s, _u: "query router" in s.lower(),
        '{"complexity":"complex","intent":"reasoning","confidence":0.9,'
        '"requires_tools":false,"sensitive":false,"routing_decision":"big"}',
    )
    # Placeholders for later small-LLM calls in the big route
    fake_small_llm.default_response = "OK"

    resp = orch.process("Something ambiguous", user_id="u_test")

    assert resp.routed_by == "llm"
    assert resp.routing_decision == "big"
    assert resp.ml_prediction["decision"] == "small"
    assert resp.ml_prediction["confidence"] == 0.4
    assert resp.classification["routing_decision"] == "big"


# ── Buffer wiring ────────────────────────────────────────────────────────────

def test_buffer_records_each_turn(orch, fake_small_llm):
    fake_small_llm.default_response = "fine"
    orch.process("Hi there!", user_id="u_buf")
    orch.process("What is 2+2?", user_id="u_buf")

    buf = orch.get_buffer("u_buf")
    assert len(buf.recent_turns) == 4          # 2 user + 2 assistant
    contents = [t["content"] for t in buf.recent_turns]
    assert "Hi there!" in contents
    assert "What is 2+2?" in contents


def test_buffer_history_fed_to_next_call(orch, fake_small_llm):
    fake_small_llm.default_response = "ack"
    orch.process("Hi there!", user_id="u_ctx")

    # Second turn must see the first in history
    orch.process("What did I just ask?", user_id="u_ctx")

    last_call = fake_small_llm.calls[-1]
    assert last_call["history"] is not None
    history_contents = [m["content"] for m in last_call["history"]]
    assert any("Hi there!" in c for c in history_contents)


def test_clear_buffer(orch, fake_small_llm):
    fake_small_llm.default_response = "ack"
    orch.process("Hi", user_id="u_clr")
    assert len(orch.get_buffer("u_clr").recent_turns) == 2

    orch.clear_buffer("u_clr")
    assert orch.get_buffer("u_clr").recent_turns == []


def test_per_user_buffer_isolation(orch, fake_small_llm):
    fake_small_llm.default_response = "ack"
    orch.process("user A question", user_id="alice")
    orch.process("user B question", user_id="bob")

    alice_msgs = [m["content"] for m in orch.get_buffer("alice").recent_turns]
    bob_msgs = [m["content"] for m in orch.get_buffer("bob").recent_turns]

    assert any("user A" in c for c in alice_msgs)
    assert not any("user B" in c for c in alice_msgs)
    assert any("user B" in c for c in bob_msgs)
    assert not any("user A" in c for c in bob_msgs)


# ── Routing log ──────────────────────────────────────────────────────────────

def test_every_turn_is_logged(orch, fake_small_llm):
    fake_small_llm.default_response = "ack"
    orch.process("Hi", user_id="u_log")
    orch.process("Thanks", user_id="u_log")

    assert orch.routing_log.count() == 2


# ── Pre-flight bypass ────────────────────────────────────────────────────────

def test_preflight_long_query_routes_big_without_classifier(
    orch, fake_small_llm, fake_large_llm, monkeypatch,
):
    # Set LARGE_INPUT_THRESHOLD low so our test query trips it
    monkeypatch.setattr("orchestrator.LARGE_INPUT_THRESHOLD", 10)

    fake_large_llm.default_response = "long core answer"
    fake_small_llm.default_response = (
        "INTRO: x\nCLOSING: y"     # will be parsed by personalizer
    )

    long_query = " ".join(["token"] * 30)
    resp = orch.process(long_query, user_id="u_pre")

    assert resp.routing_decision == "pre-flight:large-input"
    assert resp.routed_by == "pre-flight"
    assert len(fake_large_llm.calls) == 1
