"""HTTP API tests — FastAPI TestClient over an orchestrator with fake LLMs."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(orch):
    with TestClient(create_app(orch)) as c:
        yield c


def chat(client, content, **body):
    return client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": content}], **body})


def test_health_reports_models(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["small_model"] == "fake-small"
    assert r.json()["ml_router"] is True
    assert r.headers["x-request-id"]


def test_chat_completion_is_openai_shaped_with_routing(client, fake_small_llm):
    fake_small_llm.default_response = "Hello!"
    r = chat(client, "Hi there!", user="cs_student")

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"])
    assert body["routing"]["decision"] == "small"
    assert body["routing"]["routed_by"] == "ml"
    assert [c["stage"] for c in body["routing"]["calls"]] == ["answerer"]


def test_model_large_forces_the_large_llm(client, fake_large_llm, fake_small_llm):
    fake_large_llm.default_response = "core"
    fake_small_llm.default_response = "INTRO: a\nCLOSING: b"
    r = chat(client, "Hi", model="large")
    assert r.json()["routing"]["decision"] == "forced:large"
    assert len(fake_large_llm.calls) == 1


def test_stateless_request_passes_prior_messages_as_history(client, fake_small_llm):
    fake_small_llm.default_response = "ok"
    client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Hi Sam."},
        {"role": "user", "content": "Hi again"},
    ]})
    assert fake_small_llm.calls[-1]["history"] == [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Hi Sam."},
    ]


def test_bad_requests_are_rejected(client):
    assert chat(client, "hi", model="gpt-4o").status_code == 422
    assert chat(client, "hi", stream=True).status_code == 400
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 422


def test_provider_failure_is_a_502(client, fake_small_llm):
    def boom(*a, **k):
        raise RuntimeError("provider down")
    fake_small_llm.complete = boom
    r = chat(client, "Hi there!")
    assert r.status_code == 502
    assert "provider down" not in r.json()["detail"]     # no upstream detail leaks


def test_personas_are_seeded_and_resettable(client):
    personas = {p["user_id"] for p in client.get("/v1/personas").json()}
    assert {"cs_student", "nasa_engineer", "high_school", "guest"} <= personas

    alex = client.get("/v1/profiles/cs_student").json()
    assert alex["name"] == "Alex"
    assert client.get("/v1/profiles/guest").json()["name"] == ""

    client.app.state.orchestrator.profile_db.save("cs_student", {**alex, "name": "Changed"})
    assert client.delete("/v1/profiles/cs_student").status_code == 204
    assert client.get("/v1/profiles/cs_student").json()["name"] == "Alex"


def test_usage_endpoint_summarises_logged_queries(client, fake_small_llm):
    fake_small_llm.default_response = "ok"
    chat(client, "Hi", user="u1")
    chat(client, "Thanks", user="u1")
    summary = client.get("/v1/usage").json()
    assert summary["queries"] == 2
    assert summary["unpriced"] == 2          # fake models have no price


def test_clear_conversation(client, fake_small_llm):
    fake_small_llm.default_response = "ok"
    chat(client, "Hi", user="u2")
    assert client.delete("/v1/conversations/u2").status_code == 204
    assert client.app.state.orchestrator.get_buffer("u2").recent_turns == []
