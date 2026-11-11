"""Provider adapters, exercised against stub SDK clients (no network)."""

from types import SimpleNamespace as NS

import pytest

from llm.providers import AnthropicProvider, OpenAIProvider, make_provider
from llm.usage import track_usage


class _Recorder:
    """Stub for an SDK method: records kwargs, returns a canned response."""
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_adapter_sends_system_first_and_records_usage():
    resp = NS(model="gpt-4o-mini-2024-07-18",
              choices=[NS(message=NS(content=" hi "))],
              usage=NS(prompt_tokens=120, completion_tokens=30,
                       prompt_tokens_details=NS(cached_tokens=100)))
    completions = _Recorder(resp)
    p = OpenAIProvider("k", 10, client=NS(chat=NS(completions=completions)))

    with track_usage() as ledger:
        out = p.complete("gpt-4o-mini", "SYS", [{"role": "user", "content": "q"}],
                         max_tokens=50, temperature=0.1, json_mode=True)

    assert out == "hi"
    assert completions.kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    call = ledger.calls[0]
    assert (call.model, call.input_tokens, call.cached_input_tokens) == (
        "gpt-4o-mini-2024-07-18", 120, 100)
    assert call.cost_usd is not None


def test_openai_reasoning_models_get_no_temperature():
    completions = _Recorder(NS(model="o3", choices=[], usage=None))
    p = OpenAIProvider("k", 10, client=NS(chat=NS(completions=completions)))
    p.complete("o3", "s", [{"role": "user", "content": "q"}], 10, temperature=0.7)
    assert "temperature" not in completions.kwargs


def test_anthropic_adapter_counts_cached_tokens_and_skips_sampling():
    pytest.importorskip("anthropic")
    resp = NS(stop_reason="end_turn",
              content=[NS(type="thinking", thinking=""), NS(type="text", text="answer")],
              usage=NS(input_tokens=20, output_tokens=7,
                       cache_read_input_tokens=100, cache_creation_input_tokens=0))
    messages = _Recorder(resp)
    p = AnthropicProvider("k", 10, client=NS(messages=messages))

    with track_usage() as ledger:
        out = p.complete("claude-haiku-5-5", "SYS", [{"role": "user", "content": "q"}],
                         max_tokens=60, temperature=0.7)

    assert out == "answer"
    kw = messages.kwargs
    assert kw["system"] == "SYS" and "temperature" not in kw
    assert kw["max_tokens"] >= 60
    assert kw["thinking"] == {"type": "disabled"}
    assert (ledger.calls[0].input_tokens, ledger.calls[0].cached_input_tokens) == (120, 100)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        make_provider("acme", "k", 10)


def test_failed_call_is_recorded_and_wrapped():
    import httpx
    from openai import APITimeoutError

    class Failing:
        def create(self, **kwargs):
            raise APITimeoutError(request=httpx.Request("POST", "https://x"))

    p = OpenAIProvider("k", 10, client=NS(chat=NS(completions=Failing())))
    with track_usage() as ledger, pytest.raises(RuntimeError, match="APITimeoutError"):
        p.complete("gpt-4o", "s", [{"role": "user", "content": "q"}], 10)
    assert ledger.calls[0].error == "APITimeoutError"
