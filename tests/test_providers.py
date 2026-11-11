"""Provider adapters, exercised against stub clients (no network, no models)."""

from types import SimpleNamespace as NS

import httpx
import pytest

from llm.providers import GeminiProvider, OllamaProvider, ProviderError, make_provider
from llm.usage import track_usage


def _ollama(handler) -> OllamaProvider:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaProvider(base_url="http://ollama", timeout=5, client=client)


def test_ollama_sends_system_first_and_records_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json={
            "model": "gemma4:e4b", "message": {"role": "assistant", "content": " hi "},
            "prompt_eval_count": 42, "eval_count": 7,
        })

    with track_usage() as ledger:
        out = _ollama(handler).complete(
            "gemma4:e4b", "SYS", [{"role": "user", "content": "q"}],
            max_tokens=50, temperature=0.1, json_mode=True)

    assert out == "hi"
    body = seen["body"]
    assert seen["path"] == "/api/chat"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["format"] == "json" and body["stream"] is False and body["think"] is False
    assert body["options"] == {"num_predict": 50, "temperature": 0.1}
    call = ledger.calls[0]
    assert (call.model, call.input_tokens, call.output_tokens) == ("gemma4:e4b", 42, 7)
    assert call.cost_usd == 0.0                       # local inference


def test_ollama_missing_model_says_how_to_fix_it():
    p = _ollama(lambda r: httpx.Response(404, json={"error": "model 'gemma4:e4b' not found"}))
    with track_usage() as ledger, pytest.raises(ProviderError, match="ollama pull gemma4:e4b"):
        p.complete("gemma4:e4b", "s", [{"role": "user", "content": "q"}], 10)
    assert ledger.calls[0].error == "HTTP404"


def test_ollama_not_running_is_a_provider_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(ProviderError, match="ollama serve"):
        _ollama(handler).complete("gemma4:e4b", "s", [{"role": "user", "content": "q"}], 10)


class _Models:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _gemini_response(text="answer", prompt=100, out=20, thoughts=30, cached=0):
    return NS(text=text, usage_metadata=NS(
        prompt_token_count=prompt, candidates_token_count=out,
        thoughts_token_count=thoughts, cached_content_token_count=cached))


def test_gemini_maps_roles_and_bills_thinking_as_output():
    models = _Models(_gemini_response())
    p = GeminiProvider(api_key="k", timeout=5, client=NS(models=models))

    with track_usage() as ledger:
        out = p.complete("gemini-3.8-flash", "SYS",
                         [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                          {"role": "user", "content": "c"}],
                         max_tokens=4096, temperature=0.7)

    assert out == "answer"
    kw = models.kwargs
    assert [c.role for c in kw["contents"]] == ["user", "model", "user"]
    assert kw["config"].system_instruction == "SYS"
    assert kw["config"].thinking_config is None           # long call: model default
    call = ledger.calls[0]
    assert (call.input_tokens, call.output_tokens) == (100, 50)
    assert call.cost_usd == pytest.approx((100 * 0.75 + 50 * 3.75) / 1e6)


def test_gemini_short_calls_get_headroom_and_low_thinking():
    from google.genai import types

    models = _Models(_gemini_response())
    GeminiProvider(api_key="k", timeout=5, client=NS(models=models)).complete(
        "gemini-3.8-flash", "s", [{"role": "user", "content": "q"}], max_tokens=60, json_mode=True)

    cfg = models.kwargs["config"]
    assert cfg.max_output_tokens >= 8192                  # thinking counts against the cap
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.response_mime_type == "application/json"


def test_gemma_on_the_gemini_api_gets_no_thinking_config():
    models = _Models(_gemini_response())
    GeminiProvider(api_key="k", timeout=5, client=NS(models=models)).complete(
        "gemma-4-31b-it", "s", [{"role": "user", "content": "q"}], max_tokens=60)
    assert models.kwargs["config"].thinking_config is None


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        make_provider("openai", timeout=10)
