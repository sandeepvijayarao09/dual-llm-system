"""
Provider adapters — one chat interface over local Gemma (Ollama) and Gemini.

Every adapter takes the same arguments, returns the reply text, and records
the call's token usage and latency in the active usage ledger
(llm/usage.py). Provider-specific request quirks live here and nowhere else.

    provider = make_provider("ollama", base_url="http://localhost:11434", timeout=60)
    text = provider.complete(model="gemma4:e4b", system="...", messages=[...])

    ollama : any model served by a local Ollama (default small role: gemma4:e4b)
    gemini : the Gemini API — Gemini models, and Gemma 4 31B / 26B A4B
             (gemma-4-31b-it, gemma-4-26b-a4b-it; free tier only)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from llm.usage import record_call

logger = logging.getLogger(__name__)

Message = dict[str, str]          # {"role": "user" | "assistant", "content": str}

# Transient failures are retried with exponential backoff (0.5 s, 1 s, ...):
# rate limits, server errors and dropped connections. Client errors (400,
# 404 model-not-pulled) fail immediately.
MAX_ATTEMPTS = 3
RETRY_STATUS = (408, 429, 500, 502, 503, 504)
BACKOFF_S = 0.5


class ProviderError(RuntimeError):
    """A provider call failed (timeout, rate limit, API error, model missing)."""


class ChatProvider(Protocol):
    name: str

    def complete(
        self,
        model: str,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str: ...


# ── Ollama (local Gemma) ─────────────────────────────────────────────────────

class OllamaProvider:
    """Ollama's native /api/chat. Local inference: no API key, no per-token bill."""

    name = "ollama"

    def __init__(self, base_url: str, timeout: float, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, **_: Any) -> None:
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._sleep = sleep

    def complete(self, model, system, messages, max_tokens,
                 temperature=None, json_mode=False) -> str:
        body: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            # Gemma 4 can think; the routing tiers want direct answers.
            "think": False,
            "options": {"num_predict": max_tokens},
        }
        if temperature is not None:
            body["options"]["temperature"] = temperature
        if json_mode:
            body["format"] = "json"

        started = time.perf_counter()
        try:
            r = self._post_with_retries(body)
        except httpx.HTTPStatusError as exc:
            record_call(model, started, error=f"HTTP{exc.response.status_code}")
            detail = exc.response.text[:200]
            hint = f" (run `ollama pull {model}`)" if exc.response.status_code == 404 else ""
            raise ProviderError(f"Ollama HTTP {exc.response.status_code}: {detail}{hint}") from exc
        except httpx.HTTPError as exc:
            record_call(model, started, error=type(exc).__name__)
            raise ProviderError(
                f"Ollama {type(exc).__name__}: is `ollama serve` running? ({exc})") from exc

        data = r.json()
        record_call(
            data.get("model") or model, started,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )
        return (data.get("message", {}).get("content") or "").strip()

    def _post_with_retries(self, body: dict) -> httpx.Response:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                r = self.client.post("/api/chat", json=body)
                if r.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                    raise _Retry(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r
            except (_Retry, httpx.TransportError) as exc:
                if attempt == MAX_ATTEMPTS:
                    raise
                delay = BACKOFF_S * 2 ** (attempt - 1)
                logger.warning("Ollama %s; retry %d/%d in %.1fs", exc, attempt,
                               MAX_ATTEMPTS - 1, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")


class _Retry(Exception):
    """Internal: a retryable HTTP status."""


# ── Gemini API ───────────────────────────────────────────────────────────────

# Gemini 3.x counts thinking tokens against max_output_tokens, so a 60-token
# summary request could be spent entirely on thinking. Give every call
# headroom (it's a ceiling, not a charge) and keep short support calls on low
# thinking.
_GEMINI_MIN_OUTPUT_TOKENS = 8192
_SHORT_CALL_TOKENS = 512


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, timeout: float, client: Any = None, **_: Any) -> None:
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key or None, http_options=gemini_http_options(timeout))
        self.client = client

    def complete(self, model, system, messages, max_tokens,
                 temperature=None, json_mode=False) -> str:
        from google.genai import errors, types

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        thinking = None
        if model.startswith("gemini-") and max_tokens <= _SHORT_CALL_TOKENS:
            thinking = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max(max_tokens, _GEMINI_MIN_OUTPUT_TOKENS),
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=thinking,
        )

        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=model, contents=contents, config=config)
        except errors.APIError as exc:
            record_call(model, started, error=type(exc).__name__)
            raise ProviderError(f"Gemini {type(exc).__name__}: {exc}") from exc

        usage = response.usage_metadata
        record_call(
            model, started,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            # thinking tokens are billed as output
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        )
        return (response.text or "").strip()


def gemini_http_options(timeout: float):
    """Timeout (ms) plus the SDK's built-in exponential-backoff retries."""
    from google.genai import types

    return types.HttpOptions(
        timeout=int(timeout * 1000),
        retry_options=types.HttpRetryOptions(
            attempts=MAX_ATTEMPTS, initial_delay=BACKOFF_S,
            http_status_codes=list(RETRY_STATUS),
        ),
    )


_PROVIDERS: dict[str, Callable[..., ChatProvider]] = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
}


def make_provider(name: str, *, timeout: float, api_key: str = "",
                  base_url: str = "") -> ChatProvider:
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider {name!r}; expected one of {sorted(_PROVIDERS)}") from None
    return cls(api_key=api_key, base_url=base_url, timeout=timeout)
