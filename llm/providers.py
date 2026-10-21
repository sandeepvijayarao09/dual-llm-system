"""
Provider adapters — one chat interface over OpenAI, Anthropic and Gemini.

Every adapter takes the same arguments, returns the reply text, and records
the call's token usage and latency in the active usage ledger
(llm/usage.py). Provider-specific request quirks live here and nowhere else.

    provider = make_provider("anthropic", api_key=..., timeout=60)
    text = provider.complete(model="claude-haiku-5-5", system="...", messages=[...])

The Anthropic and Gemini SDKs are optional extras (`uv sync --extra
anthropic`, `--extra gemini`) and are imported only when that provider is
configured.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

from llm.usage import record_call

logger = logging.getLogger(__name__)

Message = dict[str, str]          # {"role": "user" | "assistant", "content": str}


class ProviderError(RuntimeError):
    """A provider call failed (timeout, rate limit, API error)."""


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


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, timeout: float, client=None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=timeout)
        self.client = client

    def complete(self, model, system, messages, max_tokens,
                 temperature=None, json_mode=False) -> str:
        from openai import APIError

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens,
        }
        # o-series reasoning models reject temperature
        if temperature is not None and not model.startswith("o"):
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except APIError as exc:
            record_call(model, started, error=type(exc).__name__)
            raise ProviderError(f"OpenAI {type(exc).__name__}: {exc}") from exc

        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        record_call(
            getattr(response, "model", None) or model, started,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
        )
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else ""


# ── Anthropic ────────────────────────────────────────────────────────────────

# Current Claude models think by default and reject sampling parameters.
# Thinking tokens count against max_tokens, so short support calls (a 60-token
# summary) need headroom; max_tokens is a ceiling, not a charge.
_CLAUDE_MIN_MAX_TOKENS = 4096


def _claude_thinking_kwargs(model: str) -> dict:
    """Cheapest valid thinking setup per model family for short, simple calls."""
    if model.startswith("claude-haiku-5"):
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    if model.startswith("claude-sonnet-5-5"):
        return {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}
    if model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5")):
        # thinking can't be disabled here; effort is the only lever
        return {"output_config": {"effort": "low"}}
    return {}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, timeout: float, client=None) -> None:
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or None, timeout=timeout)
        self.client = client

    def complete(self, model, system, messages, max_tokens,
                 temperature=None, json_mode=False) -> str:
        import anthropic

        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        kwargs: dict = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max(max_tokens, _CLAUDE_MIN_MAX_TOKENS),
            **_claude_thinking_kwargs(model),
        }

        started = time.perf_counter()
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            record_call(model, started, error=type(exc).__name__)
            raise ProviderError(f"Anthropic {type(exc).__name__}: {exc}") from exc

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        record_call(
            model, started,
            # Anthropic's input_tokens excludes cached tokens; report the total
            input_tokens=(usage.input_tokens or 0) + cache_read + cache_write,
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=cache_read,
        )
        if response.stop_reason == "refusal":
            logger.warning("Claude declined the request (%s)", getattr(response, "stop_details", None))
        return "".join(b.text for b in response.content if b.type == "text").strip()


# ── Gemini ───────────────────────────────────────────────────────────────────

class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, timeout: float, client=None) -> None:
        if client is None:
            from google import genai
            client = genai.Client(
                api_key=api_key or None,
                http_options={"timeout": int(timeout * 1000)},   # milliseconds
            )
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
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
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
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        )
        return (response.text or "").strip()


_PROVIDERS: dict[str, Callable[..., ChatProvider]] = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}


def make_provider(name: str, api_key: str, timeout: float) -> ChatProvider:
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider {name!r}; expected one of {sorted(_PROVIDERS)}") from None
    return cls(api_key=api_key, timeout=timeout)
