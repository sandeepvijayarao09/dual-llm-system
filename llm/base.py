"""
ChatLLM — the model-role client the modules talk to (SmallLLM / LargeLLM).

A role is a (provider, model) pair from config. The modules only ever call
`complete(system, user_message, history, max_tokens, json_mode)`; which
vendor answers is decided by SMALL_PROVIDER / LARGE_PROVIDER.
"""

from __future__ import annotations

import logging

from config import settings
from llm.providers import ChatProvider, make_provider

logger = logging.getLogger(__name__)


class ChatLLM:
    role = "llm"
    temperature: float = 0.7

    def __init__(self, provider: str, model: str, timeout: float) -> None:
        api_key = settings.api_key_for(provider)  # type: ignore[arg-type]
        # The OpenAI SDK has no other credential source; Anthropic and Gemini
        # also resolve credentials from their own env vars / profiles.
        if provider == "openai" and not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.provider_name = provider
        self.model = model
        self.provider: ChatProvider = make_provider(provider, api_key, timeout)

    def complete(
        self,
        system: str,
        user_message: str,
        history: list[dict] | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
        use_thinking: bool = False,
    ) -> str:
        messages = [{"role": t["role"], "content": t["content"]} for t in history or []]
        messages.append({"role": "user", "content": user_message})
        return self.provider.complete(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1 if json_mode else self.temperature,
            json_mode=json_mode,
        )
