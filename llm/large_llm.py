"""
Large LLM client — GPT-4o (or o3) via OpenAI API.

Used for complex queries requiring deep reasoning.
Personalization is handled via system prompt injection (user profile).
"""

import logging

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from config import OPENAI_API_KEY, LARGE_MODEL, LARGE_LLM_TIMEOUT

logger = logging.getLogger(__name__)


class LargeLLM:
    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=LARGE_LLM_TIMEOUT,
        )
        self.model = LARGE_MODEL

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user_message: str,
        history: list[dict] | None = None,
        use_thinking: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        """
        Call the Large LLM and return the full response text.
        Automatically handles reasoning models (o-series) that don't
        support temperature or max_tokens params.
        """
        messages = self._build_messages(system, history, user_message)

        # o-series reasoning models don't support temperature
        is_reasoning = self.model.startswith("o")
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = 0.7

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content if response.choices else None
            return content.strip() if content else ""
        except APITimeoutError:
            logger.error("Large LLM timed out after %ss", LARGE_LLM_TIMEOUT)
            raise
        except RateLimitError as exc:
            logger.error("Large LLM rate limited: %s", exc)
            raise
        except APIError as exc:
            logger.error("Large LLM API error: %s", exc)
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system: str, history: list[dict] | None, user_message: str
    ) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
