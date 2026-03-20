"""
Large LLM client — Claude Opus 4.6 via Anthropic API.

Uses adaptive thinking for complex queries and streaming to avoid timeouts.
"""

import anthropic
from config import ANTHROPIC_API_KEY, LARGE_MODEL


class LargeLLM:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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
        Call Claude Opus with optional adaptive thinking.
        Streams internally and returns the full text response.
        """
        messages = self._build_messages(history, user_message)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        full_text = ""
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                full_text += text

        return full_text.strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, history: list[dict] | None, user_message: str
    ) -> list[dict]:
        messages = []
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
