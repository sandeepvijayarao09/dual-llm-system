"""
Large LLM client — GPT o3 (or GPT-4o) via OpenAI API.

Used for complex queries requiring deep reasoning.
Personalization is handled via system prompt injection (user profile).

Install: pip install openai
Docs:    https://platform.openai.com/docs
"""

from openai import OpenAI
from config import OPENAI_API_KEY, LARGE_MODEL, LARGE_LLM_TIMEOUT


class LargeLLM:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
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
        use_thinking is accepted for interface compatibility but
        OpenAI reasoning models handle this internally.
        """
        messages = self._build_messages(system, history, user_message)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, system: str, history: list[dict] | None, user_message: str
    ) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages
