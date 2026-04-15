"""
Small LLM client — GPT-4o Mini via OpenAI API.

Used for:
  - Query classification / routing
  - Direct answers to simple queries (personalized via user profile)
  - Adding short context notes after Large LLM answers
  - Profile updates at end of session

Install: pip install openai
Docs:    https://platform.openai.com/docs
"""

from openai import OpenAI
from config import OPENAI_API_KEY, SMALL_MODEL, SMALL_LLM_TIMEOUT


class SmallLLM:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = SMALL_MODEL

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user_message: str,
        history: list[dict] | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Call GPT-4o Mini and return the response text.
        Set json_mode=True to request a JSON-formatted response.
        """
        messages = self._build_messages(system, history, user_message)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1 if json_mode else 0.7,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

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
