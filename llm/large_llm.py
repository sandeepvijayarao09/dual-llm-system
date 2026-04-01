"""
Large LLM client — Gemini 2.0 Pro via Google GenAI API.

Used for complex queries requiring deep reasoning.
Personalization is handled via system prompt injection (user profile).

Install: pip install google-genai
Docs:    https://ai.google.dev/gemini-api/docs
"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, LARGE_MODEL, LARGE_LLM_TIMEOUT


class LargeLLM:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
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
        Call Gemini 2.0 Pro and return the full response text.
        use_thinking enables Gemini's thinking mode for complex reasoning.
        """
        contents = self._build_contents(history, user_message)

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": 0.7,
            "system_instruction": system,
        }

        if use_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=8192,
            )

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return response.text.strip() if response.text else ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_contents(
        self, history: list[dict] | None, user_message: str
    ) -> list:
        """Build the contents list for the Gemini API."""
        contents = []
        if history:
            for turn in history:
                role = "user" if turn["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=turn["content"])]
                    )
                )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            )
        )
        return contents
