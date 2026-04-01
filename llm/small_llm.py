"""
Small LLM client — Gemini 2.5 Flash via Google GenAI API.

Used for:
  - Query classification / routing
  - Direct answers to simple queries (personalized via user profile)
  - Adding short context notes after Large LLM answers

Install: pip install google-genai
Docs:    https://ai.google.dev/gemini-api/docs
"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, SMALL_MODEL, SMALL_LLM_TIMEOUT


class SmallLLM:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or environment."
            )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
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
        Call Gemini 2.5 Flash and return the response text.
        Set json_mode=True to request a JSON-formatted response.
        """
        contents = self._build_contents(history, user_message)

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": 0.1 if json_mode else 0.7,
            "system_instruction": system,
        }

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

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
