"""
Simple Answerer — runs on the Small LLM (GPT-4o Mini via OpenAI).
Used for queries classified as "simple" that don't need the Large LLM.

Personalization: the raw user profile is passed to the model as-is.
The model decides how to adapt — no hardcoded rules.
"""

import json
from llm.small_llm import SmallLLM

ANSWERER_SYSTEM = """You are a helpful, concise assistant.
Answer the user's question directly and clearly.
Keep responses brief unless detail is clearly needed.
Do not add unnecessary caveats or padding."""


class SimpleAnswerer:
    def __init__(self, small_llm: SmallLLM):
        self.llm = small_llm

    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> str:
        """Generate a direct answer using the Small LLM."""
        system = ANSWERER_SYSTEM
        if user_profile:
            system += "\n\nUser profile:\n" + json.dumps(user_profile, indent=2)

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            max_tokens=512,
        )
