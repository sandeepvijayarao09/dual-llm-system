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

ANSWERER_SYSTEM_WITH_PROFILE = """You are a helpful, concise assistant with full awareness of who you are talking to.

CRITICAL RULES when a user profile is provided:
- Tailor your answer SPECIFICALLY to the user's domain, background, and interests.
- If the question is broad (e.g. "career options", "best tools", "learning path"),
  do NOT give a generic answer — filter and focus only on what is relevant to this user.
- Match your vocabulary and depth to their expertise level (novice / intermediate / expert).
- Match your format to their preferred_format (bullets / prose / plain).
- Never list options outside their domain unless they explicitly ask for it.

Keep responses concise unless detail is clearly needed."""


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
        if user_profile:
            system = ANSWERER_SYSTEM_WITH_PROFILE
            system += "\n\nUser profile:\n" + json.dumps(user_profile, indent=2)
        else:
            system = ANSWERER_SYSTEM

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            max_tokens=512,
        )
