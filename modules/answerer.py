"""
Simple Answerer — runs on the Small LLM (GPT-4o Mini via OpenAI).
Used for queries classified as "simple" that don't need the Large LLM.

Personalization: the raw user profile is passed to the model as-is.
The model decides how to adapt — no hardcoded rules.
"""

import json
from datetime import date
from llm.small_llm import SmallLLM

def _today() -> str:
    return date.today().strftime("%B %d, %Y")   # e.g. "April 21, 2026"

ANSWERER_SYSTEM = f"""You are a helpful, concise assistant.
Answer the user's question directly and clearly.
Keep responses brief unless detail is clearly needed.
Do not add unnecessary caveats or padding.
Today's date is {{today}}.
If a user_name is provided, use it naturally in greetings. Otherwise don't guess it."""

ANSWERER_SYSTEM_WITH_PROFILE = f"""You are a helpful, concise assistant with full awareness of who you are talking to.

The user's profile is embedded directly in their message inside [User context: ...].
Use it to tailor your answer — do NOT ask for information that is already in the profile.

CRITICAL RULES:
- Answer specifically for THIS user based on their domain, background, and interests.
- If the question is broad (e.g. "career options", "best tools", "learning path"),
  filter and focus only on what is relevant to their profile — never give a generic list.
- Match vocabulary and depth to their expertise level (novice / intermediate / expert).
- Match format to their preferred_format (bullets / prose / plain).
- Never ask the user to clarify things already stated in their profile.

Today's date is {{today}}.
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
        today = _today()
        if user_profile:
            system = ANSWERER_SYSTEM_WITH_PROFILE.format(today=today)
            augmented_query = self._augment_query(query, user_profile)
        else:
            system = ANSWERER_SYSTEM.format(today=today)
            augmented_query = query

        return self.llm.complete(
            system=system,
            user_message=augmented_query,
            history=history,
            max_tokens=512,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _augment_query(query: str, user_profile: dict) -> str:
        """Embed compact profile context directly into the user message."""
        parts = []
        if user_profile.get("name"):
            parts.append(f"Name: {user_profile['name']}")
        if user_profile.get("expertise"):
            parts.append(f"Expertise: {user_profile['expertise']}")
        if user_profile.get("domain"):
            parts.append(f"Domain: {user_profile['domain']}")
        if user_profile.get("interests"):
            interests = user_profile["interests"]
            if isinstance(interests, list):
                parts.append(f"Interests: {', '.join(interests[:5])}")
        if user_profile.get("background"):
            parts.append(f"Background: {user_profile['background']}")
        if user_profile.get("preferred_format"):
            parts.append(f"Preferred format: {user_profile['preferred_format']}")

        if not parts:
            return query

        context_block = "\n".join(parts)
        return (
            f"{query}\n\n"
            f"[User context — use this to tailor your answer specifically to this person:\n"
            f"{context_block}]"
        )
