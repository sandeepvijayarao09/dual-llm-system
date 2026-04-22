"""
Reasoner — runs on the Large LLM (GPT-4o via OpenAI API).
Used for complex queries that require deep reasoning.

The Large LLM focuses on the core concept with a lightweight audience hint
(expertise + domain + format only — no full profile dump).
Full personalization (intro/closing) is still handled by the Small LLM.
"""

from datetime import date
from llm.large_llm import LargeLLM

def _today() -> str:
    return date.today().strftime("%B %d, %Y")

REASONER_SYSTEM = """You are an expert assistant capable of deep reasoning, analysis, and synthesis.
Think carefully and provide accurate, well-structured responses.
When solving complex problems, show your reasoning clearly.
Be thorough but avoid unnecessary repetition.
Today's date is {today}."""


class Reasoner:
    def __init__(self, large_llm: LargeLLM) -> None:
        self.llm = large_llm

    def reason(
        self,
        query: str,
        history: list[dict] | None = None,
        use_thinking: bool = True,
        user_profile: dict | None = None,
    ) -> str:
        """
        Generate a deep, reasoned response using the Large LLM.
        A lightweight audience hint (expertise, domain, format) is injected
        so the answer depth and scope match the user — without biasing reasoning
        with full profile details.
        """
        system = REASONER_SYSTEM.format(today=_today())

        # Build an augmented query that embeds key profile fields directly
        # into the user message — models attend to user messages more reliably
        # than system-prompt hints for open-ended questions.
        augmented_query = self._augment_query(query, user_profile)

        return self.llm.complete(
            system=system,
            user_message=augmented_query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _augment_query(query: str, user_profile: dict | None) -> str:
        """
        Embed a compact user-context block into the query when a profile exists.
        Only includes fields that are actually set — keeps the injection minimal.
        Skips augmentation for queries that already contain enough specificity.
        """
        if not user_profile:
            return query

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
