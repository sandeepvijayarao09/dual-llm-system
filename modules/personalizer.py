"""
ContextAdder — runs on the Small LLM (local Ollama).

Adds a short personalised context note to any answer (small or large LLM).

KEY DESIGN:
  - Never receives the Large LLM output (avoids large input problem).
  - Only receives: user query + user profile (~150 tokens total).
  - Outputs: 2-3 sentence personalised note that connects the topic
    to the user's background, experience, or interests.
"""

from llm.small_llm import SmallLLM

CONTEXT_ADDER_SYSTEM = """You are a personalisation assistant.

Your ONLY job is to write a short context note (2-3 sentences MAX) that:
1. Connects the topic of the user's query to their personal background or interests.
2. Adds relevant insight based on their expertise level.
3. Makes the answer feel tailored specifically to them.

STRICT RULES:
- Do NOT repeat or summarise the main answer.
- Do NOT answer the question yourself.
- Do NOT exceed 3 sentences.
- Output ONLY the context note. No preamble."""


class Personalizer:
    """
    Renamed internally to ContextAdder but kept as Personalizer
    for backwards compatibility with orchestrator imports.
    """

    def __init__(self, small_llm: SmallLLM):
        self.llm = small_llm

    def add_context(
        self,
        query: str,
        user_profile: dict | None = None,
    ) -> str:
        """
        Generate a short personalised context note based on the query
        and user profile. Does NOT receive or process the main answer.
        Returns empty string if no profile is provided.
        """
        if not user_profile:
            return ""

        profile_summary = self._build_profile_summary(user_profile)
        if not profile_summary:
            return ""

        prompt = (
            f"User profile: {profile_summary}\n\n"
            f"User asked about: {query}\n\n"
            "Write a short personalised context note (2-3 sentences) "
            "that connects this topic to the user's background."
        )

        result = self.llm.complete(
            system=CONTEXT_ADDER_SYSTEM,
            user_message=prompt,
            max_tokens=150,
        )

        return result.strip() if result else ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_profile_summary(self, profile: dict) -> str:
        """Build a compact profile string to inject into the prompt."""
        parts = []
        expertise = profile.get("expertise")
        tone      = profile.get("tone")
        interests = profile.get("interests")
        domain    = profile.get("domain")
        name      = profile.get("name")

        if name:
            parts.append(f"Name: {name}")
        if expertise:
            parts.append(f"Expertise: {expertise}")
        if domain:
            parts.append(f"Domain: {domain}")
        if interests:
            if isinstance(interests, list):
                parts.append(f"Interests: {', '.join(interests)}")
            else:
                parts.append(f"Interests: {interests}")
        if tone:
            parts.append(f"Preferred tone: {tone}")

        return " | ".join(parts)
