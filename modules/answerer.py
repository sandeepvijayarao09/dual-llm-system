"""
Simple Answerer — runs on the Small LLM (local Ollama).
Used for queries classified as "simple" that don't need the Large LLM.
"""

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
            system += self._profile_hint(user_profile)

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            max_tokens=512,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _profile_hint(self, profile: dict) -> str:
        hints = []
        tone = profile.get("tone")
        length = profile.get("length")
        expertise = profile.get("expertise")
        if tone:
            hints.append(f"Use a {tone} tone.")
        if length == "brief":
            hints.append("Keep the answer very short.")
        elif length == "detailed":
            hints.append("Provide a thorough answer.")
        if expertise == "expert":
            hints.append("Assume the user has expert knowledge.")
        elif expertise == "novice":
            hints.append("Explain clearly without jargon.")
        return (" " + " ".join(hints)) if hints else ""
