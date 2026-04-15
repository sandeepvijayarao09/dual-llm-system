"""
Reasoner — runs on the Large LLM (Claude Opus 4.6 via Anthropic API).
Used for complex queries that require deep reasoning.

Personalization is handled HERE by injecting the user profile directly
into the system prompt — so the Large LLM answers in a personalised way
without needing a second LLM pass.
"""

from llm.large_llm import LargeLLM

REASONER_SYSTEM_BASE = """You are an expert assistant capable of deep reasoning, analysis, and synthesis.
Think carefully and provide accurate, well-structured responses.
When solving complex problems, show your reasoning clearly.
Be thorough but avoid unnecessary repetition."""


class Reasoner:
    def __init__(self, large_llm: LargeLLM):
        self.llm = large_llm

    def reason(
        self,
        query: str,
        history: list[dict] | None = None,
        user_profile: dict | None = None,
        use_thinking: bool = True,
    ) -> str:
        """
        Generate a deep, reasoned response using the Large LLM.
        If user_profile is provided, the answer is personalised directly
        by the Large LLM — no second LLM pass needed.
        """
        system = REASONER_SYSTEM_BASE
        if user_profile:
            system += self._build_persona_hint(user_profile)

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_persona_hint(self, profile: dict) -> str:
        """
        Build a compact persona instruction (~50-80 tokens) appended to
        the system prompt. Keeps token overhead minimal.
        """
        parts = []
        name      = profile.get("name")
        expertise = profile.get("expertise")
        tone      = profile.get("tone")
        length    = profile.get("length")
        fmt       = profile.get("format")
        domain    = profile.get("domain")
        interests = profile.get("interests")

        hint = "\n\nUser persona (adapt your answer accordingly):"
        if name:
            hint += f"\n- Name: {name}"
        if expertise == "expert":
            hint += "\n- Expert level: skip basics, use technical terms freely."
        elif expertise == "novice":
            hint += "\n- Novice level: use simple language, avoid jargon, add analogies."
        elif expertise == "intermediate":
            hint += "\n- Intermediate level: balance depth and clarity."
        if domain:
            hint += f"\n- Domain background: {domain}"
        if interests:
            if isinstance(interests, list):
                hint += f"\n- Interests: {', '.join(interests)}"
            else:
                hint += f"\n- Interests: {interests}"
        if tone == "casual":
            hint += "\n- Tone: friendly and conversational."
        elif tone == "formal":
            hint += "\n- Tone: formal and professional."
        elif tone == "technical":
            hint += "\n- Tone: precise and technical."
        if length == "brief":
            hint += "\n- Length: be concise, essentials only."
        elif length == "detailed":
            hint += "\n- Length: thorough and detailed."
        if fmt == "bullets":
            hint += "\n- Format: use bullet points."
        elif fmt == "prose":
            hint += "\n- Format: flowing prose, no bullet points."

        background = profile.get("background")
        if background:
            hint += f"\n- Background: {background}"

        return hint
