"""
Personalizer — runs on the Small LLM (local Ollama).

Takes the Large LLM's raw output and applies stylistic transformation
based on the user's profile.

CRITICAL CONSTRAINT: Never alter facts, numbers, or specific claims.
Only adjust tone, structure, length, and format.
"""

from llm.small_llm import SmallLLM

PERSONALIZER_SYSTEM = """You are a text formatter and style adapter.

Your ONLY job is to reformat or rephrase the provided text to match the requested style.

STRICT RULES — never break these:
1. Do NOT change any facts, numbers, dates, names, or specific claims.
2. Do NOT add new information that wasn't in the original.
3. Do NOT remove key information unless the user explicitly wants a shorter version.
4. ONLY change: tone, sentence structure, paragraph layout, length, and formatting style.

Output ONLY the reformatted text. No preamble like "Here is the reformatted text:"."""


class Personalizer:
    def __init__(self, small_llm: SmallLLM):
        self.llm = small_llm

    def personalize(
        self,
        raw_response: str,
        user_profile: dict | None = None,
        original_query: str = "",
    ) -> str:
        """
        Reformat the Large LLM's response to match user preferences.
        Returns the raw response unchanged if no profile is provided.
        """
        if not user_profile:
            return raw_response

        style_request = self._build_style_request(user_profile)
        if not style_request:
            return raw_response

        prompt = (
            f"Original query: {original_query}\n\n"
            f"Text to reformat:\n{raw_response}\n\n"
            f"Style instructions: {style_request}"
        )

        # Cap output at 110 % of input length to prevent expansion
        max_tokens = int(len(raw_response.split()) * 1.1) + 100
        max_tokens = min(max_tokens, 2048)

        result = self.llm.complete(
            system=PERSONALIZER_SYSTEM,
            user_message=prompt,
            max_tokens=max_tokens,
        )

        return result if result else raw_response

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_style_request(self, profile: dict) -> str:
        parts = []
        tone = profile.get("tone")
        length = profile.get("length")
        fmt = profile.get("format")
        expertise = profile.get("expertise")

        if tone == "casual":
            parts.append("Use a friendly, casual conversational tone.")
        elif tone == "formal":
            parts.append("Use a formal, professional tone.")
        elif tone == "technical":
            parts.append("Use precise technical language.")

        if length == "brief":
            parts.append("Shorten the response significantly, keeping only the essentials.")
        elif length == "detailed":
            parts.append("Keep all detail; structure it clearly.")

        if fmt == "bullets":
            parts.append("Use bullet points for the main content.")
        elif fmt == "prose":
            parts.append("Use flowing prose paragraphs, no bullet points.")
        elif fmt == "plain":
            parts.append("Use plain text with no markdown formatting.")

        if expertise == "novice":
            parts.append("Simplify technical terms and add brief explanations.")
        elif expertise == "expert":
            parts.append("Assume expert knowledge; skip basic explanations.")

        return " ".join(parts)
