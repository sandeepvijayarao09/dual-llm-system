"""
ContextAdder — runs on the Small LLM.

Adds a short personalised context note to any answer (small or large LLM).

KEY DESIGN:
  - Never receives the Large LLM output (avoids large input problem).
  - Only receives: user query + raw user profile.
  - The model reads the profile and decides what context to add.
  - No hardcoded rules — the LLM figures out personalization from the data.
"""

import json
from llm.small_llm import SmallLLM

CONTEXT_ADDER_SYSTEM = """You are a personalisation assistant.

You will receive a user's profile and their query.
Read the profile and write a short context note (2-3 sentences MAX)
that connects the topic to who this person is.

Rules:
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
        and raw user profile. Does NOT receive or process the main answer.
        Returns empty string if no profile is provided.
        """
        if not user_profile:
            return ""

        prompt = (
            f"User profile:\n{json.dumps(user_profile, indent=2)}\n\n"
            f"User asked about: {query}"
        )

        result = self.llm.complete(
            system=CONTEXT_ADDER_SYSTEM,
            user_message=prompt,
            max_tokens=150,
        )

        return result.strip() if result else ""
