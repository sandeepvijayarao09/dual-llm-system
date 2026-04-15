"""
Reasoner — runs on the Large LLM via OpenAI API.
Used for complex queries that require deep reasoning.

Personalization: the raw user profile is passed to the model as-is.
The model decides how to adapt — no hardcoded rules about what
"expert" or "novice" means. The LLM figures it out from the data.
"""

import json
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
        If user_profile is provided, the raw profile is passed directly.
        The model decides how to personalize — nothing is hardcoded.
        """
        system = REASONER_SYSTEM_BASE
        if user_profile:
            system += "\n\nUser profile:\n" + json.dumps(user_profile, indent=2)

        return self.llm.complete(
            system=system,
            user_message=query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )
