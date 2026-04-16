"""
Reasoner — runs on the Large LLM (GPT-4o via OpenAI API).
Used for complex queries that require deep reasoning.

The Large LLM focuses ONLY on the core concept — no user profile,
no personalization. It delivers the purest, most accurate answer.
Personalization is handled entirely by the Small LLM.
"""

from llm.large_llm import LargeLLM

REASONER_SYSTEM = """You are an expert assistant capable of deep reasoning, analysis, and synthesis.
Think carefully and provide accurate, well-structured responses.
When solving complex problems, show your reasoning clearly.
Be thorough but avoid unnecessary repetition.
Focus purely on the core concept. Do not tailor your answer to any specific audience."""


class Reasoner:
    def __init__(self, large_llm: LargeLLM) -> None:
        self.llm = large_llm

    def reason(
        self,
        query: str,
        history: list[dict] | None = None,
        use_thinking: bool = True,
    ) -> str:
        """
        Generate a deep, reasoned response using the Large LLM.
        No user profile is passed — the Large LLM focuses purely
        on delivering the best core answer.
        """
        return self.llm.complete(
            system=REASONER_SYSTEM,
            user_message=query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )
