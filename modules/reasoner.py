"""
Reasoner — runs on the Large LLM (Claude Opus 4.6 via Anthropic API).
Used for complex queries that require deep reasoning.
"""

from llm.large_llm import LargeLLM

REASONER_SYSTEM = """You are an expert assistant capable of deep reasoning, analysis, and synthesis.
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
        use_thinking: bool = True,
    ) -> str:
        """Generate a deep, reasoned response using the Large LLM."""
        return self.llm.complete(
            system=REASONER_SYSTEM,
            user_message=query,
            history=history,
            use_thinking=use_thinking,
            max_tokens=4096,
        )
