"""
Shared pytest fixtures + environment shims.

The LLM clients (llm/small_llm.py, llm/large_llm.py) validate OPENAI_API_KEY
in __init__. For offline test runs we want to instantiate modules without
actually hitting OpenAI — so we set a dummy key before any project import
happens and provide a FakeLLM fixture for mocking completions.
"""

import os
import sys
from pathlib import Path

# Dummy API key so llm clients construct without raising
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

# Add project root to sys.path so `from memory import ...` works when pytest
# is invoked from any directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


class FakeLLM:
    """
    Duck-typed replacement for SmallLLM / LargeLLM.

    Records every `complete()` call for assertions and returns a scripted
    response. Defaults make the summarizer / classifier behave predictably.
    """

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.calls: list[dict] = []
        # Scripted responses — first match wins.
        # Each entry: (predicate(system, user), response_str).
        self.scripts: list = []
        self.default_response = "OK"

    def script(self, predicate, response: str) -> None:
        self.scripts.append((predicate, response))

    def complete(
        self,
        system: str,
        user_message: str,
        history=None,
        max_tokens: int = 512,
        json_mode: bool = False,
        use_thinking: bool = False,
    ) -> str:
        self.calls.append({
            "system": system, "user": user_message,
            "history": history, "max_tokens": max_tokens,
            "json_mode": json_mode,
        })
        for predicate, response in self.scripts:
            if predicate(system, user_message):
                return response
        return self.default_response


@pytest.fixture
def fake_small_llm():
    return FakeLLM(model="fake-small")


@pytest.fixture
def fake_large_llm():
    return FakeLLM(model="fake-large")
