"""
Shared pytest fixtures + environment shims.

The Large LLM client (Gemini) validates GEMINI_API_KEY in __init__. For
offline test runs we want to instantiate modules without calling any model,
so we set a dummy key before any project import happens and provide a
FakeLLM fixture for mocking completions.
"""

import os
import sys
from pathlib import Path

# Dummy API key so llm clients construct without raising. setdefault() is not
# enough: an exported-but-empty GEMINI_API_KEY would survive it and break the
# integration tests.
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "test-key-not-real"

# Add project root to sys.path so `from memory import ...` works when pytest
# is invoked from any directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import pytest

from llm.usage import record_call


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
        response = self.default_response
        for predicate, scripted in self.scripts:
            if predicate(system, user_message):
                response = scripted
                break
        # Report usage like the real clients do (word counts stand in for tokens).
        record_call(
            self.model, time.perf_counter(),
            input_tokens=len(f"{system} {user_message}".split()),
            output_tokens=len(response.split()),
        )
        return response


@pytest.fixture
def fake_small_llm():
    return FakeLLM(model="fake-small")


@pytest.fixture
def fake_large_llm():
    return FakeLLM(model="fake-large")


@pytest.fixture
def orch(monkeypatch, tmp_path, fake_small_llm, fake_large_llm):
    """Fully wired Orchestrator with fake LLMs + tmp DBs."""
    # Isolate databases
    monkeypatch.setenv("DB_PATH", str(tmp_path / "profiles.db"))
    monkeypatch.setenv("ROUTING_LOG_PATH", str(tmp_path / "routing.db"))

    # Force config reload (it's read at import time, but the orchestrator
    # re-reads constants by module import — we patch them directly below)
    import importlib

    import config
    importlib.reload(config)

    from orchestrator import Orchestrator

    o = Orchestrator()

    # Swap in fakes — we re-init the modules that depended on the real clients
    o.small_llm = fake_small_llm
    o.large_llm = fake_large_llm

    # Rebuild dependents that captured the original clients
    from modules.answerer import SimpleAnswerer
    from modules.classifier import Classifier
    from modules.personalizer import Personalizer
    from modules.profile_updater import ProfileUpdater
    from modules.reasoner import Reasoner

    o.classifier      = Classifier(fake_small_llm)
    o.answerer        = SimpleAnswerer(fake_small_llm)
    o.reasoner        = Reasoner(fake_large_llm)
    o.context_adder   = Personalizer(fake_small_llm)
    o.profile_updater = ProfileUpdater(fake_small_llm)

    # Re-init buffers dict so they also use the fake small LLM
    o.buffers.clear()

    # Re-init logger to tmp path
    from router.classification_logger import ClassificationLogger
    o.routing_log = ClassificationLogger(str(tmp_path / "routing.db"))

    return o
