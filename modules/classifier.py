"""
Classifier module — runs on the Small LLM (GPT-4o Mini via OpenAI).

Analyses the incoming query and returns a structured routing decision:
  {
    "complexity":        "simple" | "moderate" | "complex",
    "intent":           "greeting" | "factual" | "math" | "coding" |
                         "reasoning" | "creative" | "other",
    "confidence":        0.0 – 1.0,
    "requires_tools":    true | false,
    "sensitive":         true | false,
    "routing_decision":  "small" | "big"
  }

routing_decision meanings:
  small → answer directly with the Small LLM
  big   → send to Large LLM (personalized via profile injection)
"""

import json
import re
from dataclasses import dataclass, field
from llm.small_llm import SmallLLM

CLASSIFIER_SYSTEM = """You are a query router. Analyse the user query and output ONLY a JSON object — no prose, no markdown.

Output schema (all fields required):
{
  "complexity":       "simple" | "moderate" | "complex",
  "intent":           "greeting" | "factual" | "math" | "coding" | "reasoning" | "creative" | "other",
  "confidence":       <float 0.0–1.0>,
  "requires_tools":   <true|false>,
  "sensitive":        <true|false>,
  "routing_decision": "small" | "big"
}

Routing rules:
- "small" : complexity=simple, confidence≥0.75, no tools needed, not sensitive
- "big"   : complexity=moderate OR complex OR requires_tools=true OR sensitive=true

Output ONLY the raw JSON. No explanation."""


@dataclass
class ClassificationResult:
    complexity: str = "complex"
    intent: str = "other"
    confidence: float = 0.0
    requires_tools: bool = False
    sensitive: bool = False
    routing_decision: str = "big"
    raw: dict = field(default_factory=dict)


class Classifier:
    def __init__(self, small_llm: SmallLLM):
        self.llm = small_llm

    def classify(self, query: str) -> ClassificationResult:
        """Classify the query and return a routing decision."""
        raw_text = self.llm.complete(
            system=CLASSIFIER_SYSTEM,
            user_message=f"Query: {query}",
            max_tokens=256,
            json_mode=True,
        )

        return self._parse(raw_text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse(self, raw_text: str) -> ClassificationResult:
        """Parse the JSON from the model, with a safe fallback."""
        try:
            # Strip any stray markdown code fences
            clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
            data = json.loads(clean)

            return ClassificationResult(
                complexity=data.get("complexity", "complex"),
                intent=data.get("intent", "other"),
                confidence=float(data.get("confidence", 0.0)),
                requires_tools=bool(data.get("requires_tools", False)),
                sensitive=bool(data.get("sensitive", False)),
                routing_decision=data.get("routing_decision", "big"),
                raw=data,
            )
        except (json.JSONDecodeError, ValueError):
            # Safe fallback: route to large LLM
            return ClassificationResult(routing_decision="big")
