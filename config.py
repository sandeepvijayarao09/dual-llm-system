"""
Configuration for the Dual LLM System.
- Large LLM  : Claude Opus 4.6 via Anthropic API
- Small LLM  : Local model via Ollama (http://localhost:11434)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic (Large LLM) ────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LARGE_MODEL = "claude-opus-4-6"

# ── Ollama (Small LLM) ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Change to any model you have pulled: phi3:mini, llama3.2:3b, mistral, etc.
SMALL_MODEL = os.getenv("SMALL_MODEL", "llama3.2:3b")

# ── Routing thresholds ───────────────────────────────────────────────────────
# Minimum classifier confidence to trust a "simple" routing decision
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Token count above which we skip classification and go straight to Large LLM
LARGE_INPUT_THRESHOLD = int(os.getenv("LARGE_INPUT_THRESHOLD", "2000"))

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
SMALL_LLM_TIMEOUT = int(os.getenv("SMALL_LLM_TIMEOUT", "60"))
LARGE_LLM_TIMEOUT = int(os.getenv("LARGE_LLM_TIMEOUT", "120"))
