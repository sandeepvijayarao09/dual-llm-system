"""
Configuration for the Dual LLM System.
- Large LLM  : Gemini 2.0 Pro via Google GenAI API
- Small LLM  : Gemini 2.5 Flash via Google GenAI API
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Google GenAI (both LLMs) ─────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LARGE_MODEL    = os.getenv("LARGE_MODEL",  "gemini-2.0-pro-exp")
SMALL_MODEL    = os.getenv("SMALL_MODEL",  "gemini-2.5-flash-preview-04-17")

# ── Routing thresholds ───────────────────────────────────────────────────────
# Minimum classifier confidence to trust a "simple" routing decision
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Token count above which we skip classification and go straight to Large LLM
LARGE_INPUT_THRESHOLD = int(os.getenv("LARGE_INPUT_THRESHOLD", "2000"))

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
SMALL_LLM_TIMEOUT = int(os.getenv("SMALL_LLM_TIMEOUT", "60"))
LARGE_LLM_TIMEOUT = int(os.getenv("LARGE_LLM_TIMEOUT", "120"))
