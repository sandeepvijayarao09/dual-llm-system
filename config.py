"""
Configuration for the Dual LLM System.
- Large LLM  : GPT-5 via OpenAI API
- Small LLM  : GPT-4o Mini via OpenAI API
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI (both LLMs) ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LARGE_MODEL    = os.getenv("LARGE_MODEL",  "gpt-4o")
SMALL_MODEL    = os.getenv("SMALL_MODEL",  "gpt-4o-mini")

# ── Routing thresholds ───────────────────────────────────────────────────────
# Minimum classifier confidence to trust a "simple" routing decision
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# Token count above which we skip classification and go straight to Large LLM
LARGE_INPUT_THRESHOLD = int(os.getenv("LARGE_INPUT_THRESHOLD", "2000"))

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
SMALL_LLM_TIMEOUT = int(os.getenv("SMALL_LLM_TIMEOUT", "60"))
LARGE_LLM_TIMEOUT = int(os.getenv("LARGE_LLM_TIMEOUT", "120"))

# ── Profile Database ──────────────────────────────────────────────────────────
# SQLite file path for storing user profiles
DB_PATH = os.getenv("DB_PATH", "profiles.db")
