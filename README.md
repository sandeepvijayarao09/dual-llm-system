<div align="center">

# 🧠 Dual LLM System

**Intelligent Query Routing · Persistent User Profiles · Self-Improving ML Router**

A production-style dual-LLM architecture that routes every query through a 3-layer decision engine — routing simple queries to GPT-4o-mini and complex ones to GPT-4o — while personalizing responses using persistent user profiles and a sliding-window conversation memory.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Router-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Profile_DB-003B57?style=flat-square&logo=sqlite&logoColor=white)

</div>

---

## What It Does

Most LLM apps send every query to the most expensive model available — even `"hi"` or `"what is 2+2"`. This system solves that with a **3-layer routing cascade** that picks the right model for every query at zero extra latency, while personalizing every response using a persistent profile of who the user is.

### Key Numbers

| Metric | Value |
|---|---|
| ML Router accuracy | **92.6%** (1000-case eval) |
| Cascade rate (ML → LLM fallback) | **14.7%** |
| ML router handles alone | **85.3%** of queries (zero API cost for routing) |
| Confidence improvement | 0.706 → **0.764** mean |
| Test cases evaluated | **1,500** (500 + 1000) |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 0 — Pre-flight check                             │
│  query > 2000 words? → skip everything → Large LLM      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — ML Router  (offline, ~0ms, no API call)      │
│  TF-IDF (1-3 grams, 5000 features)                      │
│  + 11 hand-crafted feature tags                         │
│  + Logistic Regression (balanced class weights)         │
│                                                         │
│  confidence ≥ 0.65 → use decision directly              │
│  confidence < 0.65 → cascade to Layer 2                 │
└─────────────────────────────────────────────────────────┘
    │ low confidence (~15% of queries)
    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2 — LLM Classifier  (GPT-4o-mini, JSON mode)    │
│  Returns: complexity, intent, confidence,               │
│           requires_tools, sensitive,                    │
│           routing_decision, profile_relevant            │
│                                                         │
│  confidence < 0.75 → force "big"                        │
│  parse failure     → default "big"                      │
└─────────────────────────────────────────────────────────┘
    │
    ├─── "small" ──────────────────────────────────────────►  GPT-4o-mini
    │    + profile context (if profile_relevant=true)          SimpleAnswerer
    │    + name stub (always, for greeting personalization)     ≤ 512 tokens
    │
    └─── "big" ────────────────────────────────────────────►  GPT-4o
         Step 1: Reasoner (core answer, audience hint)         ≤ 4096 tokens
         Step 2: Summarizer → 1-line summary (GPT-4o-mini)
         Step 3: Personalizer → intro + closing (GPT-4o-mini)
         Step 4: Assemble: intro + core + "---" + closing
```

---

## Feature Engineering (ML Router Tags)

The ML router encodes each query as a tag-augmented string before TF-IDF:

```
encode_text("why is the sky blue?")
→ "LEN_SHORT HAS_REASONING_KW Q_SINGLE why is the sky blue?"
```

| Tag | Signal | Routing hint |
|---|---|---|
| `LEN_SHORT` | ≤ 8 words | → small |
| `LEN_MEDIUM` | 9–40 words | neutral |
| `LEN_LONG` | > 40 words | → big |
| `HAS_CODE` | `` ``` ``, `def `, `class `, `import ` | → big |
| `HAS_MATH` | digits+ops, `\int`, `\sum` | → big |
| `HAS_REASONING_KW` | why, how, compare, prove, explain, analyze… | → big |
| `HAS_FACTUAL_KW` | what is, who is, when did, define… | → small |
| `HAS_GREETING_KW` | hi, hello, thanks… | → small |
| `HAS_CREATIVE_KW` | joke, poem, story, haiku, pun… | → small |
| `HAS_OPINION_KW` | best, vs, should I, pros and cons… | → big |
| `HAS_DEEP_WHAT` | "what is" + >5 words (abstract) | → big |
| `HAS_IMPACT_KW` | impact, effect, consequence… | → big |
| `HAS_RESEARCH_KW` | research, literature, studies show… | → big |
| `HAS_SELF_REF` | my name, who am I, about me… | → profile inject |
| `Q_MULTI` | 2+ question marks | → big |
| `MULTI_SENTENCE` | 3+ sentence-ending chars | → big |

---

## ML Router Evaluation Results

### 1000-Case Evaluation (latest)

```
Overall Accuracy  : 92.6%  (↑ from 88.2% baseline)
Cascade Rate      : 14.7%  (↓ from 30.8% baseline)
ML handles alone  : 85.3%  (zero API cost for routing)

Confusion Matrix (positive = 'big'):
  TP = 438   FN = 17
  FP = 57    TN = 488

  Precision : 0.885
  Recall    : 0.963
  F1        : 0.922
```

### Per-Category Accuracy

| Category | Accuracy | Cascade Rate |
|---|---|---|
| analysis | 100% | 5.6% |
| career_learning | 100% | 0.0% |
| comparison_technical | 100% | 6.7% |
| compound_questions | 100% | 2.0% |
| conversions | 100% | 45.0% |
| definitions | 100% | 0.0% |
| domain_specific_deep | 100% | 0.0% |
| math_simple | 100% | 32.7% |
| multi_constraint | 100% | 20.0% |
| proofs | 100% | 10.0% |
| real_world_planning | 100% | 16.0% |
| research_synthesis | 100% | 0.0% |
| short_creative | 100% | 11.1% |
| system_design | 100% | 0.0% |
| factual | 96.1% | 7.9% |
| yes_no | 95.6% | 26.7% |
| ambiguous | 90.0% | 15.0% |
| greetings | 93.8% | 15.6% |
| conversational | 80.0% | 26.7% |
| ethical_philosophical | 75.0% | 20.0% |
| debug_simple | 33.3% | 24.4% *(next target)* |

### Before / After Improvement

| | Accuracy | Cascade Rate |
|---|---|---|
| Baseline (v0, 120 examples) | 88.2% | 30.8% |
| Improved (v1, 280 examples) | **92.6%** | **14.7%** |
| Delta | **+4.4%** | **-16.1%** |

Run the evaluation yourself:
```bash
python -m router.eval_500   # 500 labeled cases
python -m router.eval_1000  # 1000 labeled cases (with before/after table)
```

---

## Personalization System

### User Profiles (SQLite)

Each user has a persistent profile that evolves after every session:

```json
{
  "name": "Alex",
  "expertise": "intermediate",
  "tone": "casual",
  "domain": "Computer Science",
  "interests": ["algorithms", "web development", "machine learning basics"],
  "topics_discussed": ["OOP", "recursion", "REST APIs", "databases"],
  "preferred_format": "bullets",
  "background": "Third-year CS undergraduate...",
  "interaction_count": 5
}
```

**Profile injection rules:**
- `profile_relevant=true` → full profile embedded in user message as `[User context]` block
- `profile_relevant=false` + has name → name-only stub injected (greetings say "Hey Alex!")
- `profile_relevant=false` + no name → no profile sent (pure query to model)
- Self-referential queries (`"what is my name"`, `"what are my interests"`) → always inject regardless of classifier decision

### Conversation Memory (ConversationBuffer)

Per-user sliding window prevents unbounded context growth:

```
Turn 4 (oldest)  ──► evicted → compressed into rolling_summary (≤80 tokens)
Turn 5           ──┐
Turn 6           ──┤  verbatim window (last 3 turn-pairs)
Turn 7 (current) ──┘

History sent to model:
  [system]: [Earlier summary: Alex asked about recursion, REST APIs...]
  [user/asst]: turns 5-7 verbatim
  [user]: current query
```

Configure window size: `CONVERSATION_WINDOW_SIZE=3` in `.env`

---

## Slash Commands

Force a model directly without any routing logic:

| Command | Effect |
|---|---|
| `/large <query>` | Skip all routing → GPT-4o directly |
| `/small <query>` | Skip all routing → GPT-4o-mini directly |

Examples:
```
/large explain the CAP theorem in distributed systems
/small what is 2+2
/large    ← empty query also works
```

Routing badge shows `⚡ Forced → Large LLM (GPT-4o)` in the UI.

---

## Prompt Visibility

Every response in the UI has two expanders:

- **🔍 Routing Details** — routing decision, model used, latency, ML prediction, classifier JSON, buffer state
- **📤 Prompts Sent to Models** — exact system prompt + user message sent to each model call (Reasoner, Summarizer, Personalizer, or SimpleAnswerer)

---

## Self-Improving Router Loop

Every routing decision is logged to `routing_log.db`:

```sql
routing_events (ts, user_id, query, ml_decision, ml_confidence,
                llm_decision, llm_confidence, final_routing, model_used)
```

Export high-confidence labels and retrain:

```bash
# Export rows where ML + LLM agreed (highest-quality labels)
python -c "
from router.classification_logger import ClassificationLogger
log = ClassificationLogger('routing_log.db')
n = log.export_labeled_csv('real_traffic.csv')
print(f'Exported {n} rows')
"

# Retrain on seed data + real traffic
python -m router.train --extra real_traffic.csv
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/sandeepvijayarao09/dual-llm-system.git
cd dual-llm-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY=your_key_here

# 5a. Launch Streamlit UI
streamlit run app.py

# 5b. Or use the CLI
python main.py --user cs_student --verbose
```

### CLI Options

```bash
python main.py                          # guest mode
python main.py --user cs_student        # load saved profile
python main.py --user cs_student --verbose  # show routing metadata
python main.py --demo                   # run 5 demo queries
python main.py --user cs_student --show-profile  # print profile and exit
```

### Pre-seeded Personas

| User ID | Name | Expertise | Domain |
|---|---|---|---|
| `cs_student` | Alex | Intermediate | Computer Science |
| `nasa_engineer` | Dr. Sarah Chen | Expert | Aerospace / Embedded Systems |
| `high_school` | Jordan | Novice | High School / General |

---

## Project Structure

```
dual-llm-system/
├── app.py                      # Streamlit web UI
├── main.py                     # CLI entry point
├── orchestrator.py             # Central routing engine
├── config.py                   # All configuration + env vars
├── requirements.txt
│
├── llm/
│   ├── small_llm.py            # GPT-4o-mini client
│   └── large_llm.py            # GPT-4o client
│
├── modules/
│   ├── classifier.py           # LLM-based query classifier
│   ├── answerer.py             # Simple query handler (Small LLM)
│   ├── reasoner.py             # Deep reasoning handler (Large LLM)
│   ├── personalizer.py         # Intro/closing wrapper (Small LLM)
│   └── profile_updater.py      # Session-end profile extractor
│
├── router/
│   ├── features.py             # Hand-crafted feature tags + encode_text()
│   ├── ml_router.py            # TF-IDF + LogisticRegression pipeline
│   ├── train.py                # Training script
│   ├── seed_data.py            # 280 labeled training examples
│   ├── classification_logger.py # SQLite log for retraining
│   ├── eval_500.py             # 500-case evaluation script
│   ├── eval_1000.py            # 1000-case evaluation script
│   └── models/
│       └── router_v0.joblib    # Trained model (saved pipeline)
│
├── memory/
│   └── conversation_buffer.py  # Sliding window + rolling summary
│
└── db/
    └── profile_db.py           # SQLite CRUD for user profiles
```

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `LARGE_MODEL` | `gpt-4o` | Model for complex queries |
| `SMALL_MODEL` | `gpt-4o-mini` | Model for simple queries + all support tasks |
| `CONFIDENCE_THRESHOLD` | `0.75` | Min LLM classifier confidence to trust "small" |
| `LARGE_INPUT_THRESHOLD` | `2000` | Word count to bypass classification |
| `ML_ROUTER_CONFIDENCE_THRESHOLD` | `0.65` | Min ML router confidence before LLM fallback |
| `ENABLE_ML_ROUTER` | `true` | Enable/disable ML router (falls back to LLM only) |
| `ML_ROUTER_PATH` | `router/models/router_v0.joblib` | Path to trained model |
| `ROUTING_LOG_PATH` | `routing_log.db` | SQLite log for retraining data |
| `CONVERSATION_WINDOW_SIZE` | `3` | Turn-pairs kept verbatim in memory |
| `SMALL_LLM_TIMEOUT` | `60` | API timeout in seconds |
| `LARGE_LLM_TIMEOUT` | `120` | API timeout in seconds |
| `DB_PATH` | `profiles.db` | User profile database path |

---

## Retrain the ML Router

```bash
# Train on seed data only
python -m router.train

# Train with additional real-traffic data
python -m router.train --extra my_data.csv

# CSV format for --extra:
# query,label
# "What is HTTP?",small
# "Design a URL shortener",big
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Small LLM | GPT-4o-mini (OpenAI) |
| Large LLM | GPT-4o (OpenAI) |
| ML Router | scikit-learn — TF-IDF + Logistic Regression |
| Web UI | Streamlit |
| Profile DB | SQLite (raw, no ORM) |
| Routing Log | SQLite (ClassificationLogger) |
| Config | python-dotenv |
| Model persistence | joblib |

---

## Design Principles

1. **Large LLM never sees full profile** — only a lightweight audience hint. Personalization (intro/closing) is handled by the Small LLM after reasoning is complete.

2. **Large LLM output never goes directly to Small LLM** — compressed to 1 sentence first to prevent context overflow.

3. **Profile injection is conditional** — the classifier decides `profile_relevant` per query. Math, greetings, and universal facts skip profile injection entirely.

4. **Self-referential queries always get profile** — `"what is my name"` is detected by `_is_self_referential()` and always receives profile data regardless of classifier decision.

5. **Conservative routing** — when uncertain, always escalate to the better model. Three independent safety gates (pre-flight, confidence threshold, parse failure fallback).

6. **ML router improves over time** — every routing decision is logged. Export, retrain, redeploy in minutes.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
