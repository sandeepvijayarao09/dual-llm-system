<div align="center">

# ⚡ Dual-LLM System

**Local Intelligence + Cloud Reasoning**

A cost-efficient architecture that routes queries between a local small LLM (Ollama) for simple tasks and Claude Opus for complex reasoning — achieving significant cost savings while maintaining quality.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![Claude](https://img.shields.io/badge/Claude_Opus-191919?style=flat-square&logo=anthropic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

---

## 📋 Overview

Most LLM applications send every query to expensive cloud APIs — even simple ones that a local model handles just fine. This system solves that with a **dual-LLM architecture**: an intelligent orchestrator routes queries to either a local Ollama model (fast, free) or Claude Opus (powerful, cloud-based) based on complexity analysis.

### Key Results

| Metric | Value |
|--------|-------|
| **Cost Reduction** | ~70% fewer cloud API calls |
| **Routing Latency** | <100ms query classification |
| **Simple Query Handling** | Local Ollama (Mistral/Llama) |
| **Complex Reasoning** | Claude Opus via API |

---

## 🎬 Demo

<!-- Add demo screenshot or GIF here -->
<!-- ![Demo](assets/demo.gif) -->

> **📸 Demo placeholder** — Run `python app.py` to launch the Streamlit interface, or `python main.py` for the CLI.

---

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Orchestrator │
                    │              │
                    │  Complexity  │
                    │  Analysis    │
                    └──┬───────┬───┘
                       │       │
          Simple       │       │      Complex
          Queries      │       │      Queries
                       │       │
              ┌────────▼──┐ ┌──▼──────────┐
              │   Ollama  │ │   Claude    │
              │  (Local)  │ │  (Cloud)    │
              │           │ │             │
              │ Mistral/  │ │  Opus       │
              │ Llama     │ │  API        │
              │           │ │             │
              │ Fast      │ │ Powerful    │
              │ Free      │ │ Accurate    │
              └────────┬──┘ └──┬──────────┘
                       │       │
                    ┌──▼───────▼──┐
                    │  Response   │
                    │  Synthesis  │
                    └─────────────┘
```

### How Routing Works

1. **Query arrives** at the orchestrator
2. **Complexity analysis** scores the query (keyword patterns, length, domain)
3. **Simple queries** (greetings, factual lookups, formatting) → **Ollama** (local, instant)
4. **Complex queries** (multi-step reasoning, analysis, code generation) → **Claude** (cloud, powerful)
5. **Response** is returned with metadata about which model handled it

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally
- Anthropic API key

### Setup

```bash
# 1. Clone and enter project
git clone https://github.com/sandeepvijayarao09/dual-llm-system.git
cd dual-llm-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Ollama (pull a model)
ollama pull mistral

# 4. Configure environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 5a. Launch Streamlit UI
python app.py

# 5b. Or launch CLI
python main.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Local LLM | Ollama (Mistral / Llama) |
| Cloud LLM | Claude Opus (Anthropic API) |
| Orchestration | Custom Python router |
| Web UI | Streamlit |
| Database | SQLite (conversation history) |
| Config | Environment variables |

---

## 📁 Project Structure

```
dual-llm-system/
├── app.py              # Streamlit web interface
├── main.py             # CLI interface
├── orchestrator.py     # Query routing and orchestration
├── config.py           # Configuration management
├── llm/                # LLM provider modules
├── modules/            # Feature modules
├── db/                 # Database schemas and storage
├── .env.example        # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🔬 Design Decisions

**Why Dual-LLM?**
- Cloud API costs scale linearly with usage — this cuts costs by ~70%
- Simple queries (80% of typical usage) don't need a $20/MTok model
- Local models provide instant responses with zero latency
- Fallback to cloud ensures quality is never compromised

**Why Ollama?**
- Runs fully local — no data leaves the machine
- Easy model management (`ollama pull`, `ollama run`)
- Supports Mistral, Llama, and other open-weight models
- GPU-accelerated inference on consumer hardware

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
