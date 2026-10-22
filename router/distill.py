"""
distill.py — Grow the router's training set by distilling the LLM classifier.

Usage:
    python -m router.distill [--per-call 20] [--out data/distilled_queries.jsonl]

1. The Small LLM (default: local Gemma 4 E4B) writes new user queries for a
   fixed list of topics. Each topic is asked for twice: once for queries that
   are short but genuinely hard, once for queries that are long but easy.
   Those are the cases the word-count shortcut gets wrong (FINDINGS.md §4).
2. The same model labels every query through modules.classifier with the
   orchestrator's confidence rule, i.e. exactly what the cascade's fallback
   layer would decide.
3. Queries that duplicate or closely paraphrase an Eval-1000 or seed query are
   dropped (normalised exact match, or character TF-IDF cosine >= 0.8), so the
   eval stays held out.

The generator never sees Eval-1000. Free with local Gemma; billed if
SMALL_PROVIDER points at a paid model.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from llm.small_llm import SmallLLM
from modules.classifier import Classifier
from router.eval_data import load_cases
from router.live_cascade_eval import final_route
from router.seed_data import load_seed

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "distilled_queries.jsonl")
NEAR_DUP_COSINE = 0.8

TOPICS = [
    "greetings and small talk", "unit conversions", "world geography facts",
    "history dates and people", "definitions of everyday words", "basic arithmetic",
    "algebra and calculus", "probability and statistics", "mathematical proofs",
    "Python coding", "JavaScript and web development", "SQL and databases",
    "debugging an error message", "distributed systems design", "machine learning",
    "cooking and recipes", "personal finance", "health and fitness",
    "travel planning", "career advice", "writing emails", "poems and jokes",
    "ethics and philosophy", "physics", "chemistry", "biology", "economics",
    "law and contracts", "product and business strategy", "movies, books and music",
]

STYLES = {
    "short_hard": ("SHORT (3 to 8 words) but genuinely HARD: they need multi-step "
                   "reasoning, a proof, a design, or expert analysis"),
    "long_easy": ("LONG (15 to 40 words, with chatty context or politeness) but "
                  "genuinely EASY: a single fact, a quick lookup, or a one-line answer"),
}

GEN_SYSTEM = (
    "You write realistic messages that users type into an AI assistant. "
    "Return a JSON object {\"queries\": [...]} with exactly the requested number "
    "of distinct strings. No numbering, no commentary."
)


def _norm(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def generate(llm: SmallLLM, per_call: int) -> list[tuple[str, str, str]]:
    out = []
    for topic in TOPICS:
        for style, desc in STYLES.items():
            prompt = f"Write {per_call} user messages about {topic}. Every message must be {desc}."
            raw = llm.complete(GEN_SYSTEM, prompt, max_tokens=2048, json_mode=True)
            try:
                qs = json.loads(raw).get("queries", [])
            except (json.JSONDecodeError, AttributeError):
                qs = []
            out += [(q.strip(), topic, style) for q in qs if isinstance(q, str) and q.strip()]
            print(f"  generated {len(out)} ({topic} / {style})", flush=True)
    return out


def drop_near_duplicates(rows, reference: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ref_norm = {_norm(q) for q in reference}
    seen: set[str] = set()
    unique = []
    for row in rows:
        n = _norm(row[0])
        if n and n not in ref_norm and n not in seen:
            seen.add(n)
            unique.append(row)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(reference + [r[0] for r in unique])
    sims = cosine_similarity(vec.transform([r[0] for r in unique]), vec.transform(reference)).max(axis=1)
    kept = [r for r, s in zip(unique, sims) if s < NEAR_DUP_COSINE]
    return kept, len(rows) - len(unique), len(unique) - len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--per-call", type=int, default=20)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    llm = SmallLLM()
    print(f"Generating with {llm.provider_name}/{llm.model} "
          f"({len(TOPICS)} topics x {len(STYLES)} styles x {args.per_call})")
    rows = generate(llm, args.per_call)

    reference = [q for q, _, _ in load_cases("eval_1000.jsonl")] + [q for q, _ in load_seed()]
    rows, exact, near = drop_near_duplicates(rows, reference)
    print(f"Dropped {exact} exact duplicates and {near} near-duplicates of eval/seed queries; "
          f"labelling {len(rows)}")

    classifier = Classifier(llm)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (q, topic, style) in enumerate(rows, 1):
            clf = classifier.classify(q)
            f.write(json.dumps({
                "query": q, "label": final_route(clf), "topic": topic, "style": style,
                "llm_decision": clf.routing_decision, "llm_confidence": clf.confidence,
                "parsed": bool(clf.raw), "labeler": f"{llm.provider_name}/{llm.model}",
            }) + "\n")
            if i % 100 == 0 or i == len(rows):
                print(f"  labelled {i}/{len(rows)}", flush=True)
    print(f"Saved {len(rows)} rows to {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
