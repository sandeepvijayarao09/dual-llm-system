"""
Feature extraction for the ML router.

Strategy:
    We augment the raw query with a prefix of tag-tokens that encode hand
    features (e.g. "LEN_SHORT HAS_CODE HAS_MATH"). TF-IDF then picks these up
    alongside the actual text. This avoids a custom sklearn transformer and
    keeps serialization trivial (a single sklearn Pipeline fits everything).

Tag taxonomy:
    LEN_SHORT   : ≤ 8 words       (usually greetings / lookup)
    LEN_MEDIUM  : 9-40 words
    LEN_LONG    : > 40 words      (usually context-heavy / rewrite requests)
    HAS_CODE    : contains ``` or `def ` or `class ` or `import `
    HAS_MATH    : contains =, +, -, *, /, integral, sum, or digits+op
    HAS_REASONING_KW : why, how, compare, prove, explain, analyze, derive,
                      justify, contrast, evaluate, reason
    HAS_FACTUAL_KW   : what is, who is, when did, where is, define
    HAS_GREETING_KW  : hi, hello, hey, thanks, thank you, good morning
    Q_MULTI     : multiple '?' in query (compound question)
"""

from __future__ import annotations

import re

REASONING_KW = {
    "why", "how", "compare", "prove", "explain", "analyze", "analyse",
    "derive", "justify", "contrast", "evaluate", "reason", "design",
    "architect", "optimize", "optimise", "debug",
}

FACTUAL_KW_PATTERNS = [
    r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhen\s+(did|was|is)\b",
    r"\bwhere\s+is\b", r"\bdefine\b", r"\bcapital\s+of\b",
]

GREETING_KW_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|sup)\b", r"\bthank(s| you)\b",
    r"\bgood\s+(morning|afternoon|evening|night)\b",
]

CODE_PATTERNS = [
    r"```", r"\bdef\s+\w+", r"\bclass\s+\w+", r"\bimport\s+\w+",
    r"\bfunction\s+\w+", r"=>", r"#include",
]

MATH_PATTERNS = [
    r"\b\d+\s*[+\-*/=]\s*\d+", r"\\int", r"\\sum", r"\\frac",
    r"\bintegral\b", r"\bderivative\b", r"\bprobability\b",
]


def _has_any(text_lower: str, patterns: list[str]) -> bool:
    return any(re.search(p, text_lower) for p in patterns)


def _has_reasoning_kw(tokens_lower: list[str]) -> bool:
    return any(t in REASONING_KW for t in tokens_lower)


def encode_text(query: str) -> str:
    """
    Return the query augmented with hand-feature tag tokens.

    Example:
        >>> encode_text("why is the sky blue?")
        'LEN_SHORT HAS_REASONING_KW Q_SINGLE why is the sky blue?'
    """
    if not query or not query.strip():
        return "LEN_SHORT EMPTY "
    q = query.strip()
    q_lower = q.lower()
    tokens_lower = re.findall(r"\w+", q_lower)
    word_count = len(tokens_lower)

    tags: list[str] = []

    # Length bucket
    if word_count <= 8:
        tags.append("LEN_SHORT")
    elif word_count <= 40:
        tags.append("LEN_MEDIUM")
    else:
        tags.append("LEN_LONG")

    # Content tags
    if _has_any(q, CODE_PATTERNS):
        tags.append("HAS_CODE")
    if _has_any(q_lower, MATH_PATTERNS):
        tags.append("HAS_MATH")
    if _has_reasoning_kw(tokens_lower):
        tags.append("HAS_REASONING_KW")
    if _has_any(q_lower, FACTUAL_KW_PATTERNS):
        tags.append("HAS_FACTUAL_KW")
    if _has_any(q_lower, GREETING_KW_PATTERNS):
        tags.append("HAS_GREETING_KW")

    # Question marks
    q_count = q.count("?")
    if q_count >= 2:
        tags.append("Q_MULTI")
    elif q_count == 1:
        tags.append("Q_SINGLE")

    # Multi-sentence signal (rough)
    if q.count(".") + q.count("?") + q.count("!") >= 3:
        tags.append("MULTI_SENTENCE")

    return " ".join(tags) + " " + q_lower
