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
    HAS_CREATIVE_KW  : joke, poem, story, pun, haiku, limerick, riddle,
                       tagline, metaphor, rhyme, fiction, narrative,
                       creative, write a
    HAS_OPINION_KW   : best, better, worse, should i, recommend, vs,
                       versus, opinion, preferred, favorite, worth it,
                       pros and cons, trade-off, tradeoff
    HAS_DEEP_WHAT    : query starts with "what is" AND word_count > 5
                       (signals philosophical/opinion question, not a
                       simple factual lookup)
    HAS_IMPACT_KW    : impact, effect, implication, consequence, affect,
                       influence, cause, result in, lead to
    HAS_RESEARCH_KW  : research, literature, studies show, paper, findings,
                       evidence, survey, meta-analysis, according to
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

SELF_REF_PATTERNS = [
    r"\bmy\s+(name|background|interests|domain|expertise|profile|goals?|skills?)\b",
    r"\bwho\s+am\s+i\b",
    r"\bwhat\s+(do\s+i|am\s+i|should\s+i)\b",
    r"\babout\s+me\b",
]

CREATIVE_KW_PATTERNS = [
    r"\bjoke\b", r"\bpoem\b", r"\bstory\b", r"\bpun\b",
    r"\bhaiku\b", r"\blimerick\b", r"\briddle\b", r"\btagline\b",
    r"\bmetaphor\b", r"\brhyme\b", r"\bfiction\b", r"\bnarrative\b",
    r"\bcreative\b", r"\bwrite\s+a\b",
]

OPINION_KW_PATTERNS = [
    r"\bbest\b", r"\bbetter\b", r"\bworse\b", r"\bshould\s+i\b",
    r"\brecommend\b", r"\bvs\b", r"\bversus\b", r"\bopinion\b",
    r"\bpreferred\b", r"\bfavorite\b", r"\bfavourite\b",
    r"\bworth\s+it\b", r"\bpros\s+and\s+cons\b",
    r"\btrade-off\b", r"\btradeoff\b",
]

IMPACT_KW_PATTERNS = [
    r"\bimpact\b", r"\beffect\b", r"\bimplication\b", r"\bconsequence\b",
    r"\baffect\b", r"\binfluence\b", r"\bcause\b", r"\bresult\s+in\b",
    r"\blead\s+to\b",
]

RESEARCH_KW_PATTERNS = [
    r"\bresearch\b", r"\bliterature\b", r"\bstudies\s+show\b",
    r"\bpaper\b", r"\bfindings\b", r"\bevidence\b", r"\bsurvey\b",
    r"\bmeta-analysis\b", r"\baccording\s+to\b",
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
    if _has_any(q_lower, SELF_REF_PATTERNS):
        tags.append("HAS_SELF_REF")

    # New tags — creative, opinion, deep-what, impact, research
    if _has_any(q_lower, CREATIVE_KW_PATTERNS):
        tags.append("HAS_CREATIVE_KW")
    if _has_any(q_lower, OPINION_KW_PATTERNS):
        tags.append("HAS_OPINION_KW")
    # HAS_DEEP_WHAT: "what is ..." with more than 5 words total
    # signals an abstract philosophical question, not a simple factual lookup
    if re.search(r"^\s*what\s+is\b", q_lower) and word_count > 5:
        tags.append("HAS_DEEP_WHAT")
    if _has_any(q_lower, IMPACT_KW_PATTERNS):
        tags.append("HAS_IMPACT_KW")
    if _has_any(q_lower, RESEARCH_KW_PATTERNS):
        tags.append("HAS_RESEARCH_KW")

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
