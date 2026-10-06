"""
Ablation + external baselines for the CascadeRoute ML router.

Answers three questions the original evaluation could not:

  1. What does tag augmentation *alone* contribute?  The original Table 3
     compared v0 (120 ex, unigram, no tags, Eval-500) against v1 (239 ex,
     1-3 gram, tags, Eval-1000) and attributed the whole delta to tags.
     Here we hold the training set, hyperparameters and eval set fixed and
     vary only {tags on/off} x {unigram / 1-3 gram}.

  2. How does the router compare against trivial baselines?  Accuracy is
     meaningless without them.

  3. What fraction of queries reach the *large model*?  The paper reports
     cascade rate (routing overhead) but never the large-model rate, which
     is what dominates monetary cost.

Run:  .venv/bin/python -m router.ablation
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from router.features import encode_text
from router.seed_data import load_seed

CASCADE_THRESHOLD = 0.65
SEED = 42
N_BOOTSTRAP = 2000

# ── data ──────────────────────────────────────────────────────────────────────

CASE_RE = re.compile(
    r'\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"(small|big)"\s*,\s*"([a-z_]+)"\s*\)'
)


def load_eval(path: str) -> list[tuple[str, str, str]]:
    return [(q, l, c) for q, l, c in CASE_RE.findall(open(path).read())]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ── representations ───────────────────────────────────────────────────────────

def rep_tags(q: str) -> str:
    """Production representation: tag tokens + lowercased query."""
    return encode_text(q)


def rep_notags(q: str) -> str:
    """Control: lowercased query only. encode_text() appends q.lower(), so
    stripping the tag prefix is exactly this."""
    return q.strip().lower() if q and q.strip() else ""


# ── metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    accuracy: float
    acc_lo: float
    acc_hi: float
    precision_big: float
    recall_big: float
    f1_big: float
    large_model_rate: float
    cascade_rate: float | None
    n: int


def _prf(y_true: list[str], y_pred: list[str]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "big" and p == "big")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "small" and p == "big")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "big" and p == "small")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _bootstrap_ci(correct: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(correct)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def score(name: str, y_true: list[str], y_pred: list[str],
          conf: np.ndarray | None, rng: np.random.Generator) -> Result:
    correct = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    lo, hi = _bootstrap_ci(correct, rng)
    prec, rec, f1 = _prf(y_true, y_pred)
    return Result(
        name=name,
        accuracy=float(correct.mean()),
        acc_lo=lo, acc_hi=hi,
        precision_big=prec, recall_big=rec, f1_big=f1,
        large_model_rate=sum(1 for p in y_pred if p == "big") / len(y_pred),
        cascade_rate=(float((conf < CASCADE_THRESHOLD).mean())
                      if conf is not None else None),
        n=len(y_true),
    )


# ── model ─────────────────────────────────────────────────────────────────────

def build(ngram: tuple[int, int]) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=1,
                                  max_features=5000, sublinear_tf=True,
                                  analyzer="word")),
        ("clf", LogisticRegression(C=2.0, class_weight="balanced",
                                   max_iter=2000, solver="lbfgs")),
    ])


def fit_predict(repfn, ngram, train, eval_q):
    pipe = build(ngram)
    pipe.fit([repfn(q) for q, _ in train], [l for _, l in train])
    X = [repfn(q) for q in eval_q]
    pred = list(pipe.predict(X))
    conf = pipe.predict_proba(X).max(axis=1)
    return pred, conf


# ── baselines ─────────────────────────────────────────────────────────────────

BIG_HINT_TAGS = {"HAS_CODE", "HAS_MATH", "HAS_REASONING_KW", "HAS_OPINION_KW",
                 "HAS_DEEP_WHAT", "HAS_IMPACT_KW", "HAS_RESEARCH_KW",
                 "Q_MULTI", "MULTI_SENTENCE", "LEN_LONG"}


def baseline_tag_heuristic(q: str) -> str:
    tags = {t for t in encode_text(q).split() if t.isupper() and t.isidentifier()}
    return "big" if tags & BIG_HINT_TAGS else "small"


def baseline_length(q: str, thresh: int) -> str:
    return "small" if len(re.findall(r"\w+", q)) <= thresh else "big"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)
    train = load_seed()
    ev = load_eval("router/eval_1000.py")
    train_norm = {norm(q) for q, _ in train}
    clean = [(q, l, c) for q, l, c in ev if norm(q) not in train_norm]

    print(f"train={len(train)}  eval={len(ev)}  "
          f"eval_clean(no train overlap)={len(clean)}\n")

    for label, dataset in (("FULL EVAL-1000", ev), ("CLEAN (train overlap removed)", clean)):
        qs = [q for q, _, _ in dataset]
        ys = [l for _, l, _ in dataset]
        rows: list[Result] = []

        # ── trivial baselines
        rows.append(score("all-small", ys, ["small"] * len(ys), None, rng))
        rows.append(score("all-big", ys, ["big"] * len(ys), None, rng))
        r = np.random.default_rng(SEED)
        rows.append(score("random (p=0.5)", ys,
                          list(r.choice(["small", "big"], size=len(ys))), None, rng))
        best = max(range(3, 25),
                   key=lambda t: np.mean([baseline_length(q, t) == y
                                          for q, y in zip(qs, ys)]))
        rows.append(score(f"length threshold (<={best} words -> small)", ys,
                          [baseline_length(q, best) for q in qs], None, rng))
        rows.append(score("tag heuristic (rules only, no learning)", ys,
                          [baseline_tag_heuristic(q) for q in qs], None, rng))

        # ── 2x2 ablation, identical training set + hyperparameters
        for ng, ngname in (((1, 1), "unigram"), ((1, 3), "1-3 gram")):
            for repfn, rname in ((rep_notags, "no tags"), (rep_tags, "+ tags")):
                pred, conf = fit_predict(repfn, ng, train, qs)
                rows.append(score(f"LR + TF-IDF {ngname}, {rname}", ys, pred, conf, rng))

        print(f"══ {label}  (n={len(ys)}) ".ljust(100, "═"))
        hdr = (f"{'system':46s} {'acc':>6s} {'95% CI':>15s} {'F1(big)':>8s} "
               f"{'large%':>7s} {'casc%':>6s}")
        print(hdr); print("-" * len(hdr))
        for x in rows:
            ci = f"[{x.acc_lo:.3f},{x.acc_hi:.3f}]"
            cs = f"{x.cascade_rate*100:5.1f}" if x.cascade_rate is not None else "    -"
            print(f"{x.name:46s} {x.accuracy:6.3f} {ci:>15s} {x.f1_big:8.3f} "
                  f"{x.large_model_rate*100:6.1f}% {cs:>6s}")
        print()

        if label.startswith("FULL"):
            json.dump([asdict(x) for x in rows],
                      open("router/ablation_results.json", "w"), indent=2)




# ── tag decomposition (added) ─────────────────────────────────────────────────

LEN_TAGS = {"LEN_SHORT", "LEN_MEDIUM", "LEN_LONG"}


def _split_tags(q: str) -> tuple[list[str], str]:
    enc = encode_text(q)
    toks = enc.split()
    tags = [t for t in toks if t.isupper() and t.isidentifier()]
    body = enc[len(" ".join(tags)):].strip() if tags else enc
    return tags, body


def rep_subset(q: str, keep: str) -> str:
    """keep in {'none','len','sem','all'} — isolates which tag family helps."""
    tags, body = _split_tags(q)
    if keep == "none":
        sel: list[str] = []
    elif keep == "len":
        sel = [t for t in tags if t in LEN_TAGS]
    elif keep == "sem":
        sel = [t for t in tags if t not in LEN_TAGS]
    else:
        sel = tags
    return (" ".join(sel) + " " + body).strip()


def decompose() -> None:
    """Is the tag gain semantic, or is it just query length re-encoded?"""
    rng = np.random.default_rng(SEED)
    train = load_seed()
    ev = load_eval("router/eval_1000.py")
    qs = [q for q, _, _ in ev]
    ys = [l for _, l, _ in ev]
    out = []
    for keep, name in (("none", "query text only (no tags)"),
                       ("len", "+ LEN_* tags only"),
                       ("sem", "+ semantic tags only (no LEN_*)"),
                       ("all", "+ all tags (production)")):
        pipe = build((1, 3))
        pipe.fit([rep_subset(q, keep) for q, _ in train], [l for _, l in train])
        X = [rep_subset(q, keep) for q in qs]
        pred = list(pipe.predict(X))
        conf = pipe.predict_proba(X).max(axis=1)
        out.append(score(name, ys, pred, conf, rng))
    print("══ TAG DECOMPOSITION (Eval-1000, 1-3 gram, 239 train) ".ljust(88, "═"))
    for x in out:
        print(f"{x.name:36s} acc {x.accuracy:.3f}  F1 {x.f1_big:.3f}  "
              f"large {x.large_model_rate*100:5.1f}%  casc {x.cascade_rate*100:5.1f}%")
    json.dump([asdict(x) for x in out],
              open("router/decomposition_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
    print()
    decompose()
