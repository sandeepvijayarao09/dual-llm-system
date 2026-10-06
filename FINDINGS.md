# dual-llm-system — experimental findings

All numbers reproducible with `.venv/bin/python -m router.ablation`.
Eval set: Eval-1000 (1,000 queries, 23 categories). Training set: 239 seed
examples. Hyperparameters identical across every learned row. Confidence
intervals are non-parametric bootstrap, 2,000 resamples.

## 1. The original ablation was confounded

The published Table 3 compared v0 against v1 while varying **four** things at
once — training-set size (120 vs 239), n-gram range, tag augmentation, and the
eval set itself (Eval-500 vs Eval-1000) — then attributed the entire delta to
tags.

Controlled 2×2, everything else held fixed:

| Representation | n-gram | Accuracy | Cascade | F1 |
|---|---|---|---|---|
| query only | unigram | 87.0% | 35.1% | 0.872 |
| query only | 1–3 gram | 85.5% | 38.2% | 0.859 |
| + tags | unigram | 92.0% | 14.1% | 0.917 |
| + tags | 1–3 gram | **92.6%** | **14.7%** | **0.922** |

Tags are worth **+7.1 pp / −23.5 pp** — *larger* than the +4.4 pp originally
claimed. The n-gram range does nothing: without tags trigrams **hurt**
(−1.5 pp); with tags they add +0.6 pp, inside the bootstrap interval.

## 2. The tag gain is entirely query length

| Representation | Accuracy | Cascade | Δ vs no tags |
|---|---|---|---|
| query text only | 85.5% | 38.2% | — |
| + `LEN_*` tags only (3 tags) | 92.4% | 17.3% | **+6.9 pp** |
| + semantic tags only (13 tags) | 85.9% | 29.5% | **+0.4 pp** |
| + all tags | 92.6% | 14.7% | +7.1 pp |

The three length buckets recover 6.9 of the 7.1 pp. The thirteen semantic tags
— reasoning keywords, code presence, question multiplicity, research/impact
markers — contribute +0.4 pp, indistinguishable from noise. This contradicts
the paper's stated motivation for the tag vocabulary.

Note also: `LEN_LONG`, `Q_MULTI` and `MULTI_SENTENCE` **never fire** on
Eval-1000, despite the paper naming compound questions as a strong predictor.

## 3. A one-line rule gets 89.7%

| Router | Accuracy | 95% CI | Large-model rate |
|---|---|---|---|
| Majority class (all-small) | 54.5% | [0.514, 0.576] | 0.0% |
| All-big | 45.5% | [0.424, 0.486] | 100.0% |
| Random | 51.6% | [0.486, 0.546] | 50.9% |
| Tag rules only, no learning | 70.3% | [0.675, 0.733] | 49.4% |
| Word-count logistic regression | 86.1% | — | 55.0% |
| **Word-count rule (≤8 words → small)** | **89.7%** | [0.879, 0.915] | 46.8% |
| CascadeRoute (full router) | **92.6%** | [0.909, 0.942] | 49.5% |

The full system beats `len(query.split()) <= 8` by **2.9 pp**, with overlapping
confidence intervals.

## 4. Root cause: the benchmark is length-confounded

Point-biserial correlation between word count and the `big` label: **r = 0.766**.

| Word count | n | % big |
|---|---|---|
| 1–5 | 207 | 2.4% |
| 6–8 | 325 | 12.3% |
| 9–12 | 309 | 81.6% |
| 13+ | 159 | 99.4% |

The label flips almost deterministically at nine words. Because a single
annotator authored both queries and labels, complexity appears to have been
operationalised as length.

Where the router actually fails: on the **103 queries (10.3%)** where the
word-count rule is wrong, the router scores **52.4%** — chance — while also
overturning 25 cases the trivial rule got right. Within fixed length buckets it
beats a length-only majority predictor by +0.5, +3.1, +5.5, +0.6 pp.

## 5. Cost claim needs restating

Cascade rate (14.7%) is *routing overhead*, not serving cost. The router
dispatches **495/1000 = 49.5%** of queries to GPT-4o (ground-truth big rate:
45.5%). Monetary savings are governed by 49.5%, not 14.7% — and RouteLLM quotes
a ~14% *strong-model dispatch* rate, so the two are easy to conflate.

## 6. Data integrity (good news)

- Train ∩ Eval-1000 = **6 of 1000** (0.6%). Removing them changes nothing
  (92.6% → 92.6%).
- Train ∩ Eval-500 = **56 of 500** (11.2%) — the old v0 baseline was measured
  on partly-seen data.
- Eval-500 ∩ Eval-1000 = **7** queries. They are near-disjoint, so v1 was
  effectively evaluated held-out. The paper says Eval-1000 "extends" Eval-500;
  it does not, and the truth is the stronger claim.
- Published confusion matrix (TP 438, FN 17, FP 57, TN 488) reproduces exactly.

## What this means

The engineering is sound and the reported numbers are honest and reproducible.
But the paper's central scientific claim — that hand-crafted semantic features
drive routing accuracy — is refuted by its own data. Two viable paths:

**A. Reframe as a cautionary result.** "Self-authored routing benchmarks are
length-confounded; semantic features add nothing once length is controlled."
Honest, useful to the routing literature, publishable at a workshop. Requires
no new data.

**B. Rebuild the benchmark length-stratified** so P(big) is constant across
word-count buckets, then re-test whether semantic tags earn their keep. This is
the real experiment, and it may yield a positive result.

Path A is available now. Path B is the stronger paper.
