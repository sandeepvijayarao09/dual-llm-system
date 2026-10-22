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

## 7. Estimated serving cost

Reproduce with `.venv/bin/python -m router.cost_eval`. Each query is priced as
the calls this system makes for it (big route: GPT-4o reasoner + GPT-4o-mini
summarizer and personalizer; cascaded queries add one classifier call), using
the real system prompts and list prices as of 2026-07. These are estimates:
prompt tokens are characters/4, and answer lengths are fixed (150 small, 600
big) because the eval has no model outputs.

| Strategy | Accuracy | GPT-4o share | $ / 1k queries | vs always GPT-4o |
|---|---|---|---|---|
| Always GPT-4o | 45.5% | 100.0% | 6.41 | 100.0% |
| Always GPT-4o-mini | 54.5% | 0.0% | 0.10 | 1.6% |
| Word-count rule (≤8 words → small) | 89.7% | 46.8% | 3.06 | 47.8% |
| CascadeRoute (ML decision) | 92.6% | 49.5% | 3.24 | 50.6% |
| CascadeRoute, cascaded → big (bound) | 85.7% | 59.0% | 3.84 | 59.9% |
| Oracle (perfect routing) | 100.0% | 45.5% | 2.98 | 46.5% |

- Cost tracks the GPT-4o share almost exactly, as §5 argued: the 14.7%
  cascade adds about $0.01 per 1k queries.
- The router costs **2.8 points more** than the word-count rule for its 2.9
  points of accuracy, and is 4.1 points above perfect routing.
- The share barely moves with answer length (51.6% / 50.6% / 50.1% at 300 /
  600 / 1,200 GPT-4o answer tokens).
- Offline we don't know what the LLM classifier would decide for the 147
  cascaded queries; the bound row sends them all to GPT-4o. Real traffic
  settles this: `routing_log.db` now records measured tokens and cost per call.

### 7a. Default pair since the Gemma/Gemini switch

The system now defaults to Gemma 4 E4B (local, via Ollama) for the small role
and every support call, and Gemini 3.8 Flash for the large role. Local calls
have no per-token price, so only Gemini calls cost money (same assumptions as
above; Gemini list prices as of 2026-10):

| Strategy | Accuracy | Gemini share | $ / 1k queries | vs always Gemini |
|---|---|---|---|---|
| Always Gemini 3.8 Flash | 45.5% | 100.0% | 2.31 | 100.0% |
| Always Gemma 4 E4B | 54.5% | 0.0% | 0.00 | 0.0% |
| Word-count rule (≤8 words → small) | 89.7% | 46.8% | 1.08 | 46.9% |
| CascadeRoute (ML decision) | 92.6% | 49.5% | 1.15 | 49.6% |
| CascadeRoute, cascaded → big (bound) | 85.7% | 59.0% | 1.37 | 59.1% |
| Oracle (perfect routing) | 100.0% | 45.5% | 1.05 | 45.6% |

The ordering and gaps are unchanged; the router still costs ~2.7 points more
than the word-count rule. These rows say nothing about answer quality: the
accuracy column is agreement with the hand-written Eval-1000 labels, and
whether Gemma 4 E4B answers the "small" queries as well as GPT-4o-mini did is
untested.

## 8. The live cascade, with a real classifier

Reproduce with `.venv/bin/python -m router.live_cascade_eval --all` (needs
Ollama with `gemma4:e4b`; local, $0). Until now every number above assumed
the ML router's own guess for the 147 queries it defers on. This run asks the
LLM classifier (Gemma 4 E4B, temperature 0.1, one run) for those 147, applies
the orchestrator's confidence rule, and also runs the classifier alone on all
1,000.

| Router | Accuracy | 95% CI | Large-model rate |
|---|---|---|---|
| Word-count rule (≤8 words → small) | 89.7% | [0.879, 0.915] | 46.8% |
| ML router alone | 92.6% | [0.909, 0.942] | 49.5% |
| LLM classifier alone (Gemma 4 E4B) | 93.8% | [0.922, 0.953] | 48.3% |
| **Full cascade (ML router → Gemma 4 E4B)** | **94.7%** | [0.932, 0.960] | 49.8% |

- On the 147 deferred queries the classifier scores **91.8%**, against 77.6%
  for the ML router's guess. That moves the cascade from 92.6% to 94.7%.
- With a real classifier the cascade's interval no longer overlaps the
  word-count rule's: +5.0 points, not +2.9. The fallback layer carries the
  gain; the ML router alone remains within reach of the one-line rule.
- The classifier alone is close (93.8%) but costs ~0.93 s of local inference on
  every query (median; p95 1.0 s), against ~0 ms for the ML router. The
  cascade pays that only on 14.7% of traffic.
- Zero unparseable replies in 1,147 calls; the low-confidence upgrade rule
  fired once in the cascade and 7 times in the classifier-only run.
- The large-model rate (49.8%) matches the offline estimate, so §7a's cost
  figure (~49.6% of always-Gemini) holds.

A second full run reproduced both accuracies exactly (94.7%, 93.8%); the
classifier-only large-model rate moved 48.3% → 48.5%, and the committed
`router/live_cascade_results.json` is from that second run.

Caveats: temperature 0.1, so expect small run-to-run variation;
the CIs are binomial bootstrap on the accuracy, not paired tests; and the
benchmark is still the length-confounded one from §4 — an LLM classifier can
read length too. This is routing accuracy, not answer quality.

## 9. Router v1: distilling the classifier into the router

Reproduce with `.venv/bin/python -m router.refine --select` (the distilled
data is committed; regenerate it with `python -m router.distill`, local Gemma).
Eval-1000 was used once, to score choices fixed in advance:

1. **Hyperparameters** were picked by 5×5-fold CV on the 239 seed examples
   (word unigrams + character 2–5-grams). This alone did not help: CV
   accuracy rose 0.936 → 0.947 and log-loss halved, but on Eval-1000 the
   seed-only v1 scored 92.3% and became overconfident (defers 5.4%), so its
   cascade fell to 93.7%.
2. **Distillation.** Local Gemma wrote 1,199 new queries over 30 topics,
   deliberately *short-but-hard* and *long-but-easy* — the cases the length
   shortcut gets wrong — and the Gemma classifier labelled them. None
   duplicated or closely paraphrased an Eval-1000 or seed query.
3. **Weighting.** At full weight these counter-examples swamp the seed set
   (cascade 90.8%). Their weight (0.1) and C (2) were chosen on a dev set,
   Eval-500 minus its overlaps with Eval-1000 and the seed data (438 queries).

| Router | ML alone | Deferred | Cascade | 95% CI | Large-model rate | Where length misleads (n=103) |
|---|---|---|---|---|---|---|
| v0 (published) | 92.6% | 14.7% | 94.7% | [0.933, 0.961] | 49.8% | 68.0% |
| v1, seed only | 92.3% | 5.4% | 93.7% | [0.922, 0.952] | 51.2% | 61.2% |
| v1, + distilled at weight 1.0 | 88.7% | 9.3% | 90.8% | [0.890, 0.926] | 51.3% | 75.7% |
| **v1, + distilled at weight 0.1** | **93.2%** | 14.6% | **95.8%** | [0.946, 0.970] | 49.1% | **79.6%** |

Cascade figures use the cached Gemma 4 E4B verdicts
(`data/classifier_verdicts_1000.jsonl`) at the unchanged 0.65 threshold.

- Paired on the same 1,000 queries, v1's cascade fixes 12 of v0's errors and
  introduces 1 (exact McNemar p = 0.003). All 12 fixes are among the 103
  queries where the word-count rule is wrong, and none of those regress.
- The ML router alone is +0.6 points (fixes 19, breaks 13; p = 0.38): not
  significant. The gain is in *what reaches the classifier*, not in the
  router's own accuracy.
- v1 sends slightly less to the large model (49.1% vs 49.8%), so it is not
  buying accuracy with cost.

Caveats: the distilled labels come from the same Gemma classifier the cascade
falls back to, so v1 is partly learning to agree with it; Eval-500 now serves
as v1's dev set and is no longer a clean test for v1; one distillation run.
`router_v0.joblib` is unchanged and `router.eval_1000` still scores it, so the
published 92.6% reproduces. The app now loads `router_v1.joblib` by default.

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
