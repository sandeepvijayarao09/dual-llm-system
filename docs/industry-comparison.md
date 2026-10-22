# How CascadeRoute compares with industry routers

Desk research as of 2026-10-21 from papers and vendor docs (links inline).
Anything marked **[unverified]** could not be confirmed from a primary source.

## Comparison

| System | Approach | Training signal | What's evaluated | Headline result | Cost claim | Routing overhead | Open source |
|---|---|---|---|---|---|---|---|
| **CascadeRoute (this repo)** | Binary. Length pre-flight → TF-IDF + LR → local Gemma 4 E4B classifier when router confidence < 0.65 | 239 hand labels + 1,199 LLM-distilled labels (weight 0.1) | Agreement with its own Eval-1000 routing labels | 95.8% cascade, 93.2% router alone, 89.7% for "≤8 words" ([FINDINGS.md](../FINDINGS.md) §9) | ~49.6% of always-large (estimated; answer quality not measured) | ~0 ms; ~0.93 s on the ~15% escalated | Yes |
| [RouteLLM](https://arxiv.org/abs/2406.18665) (LMSYS, 2024) | Binary strong/weak: matrix factorization, BERT, causal-LLM, similarity-weighted | Chatbot Arena preferences + GPT-4-judge labels + gold labels | Answer quality (MT-Bench judge, MMLU, GSM8K) vs % strong-model calls | 14% GPT-4 calls for 95% of GPT-4 quality on MT-Bench ([blog](https://lmsys.org/blog/2024-07-01-routellm/)) | >85% / 45% / 35% on MT-Bench / MMLU / GSM8K | [unverified] | Yes |
| [FrugalGPT](https://arxiv.org/abs/2305.05176) (2023) | Sequential cascade, learned answer-reliability scorer | Per-query correctness | Accuracy vs cost | Matches GPT-4 | Up to 98% | Sequential calls | [unverified] |
| [RouterBench](https://arxiv.org/abs/2403.12031) / [LLMRouterBench](https://arxiv.org/abs/2601.07206) | Benchmarks | Precomputed outputs (405k; 400k+) | Cost-quality hull (AIQ) | Many routers ≈ simple baselines | — | — | Yes |
| [Not Diamond](https://www.notdiamond.ai/) | Learned multi-model router | Proprietary | Not published | +5% accuracy (vendor claim) | 50%+ (vendor claim) | [unverified] | No |
| [OpenRouter auto](https://openrouter.ai/docs/guides/routing/auto-model-selection) | Task-type classifier + ranking by community spend | Usage data | Not published | — | No extra fee | [unverified] | No |
| [Azure model router](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router-how-it-works) | Trained non-LLM router, Balanced/Cost/Quality modes | "Hundreds of thousands of examples" | Not published | — | Not stated | "Minimal" | No |
| [Bedrock Intelligent Prompt Routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html) | Pairwise quality prediction within one model family + threshold | Proprietary | ARQGC (0.5 random, 1 oracle), savings | ARQGC 0.86 on the Anthropic family ([blog](https://aws.amazon.com/blogs/machine-learning/use-amazon-bedrock-intelligent-prompt-routing-for-cost-and-latency-benefits/)) | 16–56% by family; "up to 30%" on the product page | ~85 ms P90 | No |
| [Vertex AI Model Optimizer](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/vertex-ai-model-optimizer) | Picks Gemini Pro or Flash per prompt | Proprietary | Not published | — | [unverified] | [unverified] | No |

## Where CascadeRoute matches practice

- A cheap non-LLM router in front, escalating only when unsure — the same
  shape as Azure's and Bedrock's routers. Its ~0 ms ML path is cheaper than
  Bedrock's ~85 ms P90, and the LLM escalation touches ~15% of traffic.
- Per-call token, latency and cost accounting, and an OpenAI-format endpoint
  (as Azure, OpenRouter and RouteLLM expose).
- Escalating to the large model when uncertain, like Bedrock's fallback.

## Where it falls short

1. **The metric.** Industry and research routers are judged on the quality
   of the answers users get (benchmark accuracy or an LLM judge) against cost.
   CascadeRoute reports agreement with labels its author wrote.
2. **The benchmark.** Those labels correlate with word count at r = 0.766
   (FINDINGS.md §4). RouterBench and LLMRouterBench both find simple baselines
   hard to beat, so the gain over the length rule has to be shown on answer
   quality.
3. **The training signal.** No preference or correctness data. RouteLLM's
   largest gains came from judge-labelled data (MT-Bench CPT(50%) 25% → 13%).
4. **One operating point.** Fixed 0.65 / 0.75 thresholds, no cost-quality or
   deferral curve, no random or oracle baseline.
5. **Scope.** Binary, one model pair, not tested for transfer to other pairs
   (RouteLLM shows its routers transfer).

## Next steps that would close the gap

1. **Answer-quality evaluation on public benchmarks** in RouteLLM's setup
   (MT-Bench with an LLM judge, MMLU, GSM8K), Gemma 4 E4B as weak and Gemini
   as strong, reporting APGR and CPT(50%/80%) against random, oracle and the
   ≤8-words rule; and on RouterBench or LLMRouterBench.
2. **A deferral curve**: sweep the router threshold, plot large-model share
   against judged quality, report the area and mark the chosen point.
3. **Outcome labels instead of authored labels**: answer a few thousand
   public prompts with both models, label "small was good enough" with a
   judge, retrain, and re-check the length confound and transfer to another
   model pair.
