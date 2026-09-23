# JEV vs Laya: MMLU-Pro routing results (qwen/qwen3.5-9b vs google/gemini-2.5-flash-lite)

## Results

Equal-category sample: **280** questions.

### Routing-choice accuracy (distinguishable prompts)

Exactly one candidate answered gold on **76** prompts. On these prompts, selecting that candidate is an identifiable correct routing decision. JEV **42/76 (55.3%)**; Laya **35/76 (46.1%)**. JEV minus Laya **9.21 pp**, stratified paired bootstrap 95% CI **[-7.89, 27.63] pp**. Fixed qwen/qwen3.5-9b would score **42/76** and fixed google/gemini-2.5-flash-lite **34/76**. JEV chose qwen/qwen3.5-9b on **278/280** prompts; Laya chose google/gemini-2.5-flash-lite on **265/280**. The 10,000 seeded paired bootstrap resamples within each observed category. Both-correct and both-wrong prompts cannot identify a preferable route from gold.

### Downstream routed accuracy (all prompts)

This measures whether the selected candidate answered gold; it also depends on candidate ability and is **not** pure route-choice accuracy.

| Policy | Correct / N (accuracy) |
|---|---:|
| JEV | 189/280 (67.5%) |
| Laya | 182/280 (65.0%) |
| Fixed qwen/qwen3.5-9b | 189/280 (67.5%) |
| Fixed google/gemini-2.5-flash-lite | 181/280 (64.6%) |
| Best fixed in hindsight (not deployable as selected here) | 189/280 (67.5%) |
| Per-prompt oracle (not deployable) | 223/280 (79.6%) |

JEV minus Laya downstream: **2.50 pp**, stratified paired bootstrap 95% percentile CI **[-3.21, 8.21] pp** (10,000 seeded resamples; 20/category). Oracle gaps: JEV 12.14 pp; Laya 14.64 pp. Routing failures: {'jev': 0, 'laya': 0}; invalid candidate answers: {'first': 69, 'second': 0} (first/second respectively; truncations: {'first': 65, 'second': 0}). These are answer-budget-specific outcomes, not general routing skill. Neither interval proves equivalence.

| Category | N | One-correct-only | JEV choice hits | Laya choice hits | JEV routed | Laya routed | qwen/qwen3.5-9b correct | google/gemini-2.5-flash-lite correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biology | 20 | 4 | 1 | 3 | 16 | 18 | 16 | 18 |
| business | 20 | 6 | 6 | 1 | 17 | 12 | 17 | 11 |
| chemistry | 20 | 8 | 3 | 5 | 9 | 11 | 9 | 11 |
| computer science | 20 | 6 | 5 | 1 | 15 | 11 | 15 | 11 |
| economics | 20 | 2 | 2 | 0 | 17 | 15 | 17 | 15 |
| engineering | 20 | 10 | 5 | 5 | 11 | 11 | 11 | 11 |
| health | 20 | 5 | 3 | 3 | 13 | 13 | 13 | 12 |
| history | 20 | 4 | 0 | 4 | 8 | 12 | 8 | 12 |
| law | 20 | 7 | 1 | 6 | 7 | 12 | 7 | 12 |
| math | 20 | 8 | 7 | 1 | 17 | 11 | 17 | 11 |
| other | 20 | 3 | 1 | 2 | 12 | 13 | 12 | 13 |
| philosophy | 20 | 5 | 2 | 2 | 14 | 14 | 14 | 15 |
| physics | 20 | 8 | 6 | 2 | 15 | 11 | 15 | 11 |
| psychology | 20 | 0 | 0 | 0 | 18 | 18 | 18 | 18 |

## Method and provenance

[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) MIT-licensed test split at `b189ec765aa7ed75c8acfea42df31fdae71f97be`; 20 rows selected per sorted category with Python `random.Random(20260922)` from question-ID-sorted rows, then question-ID sorted globally. Equal category weighting is **not** population-weighted MMLU-Pro. Question IDs/hash, prompt, Choice criteria, generation settings, requested model IDs, price-page links and revision are frozen in the local manifest; manifest SHA-256: `7b71e589a822ca11af06624d8319e2840a9a6fe4e43a51e4c3e29139a4d79e6e`. Public [scores](per_prompt.csv) and [machine summary](summary.json) enable independent score checks. Neither router receives gold or candidate answers.

Both hosted candidates receive the same zero-shot system message `Answer the multiple-choice question. Return only a JSON object with the selected option letter in the answer field.` and `Question: ...\\nOptions: ...` state, temperature 0, max_tokens 4096; OpenRouter JSON-schema structured output requires `{"answer": "X"}` with a per-row enum of listed option letters (`provider.require_parameters=true`). Only a valid one-key JSON object with normal finish scores; invalid/truncated outputs score wrong. Routers see only the state and one identical Choice question, instructions `Which model is more likely to answer this multiple-choice question correctly? Select one model.`, criteria `{"google/gemini-2.5-flash-lite":"Gemini 2.5 Flash Lite by Google; lightweight general reasoning model","qwen/qwen3.5-9b":"Qwen3.5-9B by Qwen; 9B-parameter general reasoning model"}`. Laya `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` runs locally on MPS if available else CPU (512-token English context default); JEV `typesafe/jev-1.13` uses OpenRouter decisions. Requested/returned model IDs and provider identities actually returned: `{"google/gemini-2.5-flash-lite":[["google/gemini-2.5-flash-lite","Google"]],"jev":[["typesafe/jev-1.13-20260917","TypeSafe"]],"laya":[["convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982",null]],"qwen/qwen3.5-9b":[["qwen/qwen3.5-9b","Darkbloom"],["qwen/qwen3.5-9b","DeepInfra"],["qwen/qwen3.5-9b","Parasail"],["qwen/qwen3.5-9b","SiliconFlow"],["qwen/qwen3.5-9b","Together"],["qwen/qwen3.5-9b","Venice"]]}`. Null provider means no provider identity was reported.

HTTP 429/5xx retries at most twice when billed cost is known; 401/402/403, unknown cost, timeout and unknown billing stop. Append-only ledger includes earlier smoke, integration and both full runs. It stops at $4.95 before the next paid request; concurrent requests reserve $0.05 each strictly below the $5 ceiling, then settle to provider-reported cost. Provider-reported spend this run: `{"google/gemini-2.5-flash-lite":"0.007018011","qwen/qwen3.5-9b":"0.1227767409","typesafe/jev-1.13":"0.006562542"}`; run total $0.1363572939; shared ledger $0.2911646564. The ledger includes $0.0051852185 of key-usage reconciliation after earlier interrupted/unknown-billing calls; this is not attributed to a model or run. Local Laya has no OpenRouter cost.

## Limits

Academic multiple-choice questions do not measure production routing. Model descriptions are zero-shot metadata, not trained performance priors; one generation per candidate, even at temperature 0, can vary. The 512-token Laya context and answer budget can affect outcomes. This finite equal-category sample and bootstrap interval do not generalize automatically. Hosted JEV versus local Laya latency is not hardware-neutral. Best fixed uses hindsight; the oracle uses per-prompt gold and cannot be deployed. Router ties or overlapping uncertainty are not evidence of superiority.

## Cross-pool comparison

The same **280 question IDs, categories and gold answers** were used in both runs. Only the first candidate changed: Qwen3.5-9B was replaced by GPT-4.1 Mini. Gemini was queried anew, and both routers received the same state and Choice instructions with the new candidate's truthful model description. Prompt, JSON answer schema, temperature, token cap, dataset revision, seed and scoring remained fixed.

| Pool (first / second) | One-correct-only | JEV choice hit | Laya choice hit | JEV−Laya choice pp [95% CI] | JEV routed | Laya routed | Fixed first | Fixed Gemini | Oracle | Invalid first/second | Run USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen / Gemini | 76 | 42/76 (55.3%) | 35/76 (46.1%) | 9.21 [-7.89, 27.63] | 189/280 (67.5%) | 182/280 (65.0%) | 189/280 (67.5%) | 181/280 (64.6%) | 223/280 (79.6%) | 69/0 | $0.1363572939 |
| GPT-4.1 Mini / Gemini | 70 | 35/70 (50.0%) | 34/70 (48.6%) | 1.43 [-8.57, 11.43] | 180/280 (64.3%) | 179/280 (63.9%) | 179/280 (63.9%) | 181/280 (64.6%) | 215/280 (76.8%) | 0/0 | $0.044283153 |

The routing-choice denominator is *pool-specific*: a prompt is counted only if exactly one candidate in that pool answered gold. The confidence intervals are 10,000 seeded stratified paired bootstrap resamples within each run, **not** a test of the difference between pools. Repeated candidate inference and provider routing may vary, so cross-run differences cannot be attributed exclusively to replacing Qwen. The original Qwen run had 69 invalid answers (65 truncations); the Mini run had 0 invalid answers (0 truncations). Routing decisions are blind to candidate answers and gold.

Auditable second-pool [report](../mmlu-pro-280-mini/report.md), [summary](../mmlu-pro-280-mini/summary.json) and [per-prompt scores](../mmlu-pro-280-mini/per_prompt.csv). The original pool's [summary](summary.json) and [per-prompt scores](per_prompt.csv) are alongside this report. Shared ledger spend after both runs: **$0.2911646564**, below the $5 cap.
