# JEV vs Laya: MMLU-Pro routing results (openai/gpt-4.1-mini vs google/gemini-2.5-flash-lite)

## Results

Equal-category sample: **280** questions.

### Routing-choice accuracy (distinguishable prompts)

Exactly one candidate answered gold on **70** prompts. On these prompts, selecting that candidate is an identifiable correct routing decision. JEV **35/70 (50.0%)**; Laya **34/70 (48.6%)**. JEV minus Laya **1.43 pp**, stratified paired bootstrap 95% CI **[-8.57, 11.43] pp**. Fixed openai/gpt-4.1-mini would score **34/70** and fixed google/gemini-2.5-flash-lite **36/70**. JEV chose openai/gpt-4.1-mini on **279/280** prompts; Laya chose google/gemini-2.5-flash-lite on **62/280**. The 10,000 seeded paired bootstrap resamples within each observed category. Both-correct and both-wrong prompts cannot identify a preferable route from gold.

### Downstream routed accuracy (all prompts)

This measures whether the selected candidate answered gold; it also depends on candidate ability and is **not** pure route-choice accuracy.

| Policy | Correct / N (accuracy) |
|---|---:|
| JEV | 180/280 (64.3%) |
| Laya | 179/280 (63.9%) |
| Fixed openai/gpt-4.1-mini | 179/280 (63.9%) |
| Fixed google/gemini-2.5-flash-lite | 181/280 (64.6%) |
| Best fixed in hindsight (not deployable as selected here) | 181/280 (64.6%) |
| Per-prompt oracle (not deployable) | 215/280 (76.8%) |

JEV minus Laya downstream: **0.36 pp**, stratified paired bootstrap 95% percentile CI **[-2.14, 2.86] pp** (10,000 seeded resamples; 20/category). Oracle gaps: JEV 12.50 pp; Laya 12.86 pp. Routing failures: {'jev': 0, 'laya': 0}; invalid candidate answers: {'first': 0, 'second': 0} (first/second respectively; truncations: {'first': 0, 'second': 0}). These are answer-budget-specific outcomes, not general routing skill. Neither interval proves equivalence.

| Category | N | One-correct-only | JEV choice hits | Laya choice hits | JEV routed | Laya routed | openai/gpt-4.1-mini correct | google/gemini-2.5-flash-lite correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biology | 20 | 1 | 0 | 0 | 17 | 17 | 17 | 18 |
| business | 20 | 7 | 2 | 3 | 7 | 8 | 7 | 10 |
| chemistry | 20 | 9 | 1 | 4 | 4 | 7 | 4 | 11 |
| computer science | 20 | 5 | 5 | 4 | 16 | 15 | 16 | 11 |
| economics | 20 | 3 | 2 | 2 | 16 | 16 | 16 | 15 |
| engineering | 20 | 5 | 2 | 3 | 10 | 11 | 10 | 11 |
| health | 20 | 5 | 4 | 4 | 15 | 15 | 15 | 12 |
| history | 20 | 4 | 3 | 3 | 14 | 14 | 14 | 12 |
| law | 20 | 8 | 4 | 3 | 13 | 12 | 13 | 13 |
| math | 20 | 6 | 3 | 2 | 10 | 9 | 9 | 11 |
| other | 20 | 5 | 3 | 2 | 14 | 13 | 14 | 13 |
| philosophy | 20 | 7 | 4 | 2 | 16 | 14 | 16 | 15 |
| physics | 20 | 3 | 1 | 1 | 10 | 10 | 10 | 11 |
| psychology | 20 | 2 | 1 | 1 | 18 | 18 | 18 | 18 |

## Method and provenance

[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) MIT-licensed test split at `b189ec765aa7ed75c8acfea42df31fdae71f97be`; 20 rows selected per sorted category with Python `random.Random(20260922)` from question-ID-sorted rows, then question-ID sorted globally. Equal category weighting is **not** population-weighted MMLU-Pro. Question IDs/hash, prompt, Choice criteria, generation settings, requested model IDs, price-page links and revision are frozen in the local manifest; manifest SHA-256: `be572566c8910340e85b010dccb6fce85d683e2438d9168799630bfaf9481b3e`. Public [scores](per_prompt.csv) and [machine summary](summary.json) enable independent score checks. Neither router receives gold or candidate answers.

Both hosted candidates receive the same zero-shot system message `Answer the multiple-choice question. Return only a JSON object with the selected option letter in the answer field.` and `Question: ...\\nOptions: ...` state, temperature 0, max_tokens 4096; OpenRouter JSON-schema structured output requires `{"answer": "X"}` with a per-row enum of listed option letters (`provider.require_parameters=true`). Only a valid one-key JSON object with normal finish scores; invalid/truncated outputs score wrong. Routers see only the state and one identical Choice question, instructions `Which model is more likely to answer this multiple-choice question correctly? Select one model.`, criteria `{"google/gemini-2.5-flash-lite":"Gemini 2.5 Flash Lite by Google; lightweight general reasoning model","openai/gpt-4.1-mini":"GPT-4.1 Mini by OpenAI; mid-sized general-purpose model"}`. Laya `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` runs locally on MPS if available else CPU (512-token English context default); JEV `typesafe/jev-1.13` uses OpenRouter decisions. Requested/returned model IDs and provider identities actually returned: `{"google/gemini-2.5-flash-lite":[["google/gemini-2.5-flash-lite","Google"]],"jev":[["typesafe/jev-1.13-20260917","TypeSafe"]],"laya":[["convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982",null]],"openai/gpt-4.1-mini":[["openai/gpt-4.1-mini","OpenAI"]]}`. Null provider means no provider identity was reported.

HTTP 429/5xx retries at most twice when billed cost is known; 401/402/403, unknown cost, timeout and unknown billing stop. Append-only ledger includes earlier smoke, integration and both full runs. It stops at $4.95 before the next paid request; concurrent requests reserve $0.05 each strictly below the $5 ceiling, then settle to provider-reported cost. Provider-reported spend this run: `{"google/gemini-2.5-flash-lite":"0.007018011","openai/gpt-4.1-mini":"0.030773160","typesafe/jev-1.13":"0.006491982"}`; run total $0.044283153; shared ledger $0.2911646564. The ledger includes $0.0051852185 of key-usage reconciliation after earlier interrupted/unknown-billing calls; this is not attributed to a model or run. Local Laya has no OpenRouter cost.

## Limits

Academic multiple-choice questions do not measure production routing. Model descriptions are zero-shot metadata, not trained performance priors; one generation per candidate, even at temperature 0, can vary. The 512-token Laya context and answer budget can affect outcomes. This finite equal-category sample and bootstrap interval do not generalize automatically. Hosted JEV versus local Laya latency is not hardware-neutral. Best fixed uses hindsight; the oracle uses per-prompt gold and cannot be deployed. Router ties or overlapping uncertainty are not evidence of superiority.
