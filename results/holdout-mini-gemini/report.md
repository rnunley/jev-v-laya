# JEV vs Laya: MMLU-Pro routing results (openai/gpt-4.1-mini vs google/gemini-2.5-flash-lite)

## Results

Disjoint public-dataset holdout: **1400** questions.

### Routing-choice accuracy (distinguishable prompts)

Exactly one candidate answered gold on **302** prompts. On these prompts, selecting that candidate is an identifiable correct routing decision. JEV **156/302 (51.7%)**; Laya **178/302 (58.9%)**. JEV minus Laya **-7.28 pp**, stratified paired bootstrap 95% CI **[-11.92, -2.65] pp**. Fixed openai/gpt-4.1-mini would score **156/302** and fixed google/gemini-2.5-flash-lite **146/302**. JEV chose openai/gpt-4.1-mini on **1394/1400** prompts; Laya chose google/gemini-2.5-flash-lite on **301/1400**. The 10,000 seeded paired bootstrap resamples within each observed category. Both-correct and both-wrong prompts cannot identify a preferable route from gold.

### Downstream routed accuracy (all prompts)

This measures whether the selected candidate answered gold; it also depends on candidate ability and is **not** pure route-choice accuracy.

| Policy | Correct / N (accuracy) |
|---|---:|
| JEV | 816/1400 (58.3%) |
| Laya | 838/1400 (59.9%) |
| Fixed openai/gpt-4.1-mini | 816/1400 (58.3%) |
| Fixed google/gemini-2.5-flash-lite | 806/1400 (57.6%) |
| Pilot-fixed google/gemini-2.5-flash-lite | 806/1400 (57.6%) |
| Pilot category rule | 817/1400 (58.4%) |
| Best fixed in hindsight (not deployable as selected here) | 816/1400 (58.3%) |
| Per-prompt oracle (not deployable) | 962/1400 (68.7%) |

JEV minus Laya downstream: **-1.57 pp**, stratified paired bootstrap 95% percentile CI **[-2.64, -0.50] pp** (10,000 seeded resamples; 100/category). Oracle gaps: JEV 10.43 pp; Laya 8.86 pp. Routing failures: {'jev': 0, 'laya': 0}; invalid candidate answers: {'first': 0, 'second': 2} (first/second; finish reasons: {'first': {}, 'second': {'error': 2}}; truncations: {'first': 0, 'second': 0}). These are answer-budget-specific outcomes, not general routing skill. Neither interval proves equivalence.

**Laya leads on the predeclared all-question contrast**, but the lower confidence bound for that lead is 0.50 pp. This does not establish the predeclared 2 pp minimum useful gain.

### Value over a fixed route (all prompts)

Positive gain means the router answers more prompts correctly than always using that candidate. Wins / losses count paired prompts on which only the router / only the fixed policy answers correctly. The interval resamples paired prompts within category, not independently generated candidate responses.

| Router | Fixed policy | Wins / losses | Net gain pp [95% paired CI] |
|---|---|---:|---:|
| JEV | Fixed openai/gpt-4.1-mini | 0 / 0 | +0.00 [0.00, 0.00] |
| JEV | Fixed google/gemini-2.5-flash-lite | 156 / 146 | +0.71 [-1.71, 3.21] |
| Laya | Fixed openai/gpt-4.1-mini | 41 / 19 | +1.57 [0.50, 2.64] |
| Laya | Fixed google/gemini-2.5-flash-lite | 137 / 105 | +2.29 [0.14, 4.43] |

Improvement over **both** fixed policies is required to claim useful accuracy routing for this pool; the best fixed policy selected on this same sample is descriptive, not a pre-registered significance test. Router failures count as incorrect. The oracle is an unattainable upper bound.

### Frozen pilot policies

The disjoint 280-question Mini/Gemini pilot selected fixed **google/gemini-2.5-flash-lite** and a category-to-model rule before this holdout was collected (pilot summary SHA-256 `68d838890dbf753ce4e19c671a6b05153f55287a1c4d57ca3d0d7a7c747a7709`). Paired wins/losses and intervals below use all holdout questions.

| Router | Pilot policy | Wins / losses | Net gain pp [95% paired CI] |
|---|---|---:|---:|
| JEV | Pilot-fixed google/gemini-2.5-flash-lite | 156 / 146 | +0.71 [-1.71, 3.21] |
| JEV | Pilot category rule | 93 / 94 | -0.07 [-2.00, 1.86] |
| Laya | Pilot-fixed google/gemini-2.5-flash-lite | 137 / 105 | +2.29 [0.14, 4.43] |
| Laya | Pilot category rule | 95 / 74 | +1.50 [-0.29, 3.36] |

### Query-independent controls

The random control permutes each router's observed numbers of first- and second-model selections over the same questions (failures remain zero). Its table is an *expectation*, not a sampled random run or an inferential interval; it measures item sensitivity at the observed call mix.

| Router | Observed correct | Matched-mix expected correct | Observed − expected pp |
|---|---:|---:|---:|
| JEV | 816/1400 | 815.96/1400 | +0.003 |
| Laya | 838/1400 | 813.85/1400 | +1.725 |

### API cost at the observed operating points

Per-question USD below is the selected answer model's actual billed cost plus the hosted JEV decision fee when applicable. Fixed and pilot policies would call only one answer model per query. Laya's local GPU/CPU, energy, and amortization cost is **unpriced**; these are not full comparable deployment costs. Collecting counterfactual answers for both models on every question, failed calls, and retries are separate experimental expenses, not included in these one-successful-call policy points. Uncertain charges are bounded separately in the ledger.

| Policy | API USD / question | Accuracy |
|---|---:|---:|
| Fixed openai/gpt-4.1-mini | $0.00010706 | 816/1400 (58.3%) |
| Fixed google/gemini-2.5-flash-lite | $0.00002436 | 806/1400 (57.6%) |
| Pilot category rule | $0.00005773 | 817/1400 (58.4%) |
| JEV + selected candidate | $0.00012972 | 816/1400 (58.3%) |
| Laya + selected candidate | $0.00008706 | 838/1400 (59.9%) |

At each router's observed API-dollar budget, the query-independent fixed-model mixture has:

| Router budget | Expected fixed-mixture accuracy |
|---|---:|
| JEV | outside the fixed-model cost span |
| Laya | 58.11% (first-model share 75.8%) |

The cost-matched mixture is descriptive and selected from holdout costs; it is not a tuned-router curve or a pre-registered significance comparison. Option order was balanced 50/50 within each category (canonical un-reversed criteria list `['google/gemini-2.5-flash-lite', 'openai/gpt-4.1-mini']`):

| Router | Option order | openai/gpt-4.1-mini selected |
|---|---|---:|
| JEV | google/gemini-2.5-flash-lite first | 694/700 |
| JEV | openai/gpt-4.1-mini first | 700/700 |
| Laya | google/gemini-2.5-flash-lite first | 540/700 |
| Laya | openai/gpt-4.1-mini first | 559/700 |

Measured warm decision-call latency (Laya's local inference after model load versus JEV's hosted network request) is JEV p50/p95 **0.263/0.368 s** and Laya **0.057/0.101 s**. Hardware and network differ; this is not a hardware-normalized speed ranking.

Eligibility required the full state, both model descriptions and question instructions to fit Laya's pinned 512-token context; 504 remaining source rows were too long. The holdout also excludes 430 previously sampled MMLU-Pro IDs and normalized exact duplicate question texts; 385 candidate rows were removed by text deduplication. Sampled questions are unique by ID and normalized text. This defines a short-input, equally weighted 14-category estimand, not the original MMLU-Pro population.

The design uses [RouterBench](https://arxiv.org/abs/2403.12031)'s cached outcomes, fixed-policy and oracle controls, [RouteLLM](https://arxiv.org/abs/2406.18665)'s matched random-call-rate check, and [LLMRouterBench](https://arxiv.org/abs/2601.07206)'s same-pool evaluations. Neither fixed-choice router supplies a tested threshold sweep here: the two observed operating points are not a cost–quality curve.
The direct [sysone-bench comparison](https://github.com/instax-dutta/sysone-bench/blob/master/REPORT.md) likewise checks byte-identical decision inputs and pinned versions; its triage/moderation outcomes are not answer-model-routing outcomes and are not imported as scores here.

### Category breakdown

| Category | N | One-correct-only | JEV choice hits | Laya choice hits | JEV routed | Laya routed | openai/gpt-4.1-mini correct | google/gemini-2.5-flash-lite correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| biology | 100 | 11 | 2 | 6 | 76 | 80 | 76 | 83 |
| business | 100 | 24 | 11 | 12 | 43 | 44 | 43 | 45 |
| chemistry | 100 | 30 | 16 | 17 | 46 | 47 | 46 | 44 |
| computer science | 100 | 22 | 11 | 12 | 64 | 65 | 64 | 64 |
| economics | 100 | 19 | 10 | 9 | 73 | 72 | 73 | 72 |
| engineering | 100 | 31 | 14 | 18 | 43 | 47 | 43 | 46 |
| health | 100 | 18 | 14 | 14 | 72 | 72 | 72 | 62 |
| history | 100 | 19 | 7 | 16 | 62 | 71 | 62 | 67 |
| law | 100 | 26 | 15 | 12 | 48 | 45 | 48 | 44 |
| math | 100 | 27 | 14 | 15 | 48 | 49 | 48 | 47 |
| other | 100 | 16 | 11 | 12 | 68 | 69 | 68 | 62 |
| philosophy | 100 | 21 | 10 | 13 | 60 | 63 | 60 | 61 |
| physics | 100 | 16 | 7 | 9 | 34 | 36 | 34 | 36 |
| psychology | 100 | 22 | 14 | 13 | 79 | 78 | 79 | 73 |

## Method and provenance

[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) MIT-licensed test split at `b189ec765aa7ed75c8acfea42df31fdae71f97be`; 100 rows selected per sorted category with Python `random.Random(20260922)` from question-ID-sorted rows after excluding 430 previously sampled question IDs, then question-ID sorted globally. Equal category weighting is **not** population-weighted MMLU-Pro. Question IDs/hash, excluded-ID and prior-manifest hashes, prompt, Choice criteria, generation settings, requested model IDs, price-page links and revision are frozen in the local manifest; manifest SHA-256: `dca1bac7ec84d52d4ff62017fe43a2ab3174d40f43f050b066cbb17e67ec5be8`. Public [scores](per_prompt.csv) and [machine summary](summary.json) enable independent score checks. Neither router receives gold or candidate answers.
Both hosted candidates receive the same zero-shot system message `Answer the multiple-choice question. Return only a JSON object with the selected option letter in the answer field.` and `Question: ...\\nOptions: ...` state, temperature 0, max_tokens 4096; OpenRouter JSON-schema structured output requires `{"answer": "X"}` with a per-row enum of listed option letters (`provider.require_parameters=true`). Only a valid one-key JSON object with normal finish scores; invalid/truncated outputs score wrong. Routers see only the state and one identical Choice question, instructions `Which model is more likely to answer this multiple-choice question correctly? Select one model.`, criteria `{"google/gemini-2.5-flash-lite":"Gemini 2.5 Flash Lite by Google; lightweight general reasoning model","openai/gpt-4.1-mini":"GPT-4.1 Mini by OpenAI; mid-sized general-purpose model"}`. Laya `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` runs locally on MPS if available else CPU (512-token English context default); JEV `typesafe/jev-1.13` uses OpenRouter decisions. Requested/returned model IDs and provider identities actually returned: `{"google/gemini-2.5-flash-lite":[["google/gemini-2.5-flash-lite","Google"]],"jev":[["typesafe/jev-1.13-20260917","TypeSafe"]],"laya":[["convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982",null]],"openai/gpt-4.1-mini":[["openai/gpt-4.1-mini","OpenAI"]]}`. Null provider means no provider identity was reported.

HTTP 429/5xx retries at most twice when billed cost is known; 401/402/403, unknown cost, timeout and unknown billing stop until key usage is reconciled or a conservative charge ceiling is reserved. The append-only shared ledger includes pilots and subsequent runs. It stops at $4.95 before the next paid request; concurrent requests reserve $0.05 each strictly below the $5 ceiling. Provider-reported charges this run: `{"google/gemini-2.5-flash-lite":"0.034108767","openai/gpt-4.1-mini":"0.149889564","typesafe/jev-1.13":"0.032065992"}`; recorded run total $0.216064323. A further **$0.05** is held as a conservative ceiling for timed-out requests with no per-generation billing ID; this is **not an observed charge**. Shared ledger exposure including this ceiling is $0.57804812896, below $5. Historical unattributed key-usage reconciliation: $0.0051852185. Local Laya has no OpenRouter charge.

## Limits

Academic multiple-choice questions do not measure production routing. Model descriptions are zero-shot metadata, not trained performance priors; one generation per candidate, even at temperature 0, can vary. The short-input eligibility rule changes the target population; answer budgets and public benchmark contamination remain possible. This equal-category sample and bootstrap interval do not generalize automatically. Hosted JEV versus local Laya latency is not hardware-neutral, and local compute is not free. The hindsight-best fixed score and per-prompt oracle cannot be deployed. A confidence interval crossing zero does not establish equivalence.
