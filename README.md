# JEV vs Laya: MMLU-Pro routing benchmark

## Executive summary

Two runs on the **same 280 MMLU-Pro prompts** compare locally run Laya and hosted JEV as routers between two OpenRouter answer models. The second run replaces only **Qwen3.5-9B with GPT-4.1 Mini**; Gemini, the sample, prompts, schema, settings, scoring and $5 shared spend ceiling stay fixed. Both candidates' answers and both routers' choices were collected anew.

| Candidate pool | Distinguishable prompts | JEV correct choice | Laya correct choice | JEV−Laya [95% paired CI] | JEV routed | Laya routed | Invalid first candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen / Gemini | 76 | 42/76 (55.3%) | 35/76 (46.1%) | +9.21 pp [−7.89, +27.63] | 189/280 (67.5%) | 182/280 (65.0%) | 69 (65 truncations) |
| GPT-4.1 Mini / Gemini | 70 | 35/70 (50.0%) | 34/70 (48.6%) | +1.43 pp [−8.57, +11.43] | 180/280 (64.3%) | 179/280 (63.9%) | **0** |

On the Mini pool, fixed Mini scored **179/280 (63.9%)**, fixed Gemini **181/280 (64.6%)**, and the per-prompt oracle **215/280 (76.8%)**. JEV selected Mini on **279/280** prompts; Laya selected Mini on **218/280**. JEV beat Laya on **8** prompts and lost on **7**. Neither pool's interval establishes a reliable routing-choice advantage, and neither demonstrates a clear gain over a fixed model. The one-correct-only denominator differs by pool; these are **within-pool** comparisons, not a causal test of the model swap.

The Mini run cost **$0.04428**; shared recorded spend across experiments and both full runs was **$0.29116**, below $5. Qwen's invalid/truncated outputs materially qualify the original result; Mini and Gemini each produced **0 invalid answers** in the second run.

Read the [combined comparison and original report](results/mmlu-pro-280/report.md), original [summary](results/mmlu-pro-280/summary.json) and [per-prompt CSV](results/mmlu-pro-280/per_prompt.csv); second-pool [report](results/mmlu-pro-280-mini/report.md), [summary](results/mmlu-pro-280-mini/summary.json) and [per-prompt CSV](results/mmlu-pro-280-mini/per_prompt.csv). Raw responses, API key and model weights are not published.

## Methodology (frozen)

- MMLU-Pro test @ `b189ec765aa7ed75c8acfea42df31fdae71f97be`, 20 per category sampled with seed `20260922`, question-ID sorted; exactly the same IDs and gold in both runs.
- Pools: `qwen/qwen3.5-9b` + `google/gemini-2.5-flash-lite`, then `openai/gpt-4.1-mini` + the same Gemini ID. Temperature 0, max_tokens 4096, per-row JSON-schema enum of listed A–J letters, and `provider.require_parameters=true` in both.
- Only a valid `{"answer":"X"}` JSON object with a listed letter and normal completion scores; invalid/truncated outputs count wrong.
- Routers see identical state + one Choice question with literal criteria.
- Laya: local convaiinnovations/laya @1c5edc17a7acd8701df6fc341c0d179f1c62c982 on MPS/CPU.
- JEV: typesafe/jev-1.13 via OpenRouter /alpha/decisions.
- Shared append-only ledger counts earlier experiments; stops new calls at $4.95 and reserves $0.05 per in-flight call, keeping total exposure strictly under the user-approved $5 cap.
- Primary routing-choice score: among prompts where exactly one candidate answers gold, did the router select that candidate? Downstream routed accuracy is reported separately for all prompts.

## Methodology review: what would measure routing skill

The current runs are **diagnostic pilots, not evidence of adaptive routing**. On Qwen/Gemini, 69 Qwen outputs were invalid (65 length-truncated); the measured preference partly rewards an output-budget failure. JEV's two departures from always selecting Qwen changed no outcomes. On Mini/Gemini, only 70/280 prompts distinguish candidates (34 Mini-only wins, 36 Gemini-only wins); JEV switched away from Mini once. Its 35/70 choice hits are below fixed Gemini's 36/70. Laya's 14 switches on distinguishing prompts produced seven gains and seven losses relative to fixed Mini. Equal-category MMLU-Pro and a single answer per model say little about production traffic or expected model accuracy.

The existing reports now include **paired wins/losses and net accuracy gain against *each* fixed candidate**, with prompt-paired, category-stratified bootstrap intervals. This is a useful *negative control*: a router that just picks the first model should not be credited for matching a candidate's raw accuracy. The best fixed candidate chosen on these same prompts is descriptive; don't treat selecting it post hoc as a pre-registered statistical test. Routing-choice accuracy on the one-correct-only subset is still reported, but by itself it conceals whether routing improves end-to-end decisions.

For a prospective benchmark:

1. **Declare the target and baseline first.** Use a representative task mixture, with a frozen accuracy rubric and candidate inference settings. Preselect the best fixed policy *using pilot data* (and a simple category-to-model rule learned only from pilot data). On an untouched holdout, measure accuracy gain versus both fixed models and these pilot-trained policies, plus paired JEV−Laya gain. Count missing/invalid routes as failures; report the full holdout and category slices. If optimizing cost, predefine a price/latency utility and include inference **and router** cost rather than calling raw accuracy "routing value".
2. **Pick a pair with headroom on a disjoint pilot.** Require low invalid-answer rates, comparable overall accuracy, meaningful complementary wins in both directions across task families, and enough one-correct-only prompts for a useful confidence interval. Do not choose the pair, thresholds, categories or questions using holdout candidate answers. Keep candidate IDs/provider versions, prompt, answer parser and price provenance fixed for the holdout. Qwen's invalid rate makes its current pool unsuitable for the primary claim.
3. **Size for paired information, not total prompts.** Estimate pilot rates of one-correct-only outcomes and paired router disagreements, then simulate/power the predeclared smallest worthwhile gain and sample enough independent holdout prompts that its interval can distinguish gain from zero. At the Mini pilot's 70/280 informative rate, another 280 similarly drawn questions would again yield only about 70 informative decisions; a 1/70 hit difference cannot resolve a small advantage. Do not stop on favorable interim results. If balancing task categories rather than production frequency, state that estimand and publish category weights.
4. **Audit shortcut policies.** Supply both routers the same full question, candidate descriptions and decision schema; check what actually fits Laya's 512-token context, or restrict both to a common short-prompt subset *before* outcome collection. Randomize/counterbalance candidate order and opaque option labels, mapping choices back to models, and measure flip consistency. Run a frozen metadata-only control and the pilot-trained category rule on the same holdout to distinguish item-sensitive decisions from always-first, model-brand or category priors. Do not let either router see gold or candidate answers.
5. **Measure answer uncertainty.** Where provider nondeterminism matters, obtain several independent candidate completions per question under the same settings; estimate each candidate's success rate, not a single lucky outcome. Pair both routers on exactly the same questions and candidate observations, resample by *question* (never by individual completions), and disclose invalid rates separately. This requires a new, authorized paid run; the published 280-question scores cannot retroactively establish repeatability or remove selection bias.

Until that holdout exists, the defensible conclusion is **no demonstrated advantage over fixed routing or between JEV and Laya** on these two academic pools, not that the routers are equivalent.

## Reproduction

Offline scoring from **locally retained raw run records** (not published):
```bash
uv sync --extra dev --python 3.12
uv run pytest -q tests/test_benchmark.py
uv run python benchmark.py report --run runs/main-structured --out results/mmlu-pro-280
uv run python benchmark.py report --run runs/main-mini-gemini --out results/mmlu-pro-280-mini
uv run python benchmark.py compare --original results/mmlu-pro-280 --alternate results/mmlu-pro-280-mini
```

Fresh collection below makes **paid external calls**; run only with your own authorization and budget. It reproduces the existing Mini/Gemini pilot, **not** the proposed held-out benchmark.

```bash
cp .env.example .env  # replace placeholder with your OpenRouter key
BENCHMARK_INTEGRATION_POOL=mini-gemini RUN_OPENROUTER_INTEGRATION=1 uv run pytest -q -m integration tests/test_integration.py
uv run python benchmark.py prepare --run runs/main-mini-gemini --pool mini-gemini
uv run python benchmark.py answer --run runs/main-mini-gemini --model openai/gpt-4.1-mini
uv run python benchmark.py answer --run runs/main-mini-gemini --model google/gemini-2.5-flash-lite
uv run python benchmark.py route --run runs/main-mini-gemini --router laya
uv run python benchmark.py route --run runs/main-mini-gemini --router jev
```

The original Qwen results are published in `results/mmlu-pro-280/`; the fresh collection commands **do not call Qwen**. The opt-in three-question integration test verifies both candidates, both routers, the ledger and the joined report before the full run. Matching records resume without truncating the shared ledger. Eight candidate requests run concurrently by default; each reserves $0.05 and settles to provider-reported cost. Without local raw records, the published per-prompt CSV and summary permit independent recalculation of scores, but cannot reconstruct inference or model outputs.

## Limits

Academic multiple-choice questions, one generation per candidate, zero-shot router metadata, and local-versus-hosted hardware differences restrict interpretation. A timeout or missing provider-reported cost stops paid calls until actual key usage is reconciled.

## License

MIT © 2026 rnunley

MMLU-Pro (MIT) https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro  
Laya (Apache-2.0) https://github.com/NandhaKishorM/laya

No weights or raw responses in this repo.
