# JEV vs Laya: MMLU-Pro routing benchmark

## Executive summary

On 280 equally sampled MMLU-Pro test questions, **76 distinguished the two candidates** (exactly one answered gold). JEV chose the correct candidate on **42/76 (55.3%)**; local Laya on **35/76 (46.1%)**. JEV–Laya routing-choice difference: **+9.21 percentage points**, 95% stratified paired bootstrap interval **[−7.89, +27.63] pp**. This interval includes zero; the sample does not establish a reliable advantage. Fixed Qwen also scores **42/76** on this conditional metric, fixed Gemini **34/76**; JEV chose Qwen on **278/280** prompts, Laya chose Gemini on **265/280**.

Downstream routed accuracy across all 280: **JEV 189/280 (67.5%)**, **Laya 182/280 (65.0%)**; difference **+2.50 pp**, 95% interval **[−3.21, +8.21] pp**. Fixed Qwen scored **189/280 (67.5%)**, fixed Gemini **181/280 (64.6%)**, best fixed *in hindsight* **189/280**, and an undeployable per-prompt oracle **223/280 (79.6%)**. JEV's downstream score matched fixed Qwen; this run does not demonstrate a routing gain over that baseline.

**Qualification:** Qwen produced **69 invalid answers**, including **65 4,096-token truncations**; Gemini produced zero invalid answers. Routing-choice accuracy here measures selection under this answer budget and model pool, not general routing quality. The full run cost **$0.13636**; the shared ledger including abandoned protocol and integration checks recorded **$0.24647**, below the $5 cap.

Read the [full report](results/mmlu-pro-280/report.md), [machine-readable summary](results/mmlu-pro-280/summary.json), and [auditable per-prompt scores](results/mmlu-pro-280/per_prompt.csv). Raw responses, API key and model weights are not published.

## Methodology (frozen)

- MMLU-Pro test @ b189ec765aa7ed75c8acfea42df31fdae71f97be , 20 per category sampled with seed 20260922, question_id sorted.
- Candidates: `qwen/qwen3.5-9b` and `google/gemini-2.5-flash-lite` on OpenRouter, temperature 0, max_tokens 4096, a per-row JSON-schema enum of listed A–J letters, and `provider.require_parameters=true`.
- Only a valid `{"answer":"X"}` JSON object with a listed letter and normal completion scores; invalid/truncated outputs count wrong.
- Routers see identical state + one Choice question with literal criteria.
- Laya: local convaiinnovations/laya @1c5edc17a7acd8701df6fc341c0d179f1c62c982 on MPS/CPU.
- JEV: typesafe/jev-1.13 via OpenRouter /alpha/decisions.
- Shared append-only ledger counts earlier experiments; stops new calls at $4.95 and reserves $0.05 per in-flight call, keeping total exposure strictly under the user-approved $5 cap.
- Primary routing-choice score: among prompts where exactly one candidate answers gold, did the router select that candidate? Downstream routed accuracy is reported separately for all prompts.

## Reproduction

```bash
uv sync --extra dev --python 3.12
cp .env.example .env  # replace placeholder with your OpenRouter key
uv run pytest -q tests/test_benchmark.py
RUN_OPENROUTER_INTEGRATION=1 uv run pytest -q -m integration tests/test_integration.py
uv run python benchmark.py prepare --run runs/main-structured
uv run python benchmark.py answer --run runs/main-structured --model qwen/qwen3.5-9b
uv run python benchmark.py answer --run runs/main-structured --model google/gemini-2.5-flash-lite
uv run python benchmark.py route --run runs/main-structured --router laya
uv run python benchmark.py route --run runs/main-structured --router jev
uv run python benchmark.py report --run runs/main-structured --out results/mmlu-pro-280
```

For a three-question smoke check, `prepare --run runs/smoke-structured --smoke 3` selects the first three full-sample IDs. The same four answer/router commands with `runs/smoke-structured` precede `report --run runs/smoke-structured` (no public `--out`). Candidate and JEV calls still incur OpenRouter charges; only Laya runs locally. Repeated commands reuse matching records and never truncate the shared ledger.

The opt-in integration test issues fresh paid calls against three public questions, runs Laya locally, and verifies the joined report. Candidate collection uses eight concurrent requests per model by default; the shared ledger reserves $0.05 per in-flight call and settles to provider-reported cost. It fails closed if billing becomes unknown.

## Limits

Academic multiple-choice questions, one generation per candidate, zero-shot router metadata, and local-versus-hosted hardware differences restrict interpretation. A timeout or missing provider-reported cost stops paid calls until actual key usage is reconciled.

## License

MIT © 2026 rnunley

MMLU-Pro (MIT) https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro  
Laya (Apache-2.0) https://github.com/NandhaKishorM/laya

No weights or raw responses in this repo.
