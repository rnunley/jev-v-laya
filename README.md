# JEV vs Laya: held-out MMLU-Pro routing benchmark

## Executive summary

On a **disjoint 1,400-question, short-input MMLU-Pro holdout**, locally run Laya routed **838/1,400 (59.9%)** answers correctly; hosted JEV routed **816/1,400 (58.3%)**. The predeclared JEV−Laya difference is **−1.57 percentage points**, with a question-paired, category-stratified 95% bootstrap interval of **[−2.64, −0.50] pp**. Laya leads on this particular benchmark, but the interval does **not** establish the predeclared **2 pp minimum useful gain**. This is not a general claim about either model's other typed-decision tasks.

| Policy | Correct / 1,400 | API USD / query |
|---|---:|---:|
| Laya + selected candidate | 838 (59.9%) | $0.00008706 |
| JEV + selected candidate | 816 (58.3%) | $0.00012972 |
| Fixed GPT-4.1 Mini | 816 (58.3%) | $0.00010706 |
| Fixed Gemini 2.5 Flash Lite | 806 (57.6%) | $0.00002436 |
| Pilot-trained category rule | 817 (58.4%) | $0.00005773 |
| Per-question oracle (unavailable in practice) | 962 (68.7%) | — |

API cost includes the selected answer-model charge and JEV's hosted decision charge; **Laya's local compute is not priced**. These are observed one-successful-call operating points, not a threshold sweep or full deployment cost. JEV selected Mini on **1,394/1,400** items and tied always-Mini accuracy while adding an API fee. Laya selected Mini on **1,099/1,400**. Among **302** questions where exactly one candidate was correct, Laya selected that candidate **178** times versus JEV's **156** (Laya−JEV **+7.28 pp**, paired 95% CI **[+2.65, +11.92] pp**).

Laya's secondary paired gains over fixed Mini (**+1.57 pp**, 95% CI **[+0.50, +2.64]**) and fixed Gemini (**+2.29 pp**, **[+0.14, +4.43]**) are positive at nominal 95% confidence; these multiple secondary intervals were not multiplicity-adjusted. Its gain over the **pilot-trained category rule** is **+1.50 pp [−0.29, +3.36]**, unresolved. The fixed model chosen on the 280-question pilot was Gemini; it underperformed Mini on this holdout. Two Gemini answers ended with `finish_reason=error` and scored wrong; no router decision was missing.

Read the [full held-out findings](results/holdout-mini-gemini/report.md), [machine summary](results/holdout-mini-gemini/summary.json), and [per-question scores](results/holdout-mini-gemini/per_prompt.csv). The earlier [Mini/Gemini](results/mmlu-pro-280-mini/report.md) and [Qwen/Gemini](results/mmlu-pro-280/report.md) runs are **development pilots**, not additional held-out tests; Qwen produced 69 invalid answers in its pilot (65 truncations).

## Frozen protocol and external precedent

- The MMLU-Pro test revision is `b189ec765aa7ed75c8acfea42df31fdae71f97be`. Before requesting holdout outcomes, the manifest froze **100 questions per category**, seed `20260922`, the Mini/Gemini pool, a 2 pp minimum useful **absolute** router difference, the primary all-question JEV−Laya accuracy contrast, and a complete-sample stop rule. It excluded **430 previously sampled MMLU-Pro IDs** plus normalized duplicate question texts; 385 additional candidate rows were removed by text deduplication. All 1,400 published questions have unique IDs and normalized texts and no overlap with the retained pilot questions. The complete state and descriptions fit Laya's pinned 512-token context; 504 otherwise eligible source rows were too long. This estimates an **equally weighted, short-input 14-category population**, not native MMLU-Pro frequencies or production traffic.
- Each answer model received the same zero-shot multiple-choice prompt, temperature 0, `max_tokens=4096`, strict per-row JSON-schema letter enum, and `provider.require_parameters=true`. Invalid or non-normal completions scored wrong. Both routers received the same question and candidate descriptions, with candidate presentation order balanced **700/700 within categories**. They saw neither gold nor candidate answers. Laya used `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` locally; JEV used `typesafe/jev-1.13` via OpenRouter decisions. Neither router was trained on this holdout; the fixed-model and category policies were selected from the separate Mini/Gemini pilot.
- The report compares both fixed candidates, the pilot-selected fixed Gemini and category rule, a query-independent random mixture at each router's observed selection rate, and an unattainable per-question oracle. The primary interval resamples paired **questions within category**, not independent candidate completions. The random and cost-matched mixtures are descriptive expectations, not inferential claims; no router threshold was swept. Candidate outputs are cached so both routers face the same observed answer outcomes.
- This adapts [RouterBench](https://arxiv.org/abs/2403.12031)'s fixed-mixture/oracle and cost–quality controls, [RouteLLM](https://arxiv.org/abs/2406.18665)'s held-out and matched-random-call-rate design, and [LLMRouterBench](https://arxiv.org/abs/2601.07206)'s common-pool baseline comparisons. The direct [sysone-bench](https://github.com/instax-dutta/sysone-bench/blob/master/REPORT.md) comparison demonstrates pinned, byte-identical JEV/Laya inputs, but its classification results do not measure this answer-model-selection task.

## Reproduce

With locally retained `runs/` records (not distributed), **no paid requests**:

```bash
uv sync --extra dev --python 3.12
uv run pytest -q tests/test_benchmark.py
uv run python benchmark.py report --run runs/holdout-mini-gemini --out results/holdout-mini-gemini
```

The published CSV and summary allow independent recounts without raw inference records. To reconstruct the frozen sample locally, the five excluded pilot runs must also be retained:

```bash
uv run python benchmark.py prepare --run runs/holdout-mini-gemini --pool mini-gemini \
  --per-category 100 --exclude-run runs/main-structured \
  --exclude-run runs/pilot-math-history --exclude-run runs/pilot-mathqa-history \
  --exclude-run runs/pilot-code-history --exclude-run runs/pilot-phi-gemini \
  --full-laya-context --counterbalance-order \
  --pilot-summary results/mmlu-pro-280-mini/summary.json
```

Fresh `benchmark.py answer` and hosted `benchmark.py route --router jev` requests **incur charges** and require an OpenRouter key; the existing run records resume without paying twice for completed rows. The append-only shared ledger stopped unknown billing until key usage was checked. The holdout's provider-reported run charges were **$0.216064323**, plus a **$0.05 conservative ceiling** for one transport-failed JEV request with no per-generation billing ID; that ceiling is *not an observed charge*. Cumulative ledger exposure is **$0.57804812896**, below the approved **$5 cap**. Raw responses, the API key, and model weights remain local.

## Limits

Only one candidate completion per question was collected, even at temperature 0. Public MMLU-Pro items may occur in model pretraining; ID/text disjointness from our pilots does not prove decontamination. The short-input filter, hand-written model descriptions, equal-category weights, and local-Laya-versus-hosted-JEV hardware/network differences limit transfer. JEV's returned provider/model and Laya's checkpoint are in the machine summary. Laya's local hardware cost is unpriced, so API-dollar comparisons do not establish total cost efficiency. A confidence interval crossing zero on another task would not prove equivalence.

## License

MIT © 2026 rnunley

MMLU-Pro (MIT) https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro  
Laya (Apache-2.0) https://github.com/NandhaKishorM/laya

No weights or raw responses in this repo.
