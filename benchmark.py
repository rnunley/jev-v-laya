"""Paired, budget-limited MMLU-Pro routing benchmark. Raw responses remain local."""
import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from uuid import uuid4

import httpx

DATASET = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
SEED = 20260922
QWEN = "qwen/qwen3.5-9b"
MINI = "openai/gpt-4.1-mini"
GEMINI = "google/gemini-2.5-flash-lite"
JEV = "typesafe/jev-1.13"
LAYA = "convaiinnovations/laya"
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
MODEL_POOLS = {"qwen-gemini": (QWEN, GEMINI), "mini-gemini": (MINI, GEMINI)}
MODEL_DESCRIPTIONS = {
    QWEN: "Qwen3.5-9B by Qwen; 9B-parameter general reasoning model",
    MINI: "GPT-4.1 Mini by OpenAI; mid-sized general-purpose model",
    GEMINI: "Gemini 2.5 Flash Lite by Google; lightweight general reasoning model",
}
SYSTEM = "Answer the multiple-choice question. Return only a JSON object with the selected option letter in the answer field."
ROUTE_INSTRUCTIONS = "Which model is more likely to answer this multiple-choice question correctly? Select one model."


def questions_for(models):
    return {"route": {"type": "choice", "instructions": ROUTE_INSTRUCTIONS,
                      "criteria": {model: MODEL_DESCRIPTIONS[model] for model in models}}}


PRICES = {QWEN: {"input": "0.08", "output": "0.13", "url": "https://openrouter.ai/qwen/qwen3.5-9b"}, MINI: {"input": "0.40", "output": "1.60", "url": "https://openrouter.ai/openai/gpt-4.1-mini"}, GEMINI: {"input": "0.10", "output": "0.40", "url": "https://openrouter.ai/google/gemini-2.5-flash-lite"}, JEV: {"input": "0.042", "output": "0", "url": "https://openrouter.ai/typesafe/jev-1.13/"}}
BOUND_PRICES = {QWEN: {"input": "0.20", "output": "0.30"}, MINI: {"input": "0.50", "output": "2.00"}, GEMINI: {"input": "0.20", "output": "0.80"}}
CAP = Decimal("5")
STOP = Decimal("4.95")
MAX_TOKENS = 4096


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.write("\n")
        name = tmp.name
    os.replace(name, path)


def load(path):
    return json.loads(Path(path).read_text())


def state(row):
    return "Question: " + row["question"] + "\nOptions:\n" + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(row["options"]))


def answer_format(option_count):
    return {"type": "json_schema", "json_schema": {"name": "multiple_choice_answer",
            "strict": True, "schema": {"type": "object",
            "properties": {"answer": {"type": "string",
                                      "enum": [chr(65 + i) for i in range(option_count)]}},
            "required": ["answer"], "additionalProperties": False}}}


def parse_answer(text, option_count, finish_reason="stop"):
    if finish_reason != "stop" or not isinstance(text, str):
        return None
    try:
        answer = json.loads(text)
    except ValueError:
        return None
    if not isinstance(answer, dict) or set(answer) != {"answer"}:
        return None
    letter = answer["answer"]
    return letter if isinstance(letter, str) and len(letter) == 1 and "A" <= letter <= chr(64 + option_count) else None


def prepare(run, smoke=None, candidate_pool="qwen-gemini"):
    from datasets import load_dataset
    models = MODEL_POOLS[candidate_pool]
    rows = load_dataset(DATASET, split="test", revision=REVISION)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    if len(by_category) != 14:
        raise ValueError(f"Expected 14 categories, got {len(by_category)}")
    rng = random.Random(SEED)
    chosen = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda r: r["question_id"])
        chosen.extend(rng.sample(pool, 20))
    chosen.sort(key=lambda r: r["question_id"])
    if smoke is not None:
        if not 1 <= smoke <= len(chosen):
            raise ValueError("Smoke size outside sample")
        chosen = chosen[:smoke]
    clean = []
    ids = set()
    for row in chosen:
        qid = row["question_id"]
        opts = row["options"]
        idx = row["answer_index"]
        if qid in ids or not 1 <= len(opts) <= 10 or not 0 <= idx < len(opts) or row["answer"] != chr(65 + idx):
            raise ValueError(f"Invalid gold/options/duplicate for {qid}")
        ids.add(qid)
        clean.append({"question_id": qid, "question": row["question"], "options": opts, "gold": row["answer"], "category": row["category"]})
    manifest = {"dataset": DATASET, "dataset_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro", "revision": REVISION, "seed": SEED, "sample_per_category": 20, "smoke": smoke, "sample_ids": [r["question_id"] for r in clean], "questions_sha256": digest(clean), "candidate_models": list(models), "laya": {"model": LAYA, "revision": LAYA_REVISION, "device": "mps or cpu"}, "jev": JEV, "prices_usd_per_million": {model: PRICES[model] for model in (*models, JEV)}, "system": SYSTEM, "questions": questions_for(models), "temperature": 0, "max_tokens": MAX_TOKENS, "response_format_example": answer_format(10), "response_format_enum_policy": "A through the last listed option letter, per row", "provider_preferences": {"require_parameters": True}, "answer_endpoint": "https://openrouter.ai/api/v1/chat/completions", "route_endpoint": "https://openrouter.ai/api/alpha/decisions"}
    run.mkdir(parents=True, exist_ok=True)
    for name, content in (("manifest.json", manifest), ("questions.json", clean)):
        path = run / name
        if path.exists():
            if load(path) != content:
                raise ValueError(f"Conflicting existing {path}")
        else:
            atomic_json(path, content)
    ledger = run.parent / "openrouter_usage.jsonl"
    ledger.touch(exist_ok=True)
    print(f"Prepared {len(clean)} questions in {run}; hash {digest(manifest)}")


def run_data(run):
    manifest, rows = load(run / "manifest.json"), load(run / "questions.json")
    if digest(rows) != manifest["questions_sha256"] or [r["question_id"] for r in rows] != manifest["sample_ids"]:
        raise ValueError("Manifest and questions mismatch")
    return manifest, rows, digest(manifest)


def ledger_total(entries):
    total = Decimal(0)
    pending = {}
    resolved = {entry["id"] for entry in entries if entry.get("event") == "reconcile"}
    unknown = set()
    if len(resolved) != sum(entry.get("event") == "reconcile" for entry in entries):
        raise ValueError("Duplicate billing reconciliation")
    for entry in entries:
        event = entry.get("event")
        if event == "reserve":
            reserve = Decimal(str(entry["reserve"]))
            if not reserve.is_finite() or reserve <= 0 or entry["id"] in pending:
                raise ValueError("Invalid reservation")
            pending[entry["id"]] = reserve
            continue
        cost = entry.get("cost")
        if cost is None and event == "settle" and entry["id"] in resolved:
            unknown.add(entry["id"])
            cost = "0"
        if cost is None or isinstance(cost, bool):
            raise ValueError("Unknown billing in ledger; check actual spend before further paid calls")
        amount = Decimal(str(cost))
        if not amount.is_finite() or amount < 0:
            raise ValueError("Invalid billing entry")
        if event == "settle":
            if entry["id"] not in pending:
                raise ValueError("Unmatched settlement")
            del pending[entry["id"]]
        elif event not in (None, "reconcile"):
            raise ValueError("Unknown ledger event")
        total += amount
    if resolved != unknown:
        raise ValueError("Reconciliation without unknown settlement")
    return total + sum(pending.values(), Decimal(0))


@contextmanager
def ledger_lock(run):
    path = run.parent / "openrouter_usage.jsonl"
    if not path.exists():
        raise ValueError(f"Missing ledger {path}; refusing paid request")
    with open(path, "r+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        entries = [json.loads(line) for line in file if line.strip()]
        yield file, entries


def append_ledger(file, entry):
    file.seek(0, 2)
    file.write(canonical(entry) + "\n")
    file.flush()
    os.fsync(file.fileno())


def call_paid(run, qid, model, url, payload, attempt):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY must be set before paid calls")
    # Reserve five cents so concurrent requests cannot reach the $5 cap.
    price = BOUND_PRICES.get(model, PRICES[model])
    maximum = (Decimal(len(canonical(payload).encode())) * Decimal(price["input"]) +
               Decimal(MAX_TOKENS if model != JEV else 512) * Decimal(price["output"])) / 1000000
    reserve = Decimal("0.05")
    if maximum >= reserve:
        raise ValueError(f"Request {qid} may exceed $0.05 at current rates")
    request_id = uuid4().hex
    with ledger_lock(run) as (file, entries):
        exposure = ledger_total(entries)
        if exposure >= STOP or exposure + reserve >= CAP:
            raise ValueError(f"Budget stop: ledger exposure ${exposure}; no request started")
        append_ledger(file, {"event": "reserve", "id": request_id, "reserve": str(reserve),
                             "run": str(run), "question_id": qid, "model": model, "attempt": attempt})

    started = time.monotonic()
    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload)
    except httpx.RequestError as exc:
        with ledger_lock(run) as (file, _):
            append_ledger(file, {"event": "settle", "id": request_id, "run": str(run),
                                 "question_id": qid, "model": model, "attempt": attempt,
                                 "status": None, "cost": None})
        raise ValueError("Transport or timeout with unknown billing; stop paid requests and check actual spend") from exc

    duration = time.monotonic() - started
    try:
        data = response.json()
    except ValueError:
        data = {}
    usage = data.get("usage") if isinstance(data, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    with ledger_lock(run) as (file, _):
        append_ledger(file, {"event": "settle", "id": request_id, "run": str(run),
                             "question_id": qid, "model": model, "attempt": attempt,
                             "status": response.status_code, "cost": cost})
    if cost is None:
        raise ValueError(f"HTTP {response.status_code} lacks usage.cost; stop paid requests and check actual spend")
    amount = Decimal(str(cost))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid billing entry; stop paid requests")
    if response.status_code in (401, 402, 403):
        raise ValueError(f"HTTP {response.status_code}; abort")
    if response.status_code in (429, 500, 502, 503, 504):
        return None, duration, response.status_code, cost
    if response.status_code >= 400:
        raise ValueError(f"HTTP {response.status_code}; abort")
    return data, duration, response.status_code, cost


def paid_retry(run, qid, model, url, payload):
    for attempt in range(1, 4):
        data, duration, status, cost = call_paid(run, qid, model, url, payload, attempt)
        if data is not None:
            return data, duration, cost
        if attempt < 3:
            time.sleep(attempt)
    raise ValueError(f"HTTP {status} after three attempts; last billed cost {cost}")


def record_path(run, kind, model, qid):
    return run / kind / model / f"{qid}.json"


def collect_answers(run, model, retry_errors=False, workers=8):
    """Collect and grade one hosted candidate's answers, with bounded concurrency."""
    manifest, rows, manifest_hash = run_data(run)
    if model not in manifest["candidate_models"] or not 1 <= workers <= 16:
        raise ValueError("Unsupported model or worker count")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY must be set before paid calls")

    def answer_one(row):
        qid = row["question_id"]
        path = record_path(run, "answers", model, qid)
        if path.exists():
            old = load(path)
            if old.get("manifest_hash") != manifest_hash or old.get("requested_model") != model:
                raise ValueError(f"Conflicting record {path}")
            if not retry_errors or not old.get("error") or old.get("error_kind") != "transport":
                return
        result = {"question_id": qid, "manifest_hash": manifest_hash, "requested_model": model,
                  "returned_model": None, "provider": None, "duration_seconds": None,
                  "cost": None, "error": None}
        try:
            payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": state(row)}], "temperature": 0, "max_tokens": MAX_TOKENS,
                       "response_format": answer_format(len(row["options"])),
                       "provider": {"require_parameters": True}}
            data, result["duration_seconds"], result["cost"] = paid_retry(
                run, qid, model, manifest["answer_endpoint"], payload)
            result["returned_model"] = data.get("model")
            result["provider"] = data.get("provider")
            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content")
            result.update({"text": text, "finish_reason": choice.get("finish_reason"),
                           "choice": parse_answer(text, len(row["options"]), choice.get("finish_reason")),
                           "gold": row["gold"]})
            result["correct"] = result["choice"] == row["gold"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            message = (str(exc) if isinstance(exc, ValueError) and
                       str(exc).startswith(("HTTP ", "Budget stop:", "Transport or timeout with unknown billing",
                                            "Unknown billing in ledger"))
                       else type(exc).__name__)
            result["error"] = message
            result["error_kind"] = "transport" if message.startswith(("HTTP ", "Transport or timeout", "Unknown billing in ledger")) else "fatal"
            atomic_json(path, result)
            raise
        atomic_json(path, result)
        print(f"answers {model} {qid}: {result['choice']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(answer_one, rows):
            pass


def collect_routes(run, router, retry_errors=False):
    """Collect routes from one router (laya local or jev hosted). Single responsibility: routing decision collection."""
    manifest, rows, manifest_hash = run_data(run)
    if router not in ("laya", "jev"):
        raise ValueError("Unsupported router")
    if router == "laya":
        from huggingface_hub import snapshot_download
        import laya
        import torch
        path = snapshot_download(LAYA, revision=LAYA_REVISION, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"], local_dir=".cache/laya")
        agent = laya.load(path, device="mps" if torch.backends.mps.is_available() else "cpu")
    elif not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY must be set before paid calls")
    for row in rows:
        qid = row["question_id"]
        path = record_path(run, "routes", router, qid)
        if path.exists():
            old = load(path)
            if old.get("manifest_hash") != manifest_hash or old.get("requested_model") != (JEV if router == "jev" else LAYA):
                raise ValueError(f"Conflicting record {path}")
            if not retry_errors or not old.get("error") or old.get("error_kind") != "transport":
                continue
        requested = JEV if router == "jev" else LAYA
        result = {"question_id": qid, "manifest_hash": manifest_hash, "requested_model": requested, "returned_model": None, "provider": None, "duration_seconds": None, "cost": None, "error": None}
        try:
            if router == "laya":
                start = time.monotonic()
                data = agent.predict(state(row), manifest["questions"])
                result["duration_seconds"] = time.monotonic() - start
                result["returned_model"] = LAYA + "@" + LAYA_REVISION
            else:
                payload = {"model": JEV, "state": state(row), "questions": manifest["questions"]}
                data, result["duration_seconds"], result["cost"] = paid_retry(run, qid, requested, manifest["route_endpoint"], payload)
                result["returned_model"] = data.get("model")
                result["provider"] = data.get("provider")
            answer = data.get("answers", {}).get("route", {})
            selection = answer.get("choice") if isinstance(answer, dict) else None
            result.update({"selected_model": selection if selection in manifest["candidate_models"] else None, "probabilities": answer.get("probabilities") if isinstance(answer, dict) else None})
            if result["selected_model"] is None:
                result["error"] = "Invalid or missing route choice"
                result["error_kind"] = "content"
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            message = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("HTTP ", "Budget stop:", "Transport or timeout with unknown billing")) else type(exc).__name__
            result["error"] = message
            result["error_kind"] = "transport" if message.startswith(("HTTP ", "Transport or timeout")) else "fatal"
            atomic_json(path, result)
            raise
        atomic_json(path, result)
        print(f"routes {router} {qid}: {result.get('selected_model')}", flush=True)


def score_rows(rows):
    n = len(rows)
    keys = {"jev": "jev_correct", "laya": "laya_correct", "first": "first_correct", "second": "second_correct", "oracle": "oracle_correct"}
    counts = {k: sum(bool(r[v]) for r in rows) for k, v in keys.items()}
    counts["best_fixed_in_hindsight"] = max(counts["first"], counts["second"])
    distinguish = [r for r in rows if bool(r["first_correct"]) != bool(r["second_correct"])]
    counts["one_correct_only"] = len(distinguish)
    hits = {policy: sum(bool(r[f"{policy}_correct"]) for r in distinguish)
            for policy in ("jev", "laya", "first", "second")}
    return {
        "n": n, "counts": counts,
        "accuracy": {k: v / n for k, v in counts.items() if k != "one_correct_only"},
        "choice_hit_counts": hits,
        "choice_hit_rate": {k: (v / len(distinguish) if distinguish else None) for k, v in hits.items()},
        "gap_to_oracle": {k: (counts["oracle"] - counts[k]) / n for k in ("jev", "laya")},
    }


def bootstrap(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(int(row["jev_correct"]) - int(row["laya_correct"]))
    rng = random.Random(SEED)
    draws = []
    for _ in range(10000):
        draws.append(100 * sum(rng.choice(values) for values in groups.values() for _ in values) / len(rows))
    draws.sort()
    return [draws[249], draws[9749]]


def report(run, out=None):
    manifest, questions, manifest_hash = run_data(run)
    if out is not None and len(questions) != 280:
        raise ValueError("Curated output requires the complete 280-question sample")
    first, second = manifest["candidate_models"]
    models = (first, second)
    records = {model: {} for model in (*models, "jev", "laya")}
    for model in records:
        kind = "answers" if model in models else "routes"
        for question in questions:
            path = record_path(run, kind, model, question["question_id"])
            if not path.exists():
                raise ValueError(f"Incomplete run: missing {path}")
            rec = load(path)
            requested = JEV if model == "jev" else LAYA if model == "laya" else model
            if (rec.get("manifest_hash") != manifest_hash or rec.get("requested_model") != requested
                    or rec.get("error_kind") in ("fatal", "transport")):
                raise ValueError(f"Conflicting/unfinished record {path}")
            records[model][question["question_id"]] = rec

    rows = []
    for question in questions:
        qid = question["question_id"]
        a, b, j, l = (records[k][qid] for k in (*models, "jev", "laya"))
        for answer in (a, b):
            if (answer.get("gold") != question["gold"]
                    or answer.get("correct") != (answer.get("choice") == question["gold"])):
                raise ValueError(f"Inconsistent candidate grading {qid}")
        row = {"question_id": qid, "category": question["category"], "gold": question["gold"],
               "first_model": first, "first_choice": a["choice"], "first_correct": bool(a["correct"]),
               "second_model": second, "second_choice": b["choice"], "second_correct": bool(b["correct"]),
               "jev_selected_model": j["selected_model"], "jev_error": j["error"],
               "laya_selected_model": l["selected_model"], "laya_error": l["error"]}
        for router, rec in (("jev", j), ("laya", l)):
            selected = rec["selected_model"]
            row[f"{router}_correct"] = bool(selected and records[selected][qid]["correct"])
        row["oracle_correct"] = row["first_correct"] or row["second_correct"]
        rows.append(row)

    stats = score_rows(rows)
    stats["candidate_models"] = list(models)
    stats["paired_difference_pp"] = 100 * (stats["accuracy"]["jev"] - stats["accuracy"]["laya"])
    stats["paired_bootstrap_95_ci_pp"] = bootstrap(rows)
    distinguish = [r for r in rows if r["first_correct"] != r["second_correct"]]
    stats["choice_difference_pp"] = (100 * (stats["choice_hit_rate"]["jev"] - stats["choice_hit_rate"]["laya"])
                                     if distinguish else None)
    stats["choice_bootstrap_95_ci_pp"] = bootstrap(distinguish) if distinguish else None
    stats["category"] = {category: score_rows([r for r in rows if r["category"] == category])
                         for category in sorted({r["category"] for r in rows})}
    stats["routing_failures"] = {router: sum(records[router][r["question_id"]]["selected_model"] is None for r in rows)
                                 for router in ("jev", "laya")}
    stats["selection_counts"] = {router: {model: sum(r[f"{router}_selected_model"] == model for r in rows)
                                           for model in models} for router in ("jev", "laya")}
    stats["invalid_answers"] = {slot: sum(r[f"{slot}_choice"] is None for r in rows)
                                for slot in ("first", "second")}
    stats["truncated_answers"] = {slot: sum(rec.get("finish_reason") == "length"
                                               for rec in records[model].values())
                                  for slot, model in (("first", first), ("second", second))}
    with ledger_lock(run) as (_, entries):
        stats["shared_ledger_usd"] = str(ledger_total(entries))
        unsettled = {entry["id"] for entry in entries if entry.get("event") == "reserve"} - {
            entry["id"] for entry in entries if entry.get("event") == "settle"}
        if unsettled:
            raise ValueError("Unsettled paid requests; reconcile before publishing")
        spend = defaultdict(Decimal)
        for entry in entries:
            if entry.get("event") != "reserve" and entry.get("cost") is not None:
                spend[(entry["run"], entry["model"])] += Decimal(str(entry["cost"]))
        stats["spend_usd"] = {model: str(spend[(str(run), model)]) for model in (*models, JEV)}
        stats["run_spend_usd"] = str(sum((spend[(str(run), model)] for model in (*models, JEV)), Decimal(0)))
        stats["unattributed_reconciled_usd"] = str(sum(
            (Decimal(str(entry["cost"])) for entry in entries if entry["model"] == "unattributed-key-usage"),
            Decimal(0)))
    stats["manifest_sha256"] = manifest_hash
    stats["returned_models_and_providers"] = {
        model: sorted({(rec.get("returned_model"), rec.get("provider")) for rec in records[model].values()},
                      key=canonical) for model in records}

    dest = out or run
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "summary.json", stats)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=dest, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        name = tmp.name
    os.replace(name, dest / "per_prompt.csv")

    fmt = lambda k: f"{stats['counts'][k]}/{stats['n']} ({100*stats['accuracy'][k]:.1f}%)"
    cat_lines = [
        f"| {category} | {s['n']} | {s['counts']['one_correct_only']} | "
        f"{s['choice_hit_counts']['jev']} | {s['choice_hit_counts']['laya']} | "
        f"{s['counts']['jev']} | {s['counts']['laya']} | {s['counts']['first']} | {s['counts']['second']} |"
        for category, s in stats["category"].items()]
    n_choice = stats["counts"]["one_correct_only"]
    choice_summary = (f"JEV **{stats['choice_hit_counts']['jev']}/{n_choice} "
                      f"({100*stats['choice_hit_rate']['jev']:.1f}%)**; Laya "
                      f"**{stats['choice_hit_counts']['laya']}/{n_choice} "
                      f"({100*stats['choice_hit_rate']['laya']:.1f}%)**. JEV minus Laya "
                      f"**{stats['choice_difference_pp']:.2f} pp**, stratified paired bootstrap 95% CI "
                      f"**[{stats['choice_bootstrap_95_ci_pp'][0]:.2f}, "
                      f"{stats['choice_bootstrap_95_ci_pp'][1]:.2f}] pp**."
                      if n_choice else "No prompt distinguished the candidates; routing-choice accuracy is null.")
    text = f"""# JEV vs Laya: MMLU-Pro routing results ({first} vs {second})

## Results

Equal-category sample: **{stats['n']}** questions.

### Routing-choice accuracy (distinguishable prompts)

Exactly one candidate answered gold on **{n_choice}** prompts. On these prompts, selecting that candidate is an identifiable correct routing decision. {choice_summary} Fixed {first} would score **{stats['choice_hit_counts']['first']}/{n_choice}** and fixed {second} **{stats['choice_hit_counts']['second']}/{n_choice}**. JEV chose {first} on **{stats['selection_counts']['jev'][first]}/{stats['n']}** prompts; Laya chose {second} on **{stats['selection_counts']['laya'][second]}/{stats['n']}**. The 10,000 seeded paired bootstrap resamples within each observed category. Both-correct and both-wrong prompts cannot identify a preferable route from gold.

### Downstream routed accuracy (all prompts)

This measures whether the selected candidate answered gold; it also depends on candidate ability and is **not** pure route-choice accuracy.

| Policy | Correct / N (accuracy) |
|---|---:|
| JEV | {fmt('jev')} |
| Laya | {fmt('laya')} |
| Fixed {first} | {fmt('first')} |
| Fixed {second} | {fmt('second')} |
| Best fixed in hindsight (not deployable as selected here) | {fmt('best_fixed_in_hindsight')} |
| Per-prompt oracle (not deployable) | {fmt('oracle')} |

JEV minus Laya downstream: **{stats['paired_difference_pp']:.2f} pp**, stratified paired bootstrap 95% percentile CI **[{stats['paired_bootstrap_95_ci_pp'][0]:.2f}, {stats['paired_bootstrap_95_ci_pp'][1]:.2f}] pp** (10,000 seeded resamples; 20/category). Oracle gaps: JEV {100*stats['gap_to_oracle']['jev']:.2f} pp; Laya {100*stats['gap_to_oracle']['laya']:.2f} pp. Routing failures: {stats['routing_failures']}; invalid candidate answers: {stats['invalid_answers']} (first/second respectively; truncations: {stats['truncated_answers']}). These are answer-budget-specific outcomes, not general routing skill. Neither interval proves equivalence.

| Category | N | One-correct-only | JEV choice hits | Laya choice hits | JEV routed | Laya routed | {first} correct | {second} correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(cat_lines) + f"""

## Method and provenance

[MMLU-Pro]({manifest['dataset_url']}) MIT-licensed test split at `{REVISION}`; 20 rows selected per sorted category with Python `random.Random({SEED})` from question-ID-sorted rows, then question-ID sorted globally. Equal category weighting is **not** population-weighted MMLU-Pro. Question IDs/hash, prompt, Choice criteria, generation settings, requested model IDs, price-page links and revision are frozen in the local manifest; manifest SHA-256: `{manifest_hash}`. Public [scores](per_prompt.csv) and [machine summary](summary.json) enable independent score checks. Neither router receives gold or candidate answers.

Both hosted candidates receive the same zero-shot system message `{SYSTEM}` and `Question: ...\\\\nOptions: ...` state, temperature 0, max_tokens {MAX_TOKENS}; OpenRouter JSON-schema structured output requires `{{"answer": "X"}}` with a per-row enum of listed option letters (`provider.require_parameters=true`). Only a valid one-key JSON object with normal finish scores; invalid/truncated outputs score wrong. Routers see only the state and one identical Choice question, instructions `{manifest['questions']['route']['instructions']}`, criteria `{canonical(manifest['questions']['route']['criteria'])}`. Laya `{LAYA}` at `{LAYA_REVISION}` runs locally on MPS if available else CPU (512-token English context default); JEV `{JEV}` uses OpenRouter decisions. Requested/returned model IDs and provider identities actually returned: `{canonical(stats['returned_models_and_providers'])}`. Null provider means no provider identity was reported.

HTTP 429/5xx retries at most twice when billed cost is known; 401/402/403, unknown cost, timeout and unknown billing stop. Append-only ledger includes earlier smoke, integration and both full runs. It stops at $4.95 before the next paid request; concurrent requests reserve $0.05 each strictly below the $5 ceiling, then settle to provider-reported cost. Provider-reported spend this run: `{canonical(stats['spend_usd'])}`; run total ${stats['run_spend_usd']}; shared ledger ${stats['shared_ledger_usd']}. The ledger includes ${stats['unattributed_reconciled_usd']} of key-usage reconciliation after earlier interrupted/unknown-billing calls; this is not attributed to a model or run. Local Laya has no OpenRouter cost.

## Limits

Academic multiple-choice questions do not measure production routing. Model descriptions are zero-shot metadata, not trained performance priors; one generation per candidate, even at temperature 0, can vary. The 512-token Laya context and answer budget can affect outcomes. This finite equal-category sample and bootstrap interval do not generalize automatically. Hosted JEV versus local Laya latency is not hardware-neutral. Best fixed uses hindsight; the oracle uses per-prompt gold and cannot be deployed. Router ties or overlapping uncertainty are not evidence of superiority.
"""
    (dest / "report.md").write_text(text)
    print(f"Reported {stats['n']} questions at {dest}; shared spend ${stats['shared_ledger_usd']}")


def compare_reports(original_dir, alternate_dir):
    """Append a deterministic same-sample comparison to the original report."""
    original_dir, alternate_dir = Path(original_dir), Path(alternate_dir)
    original, alternate = load(original_dir / "summary.json"), load(alternate_dir / "summary.json")
    if (original["candidate_models"] != [QWEN, GEMINI]
            or alternate["candidate_models"] != [MINI, GEMINI]
            or original["n"] != 280 or alternate["n"] != 280):
        raise ValueError("Comparison requires complete Qwen/Gemini and Mini/Gemini reports")
    def sample_keys(directory):
        with (directory / "per_prompt.csv").open(newline="") as file:
            return [(r["question_id"], r["category"], r["gold"]) for r in csv.DictReader(file)]
    keys = sample_keys(original_dir)
    if len(keys) != 280 or keys != sample_keys(alternate_dir):
        raise ValueError("Cannot compare mismatched question IDs, categories or gold")

    def metric(summary, policy):
        return f"{summary['counts'][policy]}/{summary['n']} ({100*summary['accuracy'][policy]:.1f}%)"

    def choice(summary, policy):
        n = summary["counts"]["one_correct_only"]
        return f"{summary['choice_hit_counts'][policy]}/{n} ({100*summary['choice_hit_rate'][policy]:.1f}%)" if n else "not identifiable"

    def line(label, s):
        return (f"| {label} | {s['counts']['one_correct_only']} | {choice(s,'jev')} | {choice(s,'laya')} | "
                f"{s['choice_difference_pp']:.2f} [{s['choice_bootstrap_95_ci_pp'][0]:.2f}, {s['choice_bootstrap_95_ci_pp'][1]:.2f}] | "
                f"{metric(s,'jev')} | {metric(s,'laya')} | {metric(s,'first')} | {metric(s,'second')} | "
                f"{metric(s,'oracle')} | {s['invalid_answers']['first']}/{s['invalid_answers']['second']} | "
                f"${s['run_spend_usd']} |")

    base = (original_dir / "report.md").read_text().split("\n## Cross-pool comparison", 1)[0].rstrip()
    appendix = f"""

## Cross-pool comparison

The same **280 question IDs, categories and gold answers** were used in both runs. Only the first candidate changed: Qwen3.5-9B was replaced by GPT-4.1 Mini. Gemini was queried anew, and both routers received the same state and Choice instructions with the new candidate's truthful model description. Prompt, JSON answer schema, temperature, token cap, dataset revision, seed and scoring remained fixed.

| Pool (first / second) | One-correct-only | JEV choice hit | Laya choice hit | JEV−Laya choice pp [95% CI] | JEV routed | Laya routed | Fixed first | Fixed Gemini | Oracle | Invalid first/second | Run USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{line('Qwen / Gemini', original)}
{line('GPT-4.1 Mini / Gemini', alternate)}

The routing-choice denominator is *pool-specific*: a prompt is counted only if exactly one candidate in that pool answered gold. The confidence intervals are 10,000 seeded stratified paired bootstrap resamples within each run, **not** a test of the difference between pools. Repeated candidate inference and provider routing may vary, so cross-run differences cannot be attributed exclusively to replacing Qwen. The original Qwen run had {original['invalid_answers']['first']} invalid answers ({original['truncated_answers']['first']} truncations); the Mini run had {alternate['invalid_answers']['first']} invalid answers ({alternate['truncated_answers']['first']} truncations). Routing decisions are blind to candidate answers and gold.

Auditable second-pool [report](../{alternate_dir.name}/report.md), [summary](../{alternate_dir.name}/summary.json) and [per-prompt scores](../{alternate_dir.name}/per_prompt.csv). The original pool's [summary](summary.json) and [per-prompt scores](per_prompt.csv) are alongside this report. Shared ledger spend after both runs: **${alternate['shared_ledger_usd']}**, below the $5 cap.
"""
    (original_dir / "report.md").write_text(base + appendix)
    print(f"Compared two 280-question pools in {original_dir / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Budgeted MMLU-Pro candidate and router benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze stratified public sample and manifest")
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--smoke", type=int)
    prep.add_argument("--pool", choices=tuple(MODEL_POOLS), default="qwen-gemini")
    answer = commands.add_parser("answer", help="Collect one hosted candidate's answers")
    answer.add_argument("--run", type=Path, required=True)
    answer.add_argument("--model", choices=tuple(MODEL_DESCRIPTIONS), required=True)
    answer.add_argument("--retry-errors", action="store_true")
    answer.add_argument("--workers", type=int, default=8, help="Concurrent requests for this candidate (1–16; default 8)")
    route = commands.add_parser("route", help="Collect local Laya or hosted JEV routes")
    route.add_argument("--run", type=Path, required=True)
    route.add_argument("--router", choices=("laya", "jev"), required=True)
    route.add_argument("--retry-errors", action="store_true")
    rep = commands.add_parser("report", help="Join complete run and publish scores")
    rep.add_argument("--run", type=Path, required=True)
    rep.add_argument("--out", type=Path)
    comparison = commands.add_parser("compare", help="Append same-sample pool comparison to original report")
    comparison.add_argument("--original", type=Path, required=True)
    comparison.add_argument("--alternate", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("answer", "route"):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    if args.command == "prepare":
        prepare(args.run, args.smoke, args.pool)
    elif args.command == "answer":
        collect_answers(args.run, args.model, args.retry_errors, args.workers)
    elif args.command == "route":
        collect_routes(args.run, args.router, args.retry_errors)
    elif args.command == "report":
        report(args.run, args.out)
    else:
        compare_reports(args.original, args.alternate)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
