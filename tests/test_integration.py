"""Opt-in live integration check. Uses the same shared cost ledger as the full run."""
import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from benchmark import MODEL_POOLS, collect_answers, collect_routes, load, prepare, report


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_OPENROUTER_INTEGRATION") != "1", reason="Paid integration test is opt-in")
def test_three_public_questions_end_to_end():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "Configure the git-ignored .env before running integration"
    pool = os.environ.get("BENCHMARK_INTEGRATION_POOL", "qwen-gemini")
    models = MODEL_POOLS[pool]
    run = Path("runs/integration-structured" if pool == "qwen-gemini" else "runs/integration-mini-gemini")
    prepare(run, smoke=3, candidate_pool=pool)
    for model in models:
        collect_answers(run, model, retry_errors=True, workers=2)
    collect_routes(run, "laya")
    collect_routes(run, "jev", retry_errors=True)
    report(run)

    summary = load(run / "summary.json")
    with (run / "per_prompt.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == summary["n"] == 3
    assert summary["candidate_models"] == list(models)
    for router in ("jev", "laya"):
        calculated = sum(row[f"{router}_correct"] == "True" for row in rows)
        assert calculated == summary["counts"][router]
        for row in rows:
            selected = row[f"{router}_selected_model"]
            expected = selected in models and row["first_correct" if selected == models[0] else "second_correct"] == "True"
            assert (row[f"{router}_correct"] == "True") == expected
        distinguishing = [row for row in rows if row["first_correct"] != row["second_correct"]]
        assert summary["choice_hit_counts"][router] == sum(row[f"{router}_correct"] == "True" for row in distinguishing)
    assert summary["routing_failures"] == {"jev": 0, "laya": 0}
    assert float(summary["shared_ledger_usd"]) < 5
    assert os.environ["OPENROUTER_API_KEY"] not in (run / "report.md").read_text()
