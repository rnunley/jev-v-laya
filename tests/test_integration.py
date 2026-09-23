"""Opt-in live integration check. Uses the same shared cost ledger as the full run."""
import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from benchmark import GEMINI, QWEN, collect_answers, collect_routes, load, prepare, report


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_OPENROUTER_INTEGRATION") != "1", reason="Paid integration test is opt-in")
def test_three_public_questions_end_to_end():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "Configure the git-ignored .env before running integration"
    run = Path("runs/integration-structured")
    prepare(run, smoke=3)
    collect_answers(run, QWEN, retry_errors=True, workers=2)
    collect_answers(run, GEMINI, retry_errors=True, workers=2)
    collect_routes(run, "laya")
    collect_routes(run, "jev", retry_errors=True)
    report(run)

    summary = load(run / "summary.json")
    with (run / "per_prompt.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == summary["n"] == 3
    for router in ("jev", "laya"):
        calculated = sum(row[f"{router}_correct"] == "True" for row in rows)
        assert calculated == summary["counts"][router]
        for row in rows:
            selected = row[f"{router}_selected_model"]
            expected = selected in (QWEN, GEMINI) and row["qwen_correct" if selected == QWEN else "gemini_correct"] == "True"
            assert (row[f"{router}_correct"] == "True") == expected
        distinguishing = [row for row in rows if row["qwen_correct"] != row["gemini_correct"]]
        assert summary["choice_hit_counts"][router] == sum(row[f"{router}_correct"] == "True" for row in distinguishing)
    assert summary["routing_failures"] == {"jev": 0, "laya": 0}
    assert float(summary["shared_ledger_usd"]) < 5
    assert os.environ["OPENROUTER_API_KEY"] not in (run / "report.md").read_text()
