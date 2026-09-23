from decimal import Decimal

import pytest

from benchmark import bootstrap, ledger_total, parse_answer, score_rows


def test_answer_requires_exact_json_enum_and_normal_finish():
    assert parse_answer('{"answer":"B"}', 3) == "B"
    for text, count, finish in [
        ('{"answer":"D"}', 3, "stop"),
        ('{"answer":"B"}', 3, "length"),
        ('{"answer":"B","reason":"extra"}', 3, "stop"),
        ('{"answer":2}', 3, "stop"),
        ('{"answer":"K"}', 10, "stop"),
        ('ANSWER: B', 3, "stop"),
        ('{"answer":"B"} trailing', 3, "stop"),
    ]:
        assert parse_answer(text, count, finish) is None


def test_router_failure_scores_zero_without_defaulting_to_a_candidate():
    rows = [
        {"category": "a", "qwen_correct": True, "gemini_correct": False, "jev_correct": False, "laya_correct": True, "oracle_correct": True},
        {"category": "a", "qwen_correct": False, "gemini_correct": True, "jev_correct": True, "laya_correct": False, "oracle_correct": True},
        {"category": "b", "qwen_correct": True, "gemini_correct": True, "jev_correct": False, "laya_correct": True, "oracle_correct": True},
    ]
    score = score_rows(rows)
    assert score["counts"]["jev"] == 1
    assert score["counts"]["laya"] == 2
    assert score["counts"]["oracle"] == 3
    assert score["choice_hit_rate"] == {"jev": 0.5, "laya": 0.5, "qwen": 0.5, "gemini": 0.5}
    assert score["gap_to_oracle"]["jev"] == pytest.approx(2 / 3)


def test_choice_hit_rate_is_null_without_distinguishing_question():
    row = {"category": "a", "qwen_correct": False, "gemini_correct": False, "jev_correct": False, "laya_correct": False, "oracle_correct": False}
    score = score_rows([row])
    assert score["choice_hit_rate"] == {"jev": None, "laya": None, "qwen": None, "gemini": None}
    assert score["counts"]["one_correct_only"] == 0


def test_shared_ledger_includes_retries_and_rejects_unknown_or_invalid_cost():
    assert ledger_total([{"cost": "0.02"}, {"cost": "0.03"}, {"cost": "0.001"}]) == Decimal("0.051")
    for entry in ({"cost": None}, {"cost": -1}, {"cost": "NaN"}, {"cost": True}):
        with pytest.raises(ValueError):
            ledger_total([entry])


def test_paired_bootstrap_preserves_category_sizes_and_direction():
    rows = [
        {"category": "a", "jev_correct": True, "laya_correct": False},
        {"category": "a", "jev_correct": True, "laya_correct": False},
        {"category": "b", "jev_correct": False, "laya_correct": True},
    ]
    assert bootstrap(rows) == [100 / 3, 100 / 3]


def test_curated_report_rejects_smoke_sample(tmp_path):
    from benchmark import atomic_json, digest, report

    run = tmp_path / "smoke"
    row = {"question_id": 1, "question": "Q?", "options": ["A", "B"], "gold": "A", "category": "test"}
    atomic_json(run / "questions.json", [row])
    atomic_json(run / "manifest.json", {"questions_sha256": digest([row]), "sample_ids": [1]})
    with pytest.raises(ValueError, match="280-question"):
        report(run, tmp_path / "published")
    assert not (tmp_path / "published").exists()


def test_timeout_records_unknown_charge_and_blocks_next_call(tmp_path, monkeypatch):
    import httpx
    from benchmark import QWEN, call_paid

    run = tmp_path / "main"
    run.mkdir()
    (tmp_path / "openrouter_usage.jsonl").touch()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")

    class TimedOut:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def post(self, *_args, **_kwargs):
            raise httpx.ReadTimeout("test-only")

    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: TimedOut())
    with pytest.raises(ValueError, match="unknown billing"):
        call_paid(run, 1, QWEN, "https://example.invalid", {"model": QWEN}, 1)
    with pytest.raises(ValueError, match="Unknown billing in ledger"):
        call_paid(run, 2, QWEN, "https://example.invalid", {"model": QWEN}, 1)


def test_reservations_count_as_exposure_until_settled():
    entries = [
        {"cost": "1.00"},
        {"event": "reserve", "id": "a", "reserve": "0.05"},
        {"event": "reserve", "id": "b", "reserve": "0.05"},
    ]
    assert ledger_total(entries) == Decimal("1.10")
    entries.append({"event": "settle", "id": "a", "cost": "0.002"})
    assert ledger_total(entries) == Decimal("1.052")
    entries.append({"event": "settle", "id": "b", "cost": "0.003"})
    assert ledger_total(entries) == Decimal("1.005")


def test_unknown_settlement_requires_explicit_key_usage_reconciliation():
    entries = [{"event": "reserve", "id": "a", "reserve": "0.05"},
               {"event": "settle", "id": "a", "cost": None}]
    with pytest.raises(ValueError, match="Unknown billing"):
        ledger_total(entries)
    entries.append({"event": "reconcile", "id": "a", "cost": "0.002"})
    assert ledger_total(entries) == Decimal("0.002")
    entries.append({"event": "reconcile", "id": "a", "cost": "0"})
    with pytest.raises(ValueError, match="Duplicate"):
        ledger_total(entries)


def test_two_paid_calls_can_overlap_without_overspending(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import httpx
    from benchmark import QWEN, call_paid

    run = tmp_path / "main"
    run.mkdir()
    (tmp_path / "openrouter_usage.jsonl").touch()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    barrier = Barrier(2, timeout=3)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def post(self, *_args, **_kwargs):
            barrier.wait()
            return httpx.Response(200, json={"usage": {"cost": 0.001}})

    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: FakeClient())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call_paid, run, n, QWEN, "https://example.invalid", {"model": QWEN}, 1) for n in (1, 2)]
        assert all(f.result(timeout=5)[0]["usage"]["cost"] == 0.001 for f in futures)
    import json
    entries = [json.loads(line) for line in (tmp_path / "openrouter_usage.jsonl").read_text().splitlines()]
    assert ledger_total(entries) == Decimal("0.002")
