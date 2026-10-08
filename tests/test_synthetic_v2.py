import json

from robust_likelihood.prompting import load_hypotheses
from robust_likelihood.storage import read_json, read_jsonl, repo_root
from robust_likelihood.synthetic_v2_items import (
    ATOMIC,
    DATASET_VERSION,
    EVALUATION_RULE_VERSION,
    HYPOTHESIS_VERSION,
    PROMPT_VERSION,
    authored_items,
    review_manifest,
)


def _canonical(rows: list[dict]) -> list[str]:
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]


def test_synthetic_v2_versions_and_reviewed_rows_match_the_files():
    root = repo_root()
    items = authored_items()
    stored = read_jsonl(root / "data" / "synthetic" / "alarm_v2" / "items.jsonl")
    review = read_json(root / "data" / "synthetic" / "alarm_v2" / "review_manifest.json")
    hypotheses = load_hypotheses(root, HYPOTHESIS_VERSION)
    assert _canonical(stored) == _canonical(items)
    assert _canonical(review["items"]) == _canonical(review_manifest()["items"])
    assert hypotheses.version == HYPOTHESIS_VERSION
    assert set(hypotheses.descriptions) == set(ATOMIC)
    assert "previous alarm instructions" in hypotheses.descriptions["intend_query"]
    assert "alarm-changing operation" in hypotheses.descriptions["intend_query"]
    assert all(row["approved"] is True for row in review["items"])
    assert {row["dataset_version"] for row in items} == {DATASET_VERSION}
    assert {row["prompt_version"] for row in items} == {PROMPT_VERSION}
    assert {row["evaluation_rule_version"] for row in items} == {EVALUATION_RULE_VERSION}
    report = (root / "templates" / "synthetic_alarm_report.md").read_text(encoding="utf-8")
    assert "synthetic-20261001T150250Z" in report
    assert "synthetic_v2" not in report


def test_polite_questions_request_the_operation_and_bare_cancel_is_unscored():
    items = {row["example_id"]: row for row in authored_items()}
    assert items["v2-op-create-polite"]["support"]["intend_create"] == "supported"
    assert items["v2-op-create-polite"]["support"]["intend_query"] == "rejected"
    assert items["v2-op-delete-polite"]["support"]["intend_delete"] == "supported"
    assert items["v2-op-delete-polite"]["support"]["intend_query"] == "rejected"
    assert items["v2-op-turn-off"]["support"]["intend_disable"] == "supported"
    assert items["v2-op-cancel-explicit-delete"]["support"]["intend_delete"] == "supported"
    assert items["v2-amb-cancel"]["evaluation"] == "not_scored"
    assert items["v2-amb-cancel"]["support"] is None
    assert items["v2-quote-delete"]["support"]["intend_query"] == "supported"
    assert items["v2-quote-delete"]["support"]["intend_delete"] == "rejected"
    assert items["v2-quote-delete"]["utt"].find("delete my alarm entirely") > 0


def test_reversed_plan_swaps_the_complete_hypothesis_and_rejects_partials():
    items = {row["example_id"]: row for row in authored_items()}
    forward = {row["id"]: row for row in items["v2-plan-delete-then-create"]["plan_hypotheses"]}
    reverse = {row["id"]: row for row in items["v2-plan-create-then-delete"]["plan_hypotheses"]}
    assert forward["plan_delete_then_create"]["expected"] == "supported"
    assert reverse["plan_delete_then_create"]["expected"] == "rejected"
    assert forward["plan_create_then_delete"]["expected"] == "rejected"
    assert reverse["plan_create_then_delete"]["expected"] == "supported"
    for bundle in (forward, reverse):
        assert bundle["plan_only_delete"]["expected"] == "rejected"
        assert bundle["plan_only_create"]["expected"] == "rejected"
        assert bundle["plan_only_delete"]["text"].startswith("Currently requests only")
        assert bundle["plan_delete_then_create"]["order"] == [
            "delete the morning alarm",
            "create a new alarm for 9:00",
        ]


def test_synthetic_v2_dry_run_plans_atomic_and_complete_plan_calls():
    from robust_likelihood.runner import run_comparison

    root = repo_root()
    summary = run_comparison(
        root,
        providers=("jev", "gpt", "claude"),
        dry_run=True,
        dataset="synthetic_v2",
    )
    assert summary["api_calls"] == 0
    assert summary["planned_requests"] == {"classification": 0, "pairwise": 219, "total": 219}
    assert summary["dataset"] == "synthetic_alarm_v2"
    assert summary["prompt_version"] == "v2"
    assert summary["hypothesis_description_version"] == "synthetic_v2"
    assert summary["unapproved_items"] == 0
    assert summary["live_run_allowed"] is True


def _pair(provider, example_id, hypothesis, probability):
    return {
        "cache_key": f"{provider}|{example_id}|{hypothesis}",
        "status": "success",
        "task": "pairwise",
        "provider": provider,
        "example_id": example_id,
        "hypothesis": hypothesis,
        "repeat_id": None,
        "attempt_id": 0,
        "parsed": {"plausibility_yes_probability": probability},
    }


def test_plan_scores_are_primary_and_bare_cancel_is_not_a_pass_or_failure():
    from robust_likelihood.synthetic_v2 import evaluate_synthetic_v2_records

    items = authored_items()
    by_id = {row["example_id"]: row for row in items}
    low = {name: 0.1 for name in ATOMIC}

    def atomic(provider, example_id, high, high_value=0.9):
        scores = dict(low)
        scores[high] = high_value
        return [_pair(provider, example_id, name, value) for name, value in scores.items()]

    def plans(provider, example_id, supported, rejected_value=0.1):
        rows = []
        for hypothesis in by_id[example_id]["plan_hypotheses"]:
            value = 0.9 if hypothesis["id"] == supported else rejected_value
            rows.append(_pair(provider, example_id, hypothesis["id"], value))
        return rows

    records = []
    for provider in ("jev", "gpt"):
        records.extend(atomic(provider, "v2-op-create", "intend_create"))
        records.extend(atomic(provider, "v2-quote-delete", "intend_query"))
        records.extend(atomic(provider, "v2-amb-cancel", "intend_delete"))
        records.extend(plans(provider, "v2-plan-delete-then-create", "plan_delete_then_create"))
        records.extend(atomic(provider, "v2-plan-delete-then-create", "intend_delete"))
    records.extend(plans("gpt", "v2-plan-create-then-delete", "plan_only_delete", rejected_value=0.2))
    records.extend(plans("jev", "v2-plan-create-then-delete", "plan_create_then_delete"))
    metrics, disagreements = evaluate_synthetic_v2_records(records, items)
    jev = metrics["by_provider"]["jev"]
    gpt = metrics["by_provider"]["gpt"]
    assert jev["atomic_passed"] == 2
    assert jev["atomic_rows"] == 2
    assert jev["plan_passed"] == 2
    assert jev["plan_rows"] == 2
    assert jev["not_scored"][0]["example_id"] == "v2-amb-cancel"
    assert jev["not_scored"][0]["outcome"] == "not_scored"
    assert "passed" not in jev["not_scored"][0]
    assert gpt["plan_passed"] == 1
    failed = {row["example_id"]: row for row in gpt["plans"]}
    assert failed["v2-plan-create-then-delete"]["passed"] is False
    assert failed["v2-plan-delete-then-create"]["passed"] is True
    assert any(row["example_id"] == "v2-plan-create-then-delete" for row in disagreements)
    assert jev["atomic_diagnostic"][0]["example_id"] == "v2-plan-delete-then-create"
