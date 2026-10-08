import json

from pytest import approx

from robust_likelihood.storage import read_json, read_jsonl, repo_root
from robust_likelihood.synthetic_families_items import (
    DATASET_VERSION,
    HYPOTHESIS_VERSION,
    PROMPT_VERSION,
    authored_items,
    review_manifest,
)
from robust_likelihood.synthetic_v2_items import ATOMIC
from robust_likelihood.toy_likelihood_items import TARGETS, authored_items as toy_items


def _canonical(rows):
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]


def test_family_files_match_and_splits_do_not_share_a_family():
    root = repo_root()
    items = authored_items()
    stored = read_jsonl(root / "data" / "synthetic" / "alarm_families" / "items.jsonl")
    review = read_json(root / "data" / "synthetic" / "alarm_families" / "review_manifest.json")
    assert _canonical(stored) == _canonical(items)
    assert _canonical(review["items"]) == _canonical(review_manifest()["items"])
    assert {row["dataset_version"] for row in items} == {DATASET_VERSION}
    assert {row["prompt_version"] for row in items} == {PROMPT_VERSION}
    assert {row["hypothesis_version"] for row in items} == {HYPOTHESIS_VERSION}
    dev = {row["family_id"] for row in items if row["split"] == "dev"}
    holdout = {row["family_id"] for row in items if row["split"] == "holdout"}
    assert len(dev) == 8
    assert len(holdout) == 32
    assert not dev & holdout
    order = next(row for row in items if row["example_id"] == "order-01__canonical")
    assert order["surface_order_matches_temporal"] is False
    assert order["plan_hypotheses"][0]["order"][0] == "delete the morning alarm entirely"
    ambiguous = next(row for row in items if row["example_id"] == "ambig-01__ambiguous__canonical")
    assert ambiguous["evaluation"] == "not_scored"
    assert ambiguous["support"] is None
    clarified = next(row for row in items if row["example_id"] == "ambig-01__clarified__canonical")
    assert clarified["support"]["intend_disable"] == "supported"


def test_family_and_toy_dry_runs_keep_the_pilot_prompts():
    from robust_likelihood.runner import run_comparison

    root = repo_root()
    dev = run_comparison(root, providers=("jev", "gpt", "claude"), dry_run=True, dataset="synthetic_families", split="dev")
    holdout = run_comparison(
        root, providers=("jev", "gpt", "claude"), dry_run=True, dataset="synthetic_families", split="holdout"
    )
    toy = run_comparison(root, providers=("jev", "gpt", "claude"), dry_run=True, dataset="toy_likelihood")
    assert dev["planned_requests"] == {"classification": 0, "pairwise": 522, "total": 522}
    assert holdout["planned_requests"]["total"] == 2088
    assert dev["prompt_version"] == "v2"
    assert dev["hypothesis_description_version"] == "synthetic_v2"
    assert dev["unapproved_items"] == 0
    assert toy["planned_requests"] == {"classification": 0, "pairwise": 81, "total": 81}
    assert toy["prompt_version"] == "toy_v1"
    assert toy["hypothesis_description_version"] == "toy_v1"
    report = (root / "templates" / "synthetic_alarm_report.md").read_text(encoding="utf-8")
    assert "synthetic_v2" not in report


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


def test_family_metrics_keep_the_pass_rule_and_add_margins():
    from robust_likelihood.synthetic_families import evaluate_family_records

    items = [
        row
        for row in authored_items()
        if row["family_id"] in {"mention-01", "order-01", "ambig-01"}
    ]
    low = {name: 0.1 for name in ATOMIC}
    records = []
    for provider in ("jev", "gpt"):
        for row in items:
            if row["family_id"] == "mention-01":
                scores = dict(low)
                scores["intend_query"] = 0.95 if row["variant"] == "canonical" else 0.80
                scores["intend_delete"] = 0.70 if row["variant"] == "typo" else 0.20
                records.extend(_pair(provider, row["example_id"], name, value) for name, value in scores.items())
            elif row["evaluation"] == "scored_plan":
                for hypothesis in row["plan_hypotheses"]:
                    value = 0.90 if hypothesis["id"] == "plan_temporal" else 0.15
                    records.append(_pair(provider, row["example_id"], hypothesis["id"], value))
                scores = dict(low)
                scores["intend_delete"] = 0.80
                scores["intend_create"] = 0.70
                records.extend(_pair(provider, row["example_id"], name, value) for name, value in scores.items())
            elif row["side"] == "ambiguous":
                scores = dict(low)
                scores["intend_disable"] = 0.40
                records.extend(_pair(provider, row["example_id"], name, value) for name, value in scores.items())
            else:
                scores = dict(low)
                scores["intend_disable"] = 0.93
                records.extend(_pair(provider, row["example_id"], name, value) for name, value in scores.items())
    metrics, disagreements = evaluate_family_records(records, items)
    jev = metrics["by_provider"]["jev"]
    families = {(row["family_id"], row["side"]): row for row in jev["families"]}
    assert families[("mention-01", "request")]["family_passed"] is False
    assert families[("mention-01", "request")]["max_highest_rejected"] == 0.70
    assert families[("mention-01", "request")]["supported_score_range"] == approx(0.15)
    assert families[("order-01", "request")]["family_passed"] is True
    assert families[("order-01", "request")]["mean_plan_margin"] == approx(0.75)
    assert families[("ambig-01", "ambiguous")]["family_passed"] is None
    assert families[("ambig-01", "clarified")]["family_passed"] is True
    assert jev["clarification_shifts"][0]["shift"] == approx(0.53)
    assert jev["summary"]["categories"]["action_order"]["families_passed"] == 1
    assert disagreements == []


def test_toy_error_is_the_distance_from_the_stated_rules():
    from robust_likelihood.toy_likelihood import evaluate_toy_records

    items = toy_items()
    assert TARGETS["change"] == {"toy_create": 0.60, "toy_disable": 0.30, "toy_query": 0.10}
    records = []
    for item in items:
        for hypothesis, target in item["targets"].items():
            value = target if item["rules_id"] == "by_intention" else target + 0.10
            records.append(_pair("jev", item["example_id"], hypothesis, value))
    metrics, _disagreements = evaluate_toy_records(records, items)
    jev = metrics["by_provider"]["jev"]
    assert jev["complete_cells"] == 9
    assert jev["by_rules"]["by_intention"]["mean_absolute_error"] == 0.0
    assert jev["ranking_matches"] == 9
    shifted = next(cell for cell in jev["cells"] if cell["example_id"] == "toy-by_sentence-change")
    assert shifted["absolute_errors"]["toy_create"] == approx(0.10)
    assert shifted["score_sum"] == approx(1.3)
