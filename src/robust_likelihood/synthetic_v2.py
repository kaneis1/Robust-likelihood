"""Follow-up synthetic alarm set. Metrics stay separate from synthetic_v1 and MASSIVE."""

from __future__ import annotations

import json
from pathlib import Path

from robust_likelihood.constants import YES_THRESHOLD
from robust_likelihood.planning import CallSpec, _spec
from robust_likelihood.prompting import load_hypotheses
from robust_likelihood.scoring import predict
from robust_likelihood.storage import read_json, read_jsonl
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


def load_synthetic_v2(root: Path) -> tuple[list[dict], dict, dict[str, str]]:
    directory = root / "data" / "synthetic" / "alarm_v2"
    stored = read_jsonl(directory / "items.jsonl")
    expected = authored_items()
    if _canonical(stored) != _canonical(expected):
        raise ValueError("data/synthetic/alarm_v2/items.jsonl does not match the authored items")
    review = read_json(directory / "review_manifest.json")
    if _canonical(review.get("items", [])) != _canonical(review_manifest()["items"]):
        raise ValueError("data/synthetic/alarm_v2/review_manifest.json does not match the authored review")
    hypotheses = load_hypotheses(root, HYPOTHESIS_VERSION)
    if set(hypotheses.descriptions) != set(ATOMIC):
        raise ValueError("synthetic_v2 atomic hypotheses do not match the authored set")
    if hypotheses.version != HYPOTHESIS_VERSION:
        raise ValueError("synthetic_v2 hypothesis version mismatch")
    return stored, review, hypotheses.descriptions


def _rendered(item: dict, dataset_name: str = DATASET_VERSION) -> dict:
    return {
        "example_id": item["example_id"],
        "original_example_id": item["pair_id"],
        "transformation": item["role"],
        "utt": item["utt"],
        "intent": "",
        "original_intent": "",
        "label_changed": False,
        "dataset": dataset_name,
        "family": item["family"],
        "pair_id": item["pair_id"],
        "role": item["role"],
        "massive_lump_label": "",
    }


def build_synthetic_v2_plan(
    items: list[dict],
    providers: tuple[str, ...] | list[str],
    model_ids: dict[str, str],
    atomic_descriptions: dict[str, str],
    prompt_version: str,
    inference_by_provider: dict[str, dict],
    repeats: int = 0,
    dataset_name: str = DATASET_VERSION,
) -> list[CallSpec]:
    if prompt_version not in {PROMPT_VERSION, "likely", "typical"}:
        raise ValueError("synthetic_alarm_v2 uses prompt template v2, or the likely or typical wording")
    if repeats < 0:
        raise ValueError("repeats must be >= 0")
    repeat_ids: list[int | None] = [None, *range(1, repeats + 1)]
    specs: list[CallSpec] = []
    for item in items:
        rendered = _rendered(item, dataset_name)
        for provider in providers:
            for repeat_id in repeat_ids:
                specs.extend(
                    _atomic_specs(
                        rendered,
                        provider,
                        model_ids[provider],
                        atomic_descriptions,
                        inference_by_provider[provider],
                        repeat_id,
                        prompt_version,
                    )
                )
                if item["evaluation"] == "scored_plan":
                    specs.extend(
                        _plan_specs(
                            rendered,
                            item["plan_hypotheses"],
                            provider,
                            model_ids[provider],
                            inference_by_provider[provider],
                            repeat_id,
                            prompt_version,
                        )
                    )
    return specs


def _atomic_specs(item, provider, model_id, descriptions, settings, repeat_id, prompt_version) -> list[CallSpec]:
    return [
        _spec(
            item,
            provider,
            model_id,
            "pairwise",
            hypothesis,
            descriptions,
            prompt_version,
            HYPOTHESIS_VERSION,
            settings,
            repeat_id,
        )
        for hypothesis in ATOMIC
    ]


def _plan_specs(item, hypotheses, provider, model_id, settings, repeat_id, prompt_version) -> list[CallSpec]:
    descriptions = {row["id"]: row["text"] for row in hypotheses}
    return [
        _spec(
            item,
            provider,
            model_id,
            "pairwise",
            row["id"],
            descriptions,
            prompt_version,
            HYPOTHESIS_VERSION,
            settings,
            repeat_id,
        )
        for row in hypotheses
    ]


def _yes(record: dict) -> float | None:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("plausibility_yes_probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0.0 or number > 1.0:
        return None
    return number


def _support_ok(expected: dict[str, str], scores: dict[str, float]) -> tuple[bool, str]:
    supported = [key for key, level in expected.items() if level == "supported"]
    rejected = [key for key, level in expected.items() if level == "rejected"]
    if any(scores[key] >= YES_THRESHOLD for key in rejected):
        return False, "a rejected hypothesis is at or above 0.5"
    if any(scores[key] <= YES_THRESHOLD for key in supported):
        return False, "a supported hypothesis is not above 0.5"
    for key in supported:
        if any(scores[key] <= scores[other] for other in rejected):
            return False, "a supported hypothesis does not rank above a rejected one"
    return True, "the supported hypothesis ranks above the rejected ones"


def evaluate_synthetic_v2_records(records: list[dict], items: list[dict]) -> tuple[dict, list[dict]]:
    from robust_likelihood.evaluate import final_records

    finals = [row for row in final_records(records) if row.get("repeat_id") is None] if records else []
    by_item = {item["example_id"]: item for item in items}
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    providers = []
    for record in finals:
        if record.get("task") != "pairwise":
            continue
        provider = record.get("provider")
        example_id = record.get("example_id")
        if provider not in providers:
            providers.append(provider)
        kind = "plan" if str(record.get("hypothesis", "")).startswith("plan_") else "atomic"
        grouped.setdefault((provider, example_id, kind), []).append(record)

    blocks: dict[str, dict] = {}
    disagreements = []
    jev_gpt = {}
    for (provider, example_id, kind), rows in grouped.items():
        blocks.setdefault(provider, {"scored": [], "not_scored": [], "plans": [], "atomic_diagnostic": []})
        item = by_item.get(example_id)
        if item is None:
            continue
        if kind == "atomic" and item["evaluation"] == "scored_plan":
            scores = _atomic_only(rows)
            if scores is None:
                continue
            blocks[provider]["atomic_diagnostic"].append(
                {"example_id": example_id, "scores": scores, "winner": predict(scores)}
            )
            continue
        if item["evaluation"] == "not_scored" and kind == "atomic":
            scores = _atomic_only(rows)
            if scores is None:
                continue
            blocks[provider]["not_scored"].append(
                {"example_id": example_id, "outcome": "not_scored", "scores": scores, "winner": predict(scores)}
            )
            continue
        if item["evaluation"] == "scored" and kind == "atomic":
            scores = _atomic_only(rows)
            if scores is None:
                blocks[provider]["scored"].append({"example_id": example_id, "complete": False, "passed": False})
                continue
            passed, detail = _support_ok(item["support"], scores)
            row = {
                "example_id": example_id,
                "complete": True,
                "passed": passed,
                "detail": detail,
                "winner": predict(scores),
                "scores": scores,
            }
            blocks[provider]["scored"].append(row)
            jev_gpt.setdefault((example_id, "atomic"), {})[provider] = row
        if item["evaluation"] == "scored_plan" and kind == "plan":
            scores = _plan_only(item, rows)
            if scores is None:
                blocks[provider]["plans"].append({"example_id": example_id, "complete": False, "passed": False})
                continue
            expected = {row["id"]: row["expected"] for row in item["plan_hypotheses"]}
            passed, detail = _support_ok(expected, scores)
            row = {
                "example_id": example_id,
                "complete": True,
                "passed": passed,
                "detail": detail,
                "winner": predict(scores),
                "scores": scores,
            }
            blocks[provider]["plans"].append(row)
            jev_gpt.setdefault((example_id, "plan"), {})[provider] = row

    for (example_id, kind), sides in sorted(jev_gpt.items()):
        left = sides.get("jev")
        right = sides.get("gpt")
        if not left or not right or not left.get("complete") or not right.get("complete"):
            continue
        if left["winner"] == right["winner"] and left["passed"] == right["passed"]:
            continue
        disagreements.append(
            {
                "example_id": example_id,
                "score_set": kind,
                "jev_winner": left["winner"],
                "gpt_winner": right["winner"],
                "jev_passed": left["passed"],
                "gpt_passed": right["passed"],
                "jev_scores": left["scores"],
                "gpt_scores": right["scores"],
                "inspection_notes": "",
            }
        )

    by_provider = {}
    for provider, block in blocks.items():
        scored = [row for row in block["scored"] if row.get("complete")]
        plans = [row for row in block["plans"] if row.get("complete")]
        by_provider[provider] = {
            "atomic_passed": sum(1 for row in block["scored"] if row.get("passed")),
            "atomic_complete": len(scored),
            "atomic_rows": len(block["scored"]),
            "plan_passed": sum(1 for row in block["plans"] if row.get("passed")),
            "plan_complete": len(plans),
            "plan_rows": len(block["plans"]),
            "not_scored": block["not_scored"],
            "scored": block["scored"],
            "plans": block["plans"],
            "atomic_diagnostic": block.get("atomic_diagnostic", []),
        }
    metrics = {
        "dataset": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "quantity": "These outputs are elicited plausibility judgments, not verified P(e|h).",
        "primary_test": "complete current request versus a mentioned, partial, or differently ordered request",
        "by_provider": by_provider,
        "disagreement_count": len(disagreements),
    }
    return metrics, disagreements


def _atomic_only(records: list[dict]) -> dict[str, float] | None:
    found = {}
    for record in records:
        if record.get("status") != "success":
            return None
        hypothesis = record.get("hypothesis")
        value = _yes(record)
        if hypothesis not in ATOMIC or value is None:
            return None
        found[hypothesis] = value
    if set(found) != set(ATOMIC):
        return None
    return found


def _plan_only(item: dict, records: list[dict]) -> dict[str, float] | None:
    expected = {row["id"] for row in item["plan_hypotheses"]}
    found = {}
    for record in records:
        if record.get("status") != "success":
            return None
        hypothesis = record.get("hypothesis")
        value = _yes(record)
        if hypothesis not in expected or value is None:
            return None
        found[hypothesis] = value
    if set(found) != expected:
        return None
    return found
