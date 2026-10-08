"""Family expansion of the alarm pilot. The pass rule matches synthetic_eval_v2."""

from __future__ import annotations

import json
from pathlib import Path

from robust_likelihood.prompting import load_hypotheses
from robust_likelihood.scoring import predict
from robust_likelihood.storage import read_json, read_jsonl
from robust_likelihood.synthetic_families_items import (
    DATASET_VERSION,
    EVALUATION_RULE_VERSION,
    HYPOTHESIS_VERSION,
    PROMPT_VERSION,
    authored_items,
    review_manifest,
)
from robust_likelihood.synthetic_v2 import (
    _atomic_only,
    _plan_only,
    _support_ok,
    build_synthetic_v2_plan,
)


def _canonical(rows: list[dict]) -> list[str]:
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]


def load_synthetic_families(root: Path) -> tuple[list[dict], dict, dict[str, str]]:
    directory = root / "data" / "synthetic" / "alarm_families"
    stored = read_jsonl(directory / "items.jsonl")
    expected = authored_items()
    if _canonical(stored) != _canonical(expected):
        raise ValueError("data/synthetic/alarm_families/items.jsonl does not match the authored items")
    review = read_json(directory / "review_manifest.json")
    if _canonical(review.get("items", [])) != _canonical(review_manifest()["items"]):
        raise ValueError("data/synthetic/alarm_families/review_manifest.json does not match the authored review")
    hypotheses = load_hypotheses(root, HYPOTHESIS_VERSION)
    if hypotheses.version != HYPOTHESIS_VERSION:
        raise ValueError("family items require the frozen synthetic_v2 definitions")
    return stored, review, hypotheses.descriptions


def select_split(items: list[dict], review: dict, split: str | None) -> tuple[list[dict], dict]:
    chosen = split or "all"
    if chosen not in {"all", "dev", "holdout"}:
        raise ValueError("split must be all, dev, or holdout")
    if chosen == "all":
        selected = items
    else:
        selected = [item for item in items if item["split"] == chosen]
    if not selected:
        raise ValueError(f"split {chosen} has no items")
    ids = {item["example_id"] for item in selected}
    reviewed = dict(review)
    reviewed["items"] = [row for row in review.get("items", []) if row.get("example_id") in ids]
    return selected, reviewed


def build_family_plan(items, providers, model_ids, descriptions, prompt_version, settings, repeats=0):
    if prompt_version not in {PROMPT_VERSION, "likely", "typical"}:
        raise ValueError("the family set uses frozen prompt template v2, or the likely or typical wording")
    return build_synthetic_v2_plan(
        items,
        providers,
        model_ids,
        descriptions,
        prompt_version,
        settings,
        repeats=repeats,
        dataset_name=DATASET_VERSION,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _range(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return max(values) - min(values)


def evaluate_family_records(records: list[dict], items: list[dict]) -> tuple[dict, list[dict]]:
    from robust_likelihood.evaluate import final_records

    finals = [row for row in final_records(records) if row.get("repeat_id") is None] if records else []
    by_item = {item["example_id"]: item for item in items}
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for record in finals:
        if record.get("task") != "pairwise":
            continue
        kind = "plan" if str(record.get("hypothesis", "")).startswith("plan_") else "atomic"
        grouped.setdefault((record.get("provider"), record.get("example_id"), kind), []).append(record)

    scored_rows: dict[str, list[dict]] = {}
    for item in items:
        providers = {provider for provider, example_id, _kind in grouped if example_id == item["example_id"]}
        for provider in providers:
            scored_rows.setdefault(provider, []).append(_variant_row(item, provider, grouped))

    by_provider = {}
    disagreements = []
    jev_gpt: dict[tuple[str, str], dict] = {}
    for provider, rows in scored_rows.items():
        families = _families(rows)
        by_provider[provider] = {
            "variants": rows,
            "families": families,
            "clarification_shifts": _clarification_shifts(rows),
            "summary": _summary(families),
        }
        for row in rows:
            if row["evaluation"] == "not_scored" or not row.get("complete"):
                continue
            jev_gpt.setdefault((row["example_id"], row["score_set"]), {})[provider] = row

    for (example_id, score_set), sides in sorted(jev_gpt.items()):
        left = sides.get("jev")
        right = sides.get("gpt")
        if not left or not right:
            continue
        if left["winner"] == right["winner"] and left["passed"] == right["passed"]:
            continue
        disagreements.append(
            {
                "example_id": example_id,
                "family_id": left["family_id"],
                "score_set": score_set,
                "jev_winner": left["winner"],
                "gpt_winner": right["winner"],
                "jev_passed": left["passed"],
                "gpt_passed": right["passed"],
                "jev_scores": left["scores"],
                "gpt_scores": right["scores"],
                "inspection_notes": "",
            }
        )

    metrics = {
        "dataset": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "quantity": "These outputs are elicited plausibility judgments, not verified P(e|h).",
        "analysis_unit": "family",
        "by_provider": by_provider,
        "disagreement_count": len(disagreements),
    }
    return metrics, disagreements


def _variant_row(item: dict, provider: str, grouped: dict) -> dict:
    atomic = _atomic_only(grouped.get((provider, item["example_id"], "atomic"), []))
    base = {
        "example_id": item["example_id"],
        "family_id": item["family_id"],
        "category": item["category"],
        "split": item["split"],
        "side": item["side"],
        "variant": item["variant"],
        "evaluation": item["evaluation"],
        "focus_hypothesis": item["focus_hypothesis"],
        "contrast_hypothesis": item["contrast_hypothesis"],
        "surface_order_matches_temporal": item.get("surface_order_matches_temporal"),
        "atomic_scores": atomic,
    }
    if item["evaluation"] == "not_scored":
        base.update(
            {
                "score_set": "atomic",
                "complete": atomic is not None,
                "passed": None,
                "scores": atomic,
                "winner": None if atomic is None else predict(atomic),
                "highest_rejected": None,
                "plan_margin": None,
                "focus_score": None if atomic is None else atomic.get(item["focus_hypothesis"]),
                "contrast_score": None,
            }
        )
        return base
    if item["evaluation"] == "scored_plan":
        scores = _plan_only(item, grouped.get((provider, item["example_id"], "plan"), []))
        passed = None
        detail = None
        if scores is not None:
            expected = {row["id"]: row["expected"] for row in item["plan_hypotheses"]}
            passed, detail = _support_ok(expected, scores)
        base.update(
            {
                "score_set": "plan",
                "complete": scores is not None,
                "passed": passed,
                "detail": detail,
                "scores": scores,
                "winner": None if scores is None else predict(scores),
                "highest_rejected": None if scores is None else _highest_rejected(item, scores),
                "plan_margin": None
                if scores is None
                else scores["plan_temporal"] - scores["plan_reversed"],
                "focus_score": None if scores is None else scores.get("plan_temporal"),
                "contrast_score": None if scores is None else scores.get("plan_reversed"),
            }
        )
        return base
    passed = None
    detail = None
    if atomic is not None:
        passed, detail = _support_ok(item["support"], atomic)
    base.update(
        {
            "score_set": "atomic",
            "complete": atomic is not None,
            "passed": passed,
            "detail": detail,
            "scores": atomic,
            "winner": None if atomic is None else predict(atomic),
            "highest_rejected": None if atomic is None else _highest_rejected(item, atomic),
            "plan_margin": None,
            "focus_score": None if atomic is None else atomic.get(item["focus_hypothesis"]),
            "contrast_score": None
            if atomic is None or item.get("contrast_hypothesis") is None
            else atomic.get(item["contrast_hypothesis"]),
        }
    )
    return base


def _highest_rejected(item: dict, scores: dict[str, float]) -> float:
    if item["evaluation"] == "scored_plan":
        rejected = [row["id"] for row in item["plan_hypotheses"] if row["expected"] == "rejected"]
    else:
        rejected = [key for key, level in item["support"].items() if level == "rejected"]
    return max(scores[key] for key in rejected)


def _families(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["family_id"], row["side"]), []).append(row)
    families = []
    for (family_id, side), variants in sorted(grouped.items()):
        scored = [row for row in variants if row["evaluation"] != "not_scored"]
        complete = [row for row in scored if row["complete"]]
        focus_values = [row["focus_score"] for row in complete if row.get("focus_score") is not None]
        contrast_values = [row["contrast_score"] for row in complete if row.get("contrast_score") is not None]
        shared = [row["scores"] for row in complete if row.get("scores")]
        score_ranges = []
        if len(shared) >= 2:
            keys = set.intersection(*(set(scores) for scores in shared))
            score_ranges = [_range([scores[key] for scores in shared]) for key in keys]
        rejected_values = [row["highest_rejected"] for row in complete if row.get("highest_rejected") is not None]
        margins = [row["plan_margin"] for row in complete if row.get("plan_margin") is not None]
        families.append(
            {
                "family_id": family_id,
                "side": side,
                "category": variants[0]["category"],
                "split": variants[0]["split"],
                "variants": len(variants),
                "scored_variants": len(scored),
                "complete_variants": len(complete),
                "passed_variants": sum(1 for row in complete if row["passed"]),
                "family_passed": None
                if not scored
                else len(complete) == len(scored) and all(row["passed"] for row in complete),
                "supported_score_range": _range(focus_values),
                "contrast_score_range": _range(contrast_values),
                "max_score_range": None if not score_ranges else max(score_ranges),
                "mean_highest_rejected": _mean(rejected_values),
                "max_highest_rejected": None if not rejected_values else max(rejected_values),
                "mean_plan_margin": _mean(margins),
                "min_plan_margin": None if not margins else min(margins),
                "surface_order_matches_temporal": variants[0]["surface_order_matches_temporal"],
            }
        )
    return families


def _clarification_shifts(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["category"] != "ambiguity_and_clarification" or not row["complete"]:
            continue
        if row.get("focus_score") is None:
            continue
        grouped.setdefault(row["family_id"], {}).setdefault(row["side"], []).append(row["focus_score"])
    shifts = []
    for family_id, sides in sorted(grouped.items()):
        ambiguous = sides.get("ambiguous") or []
        clarified = sides.get("clarified") or []
        if not ambiguous or not clarified:
            continue
        ambiguous_mean = _mean(ambiguous)
        clarified_mean = _mean(clarified)
        shifts.append(
            {
                "family_id": family_id,
                "ambiguous_mean": ambiguous_mean,
                "clarified_mean": clarified_mean,
                "shift": clarified_mean - ambiguous_mean,
            }
        )
    return shifts


def _summary(families: list[dict]) -> dict:
    categories = {}
    for category in sorted({row["category"] for row in families}):
        rows = [row for row in families if row["category"] == category and row["side"] != "ambiguous"]
        passed = [row for row in rows if row["family_passed"]]
        ranges = [row["supported_score_range"] for row in rows if row["supported_score_range"] is not None]
        rejected = [row["mean_highest_rejected"] for row in rows if row["mean_highest_rejected"] is not None]
        margins = [row["mean_plan_margin"] for row in rows if row["mean_plan_margin"] is not None]
        categories[category] = {
            "families": len(rows),
            "families_passed": len(passed),
            "mean_supported_score_range": _mean(ranges),
            "mean_highest_rejected": _mean(rejected),
            "mean_plan_margin": _mean(margins),
        }
    ambiguous = [row for row in families if row["side"] == "ambiguous"]
    return {"categories": categories, "ambiguous_families_not_scored": len(ambiguous)}
