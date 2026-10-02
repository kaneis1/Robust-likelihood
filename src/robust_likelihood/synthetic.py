"""Synthetic alarm extension. Metrics stay separate from the MASSIVE pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from robust_likelihood.constants import INTENTS, QUANTITY, YES_THRESHOLD
from robust_likelihood.planning import CallSpec, _spec
from robust_likelihood.prompting import HypothesisSet, load_hypotheses
from robust_likelihood.scoring import predict, validate_distribution
from robust_likelihood.storage import read_json, read_jsonl
from robust_likelihood.synthetic_items import INTENTION_HYPOTHESES, authored_items


def _canonical(rows: list[dict]) -> list[str]:
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]


def compose_input(item: dict) -> str:
    context = str(item.get("context") or "").strip()
    utterance = str(item["utt"]).strip()
    if not context:
        return utterance
    return f"Context: {context}\n\nUtterance: {utterance}"


def load_synthetic(root: Path) -> tuple[list[dict], dict, HypothesisSet, HypothesisSet]:
    directory = root / "data" / "synthetic" / "alarm"
    stored = read_jsonl(directory / "items.jsonl")
    expected = authored_items()
    if _canonical(stored) != _canonical(expected):
        raise ValueError("data/synthetic/alarm/items.jsonl does not match the authored synthetic items")
    review = read_json(directory / "review_manifest.json")
    hypotheses = load_hypotheses(root, "synthetic_v1")
    if set(hypotheses.descriptions) != set(INTENTION_HYPOTHESES):
        raise ValueError("synthetic_v1 intention hypotheses do not match the authored set")
    if set(hypotheses.state_questions) != {"alarm_is_enabled", "alarm_exists"}:
        raise ValueError("synthetic_v1 state questions must stay separate from intention hypotheses")
    massive = load_hypotheses(root, "v1")
    if set(massive.descriptions) != set(INTENTS):
        raise ValueError("MASSIVE v1 hypotheses are required for the lump classification")
    return stored, review, hypotheses, massive


def build_synthetic_plan(
    items: list[dict],
    providers: tuple[str, ...] | list[str],
    model_ids: dict[str, str],
    synthetic: HypothesisSet,
    massive: HypothesisSet,
    prompt_version: str,
    inference_by_provider: dict[str, dict],
    repeats: int = 0,
) -> list[CallSpec]:
    if repeats < 0:
        raise ValueError("repeats must be >= 0")
    repeat_ids: list[int | None] = [None, *range(1, repeats + 1)]
    specs: list[CallSpec] = []
    for item in items:
        rendered = dict(item)
        rendered["utt"] = compose_input(item)
        for provider in providers:
            for repeat_id in repeat_ids:
                if item.get("massive_lump_label"):
                    specs.append(
                        _spec(
                            rendered,
                            provider,
                            model_ids[provider],
                            "classification",
                            None,
                            massive.descriptions,
                            prompt_version,
                            massive.version,
                            inference_by_provider[provider],
                            repeat_id,
                        )
                    )
                for hypothesis in INTENTION_HYPOTHESES:
                    specs.append(
                        _spec(
                            rendered,
                            provider,
                            model_ids[provider],
                            "pairwise",
                            hypothesis,
                            synthetic.descriptions,
                            prompt_version,
                            synthetic.version,
                            inference_by_provider[provider],
                            repeat_id,
                        )
                    )
    return specs


def evaluate_synthetic_records(records: list[dict], items: list[dict]) -> tuple[dict, list[dict]]:
    from robust_likelihood.evaluate import final_records

    finals = final_records(records) if records else []
    by_id = {item["example_id"]: item for item in items}
    providers = []
    for record in finals:
        provider = record.get("provider")
        if provider and provider not in providers:
            providers.append(provider)
    coverage = {provider: {"attempted": 0, "succeeded": 0, "invalid": 0, "failed": 0} for provider in providers}
    for record in finals:
        if record.get("task") not in {"classification", "pairwise"}:
            continue
        bucket = coverage.setdefault(
            record["provider"],
            {"attempted": 0, "succeeded": 0, "invalid": 0, "failed": 0},
        )
        bucket["attempted"] += 1
        status = {"success": "succeeded"}.get(record.get("status"), record.get("status"))
        if status in bucket:
            bucket[status] += 1
    scores = _score_index(finals)
    preserved = _paired_family(items, scores, providers, "meaning_preserved", _preserved_ok)
    reversed_family = _paired_family(items, scores, providers, "meaning_reversed", _reversed_ok)
    negation = _single_family(items, scores, providers, "negation_correction", _contrast_ok)
    quoted = _single_family(items, scores, providers, "quoted_command", _contrast_ok)
    multiple = _single_family(items, scores, providers, "multiple_commands", _multiple_ok)
    underspecified = _underspecified(items, scores, providers)
    lump = _lump_classification(finals, items, providers)
    disagreements = _winner_disagreements(items, scores)
    metrics = {
        "dataset": "synthetic_alarm",
        "quantity": QUANTITY,
        "dependence": "Synthetic rows are authored pairs and variants. They are not MASSIVE observations and are not pooled with the pilot metrics.",
        "intention_hypotheses": list(INTENTION_HYPOTHESES),
        "state_questions_not_scored": True,
        "rules": {
            "yes_threshold": YES_THRESHOLD,
            "support_high_means": "yes-probability above 0.5 and strictly above every support_low hypothesis",
            "support_low_means": "yes-probability below 0.5",
            "underspecified_accuracy": None,
            "probabilities_sum_required": False,
        },
        "coverage": {"by_provider": coverage},
        "meaning_preserved": preserved,
        "meaning_reversed": reversed_family,
        "negation_correction": negation,
        "quoted_command": quoted,
        "multiple_commands": multiple,
        "underspecified": underspecified,
        "massive_lump_classification": lump,
        "disagreement_count": len(disagreements),
        "items_by_id": {item["example_id"]: item["family"] for item in by_id.values()},
    }
    return metrics, disagreements


def _score_index(records: list[dict]) -> dict[tuple[str, str], dict[str, float] | None]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("task") != "pairwise" or record.get("status") != "success":
            continue
        hypothesis = record.get("hypothesis")
        parsed = record.get("parsed") or {}
        value = parsed.get("plausibility_yes_probability")
        if hypothesis not in INTENTION_HYPOTHESES or not isinstance(value, (int, float)):
            continue
        grouped[(record["provider"], record["example_id"])][hypothesis] = float(value)
    complete = {}
    seen = {(record["provider"], record["example_id"]) for record in records if record.get("task") == "pairwise"}
    for key in seen:
        scores = grouped.get(key, {})
        complete[key] = scores if set(scores) == set(INTENTION_HYPOTHESES) else None
    return complete


def _paired_family(items, scores, providers, family, check) -> dict:
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in items:
        if item["family"] != family:
            continue
        pairs[item["pair_id"]][item["role"]] = item
    by_provider = {}
    for provider in providers:
        results = []
        for pair_id, roles in sorted(pairs.items()):
            source = roles.get("source")
            variant = roles.get("variant")
            if source is None or variant is None:
                continue
            source_scores = scores.get((provider, source["example_id"]))
            variant_scores = scores.get((provider, variant["example_id"]))
            if source_scores is None or variant_scores is None:
                results.append({"pair_id": pair_id, "complete": False, "passed": False})
                continue
            passed, detail = check(source, variant, source_scores, variant_scores)
            changes = {
                hypothesis: variant_scores[hypothesis] - source_scores[hypothesis]
                for hypothesis in INTENTION_HYPOTHESES
            }
            absolute = [abs(value) for value in changes.values()]
            results.append(
                {
                    "pair_id": pair_id,
                    "complete": True,
                    "passed": passed,
                    "detail": detail,
                    "source_winner": predict(source_scores),
                    "variant_winner": predict(variant_scores),
                    "mean_absolute_change": sum(absolute) / len(absolute),
                    "signed_change": changes,
                }
            )
        complete = [row for row in results if row["complete"]]
        by_provider[provider] = {
            "pairs": len(results),
            "complete_pairs": len(complete),
            "passed": sum(1 for row in complete if row["passed"]),
            "mean_absolute_change": _mean([row["mean_absolute_change"] for row in complete]),
            "rows": results,
        }
    return {"by_provider": by_provider}


def _preserved_ok(source, variant, source_scores, variant_scores) -> tuple[bool, str]:
    gold = source["gold_winner"]
    if variant["gold_winner"] != gold:
        return False, "pair gold winners differ"
    source_winner = predict(source_scores)
    variant_winner = predict(variant_scores)
    if source_winner == gold and variant_winner == gold:
        return True, "winner stayed on the gold intention"
    return False, f"winner moved from {source_winner} to {variant_winner}"


def _reversed_ok(source, variant, source_scores, variant_scores) -> tuple[bool, str]:
    if predict(source_scores) != source["gold_winner"]:
        return False, "source winner missed the gold intention"
    if predict(variant_scores) != variant["gold_winner"]:
        return False, "variant winner missed the gold intention"
    enable_delta = variant_scores["intend_enable"] - source_scores["intend_enable"]
    disable_delta = variant_scores["intend_disable"] - source_scores["intend_disable"]
    if enable_delta < 0 and disable_delta > 0:
        return True, "enable fell and disable rose"
    return False, "enable and disable did not move in opposite directions"


def _single_family(items, scores, providers, family, check) -> dict:
    rows = [item for item in items if item["family"] == family]
    by_provider = {}
    for provider in providers:
        results = []
        for item in rows:
            item_scores = scores.get((provider, item["example_id"]))
            if item_scores is None:
                results.append({"example_id": item["example_id"], "complete": False, "passed": False})
                continue
            passed, detail = check(item, item_scores)
            results.append(
                {
                    "example_id": item["example_id"],
                    "complete": True,
                    "passed": passed,
                    "detail": detail,
                    "winner": predict(item_scores),
                    "scores": item_scores,
                }
            )
        complete = [row for row in results if row["complete"]]
        by_provider[provider] = {
            "items": len(results),
            "complete_items": len(complete),
            "passed": sum(1 for row in complete if row["passed"]),
            "rows": results,
        }
    return {"by_provider": by_provider}


def _contrast_ok(item, scores) -> tuple[bool, str]:
    high = [key for key, level in item["support"].items() if level == "support_high"]
    low = [key for key, level in item["support"].items() if level == "support_low"]
    if any(scores[key] >= YES_THRESHOLD for key in low):
        return False, "a rejected hypothesis is at or above 0.5"
    if any(scores[key] <= YES_THRESHOLD for key in high):
        return False, "a supported hypothesis is not above 0.5"
    for supported in high:
        if any(scores[supported] <= scores[rejected] for rejected in low):
            return False, "a supported hypothesis does not rank above a rejected one"
    return True, "supported hypotheses rank above rejected ones"


def _multiple_ok(item, scores) -> tuple[bool, str]:
    high = [key for key, level in item["support"].items() if level == "support_high"]
    supported = [key for key in high if scores[key] > YES_THRESHOLD]
    if len(supported) == len(high) and _contrast_ok(item, scores)[0]:
        return True, "both actions are supported"
    if len(supported) == 1:
        return False, f"only {supported[0]} is supported"
    if not supported:
        return False, "neither action is supported"
    return False, "both actions are above 0.5 but a rejected hypothesis is also supported"


def _underspecified(items, scores, providers) -> dict:
    rows = [item for item in items if item["family"] == "underspecified"]
    by_provider = {}
    for provider in providers:
        bare = next(item for item in rows if item["role"] == "bare")
        bare_scores = scores.get((provider, bare["example_id"]))
        contexts = []
        for item in rows:
            if item["role"] == "bare":
                continue
            item_scores = scores.get((provider, item["example_id"]))
            change = None
            if bare_scores is not None and item_scores is not None:
                change = {
                    hypothesis: item_scores[hypothesis] - bare_scores[hypothesis]
                    for hypothesis in INTENTION_HYPOTHESES
                }
            contexts.append(
                {
                    "example_id": item["example_id"],
                    "role": item["role"],
                    "context": item.get("context") or "",
                    "scores": item_scores,
                    "top_gap": _top_gap(item_scores),
                    "signed_change_from_bare": change,
                }
            )
        by_provider[provider] = {
            "accuracy": None,
            "bare": {
                "example_id": bare["example_id"],
                "utt": bare["utt"],
                "scores": bare_scores,
                "top_gap": _top_gap(bare_scores),
            },
            "contexts": contexts,
        }
    return {"by_provider": by_provider}


def _lump_classification(records, items, providers) -> dict:
    gold = {
        item["example_id"]: item["massive_lump_label"]
        for item in items
        if item.get("massive_lump_label")
    }
    by_provider = {}
    for provider in providers:
        correct = 0
        scored = 0
        invalid = 0
        for record in records:
            if record.get("provider") != provider or record.get("task") != "classification":
                continue
            if record["example_id"] not in gold:
                continue
            if record.get("status") != "success":
                invalid += 1
                continue
            parsed = validate_distribution((record.get("parsed") or {}).get("distribution"))
            if parsed is None:
                invalid += 1
                continue
            scored += 1
            if predict(parsed) == gold[record["example_id"]]:
                correct += 1
        by_provider[provider] = {
            "n": scored,
            "correct": correct,
            "invalid": invalid,
            "accuracy": (correct / scored) if scored else None,
            "scored_against": "massive_lump_label",
        }
    return {"by_provider": by_provider}


def _winner_disagreements(items, scores) -> list[dict]:
    rows = []
    if not any(provider == "jev" for provider, _example in scores) or not any(
        provider == "gpt" for provider, _example in scores
    ):
        return rows
    for item in items:
        left = scores.get(("jev", item["example_id"]))
        right = scores.get(("gpt", item["example_id"]))
        if left is None or right is None:
            continue
        left_winner = predict(left)
        right_winner = predict(right)
        if left_winner == right_winner:
            continue
        rows.append(
            {
                "task": "pairwise",
                "dataset": "synthetic_alarm",
                "family": item["family"],
                "example_id": item["example_id"],
                "input_text": compose_input(item),
                "gold_winner": item.get("gold_winner"),
                "jev_prediction": left_winner,
                "gpt_prediction": right_winner,
                "jev_scores": left,
                "gpt_scores": right,
                "inspection_notes": "",
            }
        )
    return rows


def _top_gap(scores: dict[str, float] | None) -> float | None:
    if not scores:
        return None
    ordered = sorted(scores.values(), reverse=True)
    return ordered[0] - ordered[1]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
