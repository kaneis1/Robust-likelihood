"""Metrics for elicited plausibility judgments.

Ties are incorrect and counted separately. Classification probabilities are not
renormalized. Rank, margin, and ranking flips need all three hypothesis scores.
"""

from __future__ import annotations

from collections import defaultdict

from robust_likelihood.constants import (
    INTENTS,
    MEANING_PRESERVING,
    NLL_CLIP_FLOOR,
    PROBABILITY_SUM_TOLERANCE,
    QUANTITY,
    YES_THRESHOLD,
)
from robust_likelihood.scoring import (
    correct_rank,
    is_probability,
    margin,
    multiclass_brier,
    negative_log_likelihood,
    predict,
    validate_distribution,
    yes_label,
)
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

VARIANT_TRANSFORMATIONS = ("paraphrase", "minor_typo", "repetition")


def final_records(records: list[dict]) -> list[dict]:
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        key = record.get("cache_key")
        if not key:
            raise ValueError("response record is missing cache_key")
        groups[key].append((index, record))
    chosen = []
    for items in groups.values():
        successes = [pair for pair in items if pair[1].get("status") == "success"]
        if successes:
            pick = min(successes, key=lambda pair: (pair[1].get("attempt_id", 0), pair[0]))
        else:
            pick = max(items, key=lambda pair: (pair[1].get("attempt_id", 0), pair[0]))
        chosen.append(pick[1])
    return chosen


def _baseline(record: dict) -> bool:
    return record.get("repeat_id") is None


def _distribution(record: dict) -> dict[str, float] | None:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return validate_distribution(parsed.get("distribution"))


def _yes_probability(record: dict) -> float | None:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("plausibility_yes_probability")
    if not is_probability(value):
        return None
    return float(value)


def _effective_status(record: dict) -> str:
    if record.get("status") != "success":
        return record.get("status") or "failed"
    if record.get("task") == "classification":
        return "success" if _distribution(record) is not None else "invalid"
    if record.get("task") == "pairwise":
        return "success" if _yes_probability(record) is not None else "invalid"
    return "invalid"


def _empty_counts() -> dict[str, int]:
    return {"attempted": 0, "succeeded": 0, "failed": 0, "invalid": 0}


def _add_count(bucket: dict[str, int], status: str) -> None:
    bucket["attempted"] += 1
    if status == "success":
        bucket["succeeded"] += 1
    elif status == "invalid":
        bucket["invalid"] += 1
    else:
        bucket["failed"] += 1


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _classification_block(rows: list[dict]) -> dict:
    correct = 0
    ties = 0
    nlls = []
    briers = []
    for record in rows:
        scores = _distribution(record)
        if scores is None:
            continue
        prediction = predict(scores)
        if prediction == "tie":
            ties += 1
        elif prediction == record["intent"]:
            correct += 1
        nlls.append(negative_log_likelihood(scores[record["intent"]]))
        briers.append(multiclass_brier(scores, record["intent"]))
    n = len(nlls)
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "tie_count": ties,
        "nll": _mean(nlls),
        "nll_clip_floor": NLL_CLIP_FLOOR,
        "brier": _mean(briers),
    }


def _control_classification(rows: list[dict]) -> dict:
    block = _classification_block(rows)
    return {"n": block["n"], "accuracy": block["accuracy"], "tie_count": block["tie_count"]}


def _hypothesis_scores(rows: list[dict]) -> dict[str, float] | None:
    scores = {}
    for record in rows:
        if record.get("hypothesis") not in INTENTS:
            continue
        if _effective_status(record) != "success":
            continue
        probability = _yes_probability(record)
        if probability is None:
            continue
        scores[record["hypothesis"]] = probability
    if len(scores) != len(INTENTS):
        return None
    return scores


def _index(records: list[dict]) -> dict[tuple, dict]:
    index = {}
    for record in records:
        if not _baseline(record):
            continue
        key = (record["provider"], record["example_id"], record["task"], record.get("hypothesis"))
        index[key] = record
    return index


def _providers(records: list[dict]) -> list[str]:
    found = []
    for record in records:
        provider = record.get("provider")
        if provider and provider not in found:
            found.append(provider)
    return found


def _prediction_from_scores(scores: dict[str, float] | None) -> str | None:
    if scores is None:
        return None
    return predict(scores)


def _is_disagreement(left: str, right: str) -> bool:
    return left != right


def evaluate_records(records: list[dict]) -> tuple[dict, list[dict]]:
    finals = final_records(records) if records else []
    baseline = [record for record in finals if _baseline(record)]
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for record in baseline:
        provider_bucket = coverage.setdefault(
            record["provider"],
            {"classification": _empty_counts(), "pairwise": _empty_counts()},
        )
        task = record.get("task")
        if task not in provider_bucket:
            continue
        _add_count(provider_bucket[task], _effective_status(record))

    index = _index(baseline)
    providers = _providers(baseline)
    classification = {}
    for provider in providers:
        rows = [
            record
            for record in baseline
            if record["provider"] == provider
            and record["task"] == "classification"
            and record.get("label_changed") is False
            and _effective_status(record) == "success"
        ]
        classification[provider] = _classification_block(rows)

    pairwise: dict[str, dict] = {}
    for provider in providers:
        pairwise[provider] = _pairwise_provider_metrics(baseline, provider)

    paired = _paired_metrics(index, providers, coverage)
    controls = _control_metrics(baseline, providers)
    disagreements = _disagreements(index)
    metrics = {
        "quantity": QUANTITY,
        "dependence": (
            "Meaning-preserving rows come from 15 original examples. "
            "Point metrics use those rows and are not independent observations."
        ),
        "rules": {
            "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
            "nll_clip_floor": NLL_CLIP_FLOOR,
            "yes_threshold": YES_THRESHOLD,
            "renormalize": False,
            "tie_prediction": "tie",
            "tie_counts_as_correct": False,
        },
        "coverage": {"by_provider": coverage},
        "deliberate_repeats_excluded_from_point_metrics": sum(
            1 for record in finals if not _baseline(record)
        ),
        "classification": classification,
        "pairwise": pairwise,
        "paired_models": paired,
        "meaning_change_controls": controls,
        "disagreement_count": len(disagreements),
    }
    return metrics, disagreements


def _pairwise_provider_metrics(baseline: list[dict], provider: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in baseline:
        if record["provider"] != provider or record["task"] != "pairwise":
            continue
        if record.get("label_changed") is not False:
            continue
        if record["transformation"] not in MEANING_PRESERVING:
            continue
        groups[record["example_id"]].append(record)

    complete = {}
    incomplete = 0
    ranks = []
    margins = []
    winner_correct = 0
    ties = 0
    for example_id, rows in groups.items():
        scores = _hypothesis_scores(rows)
        if scores is None:
            incomplete += 1
            continue
        complete[example_id] = scores
        intent = rows[0]["intent"]
        ranks.append(correct_rank(scores, intent))
        margins.append(margin(scores))
        winner = predict(scores)
        if winner == "tie":
            ties += 1
        elif winner == intent:
            winner_correct += 1
    n = len(complete)
    originals = {
        rows[0]["original_example_id"]: example_id
        for example_id, rows in groups.items()
        if rows and rows[0]["transformation"] == "original"
    }
    ranking = {"comparisons": 0, "flips": 0, "unique_to_tie": 0, "tie_to_unique": 0, "not_comparable": 0}
    prediction_flips = {"comparisons": 0, "flips": 0}
    changes: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for example_id, rows in groups.items():
        transformation = rows[0]["transformation"]
        if transformation not in VARIANT_TRANSFORMATIONS:
            continue
        original_example_id = rows[0]["original_example_id"]
        original_id = originals.get(original_example_id)
        original_scores = complete.get(original_id) if original_id else None
        variant_scores = complete.get(example_id)
        if variant_scores is not None and original_scores is None:
            ranking["not_comparable"] += 1
        if variant_scores is not None and original_scores is not None:
            ranking["comparisons"] += 1
            before = predict(original_scores)
            after = predict(variant_scores)
            if before != after:
                ranking["flips"] += 1
                if before != "tie" and after == "tie":
                    ranking["unique_to_tie"] += 1
                if before == "tie" and after != "tie":
                    ranking["tie_to_unique"] += 1
        for hypothesis in INTENTS:
            before_row = _find_hypothesis(groups.get(original_id, []), hypothesis) if original_id else None
            after_row = _find_hypothesis(rows, hypothesis)
            if before_row is None or after_row is None:
                continue
            before_p = _yes_probability(before_row)
            after_p = _yes_probability(after_row)
            if before_p is None or after_p is None:
                continue
            if _effective_status(before_row) != "success" or _effective_status(after_row) != "success":
                continue
            delta = after_p - before_p
            changes[transformation][hypothesis].append(delta)
            prediction_flips["comparisons"] += 1
            if yes_label(before_p) != yes_label(after_p):
                prediction_flips["flips"] += 1
    score_changes = {}
    for transformation, by_hypothesis in changes.items():
        score_changes[transformation] = {}
        for hypothesis, deltas in by_hypothesis.items():
            score_changes[transformation][hypothesis] = {
                "n": len(deltas),
                "mean_signed_change": _mean(deltas),
                "mean_absolute_change": _mean([abs(value) for value in deltas]),
            }
    return {
        "complete_groups": n,
        "incomplete_groups": incomplete,
        "mean_correct_intent_rank": _mean([float(value) for value in ranks]),
        "mean_margin": _mean(margins),
        "winner_accuracy": (winner_correct / n) if n else None,
        "tie_count": ties,
        "ranking_flips": ranking,
        "hypothesis_prediction_flips": prediction_flips,
        "paired_score_changes": score_changes,
    }


def _find_hypothesis(rows: list[dict], hypothesis: str) -> dict | None:
    for record in rows:
        if record.get("hypothesis") == hypothesis:
            return record
    return None


def _paired_metrics(index: dict, providers: list[str], coverage: dict) -> dict:
    own = {provider: coverage.get(provider) for provider in ("jev", "gpt")}
    if "jev" not in providers or "gpt" not in providers:
        return {
            "providers": ["jev", "gpt"],
            "jev_gpt_comparison": "not_both_present",
            "own_coverage": own,
            "classification": {"n": 0, "accuracy": {"jev": None, "gpt": None}},
            "pairwise_winner": {"n": 0, "accuracy": {"jev": None, "gpt": None}},
        }
    class_ids = _success_ids(index, "classification", None)
    both_class = class_ids["jev"] & class_ids["gpt"]
    class_accuracy = {}
    for provider in ("jev", "gpt"):
        rows = []
        for example_id in both_class:
            record = index[(provider, example_id, "classification", None)]
            if record.get("label_changed") is False:
                rows.append(record)
        class_accuracy[provider] = _classification_block(rows)["accuracy"]
    both_class_n = len(both_class)
    pair_ids = _complete_pairwise_ids(index)
    both_pairs = pair_ids["jev"] & pair_ids["gpt"]
    winner_accuracy = {}
    winner_n = 0
    for provider in ("jev", "gpt"):
        correct = 0
        counted = 0
        for example_id in both_pairs:
            scores = {}
            intent = None
            label_changed = None
            for hypothesis in INTENTS:
                record = index[(provider, example_id, "pairwise", hypothesis)]
                scores[hypothesis] = _yes_probability(record)
                intent = record["intent"]
                label_changed = record.get("label_changed")
            if label_changed is not False or any(value is None for value in scores.values()):
                continue
            counted += 1
            winner = predict(scores)
            if winner == intent:
                correct += 1
        winner_n = counted
        winner_accuracy[provider] = (correct / counted) if counted else None
    return {
        "providers": ["jev", "gpt"],
        "jev_gpt_comparison": "intersection_of_successful_items",
        "own_coverage": own,
        "paired_item_set": "label_unchanged",
        "classification": {"n": both_class_n, "accuracy": class_accuracy},
        "pairwise_winner": {"n": winner_n, "accuracy": winner_accuracy},
    }


def _any_success_ids(index: dict, task: str, hypothesis: str | None) -> dict[str, set[str]]:
    found = {"jev": set(), "gpt": set()}
    for (provider, example_id, record_task, record_hypothesis), record in index.items():
        if provider not in found or record_task != task or record_hypothesis != hypothesis:
            continue
        if _effective_status(record) == "success":
            found[provider].add(example_id)
    return found


def _success_ids(index: dict, task: str, hypothesis: str | None) -> dict[str, set[str]]:
    found = {"jev": set(), "gpt": set()}
    for (provider, example_id, record_task, record_hypothesis), record in index.items():
        if provider not in found or record_task != task or record_hypothesis != hypothesis:
            continue
        if record.get("label_changed") is not False:
            continue
        if _effective_status(record) == "success":
            found[provider].add(example_id)
    return found


def _complete_pairwise_ids(index: dict) -> dict[str, set[str]]:
    found = {"jev": set(), "gpt": set()}
    examples: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (provider, example_id, task, hypothesis), record in index.items():
        if provider not in found or task != "pairwise" or hypothesis not in INTENTS:
            continue
        if _effective_status(record) != "success":
            continue
        examples[(provider, example_id)].add(hypothesis)
    for (provider, example_id), hypotheses in examples.items():
        if set(hypotheses) == set(INTENTS):
            found[provider].add(example_id)
    return found


def _control_metrics(baseline: list[dict], providers: list[str]) -> dict:
    classification = {}
    pairwise = {}
    for provider in providers:
        class_rows = [
            record
            for record in baseline
            if record["provider"] == provider
            and record["task"] == "classification"
            and record["transformation"] == "meaning_change"
            and _effective_status(record) == "success"
        ]
        classification[provider] = _control_classification(class_rows)
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in baseline:
            if (
                record["provider"] == provider
                and record["task"] == "pairwise"
                and record["transformation"] == "meaning_change"
            ):
                groups[record["example_id"]].append(record)
        complete = 0
        incomplete = 0
        correct = 0
        ties = 0
        for rows in groups.values():
            scores = _hypothesis_scores(rows)
            if scores is None:
                incomplete += 1
                continue
            complete += 1
            winner = predict(scores)
            if winner == "tie":
                ties += 1
            elif winner == rows[0]["intent"]:
                correct += 1
        pairwise[provider] = {
            "complete_groups": complete,
            "incomplete_groups": incomplete,
            "accuracy": (correct / complete) if complete else None,
            "tie_count": ties,
        }
    return {
        "scored_against": "updated_label",
        "classification": classification,
        "pairwise": pairwise,
    }


def _disagreements(index: dict) -> list[dict]:
    rows = []
    class_ids = _any_success_ids(index, "classification", None)
    for example_id in sorted(class_ids["jev"] & class_ids["gpt"]):
        left = index[("jev", example_id, "classification", None)]
        right = index[("gpt", example_id, "classification", None)]
        left_scores = _distribution(left)
        right_scores = _distribution(right)
        if left_scores is None or right_scores is None:
            continue
        left_prediction = predict(left_scores)
        right_prediction = predict(right_scores)
        if _is_disagreement(left_prediction, right_prediction):
            rows.append(_disagreement_row(left, right, None, left_prediction, right_prediction, left_scores, right_scores))
    pair_ids = _complete_pairwise_ids(index)
    for example_id in sorted(pair_ids["jev"] & pair_ids["gpt"]):
        left_scores = {}
        right_scores = {}
        sample = None
        usable = True
        for hypothesis in INTENTS:
            left = index.get(("jev", example_id, "pairwise", hypothesis))
            right = index.get(("gpt", example_id, "pairwise", hypothesis))
            if left is None or right is None:
                usable = False
                break
            left_p = _yes_probability(left)
            right_p = _yes_probability(right)
            if left_p is None or right_p is None:
                usable = False
                break
            left_scores[hypothesis] = left_p
            right_scores[hypothesis] = right_p
            sample = left
        if not usable or sample is None:
            continue
        left_prediction = predict(left_scores)
        right_prediction = predict(right_scores)
        if _is_disagreement(left_prediction, right_prediction):
            right = index[("gpt", example_id, "pairwise", INTENTS[0])]
            rows.append(
                _disagreement_row(
                    sample,
                    right,
                    None,
                    left_prediction,
                    right_prediction,
                    left_scores,
                    right_scores,
                )
            )
    return rows


def _disagreement_row(left, right, hypothesis, left_prediction, right_prediction, left_scores, right_scores) -> dict:
    return {
        "task": left["task"],
        "example_id": left["example_id"],
        "original_example_id": left["original_example_id"],
        "transformation": left["transformation"],
        "hypothesis": hypothesis,
        "input_text": left.get("input_text"),
        "intent": left.get("intent"),
        "jev_prediction": left_prediction,
        "gpt_prediction": right_prediction,
        "jev_scores": left_scores,
        "gpt_scores": right_scores,
        "inspection_notes": "",
    }


def write_evaluation(run_dir) -> dict:
    records = read_jsonl(run_dir / "responses.jsonl")
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest.get("dataset") == "synthetic_alarm":
        from robust_likelihood.synthetic import evaluate_synthetic_records

        metrics, disagreements = evaluate_synthetic_records(records, manifest.get("items") or [])
    elif manifest.get("dataset") == "synthetic_alarm_v2":
        from robust_likelihood.synthetic_v2 import evaluate_synthetic_v2_records

        metrics, disagreements = evaluate_synthetic_v2_records(records, manifest.get("items") or [])
    elif manifest.get("dataset") == "synthetic_alarm_families":
        from robust_likelihood.synthetic_families import evaluate_family_records

        metrics, disagreements = evaluate_family_records(records, manifest.get("items") or [])
    elif manifest.get("dataset") == "toy_likelihood":
        from robust_likelihood.toy_likelihood import evaluate_toy_records

        metrics, disagreements = evaluate_toy_records(records, manifest.get("items") or [])
    else:
        metrics, disagreements = evaluate_records(records)
    if manifest_path.is_file():
        metrics["exploratory"] = manifest.get("exploratory")
        metrics["planned_requests"] = manifest.get("planned_requests")
        metrics["stopped_reason"] = manifest.get("stopped_reason")
        metrics["http_requests"] = manifest.get("http_requests")
        planned = manifest.get("planned_requests") or {}
        if isinstance(planned.get("total"), int):
            metrics["not_attempted"] = planned["total"] - len(final_records(records))
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "disagreements.jsonl", disagreements)
    return metrics
