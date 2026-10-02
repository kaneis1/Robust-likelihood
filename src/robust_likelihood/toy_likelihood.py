"""Compare elicited emission probabilities with the toy system's known rules."""

from __future__ import annotations

import json
from pathlib import Path

from robust_likelihood.planning import _spec
from robust_likelihood.prompting import HypothesisSet, load_hypotheses, load_prompts
from robust_likelihood.storage import read_json, read_jsonl
from robust_likelihood.synthetic_v2 import _yes
from robust_likelihood.toy_likelihood_items import (
    DATASET_VERSION,
    EVALUATION_RULE_VERSION,
    HYPOTHESES,
    HYPOTHESIS_VERSION,
    PROMPT_VERSION,
    authored_items,
    review_manifest,
)


def _canonical(rows: list[dict]) -> list[str]:
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for row in rows]


def load_toy_likelihood(root: Path) -> tuple[list[dict], dict, HypothesisSet, object]:
    directory = root / "data" / "synthetic" / "toy_likelihood"
    stored = read_jsonl(directory / "items.jsonl")
    expected = authored_items()
    if _canonical(stored) != _canonical(expected):
        raise ValueError("data/synthetic/toy_likelihood/items.jsonl does not match the authored items")
    review = read_json(directory / "review_manifest.json")
    if _canonical(review.get("items", [])) != _canonical(review_manifest()["items"]):
        raise ValueError("data/synthetic/toy_likelihood/review_manifest.json does not match the authored review")
    hypotheses = load_hypotheses(root, HYPOTHESIS_VERSION)
    prompts = load_prompts(root, PROMPT_VERSION)
    if set(hypotheses.descriptions) != set(HYPOTHESES):
        raise ValueError("toy hypotheses do not match the authored set")
    return stored, review, hypotheses, prompts


def build_toy_plan(items, providers, model_ids, hypotheses: HypothesisSet, settings, repeats=0):
    if repeats < 0:
        raise ValueError("repeats must be >= 0")
    repeat_ids: list[int | None] = [None, *range(1, repeats + 1)]
    specs = []
    for item in items:
        rendered = {
            "example_id": item["example_id"],
            "original_example_id": item["pair_id"],
            "transformation": item["role"],
            "utt": item["utt"],
            "intent": "",
            "original_intent": "",
            "label_changed": False,
            "dataset": DATASET_VERSION,
            "family": item["family"],
            "pair_id": item["pair_id"],
            "role": item["role"],
            "massive_lump_label": "",
        }
        for provider in providers:
            for repeat_id in repeat_ids:
                for hypothesis in HYPOTHESES:
                    specs.append(
                        _spec(
                            rendered,
                            provider,
                            model_ids[provider],
                            "pairwise",
                            hypothesis,
                            hypotheses.descriptions,
                            PROMPT_VERSION,
                            HYPOTHESIS_VERSION,
                            settings[provider],
                            repeat_id,
                        )
                    )
    return specs


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_toy_records(records: list[dict], items: list[dict]) -> tuple[dict, list[dict]]:
    from robust_likelihood.evaluate import final_records

    finals = [row for row in final_records(records) if row.get("repeat_id") is None] if records else []
    by_item = {item["example_id"]: item for item in items}
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    providers = []
    for record in finals:
        if record.get("task") != "pairwise" or record.get("status") != "success":
            continue
        provider = record.get("provider")
        if provider not in providers:
            providers.append(provider)
        value = _yes(record)
        hypothesis = record.get("hypothesis")
        if hypothesis not in HYPOTHESES or value is None:
            grouped.setdefault((provider, record.get("example_id")), {})
            grouped[(provider, record.get("example_id"))]["__invalid__"] = 1.0
            continue
        grouped.setdefault((provider, record.get("example_id")), {})[hypothesis] = value

    by_provider = {}
    for provider in providers:
        cells = []
        for item in items:
            scores = grouped.get((provider, item["example_id"]), {})
            complete = "__invalid__" not in scores and set(scores) == set(HYPOTHESES)
            errors = {}
            if complete:
                errors = {key: scores[key] - item["targets"][key] for key in HYPOTHESES}
            ranking_match = None
            if complete:
                ranking_match = _ranking(scores) == _ranking(item["targets"])
            cells.append(
                {
                    "example_id": item["example_id"],
                    "rules_id": item["rules_id"],
                    "sentence_id": item["sentence_id"],
                    "complete": complete,
                    "scores": scores if complete else None,
                    "targets": item["targets"],
                    "errors": errors or None,
                    "absolute_errors": {key: abs(value) for key, value in errors.items()} if errors else None,
                    "ranking_match": ranking_match,
                    "score_sum": None if not complete else sum(scores.values()),
                }
            )
        complete_cells = [cell for cell in cells if cell["complete"]]
        absolute = [value for cell in complete_cells for value in cell["absolute_errors"].values()]
        by_rules = {}
        for rules_id in sorted({cell["rules_id"] for cell in complete_cells}):
            chosen = [value for cell in complete_cells if cell["rules_id"] == rules_id for value in cell["absolute_errors"].values()]
            by_rules[rules_id] = {"mean_absolute_error": _mean(chosen), "cells": sum(1 for cell in complete_cells if cell["rules_id"] == rules_id)}
        by_provider[provider] = {
            "cells": cells,
            "complete_cells": len(complete_cells),
            "mean_absolute_error": _mean(absolute),
            "max_absolute_error": None if not absolute else max(absolute),
            "within_0_05": None if not absolute else sum(1 for value in absolute if value <= 0.05) / len(absolute),
            "ranking_matches": sum(1 for cell in complete_cells if cell["ranking_match"]),
            "ranking_complete": len(complete_cells),
            "mean_score_sum": _mean([cell["score_sum"] for cell in complete_cells]),
            "by_rules": by_rules,
        }
    metrics = {
        "dataset": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "quantity": "Model scores are elicited judgments. The targets are defined toy rules, not verified P(e|h) for real users.",
        "by_provider": by_provider,
        "disagreement_count": 0,
    }
    return metrics, []


def _ranking(scores: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda key: (-scores[key], key)))
