import math

from robust_likelihood.constants import NLL_CLIP_FLOOR
from robust_likelihood.evaluate import evaluate_records, final_records


def rec(**overrides):
    base = {
        "cache_key": None,
        "attempt_id": 0,
        "repeat_id": None,
        "status": "success",
        "provider": "jev",
        "task": "classification",
        "example_id": "1__original",
        "original_example_id": "1",
        "transformation": "original",
        "hypothesis": None,
        "intent": "alarm_set",
        "original_intent": "alarm_set",
        "label_changed": False,
        "input_text": "set an alarm for seven am",
        "parsed": {
            "distribution": {"alarm_set": 1.0, "alarm_query": 0.0, "alarm_remove": 0.0}
        },
        "requested_model_id": "jev-1.13.0",
        "returned_model_id": "jev-1.13.0",
    }
    base.update(overrides)
    if base["cache_key"] is None:
        base["cache_key"] = (
            f"{base['provider']}:{base['example_id']}:{base['task']}:{base['hypothesis']}:{base['repeat_id']}"
        )
    return base


def pairwise(provider, example_id, transformation, hypothesis, probability, intent="alarm_set", label_changed=False):
    return rec(
        provider=provider,
        task="pairwise",
        example_id=example_id,
        original_example_id=example_id.split("__")[0],
        transformation=transformation,
        hypothesis=hypothesis,
        intent=intent,
        label_changed=label_changed,
        parsed={"plausibility_yes_probability": probability},
    )


def test_retry_success_is_one_final_record():
    failed = rec(status="failed", attempt_id=0, cache_key="same")
    success = rec(status="success", attempt_id=1, cache_key="same")
    finals = final_records([failed, success])
    assert len(finals) == 1
    assert finals[0]["status"] == "success"


def test_tie_is_incorrect_and_malformed_probabilities_are_excluded():
    rows = [
        rec(
            example_id="tie",
            cache_key="tie",
            parsed={"distribution": {"alarm_set": 0.5, "alarm_query": 0.5, "alarm_remove": 0.0}},
        ),
        rec(example_id="good", cache_key="good"),
        rec(
            example_id="bad-sum",
            cache_key="bad-sum",
            parsed={"distribution": {"alarm_set": 0.2, "alarm_query": 0.3, "alarm_remove": 0.6}},
        ),
        rec(
            example_id="missing",
            cache_key="missing",
            parsed={"distribution": {"alarm_query": 0.4, "alarm_remove": 0.6}},
        ),
        rec(
            example_id="zero",
            cache_key="zero",
            parsed={"distribution": {"alarm_set": 0.0, "alarm_query": 0.4, "alarm_remove": 0.6}},
        ),
    ]
    metrics, _ = evaluate_records(rows)
    block = metrics["classification"]["jev"]
    assert block["n"] == 3
    assert block["tie_count"] == 1
    assert block["correct"] == 1
    assert block["accuracy"] == 1 / 3
    assert block["nll"] == (
        -math.log(0.5) + 0.0 + (-math.log(NLL_CLIP_FLOOR))
    ) / 3
    assert metrics["coverage"]["by_provider"]["jev"]["classification"]["invalid"] == 2
    assert metrics["rules"]["renormalize"] is False
    assert "token_logprob_baseline" not in metrics


def test_incomplete_groups_still_allow_a_paired_score_change():
    rows = [
        pairwise("jev", "1__original", "original", "alarm_set", 0.9),
        pairwise("jev", "1__paraphrase", "paraphrase", "alarm_set", 0.4),
    ]
    metrics, _ = evaluate_records(rows)
    block = metrics["pairwise"]["jev"]
    assert block["complete_groups"] == 0
    assert block["incomplete_groups"] == 2
    assert block["mean_correct_intent_rank"] is None
    change = block["paired_score_changes"]["paraphrase"]["alarm_set"]
    assert change["n"] == 1
    assert change["mean_signed_change"] == -0.5


def test_ranking_flip_includes_unique_winner_becoming_a_tie():
    rows = []
    original = {"alarm_set": 0.9, "alarm_query": 0.2, "alarm_remove": 0.1}
    variant = {"alarm_set": 0.4, "alarm_query": 0.4, "alarm_remove": 0.1}
    for hypothesis, probability in original.items():
        rows.append(pairwise("jev", "1__original", "original", hypothesis, probability))
    for hypothesis, probability in variant.items():
        rows.append(pairwise("jev", "1__paraphrase", "paraphrase", hypothesis, probability))
    metrics, _ = evaluate_records(rows)
    flips = metrics["pairwise"]["jev"]["ranking_flips"]
    assert flips["comparisons"] == 1
    assert flips["flips"] == 1
    assert flips["unique_to_tie"] == 1
    assert metrics["pairwise"]["jev"]["hypothesis_prediction_flips"]["flips"] == 1
    assert metrics["pairwise"]["jev"]["complete_groups"] == 2
    assert metrics["pairwise"]["jev"]["mean_margin"] > 0


def test_controls_are_scored_against_the_new_label():
    rows = [
        rec(
            task="classification",
            example_id="9__meaning_change",
            transformation="meaning_change",
            intent="alarm_remove",
            original_intent="alarm_set",
            label_changed=True,
            parsed={"distribution": {"alarm_set": 0.0, "alarm_query": 0.0, "alarm_remove": 1.0}},
        )
    ]
    metrics, _ = evaluate_records(rows)
    assert metrics["classification"]["jev"]["n"] == 0
    assert metrics["meaning_change_controls"]["classification"]["jev"]["accuracy"] == 1.0
    assert metrics["meaning_change_controls"]["scored_against"] == "updated_label"


def test_disagreement_list_leaves_notes_blank():
    rows = [
        rec(provider="jev", cache_key="j"),
        rec(
            provider="gpt",
            cache_key="g",
            parsed={"distribution": {"alarm_set": 0.5, "alarm_query": 0.5, "alarm_remove": 0.0}},
        ),
        rec(
            provider="claude",
            cache_key="c",
            parsed={"distribution": {"alarm_set": 0.0, "alarm_query": 1.0, "alarm_remove": 0.0}},
        ),
    ]
    for hypothesis, probability in {"alarm_set": 0.2, "alarm_query": 0.2, "alarm_remove": 0.2}.items():
        rows.append(pairwise("jev", "1__original", "original", hypothesis, probability))
        rows.append(pairwise("gpt", "1__original", "original", hypothesis, probability))
    metrics, disagreements = evaluate_records(rows)
    assert metrics["disagreement_count"] == 1
    assert disagreements[0]["jev_prediction"] == "alarm_set"
    assert disagreements[0]["gpt_prediction"] == "tie"
    assert disagreements[0]["inspection_notes"] == ""
    assert metrics["paired_models"]["classification"]["n"] == 1
    assert metrics["paired_models"]["own_coverage"]["jev"]["classification"]["attempted"] == 1


def test_deliberate_repeats_are_excluded_from_point_metrics():
    rows = [rec(cache_key="base", repeat_id=None), rec(cache_key="rep", repeat_id=1)]
    metrics, _ = evaluate_records(rows)
    assert metrics["classification"]["jev"]["n"] == 1
    assert metrics["deliberate_repeats_excluded_from_point_metrics"] == 1
