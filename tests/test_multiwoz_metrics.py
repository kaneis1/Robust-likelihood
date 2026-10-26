import math
from pathlib import Path

import pytest

from robust_likelihood.multiwoz import load_config
from robust_likelihood.multiwoz.__main__ import prepare
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.metrics import (
    Bootstrap,
    analyze,
    auroc,
    bootstrap_weights,
    controls,
    goal_key,
    logit_score,
    monotonicity,
    occam_table,
    removal_pairs,
    repeat_noise,
    split_key,
    term_vector,
    wording_gap,
)
from robust_likelihood.multiwoz.reference import replicate_terms
from robust_likelihood.storage import read_json, read_jsonl, repo_root, write_json, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    folder = tmp_path_factory.mktemp("derived")
    prepare(repo_root(), load_config(repo_root()), raw=FIXTURE, derived=folder)
    return folder


def _clean(derived):
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    table = read_json(derived / "reference_table.json")
    replicates = read_json(derived / "reference_replicates.json")
    primary = table["variants"]["primary"]
    terms = [term_vector(row) for row in replicate_terms(replicates["populations"], primary["a_from"], primary["b_from"], 0.5)]
    weights = bootstrap_weights(len(items), 50, 3, "metrics-test")
    return items, probe, primary, Bootstrap(weights, term_vector(primary["slots"]), terms[:50])


def _oracle(items, probe, primary):
    """log P(turn-0 slot statuses | hypothesis) under the reference rates."""
    by_id = {item["item_id"]: item for item in items}
    scores = {}
    for row in probe:
        status = by_id[row["item_id"]]["slot_status"]
        total = 0.0
        for slot in SLOTS:
            if status[slot] not in ("named", "omitted"):
                continue
            rate = primary["slots"][slot]["a" if slot in row["values"] else "b"]
            total += math.log(rate if status[slot] == "named" else 1 - rate)
        scores.setdefault(row["item_id"], {})[row["hyp_key"]] = total
    return scores


def test_an_oracle_scorer_has_no_occam_gap(derived):
    items, probe, primary, boot = _clean(derived)
    pairs = removal_pairs(items, _oracle(items, probe, primary))
    table, _ = occam_table(pairs, boot)
    groups = [key for key in table if "slots_mean" not in key]
    assert {"impl|food|named", "expl|area|named", "impl|pricerange|omitted", "impl|all|named"} <= set(groups)
    for key in groups:
        assert table[key]["OG"] == pytest.approx(0.0, abs=1e-9)
        assert table[key]["mean_L_J"] == pytest.approx(table[key]["L_E"], abs=1e-9)


def test_a_constant_scorer_has_occam_gap_minus_l_e(derived):
    items, probe, _, boot = _clean(derived)
    flat = {}
    for row in probe:
        flat.setdefault(row["item_id"], {})[row["hyp_key"]] = 0.7
    table, _ = occam_table(removal_pairs(items, flat), boot)
    for key, group in table.items():
        assert group["mean_L_J"] == 0.0
        assert group["OG"] == pytest.approx(-group["L_E"])
    food = table["impl|food|named"]
    assert food["L_E"] == pytest.approx(read_json(derived / "reference_table.json")["variants"]["primary"]["slots"]["food"]["named"])


def test_scores_that_ignore_the_explicit_sentence_have_no_wording_gap(derived):
    items, probe, primary, boot = _clean(derived)
    gap = wording_gap(items, _oracle(items, probe, primary), boot)
    assert gap["all"]["n"] > 0
    assert gap["all"]["mean"] == 0.0 and gap["all"]["share_abs_above_1"] == 0.0
    assert gap["all"]["mean_ci95"] == [0.0, 0.0]
    assert set(gap) >= {"all", "G", "broad", "minus:food"}


def test_auroc_on_known_inputs():
    assert auroc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(3.5 / 4)
    assert auroc([1.0], [2.0]) == 0.0
    assert auroc([2.0], [2.0]) == 0.5
    assert auroc([], [1.0]) is None


def test_monotonicity_counts_violations_and_ties_apart():
    pairs = [
        {"regime": "impl", "removal": "food", "direction": "named", "narrow": 1.0, "broader": 2.0},
        {"regime": "impl", "removal": "area", "direction": "named", "narrow": 1.0, "broader": 1.0},
        {"regime": "impl", "removal": "area", "direction": "omitted", "narrow": 0.0, "broader": 5.0},
        {"regime": "impl", "removal": "all", "direction": "named", "narrow": 3.0, "broader": 1.0},
    ]
    found = monotonicity(pairs)
    assert found["impl|single_named"] == {"n": 2, "violations": 1, "violation_rate": 0.5, "ties": 1, "tie_rate": 0.5}
    assert found["impl|full_all_named"]["violations"] == 0
    assert found["expl|single_named"]["n"] == 0


def test_controls_compare_the_goal_with_each_control(derived):
    items, probe, _, boot = _clean(derived)
    changed = {(row["item_id"], row["hyp_key"]): row["changed"]["slot"] for row in probe if "changed" in row}
    scores = {}
    for row in probe:
        scores.setdefault(row["item_id"], {})[row["hyp_key"]] = 2.0 if row["hyp_key"] == "G|impl" else (
            -1.0 if row["hyp_key"] == "ctrl:contra" else 1.5
        )
    found = controls(items, scores, changed, boot)
    assert found["ctrl:contra"]["auroc"] == 1.0
    assert found["ctrl:contra"]["mean_gap"] == 3.0
    assert found["ctrl:tie"]["mean_gap"] == 0.5
    assert found["tie_offset_all"]["n"] == found["ctrl:tie"]["n"] + found["ctrl:tie2"]["n"]


def test_keys_split_into_type_and_regime():
    assert split_key("minus:food|expl") == ("minus:food", "expl")
    assert split_key("ctrl:tie") == ("ctrl:tie", "impl")
    assert goal_key(3, "expl") == "G|impl" and goal_key(2, "expl") == "G|expl"


def test_repeat_noise_reports_determinism():
    rows = [
        {"prompt_version": "v", "item_id": "a", "hyp_key": "G|impl", "repeat_id": None, "noul": 0.8},
        {"prompt_version": "v", "item_id": "a", "hyp_key": "G|impl", "repeat_id": 1, "noul": 0.8},
        {"prompt_version": "w", "item_id": "a", "hyp_key": "G|impl", "repeat_id": None, "noul": 0.8},
        {"prompt_version": "w", "item_id": "a", "hyp_key": "G|impl", "repeat_id": 1, "noul": 0.6},
    ]
    found = repeat_noise(rows, logit_score)
    assert found["v"]["deterministic"] is True and found["v"]["mean_sd"] == 0.0
    assert found["w"]["deterministic"] is False
    assert found["w"]["max_range"] == pytest.approx(logit_score(0.8) - logit_score(0.6))


def test_analyze_pairs_the_clean_bootstrap_with_the_reference(derived, tmp_path):
    probe = read_jsonl(derived / "probe.jsonl")
    items = read_jsonl(derived / "items.jsonl")
    scores = []
    for version, shift in (("multiwoz_likely", 0.0), ("multiwoz_typical", 0.05)):
        for row in probe:
            noul = min(0.95, 0.1 + 0.02 * len(row["text"]) % 0.8 + shift)
            scores.append({"item_id": row["item_id"], "hyp_key": row["hyp_key"], "prompt_version": version,
                           "repeat_id": None, "noul": noul})
    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "scores.jsonl", scores)
    write_json(run_dir / "manifest.json", {
        "selection": {"item_ids": [item["item_id"] for item in items]},
        "requested_model_id": "jev-1.13.0",
        "returned_model_ids": {"jev-1.13.0": len(scores)},
    })
    config = load_config(repo_root())
    config["bootstrap"] = {"replicates": 60, "seed": config["bootstrap"]["seed"]}
    metrics = analyze(run_dir, derived, config)
    clean = metrics["sets"]["clean"]
    assert clean["paired_with_reference"] is True
    assert clean["pairing_verified"] is True
    assert metrics["sets"]["fail_info"]["paired_with_reference"] is False
    assert set(clean["mappings"]) == {"logit", "log"}
    assert set(clean["mappings"]["logit"]) == {"multiwoz_likely", "multiwoz_typical"}
    assert set(metrics["variants"]) == {"V2", "V3", "V4"}
    assert [row["id"] for row in metrics["predictions"]] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert metrics["predictions"][-1]["pass"] is None
    assert metrics["clipped"] == {"multiwoz_likely": 0, "multiwoz_typical": 0}
