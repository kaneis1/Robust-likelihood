import math
from collections import Counter
from pathlib import Path

import pytest

from robust_likelihood.multiwoz.corpus import item_population, load_goals, load_turns, pooled_population, select_items
from robust_likelihood.multiwoz.reference import (
    a_rate,
    appendix_checks,
    b_rate,
    bootstrap_rng,
    build_reference,
    log_ratios,
    replicate_counts,
    replicate_terms,
    resample,
    slot_counts,
    wilson,
)

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"
OTHER = ("hotel", "train", "taxi", "attraction", "police", "hospital")
CONFIG = {
    "bootstrap": {"replicates": 40, "seed": 7},
    "reference": {"a_population": "match_clean_turn0", "b_population": "pooled_first_restaurant_turn", "b_smoothing": 0.5},
}


def _populations():
    goals, turns = load_goals(FIXTURE), load_turns(FIXTURE)
    items, _ = select_items(goals, turns, OTHER)
    pooled, _ = pooled_population(goals, turns)
    return {
        "match_clean_turn0": item_population([item for item in items if item["set"] == "clean"]),
        "match_all_turn0": item_population(items),
        "pooled_first_restaurant_turn": pooled,
        "pooled_no_woz": [row for row in pooled if not row["dialogue_id"].startswith("WOZ")],
    }


def test_estimators_and_smoothing():
    assert a_rate(142, 195) == pytest.approx(0.72821, abs=1e-5)
    assert a_rate(0, 0) is None
    assert b_rate(5, 819, 0.5) == pytest.approx(5.5 / 820)
    assert b_rate(0, 0, 0.5) == 0.5
    low, high = wilson(142, 195)
    assert low == pytest.approx(0.662, abs=1e-3)
    assert high == pytest.approx(0.786, abs=1e-3)
    terms = log_ratios(142 / 195, 5.5 / 820)
    assert terms["named"] == pytest.approx(4.6875, abs=1e-3)
    assert terms["omitted"] == pytest.approx(math.log((1 - 142 / 195) / (1 - 5.5 / 820)))
    assert log_ratios(1.0, 0.1) == {"named": pytest.approx(math.log(10.0)), "omitted": None}
    assert log_ratios(0.0, 0.1)["named"] is None


def test_slot_counts_split_constrained_from_no_preference():
    counts = slot_counts(_populations()["match_clean_turn0"])
    assert counts["food"] == {"constrained": 4, "named": 4, "named_mismatch": 0, "no_pref": 2, "no_pref_named": 0}
    assert counts["area"] == {"constrained": 4, "named": 2, "named_mismatch": 1, "no_pref": 2, "no_pref_named": 1}


def test_bootstrap_is_deterministic_and_matches_its_resample():
    rows = _populations()["pooled_first_restaurant_turn"]
    first = replicate_counts(rows, 25, 7, "pooled_first_restaurant_turn")
    again = replicate_counts(list(reversed(rows)), 25, 7, "pooled_first_restaurant_turn")
    other = replicate_counts(rows, 25, 7, "pooled_no_woz")
    assert first == again
    assert first != other
    ordered = sorted(rows, key=lambda row: row["dialogue_id"])
    rng = bootstrap_rng(7, "pooled_first_restaurant_turn")
    draw = [ordered[index] for index in resample(len(ordered), rng)]
    expected = slot_counts(draw)
    for slot in ("food", "area", "pricerange"):
        row = expected[slot]
        assert first[slot][0] == [row["named"], row["constrained"], row["no_pref_named"], row["no_pref"]]


def test_reference_table_variants_and_replicate_terms():
    populations = _populations()
    table, replicates = build_reference(populations, CONFIG)
    assert set(table["variants"]) == {"primary", "V2", "V3", "V4"}
    primary = table["variants"]["primary"]["slots"]["area"]
    assert primary["a"] == pytest.approx(2 / 4)
    assert primary["b"] == pytest.approx((1 + 0.5) / (3 + 1))
    assert primary["named"] == pytest.approx(math.log(0.5 / 0.375))
    assert table["populations"]["match_clean_turn0"]["size"] == 6
    assert table["populations"]["match_clean_turn0"]["mismatch_rate"]["area"] == pytest.approx(0.5)
    terms = replicate_terms(replicates["populations"], "match_clean_turn0", "pooled_first_restaurant_turn", 0.5)
    assert len(terms) == 40
    named, constrained, _, _ = replicates["populations"]["match_clean_turn0"]["area"][3]
    assert terms[3]["area"]["a"] == a_rate(named, constrained)
    checks = appendix_checks(table)
    assert checks["A2"] and checks["A3"]
    assert Counter(len(values) for values in replicates["populations"]["pooled_no_woz"].values()) == {40: 3}
