import math

import pytest

from robust_likelihood.constants import NLL_CLIP_FLOOR
from robust_likelihood.scoring import (
    is_probability,
    margin,
    multiclass_brier,
    negative_log_likelihood,
    predict,
    sums_to_one,
    validate_distribution,
    yes_label,
)


def test_probability_rejects_bools_and_out_of_range():
    assert is_probability(0) and is_probability(1) and is_probability(0.5)
    assert not is_probability(True)
    assert not is_probability(False)
    assert not is_probability(-0.1)
    assert not is_probability(1.1)
    assert not is_probability("0.5")


def test_sum_tolerance_is_absolute_and_does_not_renormalize():
    assert sums_to_one(1.0)
    assert sums_to_one(1.0005)
    assert sums_to_one(0.9995)
    assert not sums_to_one(1.002)
    assert not sums_to_one(0.998)
    valid = {"alarm_set": 0.5, "alarm_query": 0.5, "alarm_remove": 0.0}
    assert validate_distribution(valid) == {"alarm_set": 0.5, "alarm_query": 0.5, "alarm_remove": 0.0}
    assert validate_distribution({"alarm_set": 0.5, "alarm_query": 0.5, "alarm_remove": 0.1}) is None
    assert validate_distribution({"alarm_set": -0.1, "alarm_query": 0.6, "alarm_remove": 0.5}) is None
    assert validate_distribution({"alarm_set": 0.5, "alarm_query": 0.5}) is None


def test_ties_rank_margin_and_nll_floor():
    scores = {"alarm_set": 0.4, "alarm_query": 0.4, "alarm_remove": 0.2}
    assert predict(scores) == "tie"
    assert margin(scores) == 0.0
    assert correct_rank_is_one_for_tied_truth(scores)
    unique = {"alarm_set": 0.7, "alarm_query": 0.2, "alarm_remove": 0.1}
    assert predict(unique) == "alarm_set"
    assert margin(unique) == pytest.approx(0.5)
    assert negative_log_likelihood(0.0) == pytest.approx(-math.log(NLL_CLIP_FLOOR))
    assert negative_log_likelihood(1.0) == 0.0
    perfect = {"alarm_set": 1.0, "alarm_query": 0.0, "alarm_remove": 0.0}
    assert multiclass_brier(perfect, "alarm_set") == 0.0
    assert yes_label(0.5) == "tie"
    assert yes_label(0.51) == "yes"
    assert yes_label(0.49) == "no"


def correct_rank_is_one_for_tied_truth(scores):
    from robust_likelihood.scoring import correct_rank

    return correct_rank(scores, "alarm_set") == 1 and correct_rank(scores, "alarm_remove") == 3
