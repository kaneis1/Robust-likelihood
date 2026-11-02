import math
import statistics
from fractions import Fraction

import pytest

from robust_likelihood.dice import load_config as load_dice_config
from robust_likelihood.dice.design import build_pairs, draw_sequences
from robust_likelihood.multiwoz.lm import typicality_scores
from robust_likelihood.multiwoz.typv0 import (
    bin_index,
    centres,
    dice_check,
    die_distribution,
    held_out_sem,
    p_sem,
    p_sem_factorized,
    pool_stats,
    pool_weights,
    variant_scores,
)
from robust_likelihood.storage import read_json, repo_root

VARIANTS = read_json(repo_root() / "configs" / "typicality_v0.json")["variants"]


def test_bins_and_option_b_centres_follow_the_definitions():
    assert [bin_index(t, (1.0, 2.0), 2) for t in (0.5, -1.2, 2.0, -3.0)] == [0, 1, 2, 2]
    assert [bin_index(t, (-1.0, 0.0, 1.0), 1) for t in (-2.0, -1.0, -0.2, 0.0, 0.7, 5.0)] == [0, 1, 1, 2, 2, 3]
    assert centres((-1.0, 0.0, 1.0)) == [-1.5, -0.5, 0.5, 1.5]
    assert centres((-0.5, 0.5)) == [-1.0, 0.0, 1.0]
    assert centres((-2.0, -1.0, 0.0, 1.0, 2.0)) == [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


def test_pool_statistics_are_the_weighted_mixture_moments():
    moments = {"a": (1.0, 1.0), "b": (3.0, 2.0)}
    mean, spread = pool_stats(moments, {"a": 1.0, "b": 1.0})
    assert mean == pytest.approx(2.0)
    assert spread == pytest.approx(math.sqrt((1 + 1 + 4 + 1) / 2))
    mean, _ = pool_stats(moments, {"a": 3.0, "b": 1.0})
    assert mean == pytest.approx(1.5)
    probe = [{"text": "x"}, {"text": "x"}, {"text": "y"}]
    assert pool_weights(probe, "uniform") == {"x": 1.0, "y": 1.0}
    assert pool_weights(probe, "prior") == {"x": 2.0, "y": 1.0}


def test_semantic_probability_holds_each_sample_out():
    tuples = [("italian", "north", "none")] * 3 + [("none", "north", "none")]
    sem, semf = held_out_sem(tuples)
    assert sem[0] == pytest.approx(-math.log((2 + 0.5) / 4))
    assert sem[3] == pytest.approx(-math.log((0 + 0.5) / 4))
    assert semf[3] == pytest.approx(-math.log(0.5 / 4) - 2 * math.log(3.5 / 4))
    wanted = {"food": ["italian"], "area": ["north"], "pricerange": ["none"]}
    assert p_sem(tuples, wanted) == pytest.approx(3.5 / 5)
    assert p_sem_factorized(tuples, wanted) == pytest.approx((3.5 / 5) * (4.5 / 5) * (4.5 / 5))


def test_die_distribution_is_exact():
    fair = {face: Fraction(1, 6) for face in range(1, 7)}
    assert die_distribution(fair, 3) == [(pytest.approx(3 * math.log(6)), pytest.approx(1.0))]
    loaded = {**{face: Fraction(1, 10) for face in range(1, 6)}, 6: Fraction(1, 2)}
    rows = die_distribution(loaded, 2)
    assert sum(p for _, p in rows) == pytest.approx(1.0)
    assert dict(rows)[round(2 * math.log(2), 12)] == pytest.approx(0.25)


def test_per_hypothesis_typicality_ties_fair_and_even_dice():
    config = load_dice_config(repo_root())
    pairs = build_pairs(config, draw_sequences(config))
    found = dice_check(config, pairs, VARIANTS)
    assert found["per2A"]["tie_rate"] == 1.0
    assert found["per2A"]["impossible_ruled_out"] == 1.0
    assert found["glob1B"]["impossible_ruled_out"] == 0.0


class FakeInputs:
    def __init__(self, samples, cells, pool):
        self.samples, self.cells, self.pool = samples, cells, pool


def test_per2a_matches_phase8_typicality_scores():
    values = [3.0, 4.0, 5.0, 7.0, 2.5, 6.0]
    inputs = FakeInputs({"h": {"S": values}}, [{"key": ("i", "G|impl", "original", "original"), "source": "h",
                                                 "S": 9.0, "cent": 0.0, "sem": 0.0}], ["h"])
    log, bins, _ = variant_scores(inputs, VARIANTS["per2A"], {"h": 1.0})
    expected = typicality_scores(9.0, values, (1.0, 2.0))
    assert log[("i", "G|impl", "original", "original")] == pytest.approx(expected["lm_typ_bins"])
    assert bins[("i", "G|impl", "original", "original")] == 2


def test_option_b_scores_a_bin_by_its_centre():
    inputs = FakeInputs({"h": {"S": [1.0, 3.0]}, "g": {"S": [5.0, 7.0]}},
                        [{"key": ("i", "G|impl", "original", "original"), "source": "h", "S": 100.0, "cent": 0.0,
                          "sem": 0.0}], ["h", "g"])
    log, bins, pool = variant_scores(inputs, VARIANTS["glob1B"], {"h": 1.0, "g": 1.0})
    mean = statistics.fmean([2.0, 6.0])
    assert pool["H"] == pytest.approx(mean)
    assert bins[("i", "G|impl", "original", "original")] == 3
    assert log[("i", "G|impl", "original", "original")] == pytest.approx(-(mean + pool["sigma"] * 1.5))
