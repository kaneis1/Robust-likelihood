import math

import pytest

from robust_likelihood.typicality import (
    GaussianHypothesis,
    absolute_deviation,
    diagnostic_items,
    expected_log_density,
    formal_typicality_question,
    gaussian_log_density,
    hypothesis_dependent_bin_posterior,
    pivotal_bin_probability,
    point_comparison,
    signed_deviation,
    simulate_contamination,
    simulate_known_class,
    TypicalitySpecificationError,
    comparison_slice,
)


NARROW = GaussianHypothesis("narrow", 0.0, 1.0)
WIDE = GaussianHypothesis("wide", 0.0, 2.0)


def test_gaussian_mode_is_above_the_average_log_probability():
    assert signed_deviation(0.0, NARROW) == pytest.approx(0.5)
    assert signed_deviation(1.0, NARROW) == pytest.approx(0.0)
    assert signed_deviation(-1.0, NARROW) == pytest.approx(0.0)
    assert absolute_deviation(0.0, NARROW) == pytest.approx(0.5)
    assert expected_log_density(NARROW) == pytest.approx(gaussian_log_density(0.0, NARROW) - 0.5)
    assert signed_deviation(3.0, WIDE) == pytest.approx(signed_deviation(1.5, NARROW))


def test_typicality_bin_probabilities_are_the_same_for_every_gaussian():
    total = sum(pivotal_bin_probability(kind, 0.25) for kind in ("above", "near", "below"))
    assert total == pytest.approx(1.0)
    assert pivotal_bin_probability("above", 0.25) > pivotal_bin_probability("near", 0.25)


def test_hypothesis_dependent_bins_reverse_the_shell_and_ignore_the_far_tail():
    shell = point_comparison(1.0, [NARROW, WIDE], 0.25)
    assert shell["raw_map"] == "narrow"
    assert shell["fixed_bin_map"] == "narrow"
    assert shell["hypothesis_dependent_bin_map"] == "wide"
    assert shell["deviation_bin"] == {"narrow": "near", "wide": "above"}

    tail = point_comparison(10.0, [NARROW, WIDE], 0.25)
    assert tail["raw_map"] == "wide"
    assert tail["fixed_bin_map"] == "wide"
    assert tail["hypothesis_dependent_bin_map"] == "tie"
    dependent = hypothesis_dependent_bin_posterior(10.0, [NARROW, WIDE], 0.25)
    assert dependent == pytest.approx([0.5, 0.5])


def test_wider_epsilon_removes_the_shell_reversal():
    tight = point_comparison(1.0, [NARROW, WIDE], 0.1)
    wide_band = point_comparison(1.0, [NARROW, WIDE], 0.5)
    assert tight["hypothesis_dependent_bin_map"] == "wide"
    assert wide_band["deviation_bin"] == {"narrow": "near", "wide": "near"}
    assert wide_band["hypothesis_dependent_bin_map"] == "tie"
    assert wide_band["raw_map"] == "narrow"


def test_fixed_bins_keep_less_information_than_the_raw_observation():
    summary = simulate_known_class(
        [NARROW, WIDE, GaussianHypothesis("shift", 2.0, 1.0)],
        draws=2000,
        seed=20261007,
    )
    assert summary["mutual_information_fixed_bin_nats"] < summary["mutual_information_raw_nats"]
    assert summary["mean_tv_dependent_bin_vs_raw"] > summary["mean_tv_fixed_bin_vs_raw"]


def test_contaminated_tail_moves_raw_bayes_and_leaves_dependent_bins_at_the_prior():
    summary = simulate_contamination(draws=2500, seed=11)
    masses = summary["mean_posterior_on_narrow_among_outliers"]
    assert summary["outlier_count"] > 50
    assert masses["raw"] < 0.05
    assert 0.01 < masses["fixed_bin"] < 0.03
    assert masses["hypothesis_dependent_bin"] == pytest.approx(0.5, abs=1e-9)


def test_diagnostics_have_no_typicality_label_and_the_slice_is_not_sent():
    rows = diagnostic_items()
    assert {row["role"] for row in rows} >= {
        "empty",
        "very_short",
        "ordinary",
        "meaning_preserving",
        "repetition",
        "information_removed",
        "irrelevant_extension",
    }
    assert all(row["typicality_label"] is None and row["reference_distribution"] is None for row in rows)
    assert rows[0]["utt"] == ""
    slice_rows = comparison_slice()
    assert [row["kind"] for row in slice_rows] == [
        "direct_request",
        "quoted_command",
        "negation",
        "ordered_plan",
    ]
    assert all(row["status"] == "prepared_not_sent" and row["typicality_label"] is None for row in slice_rows)
    assert slice_rows[2]["utt"].startswith("Don't delete")
    assert "nine" in slice_rows[3]["utt"]


def test_formal_question_requires_a_reference_distribution():
    with pytest.raises(TypicalitySpecificationError, match="reference distribution"):
        formal_typicality_question(scale="signed_deviation", reference_distribution=None)
    with pytest.raises(TypicalitySpecificationError, match="epsilon"):
        formal_typicality_question(
            scale="within_epsilon",
            reference_distribution="a stated unigram model",
            epsilon_nats=None,
        )
    question = formal_typicality_question(
        scale="within_epsilon",
        reference_distribution="a stated unigram model",
        epsilon_nats=0.25,
    )
    assert "average log-probability" in question
    assert "0.25" in question
    assert "yes-or-no" in question
    signed = formal_typicality_question(
        scale="signed_deviation",
        reference_distribution="a stated unigram model",
    )
    assert "Report the signed typicality gap" in signed
    assert math.isfinite(signed_deviation(0.0, NARROW))
