import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from robust_likelihood.gaussian.inference import (
    cumulative_log_posterior,
    first_reaching,
    mean_and_se,
    metrics,
    posterior_mode,
    scale_objective,
    uniform_log_prior,
    unique_argmax,
)
from robust_likelihood.gaussian.model import quadratic_forms
from robust_likelihood.gaussian.scorers import score_matrix

SEED = 20261019
SCALE = np.array([0.25, 0.5, 1.0, 2.0, 4.0])


def test_posteriors_are_normalized_and_finite_at_d1000_n50_m441():
    rng = np.random.default_rng(SEED)
    d, n, side = 1000, 50, 21
    rows, cols = np.divmod(np.arange(side * side), side)
    mus = np.zeros((side * side, d))
    mus[:, 0] = rows - 10.0
    mus[:, 1] = cols - 10.0
    x = mus[100] + rng.standard_normal((n, d))
    q = quadratic_forms(x, mus, np.ones(side * side))
    prior = uniform_log_prior(side * side)
    for name in ("exact", "typ_normal", "typ_density", "bins_abs3", "clipped"):
        log_post = cumulative_log_posterior(score_matrix(name, q, np.ones(side * side), d), prior)
        assert log_post.shape == (n, side * side)
        assert np.all(np.isfinite(log_post)), name
        np.testing.assert_allclose(np.exp(log_post).sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)


def test_ties_count_as_incorrect():
    prior = uniform_log_prior(3)
    log_post = np.log(
        np.array(
            [
                [0.5, 0.25, 0.25],
                [0.4, 0.4, 0.2],
                [0.2, 0.7, 0.1],
                [0.4 + 5e-13, 0.4 - 5e-13, 0.2],
            ]
        )
    )
    assert metrics(log_post, 0, prior)["acc"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert posterior_mode(log_post).tolist() == [0, -1, 1, -1]


def test_coverage80_on_hand_built_posteriors():
    prior = uniform_log_prior(4)
    ranked = np.log(np.array([[0.1, 0.5, 0.3, 0.1]]))
    assert metrics(ranked, 1, prior)["coverage80"].tolist() == [1.0]
    assert metrics(ranked, 2, prior)["coverage80"].tolist() == [1.0]
    assert metrics(ranked, 0, prior)["coverage80"].tolist() == [0.0]
    tied = np.log(np.array([[0.2, 0.2, 0.6]]))
    assert metrics(tied, 0, uniform_log_prior(3))["coverage80"].tolist() == [0.5]
    assert metrics(tied, 2, uniform_log_prior(3))["coverage80"].tolist() == [1.0]
    flat = np.log(np.full((1, 5), 0.2))
    assert metrics(flat, 4, uniform_log_prior(5))["coverage80"][0] == pytest.approx(0.8)


def test_bits_kl_and_localization_metrics():
    prior = uniform_log_prior(4)
    log_post = np.log(np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]]))
    exact = np.log(np.array([[0.4, 0.2, 0.2, 0.2], [0.7, 0.1, 0.1, 0.1]]))
    locations = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    result = metrics(log_post, 0, prior, exact_log_post=exact, locations=locations)
    assert result["bits"].tolist() == pytest.approx([0.0, math.log2(0.7 / 0.25)])
    expected_kl = 0.4 * math.log(0.4 / 0.25) + 3 * 0.2 * math.log(0.2 / 0.25)
    assert result["kl_to_exact"].tolist() == pytest.approx([expected_kl, 0.0], abs=1e-15)
    assert result["loc_error"][0] == pytest.approx(math.hypot(0.5, 0.5))
    assert result["post_entropy"][0] == pytest.approx(2.0)


def test_metrics_accept_one_truth_per_trial():
    rng = np.random.default_rng(SEED + 1)
    scores = rng.normal(size=(6, 4, 5))
    log_post = cumulative_log_posterior(scores, uniform_log_prior(5))
    truth = np.array([0, 1, 2, 3, 4, 0])
    batched = metrics(log_post, truth, uniform_log_prior(5))
    for trial in range(6):
        single = metrics(log_post[trial], int(truth[trial]), uniform_log_prior(5))
        for key, values in single.items():
            np.testing.assert_array_equal(batched[key][trial], values)


def test_typ_normal_closed_form_limit_is_broader_than_the_truth():
    for d, expected in ((1, 4.0), (2, 2.0), (10, 1.0)):
        assert SCALE[unique_argmax(scale_objective("typ_normal", 1.0, SCALE, d))] == expected
    for d in (1, 2, 5, 10, 100, 1000):
        for truth in SCALE:
            assert SCALE[unique_argmax(scale_objective("exact", float(truth), SCALE, d))] == truth


def test_summary_helpers():
    mean, se = mean_and_se(np.array([[1.0, 0.0], [3.0, 0.0]]))
    assert mean.tolist() == [2.0, 0.0]
    assert se.tolist() == [1.0, 0.0]
    assert mean_and_se(np.array([[1.0]]))[1] is None
    assert first_reaching(np.array([0.2, 0.89, 0.9, 0.95])) == 3
    assert first_reaching(np.array([0.2, 0.5])) is None
    assert unique_argmax(np.array([1.0, 3.0, 3.0])) == -1
