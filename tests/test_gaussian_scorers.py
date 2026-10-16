import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy import stats

from robust_likelihood.gaussian.bins import NAMED_SCHEMES, BinScheme, log_masses
from robust_likelihood.gaussian.inference import cumulative_log_posterior, uniform_log_prior
from robust_likelihood.gaussian.model import quadratic_forms
from robust_likelihood.gaussian.scorers import (
    chi2_log_density,
    clipped_cap,
    content_oracle_base,
    score_matrix,
)

SEED = 20261019


def _mean_shift(rng, d: int, delta: float, m: int = 20, n: int = 50):
    directions = rng.standard_normal((m - 1, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    mus = np.vstack([np.zeros(d), delta * directions])
    x = rng.standard_normal((n, d))
    return x, mus, np.ones(m)


def _posterior(name, q, sigma2, d, **kwargs):
    return np.exp(cumulative_log_posterior(score_matrix(name, q, sigma2, d, **kwargs), uniform_log_prior(q.shape[-1])))


def test_typ_density_and_exact_give_the_same_posterior_at_d2_with_equal_variances():
    rng = np.random.default_rng(SEED)
    x, mus, sigma2 = _mean_shift(rng, 2, 1.5)
    sigma2 = np.full(mus.shape[0], 2.5)
    q = quadratic_forms(x, mus, sigma2)
    np.testing.assert_allclose(_posterior("typ_density", q, sigma2, 2), _posterior("exact", q, sigma2, 2), atol=1e-10)


def test_chi2_log_density_matches_scipy():
    q = np.array([0.01, 0.5, 1.0, 3.0, 40.0, 950.0, 1100.0])
    for d in (1, 2, 5, 1000):
        np.testing.assert_allclose(chi2_log_density(q, d), stats.chi2.logpdf(q, d), rtol=1e-12, atol=1e-12)


def test_equal_mass_posterior_stays_at_the_prior_after_50_observations():
    rng = np.random.default_rng(SEED + 1)
    for d in (1, 2, 10, 100):
        x, mus, sigma2 = _mean_shift(rng, d, 3.0)
        q = quadratic_forms(x, mus, sigma2)
        for k in (2, 3, 4):
            posterior = _posterior(f"bins_equalmass{k}", q, sigma2, d)
            assert np.max(np.abs(posterior - 1.0 / mus.shape[0])) < 1e-12, (d, k)


def test_clipped_evidence_between_two_hypotheses_never_exceeds_the_cap():
    rng = np.random.default_rng(SEED + 2)
    for d in (10, 1000):
        x, mus, sigma2 = _mean_shift(rng, d, 8.0)
        x[::5] += 50.0 * mus[1] / np.linalg.norm(mus[1])
        scores = score_matrix("clipped", quadratic_forms(x, mus, sigma2), sigma2, d)
        cap = clipped_cap(d)
        assert np.all(scores <= 0.0) and np.all(scores >= -cap)
        assert np.max(scores.max(axis=1) - scores.min(axis=1)) <= cap + 1e-12
        masses = log_masses(NAMED_SCHEMES["abs3"], d)
        assert cap == pytest.approx(float(masses.max() - masses.min()))


def test_far_observation_leaves_the_binned_posterior_unchanged():
    rng = np.random.default_rng(SEED + 3)
    d = 100
    x, mus, sigma2 = _mean_shift(rng, d, 5.0, n=10)
    direction = rng.standard_normal(d)
    far = 1e4 * direction / np.linalg.norm(direction)
    assert np.all(np.linalg.norm(mus - far, axis=1) > 0.99e4)
    before = _posterior("bins_abs3", quadratic_forms(x, mus, sigma2), sigma2, d)
    after = _posterior("bins_abs3", quadratic_forms(np.vstack([x, far]), mus, sigma2), sigma2, d)
    np.testing.assert_allclose(after[-1], before[-1], rtol=0.0, atol=1e-12)
    far_scores = score_matrix("bins_abs3", quadratic_forms(far[None, :], mus, sigma2), sigma2, d)
    assert np.all(far_scores == log_masses(NAMED_SCHEMES["abs3"], d)[-1])


def test_tempered_one_equals_exact_and_tempering_scales_the_log_likelihood():
    rng = np.random.default_rng(SEED + 4)
    x, mus, _ = _mean_shift(rng, 10, 2.0, n=5)
    sigma2 = rng.uniform(0.5, 2.0, size=mus.shape[0])
    q = quadratic_forms(x, mus, sigma2)
    exact = score_matrix("exact", q, sigma2, 10)
    np.testing.assert_array_equal(score_matrix("tempered_1.0", q, sigma2, 10), exact)
    np.testing.assert_allclose(score_matrix("tempered_0.1", q, sigma2, 10), 0.1 * exact, rtol=1e-15)


def test_typ_normal_is_minus_half_t_squared_and_bins_use_supplied_schemes():
    q = np.array([[0.0, 10.0, 20.0, 30.0]])
    d = 10
    t = (q - d) / math.sqrt(2.0 * d)
    np.testing.assert_allclose(score_matrix("typ_normal", q, np.ones(4), d), -0.5 * t**2)
    custom = BinScheme("opt2", "abs", (0.5,))
    scores = score_matrix("bins_opt2", q, np.ones(4), d, schemes={"opt2": custom})
    inner, outer = log_masses(custom, d)
    assert scores.tolist() == [[outer, inner, outer, outer]]


def test_scorer_names_are_validated():
    q = np.ones((2, 3))
    with pytest.raises(KeyError):
        score_matrix("unknown", q, np.ones(3), 4)
    with pytest.raises(ValueError):
        score_matrix("typ_content_oracle", q, np.ones(3), 4)
    with pytest.raises(ValueError):
        score_matrix("tempered_0", q, np.ones(3), 4)
    assert content_oracle_base("bins_abs3_content_oracle") == "bins_abs3"
    assert content_oracle_base("typ_content_oracle") == "typ_normal"
    assert content_oracle_base("exact") is None
