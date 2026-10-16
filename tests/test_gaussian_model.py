import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy import stats

from robust_likelihood.gaussian.model import (
    entropy,
    log_likelihood,
    quadratic_forms,
    surprisal,
    surprisal_variance,
    typicality,
    typicality_central_moments,
    typicality_lower_bound,
)
from robust_likelihood.typicality import GaussianHypothesis, signed_deviation

SEED = 20261019


def test_log_likelihood_matches_scipy_isotropic_normal():
    rng = np.random.default_rng(SEED)
    for d in (1, 3, 10):
        mus = rng.normal(size=(4, d))
        sigma2 = np.array([0.25, 1.0, 2.5, 4.0])
        x = 2.0 * rng.normal(size=(6, d))
        ours = log_likelihood(quadratic_forms(x, mus, sigma2), sigma2, d)
        for z in range(4):
            reference = stats.multivariate_normal(mean=mus[z], cov=sigma2[z] * np.eye(d)).logpdf(x)
            np.testing.assert_allclose(ours[:, z], reference, rtol=1e-12, atol=1e-10)


def test_quadratic_forms_match_a_direct_loop():
    rng = np.random.default_rng(SEED + 1)
    x = rng.normal(size=(7, 5))
    mus = 3.0 * rng.normal(size=(6, 5))
    sigma2 = rng.uniform(0.2, 5.0, size=6)
    q = quadratic_forms(x, mus, sigma2)
    assert q.shape == (7, 6)
    for i in range(7):
        for z in range(6):
            direct = float(np.sum((x[i] - mus[z]) ** 2)) / sigma2[z]
            assert q[i, z] == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_surprisal_mean_and_variance_match_entropy_and_variance():
    rng = np.random.default_rng(SEED + 2)
    draws = 100_000
    for d, sigma2 in ((1, 0.25), (3, 1.0), (10, 4.0)):
        mu = rng.normal(size=(1, d))
        variance = np.array([sigma2])
        x = mu + math.sqrt(sigma2) * rng.standard_normal((draws, d))
        values = surprisal(quadratic_forms(x, mu, variance), variance, d)[:, 0]
        h = float(entropy(variance, d)[0])
        v = surprisal_variance(d)
        assert abs(values.mean() - h) <= 5.0 * math.sqrt(v / draws)
        fourth = v**2 * typicality_central_moments(d)[4]
        assert abs(values.var(ddof=1) - v) <= 5.0 * math.sqrt((fourth - v**2) / draws)


def test_entropy_and_variance_closed_forms():
    assert float(entropy(np.array([1.0]), 1)[0]) == pytest.approx(0.5 * math.log(2.0 * math.pi * math.e))
    assert float(entropy(np.array([4.0]), 3)[0]) == pytest.approx(1.5 * math.log(2.0 * math.pi * math.e * 4.0))
    assert surprisal_variance(10) == 5.0
    moments = typicality_central_moments(10)
    assert moments[2] == pytest.approx(1.0)
    assert moments[3] == pytest.approx(math.sqrt(8.0 / 10.0))
    assert moments[4] == pytest.approx(3.0 + 12.0 / 10.0)


def test_d1_typicality_is_minus_root_two_times_signed_deviation():
    for hypothesis in (GaussianHypothesis("narrow", 0.0, 1.0), GaussianHypothesis("wide", 1.5, 2.0)):
        for x in (-3.0, -0.4, 0.0, 0.7, 1.5, 2.5, 10.0):
            q = quadratic_forms(np.array([[x]]), np.array([[hypothesis.mean]]), np.array([hypothesis.sd**2]))
            t = float(typicality(q, 1)[0, 0])
            assert t == pytest.approx(-math.sqrt(2.0) * signed_deviation(x, hypothesis), abs=1e-12)


def test_typicality_is_never_below_minus_root_half_d():
    rng = np.random.default_rng(SEED + 3)
    for d in (1, 2, 5, 100):
        mus = rng.normal(size=(4, d))
        sigma2 = rng.uniform(0.25, 4.0, size=4)
        x = np.vstack([mus, rng.normal(size=(200, d)), mus[0] + 1e-9])
        t = typicality(quadratic_forms(x, mus, sigma2), d)
        assert typicality_lower_bound(d) == pytest.approx(-math.sqrt(d / 2.0), abs=1e-12)
        assert t.min() >= typicality_lower_bound(d)
        assert t[0, 0] == pytest.approx(typicality_lower_bound(d), abs=1e-12)
