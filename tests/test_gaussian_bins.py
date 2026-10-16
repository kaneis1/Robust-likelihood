import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy import stats

from robust_likelihood.gaussian.bins import (
    NAMED_SCHEMES,
    BinScheme,
    EmptyBinError,
    bin_index,
    equal_mass_scheme,
    full_log_masses,
    log_masses,
    q_intervals,
    retained_bins,
    scheme_by_name,
)
from robust_likelihood.gaussian.model import quadratic_forms, typicality
from robust_likelihood.typicality import normal_cdf

SEED = 20261019
SCHEME_NAMES = ("abs2", "abs3", "signed5", "equalmass2", "equalmass3", "equalmass4")


def test_masses_of_every_scheme_sum_to_one():
    for d in (1, 2, 10, 1000):
        for name in SCHEME_NAMES:
            total = float(np.exp(log_masses(scheme_by_name(name, d), d)).sum())
            assert abs(total - 1.0) <= 1e-12, (name, d, total)


def test_exact_masses_match_monte_carlo_frequencies_of_t():
    rng = np.random.default_rng(SEED)
    draws = 100_000
    for d in (1, 2, 10, 100):
        mu = rng.normal(size=(1, d))
        sigma2 = np.array([rng.uniform(0.25, 4.0)])
        x = mu + math.sqrt(sigma2[0]) * rng.standard_normal((draws, d))
        t = typicality(quadratic_forms(x, mu, sigma2), d)[:, 0]
        for name in SCHEME_NAMES:
            scheme = scheme_by_name(name, d)
            masses = np.exp(log_masses(scheme, d))
            counts = np.bincount(bin_index(scheme, t, d), minlength=masses.size)
            p_value = stats.chisquare(counts, draws * masses).pvalue
            assert p_value > 0.001, (name, d, p_value)


def _d1_mass_from_normal_cdf(scheme: BinScheme, k: int, mean: float, sd: float) -> float:
    """At d = 1, q = ((x - mean) / sd)^2, so a q interval is two x intervals of the normal itself."""
    total = 0.0
    for start, stop in q_intervals(scheme, k, 1):
        inner = math.sqrt(start)
        outer = math.sqrt(stop) if math.isfinite(stop) else math.inf
        upper = 1.0 if math.isinf(outer) else normal_cdf(mean + outer * sd, mean, sd)
        total += 2.0 * (upper - normal_cdf(mean + inner * sd, mean, sd))
    return total


def test_masses_are_identical_for_every_mean_and_variance_at_the_same_d():
    for name in ("abs2", "abs3", "signed5", "equalmass3"):
        scheme = scheme_by_name(name, 1)
        exact = np.exp(full_log_masses(scheme, 1))
        for mean, sd in ((0.0, 1.0), (-3.0, 0.5), (7.5, 2.0)):
            from_normal = [_d1_mass_from_normal_cdf(scheme, k, mean, sd) for k in range(scheme.bin_count)]
            np.testing.assert_allclose(from_normal, exact, rtol=0.0, atol=1e-12)

    rng = np.random.default_rng(SEED + 1)
    d = 10
    eps = rng.standard_normal((20_000, d))
    counts = []
    for sigma2 in (0.25, 1.0, 4.0):
        mu = 5.0 * rng.normal(size=(1, d))
        t = typicality(quadratic_forms(mu + math.sqrt(sigma2) * eps, mu, np.array([sigma2])), d)[:, 0]
        counts.append(np.bincount(bin_index(NAMED_SCHEMES["signed5"], t, d), minlength=5))
    assert all(np.array_equal(counts[0], other) for other in counts[1:])


def test_signed5_drops_empty_bins_at_d1_and_never_indexes_them():
    scheme = NAMED_SCHEMES["signed5"]
    assert list(retained_bins(scheme, 1)) == [2, 3, 4]
    assert np.all(np.isneginf(full_log_masses(scheme, 1)[:2]))
    assert log_masses(scheme, 1).shape == (3,)
    rng = np.random.default_rng(SEED + 2)
    t = typicality(rng.chisquare(1, size=50_000), 1)
    t = np.concatenate([t, [-math.sqrt(0.5), -0.5, 0.999, 1.0, 2.0, 50.0]])
    index = bin_index(scheme, t, 1)
    assert index.min() >= 0 and index.max() <= 2
    assert bin_index(scheme, np.array([-math.sqrt(0.5)]), 1)[0] == 0
    with pytest.raises(EmptyBinError):
        bin_index(scheme, np.array([-1.5]), 1)
    assert list(retained_bins(scheme, 2)) == [2, 3, 4]
    assert list(retained_bins(scheme, 10)) == [0, 1, 2, 3, 4]


def test_absolute_scheme_gives_reflected_values_the_same_bin():
    t = np.linspace(-6.0, 6.0, 241)
    for name in ("abs2", "abs3", "equalmass4"):
        scheme = scheme_by_name(name, 1000)
        assert np.array_equal(bin_index(scheme, t, 1000), bin_index(scheme, -t, 1000))
    assert bin_index(NAMED_SCHEMES["abs3"], np.array([0.0, 0.999, 1.0, 1.999, 2.0, 40.0]), 100).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_equal_mass_bins_each_carry_one_kth_of_the_mass():
    for d in (1, 2, 10, 1000):
        for k in (2, 3, 4):
            scheme = equal_mass_scheme(k, d)
            assert scheme.bin_count == k
            np.testing.assert_allclose(log_masses(scheme, d), -math.log(k), rtol=0.0, atol=1e-13)


def test_bin_masses_use_the_chi_square_distribution_not_the_normal_approximation():
    inner = float(np.exp(log_masses(NAMED_SCHEMES["abs3"], 1))[0])
    assert inner == pytest.approx(stats.chi2.cdf(1.0 + math.sqrt(2.0), 1), abs=1e-14)
    assert abs(inner - (2.0 * stats.norm.cdf(1.0) - 1.0)) > 0.1


def test_scheme_validation():
    with pytest.raises(ValueError):
        BinScheme("bad", "abs", (2.0, 1.0))
    with pytest.raises(ValueError):
        BinScheme("bad", "abs", (0.0, 1.0))
    with pytest.raises(ValueError):
        BinScheme("bad", "signed", ())
    with pytest.raises(KeyError):
        scheme_by_name("unknown", 10)
