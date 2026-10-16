import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy import stats

from robust_likelihood.gaussian.families import (
    GRID_MARGIN,
    GRID_SIDE,
    grid_locations,
    localization_means,
    scale_sigma2,
    separation_threshold,
)
from robust_likelihood.gaussian.inference import cumulative_log_posterior, uniform_log_prior
from robust_likelihood.gaussian.model import quadratic_forms
from robust_likelihood.gaussian.sampling import (
    CLEAN,
    Contamination,
    TrialStreams,
    localization_trial,
    mean_shift_trial,
    scale_trial,
    trial_rng,
)
from robust_likelihood.gaussian.scorers import score_matrix

SEED = 20261019
ALL_CONTAMINATIONS = (
    CLEAN,
    Contamination("C1"),
    Contamination("C2"),
    Contamination("C3", 0.0, 50.0),
    Contamination("C3", 0.3, 50.0),
    Contamination("C4", 0.3, 50.0),
    Contamination("C5"),
)


def _trial(trial: int, d: int = 30, n: int = 40):
    return mean_shift_trial(SEED, "e4", f"d={d}|r=1.5", trial, d, 1.5 * separation_threshold(d), n, ALL_CONTAMINATIONS)


def test_same_seed_gives_identical_draws_and_trial_order_does_not_matter():
    forward = [_trial(trial) for trial in range(5)]
    backward = [_trial(trial) for trial in reversed(range(5))][::-1]
    for (family_a, x_a), (family_b, x_b) in zip(forward, backward):
        np.testing.assert_array_equal(family_a.mus, family_b.mus)
        assert x_a.keys() == x_b.keys()
        for key in x_a:
            np.testing.assert_array_equal(x_a[key], x_b[key])
    assert not np.array_equal(forward[0][1]["C0"], forward[1][1]["C0"])
    other_experiment = mean_shift_trial(SEED, "e1", "d=30|r=1.5", 0, 30, 1.0, 40)[1]["C0"]
    assert not np.array_equal(forward[0][1]["C0"], other_experiment)
    assert trial_rng(SEED, "e1", "cell", 3).random() == trial_rng(SEED, "e1", "cell", 3).random()


def test_mean_shift_family_geometry():
    d = 50
    delta = 1.25 * separation_threshold(d)
    family, _ = mean_shift_trial(SEED, "e1", "geometry", 0, d, delta, 2)
    assert family.mus.shape == (20, d) and family.true_index == 0
    np.testing.assert_array_equal(family.mus[0], np.zeros(d))
    np.testing.assert_allclose(np.linalg.norm(family.mus[1:], axis=1), delta, rtol=1e-12)
    np.testing.assert_array_equal(family.sigma2, np.ones(20))
    assert separation_threshold(100) == pytest.approx(200**0.25)


def test_contaminations_perturb_the_same_clean_draws():
    d, n, radius = 30, 400, 50.0
    family, x = mean_shift_trial(
        SEED,
        "e4",
        "perturbations",
        0,
        d,
        3.0,
        n,
        (CLEAN, Contamination("C1"), Contamination("C3", 0.0, radius), Contamination("C3", 0.3, radius),
         Contamination("C4", 0.3, radius)),
    )
    clean = x["C0"]
    np.testing.assert_array_equal(x["C3:rho=0:R=50"], clean)
    np.testing.assert_allclose(x["C1"], math.sqrt(1.1) * clean, rtol=1e-15)
    toward = family.mus[1] / np.linalg.norm(family.mus[1])
    c3_shift = x["C3:rho=0.3:R=50"] - clean
    c3_hit = np.linalg.norm(c3_shift, axis=1) > 0
    np.testing.assert_allclose(c3_shift[c3_hit], np.broadcast_to(radius * toward, (int(c3_hit.sum()), d)), atol=1e-12)
    assert 0.2 < c3_hit.mean() < 0.4
    c4_shift = x["C4:rho=0.3:R=50"] - clean
    c4_hit = np.linalg.norm(c4_shift, axis=1) > 0
    np.testing.assert_array_equal(c4_hit, c3_hit)
    np.testing.assert_allclose(np.linalg.norm(c4_shift[c4_hit], axis=1), radius, rtol=1e-12)
    assert np.max(np.abs(c4_shift[c4_hit] @ toward)) < radius


def test_heavy_tail_and_sparse_contaminations_have_the_stated_moments():
    d, n = 10, 20_000
    _, x = mean_shift_trial(SEED, "e4", "moments", 0, d, 3.0, n, (Contamination("C2"), Contamination("C5")))
    heavy = x["C2"].ravel()
    assert heavy.var() == pytest.approx(1.0, abs=0.05)
    assert stats.kurtosis(heavy, fisher=False) > 5.0
    sparse_shift = x["C5"] - mean_shift_trial(SEED, "e4", "moments", 0, d, 3.0, n)[1]["C0"]
    assert np.mean(sparse_shift != 0.0) == pytest.approx(0.01, abs=0.002)
    assert sparse_shift[sparse_shift != 0.0].std() == pytest.approx(10.0, rel=0.1)


def test_streams_are_drawn_once_and_keep_their_shape():
    streams = TrialStreams(SEED, "e4", "cell", 0)
    first = streams.normal("noise", (3, 4))
    assert streams.normal("noise", (3, 4)) is first
    with pytest.raises(ValueError):
        streams.normal("noise", (2, 4))
    with pytest.raises(ValueError):
        first[0, 0] = 1.0


def test_localization_truth_is_off_the_border_and_in_plane_data_do_not_depend_on_d():
    rows = []
    for trial in range(200):
        true_index, x2 = localization_trial(SEED, "e3", trial, 1.0, 2, 20)
        rows.append(grid_locations()[true_index])
        _, x100 = localization_trial(SEED, "e3", trial, 1.0, 100, 20)
        np.testing.assert_array_equal(x100[:, :2], x2)
        assert x100.shape == (20, 100)
    rows = np.array(rows)
    assert rows.min() == GRID_MARGIN and rows.max() == GRID_SIDE - 1 - GRID_MARGIN
    assert len({tuple(row) for row in rows}) > 100


def test_exact_posterior_of_one_trial_is_identical_at_d2_and_d100():
    posteriors = {}
    for d in (2, 100):
        true_index, x = localization_trial(SEED, "e3", 7, 1.0, d, 20)
        mus = localization_means(1.0, d)
        q = quadratic_forms(x, mus, np.ones(mus.shape[0]))
        scores = score_matrix("exact", q, np.ones(mus.shape[0]), d)
        posteriors[d] = np.exp(cumulative_log_posterior(scores, uniform_log_prior(mus.shape[0])))
    assert np.max(np.abs(posteriors[2] - posteriors[100])) < 1e-9


def test_scale_scores_from_sampled_r_match_full_simulation():
    d, draws = 10, 10_000
    sigma2 = scale_sigma2()
    for trial in range(5):
        true_index, r_direct = scale_trial(SEED, "e2", d, trial, draws)
        rng = trial_rng(SEED, "e2-shortcut-test", f"d={d}", trial)
        x = math.sqrt(sigma2[true_index]) * rng.standard_normal((draws, d))
        r_full = quadratic_forms(x, np.zeros((1, d)), np.ones(1))[:, 0]
        assert stats.ks_2samp(r_direct, r_full).pvalue > 0.001
        for name in ("exact", "typ_normal", "typ_density"):
            direct = score_matrix(name, r_direct[:, None] / sigma2, sigma2, d)
            full = score_matrix(name, quadratic_forms(x, np.zeros((5, d)), sigma2), sigma2, d)
            for z in range(5):
                assert stats.ks_2samp(direct[:, z], full[:, z]).pvalue > 0.001, (trial, name, z)
    assert [scale_trial(SEED, "e2", d, trial, 1)[0] for trial in range(10)] == [0, 1, 2, 3, 4] * 2
