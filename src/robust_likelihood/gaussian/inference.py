"""Cumulative posteriors and the metrics of the Gaussian test.

Scores have shape (..., n, M): observations on the second-to-last axis and
hypotheses on the last. The posterior after n observations is

    log pi_n(z) = log pi_0(z) + sum_{i <= n} l_z(x_i) - logsumexp_z'(.).

A tie is two posterior probabilities within 1e-12, the tolerance of
robust_likelihood.typicality.map_label. A tied top posterior is not a unique argmax.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

TIE_TOLERANCE = 1e-12
COVERAGE_MASS = 0.8
_LOG2 = math.log(2.0)


def uniform_log_prior(m: int) -> np.ndarray:
    return np.full(m, -math.log(m))


def cumulative_log_posterior(scores: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Normalized log pi_n(z) after n = 1, ..., n_max observations. scores: (..., n, M); log_prior: (M,).

    Each observation's scores are first shifted by their maximum over hypotheses.
    The shift does not depend on z and cancels in the posterior. It keeps the running
    sums near zero, so terms shared by every hypothesis, such as -(d/2) log(2 pi) at
    d = 1000, do not cost precision in the normalization.
    """
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    total = np.cumsum(shifted, axis=-2) + np.asarray(log_prior, dtype=np.float64)
    return total - special.logsumexp(total, axis=-1, keepdims=True)


def metrics(
    log_post: np.ndarray,
    true_index: int | np.ndarray,
    log_prior: np.ndarray,
    *,
    exact_log_post: np.ndarray | None = None,
    locations: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-trial metrics after each number of observations, each of shape log_post.shape[:-1].

    p_true        pi_n(z*).
    acc           1 when z* is the unique argmax of pi_n. A tie counts as incorrect.
    bits          log2 pi_n(z*) - log2 pi_0(z*).
    coverage80    Whether the smallest set of hypotheses with mass >= 0.8 contains z*.
                  Hypotheses tied with z* are ordered uniformly at random, and the
                  value is the probability that z* is in the set.
    kl_to_exact   KL(pi_n^exact || pi_n) in nats, when exact_log_post is given.
    loc_error     With locations (M, 2) in grid units: distance from the posterior
                  mean location to the location of z*.
    post_entropy  With locations: entropy of pi_n in bits.

    true_index is one index or one per trial, of shape log_post.shape[:-2].
    """
    log_post = np.asarray(log_post, dtype=np.float64)
    log_prior = np.asarray(log_prior, dtype=np.float64)
    lead = log_post.shape[:-2]
    n, m = log_post.shape[-2:]
    truth = np.broadcast_to(np.asarray(true_index, dtype=np.int64), lead)
    is_truth = np.arange(m) == truth[..., None, None]
    log_true = np.take_along_axis(log_post, np.broadcast_to(truth[..., None, None], lead + (n, 1)), axis=-1)[..., 0]
    p = np.exp(log_post)
    p_true = np.exp(log_true)
    best_other = np.where(is_truth, -np.inf, p).max(axis=-1)

    above = np.where(p > p_true[..., None] + TIE_TOLERANCE, p, 0.0).sum(axis=-1)
    tied_others = (np.abs(p - p_true[..., None]) <= TIE_TOLERANCE).sum(axis=-1) - 1
    room = COVERAGE_MASS - TIE_TOLERANCE - above
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        positions = np.where(room > 0.0, np.ceil(room / p_true), 0.0)
    coverage = np.minimum(positions, tied_others + 1) / (tied_others + 1)

    result = {
        "p_true": p_true,
        "acc": (p_true - best_other > TIE_TOLERANCE).astype(np.float64),
        "bits": (log_true - log_prior[truth][..., None]) / _LOG2,
        "coverage80": coverage,
    }
    if exact_log_post is not None:
        exact_log_post = np.asarray(exact_log_post, dtype=np.float64)
        exact_p = np.exp(exact_log_post)
        with np.errstate(invalid="ignore"):
            terms = np.where(exact_p > 0.0, exact_p * (exact_log_post - log_post), 0.0)
        result["kl_to_exact"] = terms.sum(axis=-1)
    if locations is not None:
        locations = np.asarray(locations, dtype=np.float64)
        mean_location = p @ locations
        result["loc_error"] = np.linalg.norm(mean_location - locations[truth][..., None, :], axis=-1)
        with np.errstate(invalid="ignore"):
            result["post_entropy"] = -np.where(p > 0.0, p * log_post, 0.0).sum(axis=-1) / _LOG2
    return result


def mean_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean over the first (trial) axis and its standard error, std(ddof=1) / sqrt(trials). No SE for one trial."""
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, None
    return mean, values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def first_reaching(curve: np.ndarray, threshold: float = 0.9) -> int | None:
    """Smallest n (counting from 1) with curve[n - 1] >= threshold, or None if it is never reached."""
    hits = np.flatnonzero(np.asarray(curve, dtype=np.float64) >= threshold)
    return int(hits[0]) + 1 if hits.size else None


def posterior_mode(log_post: np.ndarray) -> np.ndarray:
    """Index of the unique argmax over the last axis, or -1 when the two largest probabilities tie."""
    p = np.exp(np.asarray(log_post, dtype=np.float64))
    top_two = np.sort(p, axis=-1)[..., -2:]
    tie = top_two[..., 1] - top_two[..., 0] <= TIE_TOLERANCE
    return np.where(tie, -1, np.argmax(p, axis=-1))


def scale_objective(name: str, sigma2_true: float, sigma2: np.ndarray, d: int) -> np.ndarray:
    """Closed-form E_{x ~ N(0, sigma2_true I_d)}[l_z(x)] for each hypothesis N(0, sigma2_z I_d).

    With R = ||x||^2, E[R] = d sigma2_true and E[R^2] = sigma2_true^2 (d^2 + 2d).
    exact:      -(d/2) log(2 pi s) - E[R] / (2 s).
    typ_normal: -(E[R^2] / s^2 - 2 d E[R] / s + d^2) / (4 d), since T_s = (R/s - d) / sqrt(2d).
    """
    s = np.asarray(sigma2, dtype=np.float64)
    first = d * sigma2_true
    second = sigma2_true**2 * (d * d + 2.0 * d)
    if name == "exact":
        return -0.5 * d * np.log(2.0 * math.pi * s) - first / (2.0 * s)
    if name == "typ_normal":
        return -(second / s**2 - 2.0 * d * first / s + d * d) / (4.0 * d)
    raise KeyError(f"no closed-form scale objective for {name!r}")


def unique_argmax(values: np.ndarray, tolerance: float = TIE_TOLERANCE) -> int:
    """Index of the largest value, or -1 if another value is within tolerance of it."""
    values = np.asarray(values, dtype=np.float64)
    best = int(np.argmax(values))
    if np.sum(values >= values[best] - tolerance) > 1:
        return -1
    return best
