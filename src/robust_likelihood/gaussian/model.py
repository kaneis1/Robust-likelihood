"""Closed forms for isotropic Gaussian hypotheses, in nats.

Hypothesis z fixes x | z ~ N(mu_z, sigma2_z I_d). The scaled squared distance
q_z(x) = ||x - mu_z||^2 / sigma2_z is chi-square with d degrees of freedom when x is
drawn from z. The standardized typicality T = (q - d) / sqrt(2 d) is therefore
pivotal: it has the same distribution under every hypothesis of dimension d, with
mean 0, variance 1, skewness sqrt(8 / d), and T >= -sqrt(d / 2).

Positive T means x is less probable than a typical draw. The 1-D module
robust_likelihood.typicality reports log p - E[log p], which has the opposite sign:
for d = 1, T = -sqrt(2) * signed_deviation.
"""

from __future__ import annotations

import math

import numpy as np

LOG_TWO_PI = math.log(2.0 * math.pi)


def quadratic_forms(x: np.ndarray, mus: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """q[i, z] = ||x_i - mu_z||^2 / sigma2_z. x: (n, d); mus: (M, d); sigma2: (M,). Returns (n, M).

    Computed as ||x||^2 - 2 x . mu + ||mu||^2 with one matrix product in float64.
    Rounding can leave an exact zero slightly negative, so the squared distance is
    clipped at 0 before it is scaled.
    """
    x = np.asarray(x, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if x.ndim != 2 or mus.ndim != 2 or x.shape[1] != mus.shape[1]:
        raise ValueError("x must be (n, d) and mus (M, d) with the same d")
    if sigma2.shape != (mus.shape[0],) or not np.all(sigma2 > 0.0):
        raise ValueError("sigma2 must have shape (M,) and be positive")
    x_norm = np.einsum("ij,ij->i", x, x)
    mu_norm = np.einsum("ij,ij->i", mus, mus)
    squared = x_norm[:, None] - 2.0 * (x @ mus.T) + mu_norm[None, :]
    return np.maximum(squared, 0.0) / sigma2


def log_likelihood(q: np.ndarray, sigma2: np.ndarray, d: int) -> np.ndarray:
    """log p(x | z) = -(d/2) log(2 pi sigma2_z) - q_z(x) / 2. The last axis of q indexes z."""
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    return -0.5 * d * (LOG_TWO_PI + np.log(sigma2)) - 0.5 * np.asarray(q, dtype=np.float64)


def surprisal(q: np.ndarray, sigma2: np.ndarray, d: int) -> np.ndarray:
    """S(x, z) = -log p(x | z), in nats."""
    return -log_likelihood(q, sigma2, d)


def entropy(sigma2: np.ndarray, d: int) -> np.ndarray:
    """H(z) = E_{x~z}[S(x, z)] = (d/2) log(2 pi e sigma2_z), in nats."""
    return 0.5 * d * (LOG_TWO_PI + 1.0 + np.log(np.asarray(sigma2, dtype=np.float64)))


def surprisal_variance(d: int) -> float:
    """V(z) = Var_{x~z}[S(x, z)] = d / 2 nats^2, the same for every sigma2."""
    return 0.5 * d


def typicality(q: np.ndarray, d: int) -> np.ndarray:
    """T(x, z) = (S - H) / sqrt(V) = (q_z(x) - d) / sqrt(2 d). Positive T: less probable than typical."""
    return (np.asarray(q, dtype=np.float64) - d) / math.sqrt(2.0 * d)


def typicality_lower_bound(d: int) -> float:
    """T at q = 0, the smallest attainable value: -d / sqrt(2 d) = -sqrt(d / 2)."""
    return float(typicality(0.0, d))


def typicality_central_moments(d: int) -> dict[int, float]:
    """Central moments 2 to 6 of T under the truth.

    q ~ chi-square(d) has cumulants 2^(k-1) (k-1)! d, so T has cumulants
    kappa_k = 2^(k-1) (k-1)! d / (2 d)^(k/2): kappa_2 = 1, kappa_3 = sqrt(8 / d),
    kappa_4 = 12 / d. The central moments follow from the cumulants.
    """
    kappa = {k: 2.0 ** (k - 1) * math.factorial(k - 1) * d / (2.0 * d) ** (k / 2.0) for k in range(2, 7)}
    return {
        2: kappa[2],
        3: kappa[3],
        4: kappa[4] + 3.0 * kappa[2] ** 2,
        5: kappa[5] + 10.0 * kappa[3] * kappa[2],
        6: kappa[6] + 15.0 * kappa[4] * kappa[2] + 10.0 * kappa[3] ** 2 + 15.0 * kappa[2] ** 3,
    }
