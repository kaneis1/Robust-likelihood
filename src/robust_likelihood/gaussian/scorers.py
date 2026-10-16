"""Per-observation log-scores l_z(x) for every scorer of the Gaussian test.

Every scorer takes q[..., z] = ||x - mu_z||^2 / sigma2_z, with the last axis over
hypotheses, and returns log-scores of the same shape:

    exact          log p(x | z)
    tempered_<k>   k * log p(x | z), the power likelihood with exponent k
    clipped        max(log p(x | z) - max_z' log p(x | z'), -c), where c is the log
                   ratio of the largest to the smallest bins_abs3 mass at the same d
    typ_normal     -T^2 / 2, which is log phi(T) up to a constant
    typ_density    log f(q) for the chi-square(d) density of the statistic, without the Jacobian
    bins_<scheme>  the log-mass of the bin that T(x, z) falls into

A content-oracle scorer is its base scorer computed from the two in-plane
coordinates of the localization family, with d_eff = 2: typ_content_oracle is
typ_normal, and <base>_content_oracle is <base> otherwise. It needs in-plane q
values, so the caller scores those under the base name (see content_oracle_base).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import special

from robust_likelihood.gaussian.bins import NAMED_SCHEMES, BinScheme, bin_index, log_masses, scheme_by_name
from robust_likelihood.gaussian.model import log_likelihood, typicality

CONTENT_DIMENSIONS = 2
CONTENT_ORACLE_SUFFIX = "_content_oracle"


def clipped_cap(d: int) -> float:
    """c = log(max_k m_k / min_k m_k) over the bins_abs3 masses at dimension d, in nats."""
    masses = log_masses(NAMED_SCHEMES["abs3"], d)
    return float(masses.max() - masses.min())


def chi2_log_density(q: np.ndarray, d: int) -> np.ndarray:
    """log f(q) for the chi-square density with d degrees of freedom."""
    q = np.asarray(q, dtype=np.float64)
    half = 0.5 * d
    return special.xlogy(half - 1.0, q) - 0.5 * q - half * math.log(2.0) - special.gammaln(half)


def content_oracle_base(name: str) -> str | None:
    """The base scorer of a content-oracle scorer (typ_content_oracle -> typ_normal), or None for any other name."""
    if not name.endswith(CONTENT_ORACLE_SUFFIX):
        return None
    base = name[: -len(CONTENT_ORACLE_SUFFIX)]
    return "typ_normal" if base == "typ" else base


def resolve_scheme(name: str, d: int, schemes: Mapping[str, BinScheme] | None = None) -> BinScheme:
    """The scheme of a bins_<scheme> scorer: an entry of schemes if present, else a named or equal-mass scheme."""
    key = name.removeprefix("bins_")
    if schemes is not None and key in schemes:
        return schemes[key]
    return scheme_by_name(key, d)


def score_matrix(
    name: str,
    q: np.ndarray,
    sigma2: np.ndarray,
    d: int,
    *,
    schemes: Mapping[str, BinScheme] | None = None,
) -> np.ndarray:
    """Per-observation log-scores l_z(x_i), in nats, with the shape of q. Dispatches on the scorer name.

    schemes supplies bin schemes that a name alone does not fix, such as the
    optimized bins_opt<K> edges chosen in E5.
    """
    base = content_oracle_base(name)
    if base is not None:
        raise ValueError(f"{name} needs in-plane q; score {base} on in-plane q with d = {CONTENT_DIMENSIONS}")
    q = np.asarray(q, dtype=np.float64)
    if name == "exact":
        return log_likelihood(q, sigma2, d)
    if name.startswith("tempered_"):
        zeta = float(name.removeprefix("tempered_"))
        if not zeta > 0.0:
            raise ValueError("the tempering exponent must be positive")
        return zeta * log_likelihood(q, sigma2, d)
    if name == "clipped":
        relative = log_likelihood(q, sigma2, d)
        relative = relative - relative.max(axis=-1, keepdims=True)
        return np.maximum(relative, -clipped_cap(d))
    if name == "typ_normal":
        t = typicality(q, d)
        return -0.5 * t * t
    if name == "typ_density":
        return chi2_log_density(q, d)
    if name.startswith("bins_"):
        scheme = resolve_scheme(name, d, schemes)
        return log_masses(scheme, d)[bin_index(scheme, typicality(q, d), d)]
    raise KeyError(f"unknown scorer {name!r}")
