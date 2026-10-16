"""Partitions of the typicality axis and their exact chi-square masses.

Under the hypothesis being scored, q ~ chi-square(d), so a bin of
T = (q - d) / sqrt(2 d) has the same mass under every hypothesis of dimension d.
A signed scheme partitions T into [e_k, e_k+1); an absolute scheme partitions |T|
into [a_k, a_k+1). The binned log-score of hypothesis z is the log-mass of the bin
that T(x, z) falls into. Each hypothesis is scored on its own event, so these
scores are not the likelihood of one shared observation.

A bin whose mass is below 1e-300 at dimension d is dropped from the scheme at that
d. That happens below the bound T >= -sqrt(d / 2) at small d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import optimize, stats

EMPTY_BIN_MASS = 1e-300
_LOG_EMPTY_BIN_MASS = math.log(EMPTY_BIN_MASS)


class EmptyBinError(ValueError):
    """An observation fell into a bin that was dropped because its mass is below 1e-300."""


@dataclass(frozen=True)
class BinScheme:
    """A partition of T (kind "signed") or of |T| (kind "abs") by its interior edges."""

    name: str
    kind: Literal["abs", "signed"]
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        edges = tuple(float(edge) for edge in self.edges)
        if self.kind not in ("abs", "signed"):
            raise ValueError("kind must be abs or signed")
        if not edges or not all(math.isfinite(edge) for edge in edges):
            raise ValueError("a scheme needs at least one finite interior edge")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("edges must be strictly increasing")
        if self.kind == "abs" and edges[0] <= 0.0:
            raise ValueError("absolute edges must be positive")
        object.__setattr__(self, "edges", edges)

    @property
    def bin_count(self) -> int:
        return len(self.edges) + 1

    def bounds(self, k: int) -> tuple[float, float]:
        """[low, high) of bin k on the T axis (signed) or the |T| axis (abs)."""
        if not 0 <= k < self.bin_count:
            raise IndexError(k)
        first = 0.0 if self.kind == "abs" else -math.inf
        low = first if k == 0 else self.edges[k - 1]
        high = math.inf if k == len(self.edges) else self.edges[k]
        return low, high


NAMED_SCHEMES = {
    "abs2": BinScheme("abs2", "abs", (1.96,)),
    "abs3": BinScheme("abs3", "abs", (1.0, 2.0)),
    "signed5": BinScheme("signed5", "signed", (-2.0, -1.0, 1.0, 2.0)),
}


def absolute_scheme(edges: tuple[float, ...], name: str | None = None) -> BinScheme:
    """An absolute scheme with the given |T| edges, named after them unless a name is given."""
    label = name or "abs[" + ",".join(f"{edge:g}" for edge in edges) + "]"
    return BinScheme(label, "abs", tuple(edges))


def q_intervals(scheme: BinScheme, k: int, d: int) -> list[tuple[float, float]]:
    """The intervals of q whose T lies in bin k, clipped at q = 0. Empty pieces are left out.

    A signed bin is one interval. An absolute bin [a, b) is the union of
    (d - b sqrt(2d), d - a sqrt(2d)] and [d + a sqrt(2d), d + b sqrt(2d)).
    """
    scale = math.sqrt(2.0 * d)
    low, high = scheme.bounds(k)
    if scheme.kind == "signed":
        pieces = [(d + low * scale, d + high * scale)]
    else:
        pieces = [(d - high * scale, d - low * scale), (d + low * scale, d + high * scale)]
    kept = []
    for start, stop in pieces:
        start = max(start, 0.0)
        if stop > start:
            kept.append((start, stop))
    return kept


def _log_minus(log_big: float, log_small: float) -> float:
    """log(exp(log_big) - exp(log_small)) for log_big >= log_small; -inf when they are equal."""
    if log_small == -math.inf:
        return log_big
    if log_small >= log_big:
        return -math.inf
    return log_big + math.log1p(-math.exp(log_small - log_big))


def _log_interval_mass(low: float, high: float, d: int) -> float:
    """log P(low <= q < high) for q ~ chi-square(d), with 0 <= low < high <= inf.

    The part below the mean d is a difference of CDFs and the part above it a
    difference of survival functions, so no tail is computed as one minus a number
    close to one.
    """
    parts = []
    split = float(d)
    if low < min(high, split):
        top = min(high, split)
        log_low = float(stats.chi2.logcdf(low, d)) if low > 0.0 else -math.inf
        parts.append(_log_minus(float(stats.chi2.logcdf(top, d)), log_low))
    if max(low, split) < high:
        bottom = max(low, split)
        log_high = float(stats.chi2.logsf(high, d)) if math.isfinite(high) else -math.inf
        parts.append(_log_minus(float(stats.chi2.logsf(bottom, d)), log_high))
    if not parts:
        return -math.inf
    return float(np.logaddexp.reduce(parts))


def _read_only(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@lru_cache(maxsize=None)
def full_log_masses(scheme: BinScheme, d: int) -> np.ndarray:
    """Exact chi-square log-mass of every bin at dimension d, empty bins included (-inf when structurally empty)."""
    if d < 1:
        raise ValueError("d must be at least 1")
    masses = []
    for k in range(scheme.bin_count):
        pieces = [_log_interval_mass(start, stop, d) for start, stop in q_intervals(scheme, k, d)]
        masses.append(float(np.logaddexp.reduce(pieces)) if pieces else -math.inf)
    return _read_only(np.array(masses, dtype=np.float64))


@lru_cache(maxsize=None)
def retained_bins(scheme: BinScheme, d: int) -> np.ndarray:
    """Full indices of the bins with mass at least 1e-300 at dimension d."""
    return _read_only(np.flatnonzero(full_log_masses(scheme, d) >= _LOG_EMPTY_BIN_MASS))


@lru_cache(maxsize=None)
def log_masses(scheme: BinScheme, d: int) -> np.ndarray:
    """Exact chi-square log-masses, in nats, of the bins retained at dimension d (empty bins dropped)."""
    return _read_only(full_log_masses(scheme, d)[retained_bins(scheme, d)])


@lru_cache(maxsize=None)
def _retained_lookup(scheme: BinScheme, d: int) -> np.ndarray:
    lookup = np.full(scheme.bin_count, -1, dtype=np.int64)
    kept = retained_bins(scheme, d)
    lookup[kept] = np.arange(kept.size)
    return _read_only(lookup)


def bin_index(scheme: BinScheme, t: np.ndarray, d: int) -> np.ndarray:
    """Index into log_masses(scheme, d) of the bin that contains each value of T.

    Raises EmptyBinError if any value falls into a bin dropped as empty at d.
    """
    t = np.asarray(t, dtype=np.float64)
    if np.isnan(t).any():
        raise ValueError("typicality values must not be NaN")
    values = np.abs(t) if scheme.kind == "abs" else t
    full = np.searchsorted(np.asarray(scheme.edges), values, side="right")
    index = _retained_lookup(scheme, d)[full]
    if np.any(index < 0):
        raise EmptyBinError(f"an observation fell into a bin of {scheme.name} that is empty at d={d}")
    return index


@lru_cache(maxsize=None)
def equal_mass_scheme(k: int, d: int) -> BinScheme:
    """The absolute scheme whose k bins each have chi-square mass 1/k at dimension d.

    Edge j solves P(|T| < a_j) = j / k under chi-square(d). Every hypothesis then
    receives log-score -log k for every observation, so the posterior equals the prior.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    scale = math.sqrt(2.0 * d)

    def inner_mass(a: float) -> float:
        lower = d - a * scale
        below = float(stats.chi2.cdf(lower, d)) if lower > 0.0 else 0.0
        return float(stats.chi2.cdf(d + a * scale, d)) - below

    upper = 1.0
    while inner_mass(upper) < 1.0 - 0.5 / k:
        upper *= 2.0
    edges = []
    for j in range(1, k):
        target = j / k
        edges.append(
            optimize.brentq(
                lambda a: inner_mass(a) - target,
                0.0,
                upper,
                xtol=1e-15,
                rtol=4.0 * np.finfo(np.float64).eps,
                maxiter=500,
            )
        )
    return BinScheme(f"equalmass{k}", "abs", tuple(edges))


def scheme_by_name(name: str, d: int) -> BinScheme:
    """abs2, abs3, signed5, or equalmass<K>; equal-mass edges depend on d."""
    if name in NAMED_SCHEMES:
        return NAMED_SCHEMES[name]
    suffix = name.removeprefix("equalmass")
    if suffix != name and suffix.isdigit():
        return equal_mass_scheme(int(suffix), d)
    raise KeyError(f"unknown bin scheme {name!r}")
