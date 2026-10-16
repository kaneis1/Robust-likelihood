"""Seeding, observation sampling, and the contaminations C0-C5 of the Gaussian test.

Every (experiment, cell, trial) has its own generator from
SeedSequence([base_seed, crc32(experiment), crc32(cell_key), trial]), so results do
not depend on the order in which trials run. The separate random streams of one
trial use separate cell keys of the form "<cell_key>|<stream>".
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robust_likelihood.gaussian.families import (
    Family,
    draw_true_cell,
    grid_plane_means,
    mean_shift_family,
    scale_sigma2,
    scale_truth_index,
)

DEFAULT_BASE_SEED = 20261019
CONTAMINATION_CODES = ("C0", "C1", "C2", "C3", "C4", "C5")
C1_VARIANCE = 1.1
C2_DOF = 5
C5_PROBABILITY = 0.01
C5_SD = 10.0


@dataclass(frozen=True)
class Contamination:
    """One perturbation of observations drawn from the truth.

    C0 none. C1 x = mu* + sqrt(1.1) eps. C2 eps has t_5 coordinates scaled to unit
    variance. C3 with probability rho, add radius * u_1, where u_1 points from the
    truth to hypothesis 1. C4 with probability rho, add radius * v for a fresh uniform
    unit vector v. C5 each coordinate independently gets + N(0, 10^2) with
    probability 0.01. Only C3 and C4 take rho and radius.
    """

    code: str
    rho: float = 0.0
    radius: float = 0.0

    def __post_init__(self) -> None:
        if self.code not in CONTAMINATION_CODES:
            raise ValueError(f"contamination code must be one of {', '.join(CONTAMINATION_CODES)}")
        if self.code in ("C3", "C4"):
            if not 0.0 <= self.rho <= 1.0:
                raise ValueError("rho must lie in [0, 1]")
            if not self.radius > 0.0:
                raise ValueError("radius must be positive")
        elif self.rho != 0.0 or self.radius != 0.0:
            raise ValueError(f"{self.code} takes no rho or radius")

    @property
    def key(self) -> str:
        if self.code in ("C3", "C4"):
            return f"{self.code}:rho={self.rho:g}:R={self.radius:g}"
        return self.code

    @property
    def clean(self) -> bool:
        """True when no observation is perturbed: C0, or C3 and C4 with rho = 0."""
        return self.code == "C0" or (self.code in ("C3", "C4") and self.rho == 0.0)


CLEAN = Contamination("C0")


def trial_seed(base_seed: int, experiment: str, cell_key: str, trial: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [base_seed, zlib.crc32(experiment.encode("utf-8")), zlib.crc32(cell_key.encode("utf-8")), trial]
    )


def trial_rng(base_seed: int, experiment: str, cell_key: str, trial: int) -> np.random.Generator:
    """The generator of one (experiment, cell, trial)."""
    return np.random.default_rng(trial_seed(base_seed, experiment, cell_key, trial))


class TrialStreams:
    """The named random streams of one trial.

    Stream s draws from trial_rng(base_seed, experiment, "<cell_key>|s", trial). Each
    stream is drawn once and then reused, so every contamination and every scorer of
    the trial sees the same draws.
    """

    def __init__(self, base_seed: int, experiment: str, cell_key: str, trial: int) -> None:
        self.base_seed = base_seed
        self.experiment = experiment
        self.cell_key = cell_key
        self.trial = trial
        self._drawn: dict[str, tuple[tuple, np.ndarray]] = {}

    def generator(self, stream: str) -> np.random.Generator:
        return trial_rng(self.base_seed, self.experiment, f"{self.cell_key}|{stream}", self.trial)

    def _once(self, stream: str, spec: tuple, draw: Callable[[np.random.Generator], np.ndarray]) -> np.ndarray:
        if stream in self._drawn:
            drawn_spec, values = self._drawn[stream]
            if drawn_spec != spec:
                raise ValueError(f"stream {stream!r} was already drawn as {drawn_spec}")
            return values
        values = draw(self.generator(stream))
        values.setflags(write=False)
        self._drawn[stream] = (spec, values)
        return values

    def normal(self, stream: str, shape: tuple[int, ...]) -> np.ndarray:
        return self._once(stream, ("normal", shape), lambda rng: rng.standard_normal(shape))

    def uniform(self, stream: str, shape: tuple[int, ...]) -> np.ndarray:
        return self._once(stream, ("uniform", shape), lambda rng: rng.random(shape))

    def student_t(self, stream: str, dof: float, shape: tuple[int, ...]) -> np.ndarray:
        return self._once(stream, ("student_t", dof, shape), lambda rng: rng.standard_t(dof, shape))

    def unit_vectors(self, stream: str, count: int, d: int) -> np.ndarray:
        def draw(rng: np.random.Generator) -> np.ndarray:
            values = rng.standard_normal((count, d))
            return values / np.linalg.norm(values, axis=1, keepdims=True)

        return self._once(stream, ("unit_vectors", count, d), draw)


def contaminated_observations(
    contamination: Contamination,
    truth_mean: np.ndarray,
    toward: np.ndarray,
    streams: TrialStreams,
    n: int,
) -> np.ndarray:
    """n observations from N(truth_mean, I_d) under the contamination, with all d coordinates simulated. Returns (n, d).

    toward is the unit vector from the truth to hypothesis 1 (C3). The clean noise,
    the outlier indicators, and the outlier directions are streams of the trial, so
    every contamination of one trial perturbs the same clean draws. C3 and C4 with the
    same rho contaminate the same observations.
    """
    truth_mean = np.asarray(truth_mean, dtype=np.float64)
    d = truth_mean.shape[0]
    noise = streams.normal("noise", (n, d))
    code = contamination.code
    if code == "C1":
        return truth_mean + math.sqrt(C1_VARIANCE) * noise
    if code == "C2":
        heavy = streams.student_t("heavy_tail", C2_DOF, (n, d))
        return truth_mean + heavy / math.sqrt(C2_DOF / (C2_DOF - 2.0))
    x = truth_mean + noise
    if code in ("C3", "C4"):
        hit = streams.uniform("outlier", (n,)) < contamination.rho
        if code == "C3":
            shift = np.broadcast_to(np.asarray(toward, dtype=np.float64), (n, d))
        else:
            shift = streams.unit_vectors("outlier_direction", n, d)
        return x + contamination.radius * hit[:, None] * shift
    if code == "C5":
        hit = streams.uniform("sparse_hit", (n, d)) < C5_PROBABILITY
        return x + hit * (C5_SD * streams.normal("sparse_value", (n, d)))
    return x


def mean_shift_trial(
    base_seed: int,
    experiment: str,
    cell_key: str,
    trial: int,
    d: int,
    delta: float,
    n: int,
    contaminations: tuple[Contamination, ...] = (CLEAN,),
) -> tuple[Family, dict[str, np.ndarray]]:
    """The hypotheses of one mean-shift trial and its n observations under each contamination, keyed by contamination key."""
    streams = TrialStreams(base_seed, experiment, cell_key, trial)
    family = mean_shift_family(streams.generator("hypotheses"), d, delta)
    truth = family.mus[family.true_index]
    toward = (family.mus[1] - truth) / np.linalg.norm(family.mus[1] - truth)
    observations = {
        contamination.key: contaminated_observations(contamination, truth, toward, streams, n)
        for contamination in contaminations
    }
    return family, observations


def localization_trial(
    base_seed: int,
    experiment: str,
    trial: int,
    spacing: float,
    d: int,
    n: int,
) -> tuple[int, np.ndarray]:
    """The true grid cell of one localization trial and its n clean observations, shape (n, d).

    The true cell and the two in-plane noise coordinates come from a stream keyed
    "inplane", which depends on neither d nor the spacing. The d - 2 nuisance
    coordinates come from a stream keyed by d. The in-plane data of a trial are
    therefore identical at every d.
    """
    inplane = trial_rng(base_seed, experiment, "inplane", trial)
    true_index = draw_true_cell(inplane)
    plane = grid_plane_means(spacing)[true_index] + inplane.standard_normal((n, 2))
    nuisance = trial_rng(base_seed, experiment, f"d={d}|nuisance", trial).standard_normal((n, d - 2))
    return true_index, np.hstack([plane, nuisance])


def scale_trial(base_seed: int, experiment: str, d: int, trial: int, n: int) -> tuple[int, np.ndarray]:
    """The true hypothesis of one scale-family trial and n draws of R = ||x||^2, shape (n,).

    Every scale-family scorer depends on x only through R, and under the truth
    R ~ sigma2_true * chi-square(d) exactly, so R is sampled directly.
    """
    true_index = scale_truth_index(trial)
    rng = trial_rng(base_seed, experiment, f"d={d}", trial)
    return true_index, float(scale_sigma2()[true_index]) * rng.chisquare(d, size=n)
