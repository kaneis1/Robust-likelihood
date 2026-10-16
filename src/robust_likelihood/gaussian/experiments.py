"""Experiments E0-E5 of the Gaussian typicality test, their presets, and the run folder.

A preset fixes every grid, trial count, and the base seed. ``quick`` is a smoke run
for a laptop. ``full`` is the reportable run. Within a trial, every scorer is
evaluated on the same observations.
"""

from __future__ import annotations

import itertools
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy import stats

from robust_likelihood.gaussian.bins import (
    BinScheme,
    absolute_scheme,
    bin_index,
    log_masses,
    retained_bins,
    scheme_by_name,
)
from robust_likelihood.gaussian.families import (
    MEAN_SHIFT_HYPOTHESES,
    grid_locations,
    localization_means,
    scale_sigma2,
    separation_threshold,
)
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
from robust_likelihood.gaussian.model import (
    entropy,
    quadratic_forms,
    surprisal,
    surprisal_variance,
    typicality,
    typicality_central_moments,
)
from robust_likelihood.gaussian.sampling import (
    CLEAN,
    DEFAULT_BASE_SEED,
    Contamination,
    localization_trial,
    mean_shift_trial,
    scale_trial,
    trial_rng,
)
from robust_likelihood.gaussian.scorers import CONTENT_DIMENSIONS, content_oracle_base, score_matrix
from robust_likelihood.storage import append_jsonl, repo_root, write_json

EXPERIMENT_NAMES = ("e0", "e1", "e2", "e3", "e4", "e5")

E0_SCHEMES = ("abs2", "abs3", "signed5", "equalmass2", "equalmass3", "equalmass4")
E1_SCORERS = (
    "exact",
    "tempered_0.5",
    "tempered_0.1",
    "clipped",
    "typ_normal",
    "typ_density",
    "bins_abs2",
    "bins_abs3",
    "bins_signed5",
    "bins_equalmass3",
)
E2_SCORERS = ("exact", "typ_normal", "typ_density", "bins_abs3", "bins_signed5")
E2_N_GRID = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
E3_SCORERS = (
    "exact",
    "typ_normal",
    "typ_density",
    "bins_abs3",
    "typ_content_oracle",
    "bins_abs3_content_oracle",
)
E3_HEATMAP_SCORERS = ("exact", "typ_normal", "bins_abs3")
E4_SCORERS = (
    "exact",
    "tempered_0.5",
    "tempered_0.1",
    "clipped",
    "typ_normal",
    "bins_abs2",
    "bins_abs3",
    "bins_signed5",
)
E5_HELDOUT_SCORERS = (
    "exact",
    "bins_opt2",
    "bins_opt3",
    "bins_opt4",
    "bins_abs2",
    "bins_abs3",
    "bins_equalmass2",
    "bins_equalmass3",
    "bins_equalmass4",
)
E5_EDGE_GRID = tuple(0.25 * k for k in range(1, 17))


def contamination_grid(rhos: tuple[float, ...], radii: tuple[float, ...]) -> tuple[Contamination, ...]:
    """C0, C1, C2, then C3 and C4 over every (rho, radius), then C5."""
    grid = [Contamination("C0"), Contamination("C1"), Contamination("C2")]
    for code in ("C3", "C4"):
        grid.extend(Contamination(code, rho, radius) for radius in radii for rho in rhos)
    grid.append(Contamination("C5"))
    return tuple(grid)


@dataclass(frozen=True)
class E0Config:
    """Closed forms against Monte Carlo draws. No inference."""

    d_grid: tuple[int, ...]
    sigma2_grid: tuple[float, ...]
    sample_sizes: tuple[int, ...]
    moment_draws: int
    schemes: tuple[str, ...] = E0_SCHEMES


@dataclass(frozen=True)
class E1Config:
    """Mean-shift family, clean data. delta = r * (2d)^(1/4)."""

    d_grid: tuple[int, ...]
    r_grid: tuple[float, ...]
    n_max: int
    trials: int
    scorers: tuple[str, ...] = E1_SCORERS


@dataclass(frozen=True)
class E2Config:
    """Scale family: common mean 0, sigma2 in (0.25, 0.5, 1, 2, 4), truth cycling over trials."""

    d_grid: tuple[int, ...]
    n_max: int
    trials: int
    limit_draws: int
    scorers: tuple[str, ...] = E2_SCORERS

    @property
    def n_grid(self) -> tuple[int, ...]:
        return tuple(n for n in E2_N_GRID if n <= self.n_max)


@dataclass(frozen=True)
class E3Config:
    """Localization on a 21 x 21 grid embedded in coordinates 1-2 of R^d."""

    spacing_grid: tuple[float, ...]
    d_grid: tuple[int, ...]
    n_max: int
    trials: int
    scorers: tuple[str, ...] = E3_SCORERS
    heatmap_spacing: float = 1.0
    heatmap_d: tuple[int, ...] = (2, 100, 1000)
    heatmap_n: tuple[int, ...] = (1, 2, 5, 10, 20)
    heatmap_trial: int = 0
    heatmap_scorers: tuple[str, ...] = E3_HEATMAP_SCORERS


@dataclass(frozen=True)
class E4Config:
    """Mean-shift family at fixed r under the contaminations C0-C5."""

    d_grid: tuple[int, ...]
    r: float
    n_max: int
    trials: int
    contaminations: tuple[Contamination, ...]
    scorers: tuple[str, ...] = E4_SCORERS


@dataclass(frozen=True)
class E5Config:
    """Search over absolute bin edges, then evaluation on held-out E1 and E4 cells.

    The calibration data use their own experiment key, so no calibration draw is
    reused for evaluation. Held-out cells regenerate the E1 and E4 draws.
    """

    calibration_d: int
    calibration_r: float
    calibration_trials: int
    calibration_n_max: int
    calibration_contamination: Contamination
    heldout_e1: E1Config
    heldout_e4: E4Config
    edge_grid: tuple[float, ...] = E5_EDGE_GRID
    k_grid: tuple[int, ...] = (2, 3, 4)
    objective_n: int = 10
    constraint_n: int = 40
    min_accuracy: float = 0.9

    def __post_init__(self) -> None:
        if self.calibration_n_max < max(self.objective_n, self.constraint_n):
            raise ValueError("calibration_n_max must cover objective_n and constraint_n")
        for scorers in (self.heldout_e1.scorers, self.heldout_e4.scorers):
            for scorer in scorers:
                suffix = scorer.removeprefix("bins_opt")
                if suffix != scorer and int(suffix) not in self.k_grid:
                    raise ValueError(f"{scorer} needs K = {suffix} in k_grid")


@dataclass(frozen=True)
class Preset:
    name: str
    base_seed: int
    e0: E0Config
    e1: E1Config
    e2: E2Config
    e3: E3Config
    e4: E4Config
    e5: E5Config


FULL = Preset(
    name="full",
    base_seed=DEFAULT_BASE_SEED,
    e0=E0Config(
        d_grid=(1, 2, 10, 100, 1000),
        sigma2_grid=(0.25, 1.0, 4.0),
        sample_sizes=(10, 100, 1000, 10000),
        moment_draws=100_000,
    ),
    e1=E1Config(
        d_grid=(10, 100, 1000),
        r_grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
        n_max=50,
        trials=500,
    ),
    e2=E2Config(d_grid=(1, 2, 5, 10, 100, 1000), n_max=1000, trials=500, limit_draws=100_000),
    e3=E3Config(spacing_grid=(0.5, 1.0, 2.0), d_grid=(2, 10, 100, 1000), n_max=20, trials=300),
    e4=E4Config(
        d_grid=(100, 1000),
        r=1.5,
        n_max=40,
        trials=500,
        contaminations=contamination_grid((0.0, 0.05, 0.1, 0.2, 0.3), (20.0, 50.0)),
    ),
    e5=E5Config(
        calibration_d=100,
        calibration_r=1.5,
        calibration_trials=200,
        calibration_n_max=40,
        calibration_contamination=Contamination("C3", 0.2, 50.0),
        heldout_e1=E1Config(
            d_grid=(10, 1000),
            r_grid=(1.0, 1.5, 2.0),
            n_max=50,
            trials=500,
            scorers=E5_HELDOUT_SCORERS,
        ),
        heldout_e4=E4Config(
            d_grid=(1000,),
            r=1.5,
            n_max=40,
            trials=500,
            contaminations=tuple(
                Contamination("C3", rho, radius) for radius in (20.0, 50.0) for rho in (0.1, 0.3)
            ),
            scorers=E5_HELDOUT_SCORERS,
        ),
    ),
)

QUICK = Preset(
    name="quick",
    base_seed=DEFAULT_BASE_SEED,
    e0=FULL.e0,
    e1=E1Config(d_grid=(10, 100), r_grid=(1.0, 2.0), n_max=20, trials=50),
    e2=E2Config(d_grid=(1, 10), n_max=200, trials=50, limit_draws=100_000),
    e3=E3Config(spacing_grid=(1.0,), d_grid=(2, 100), n_max=20, trials=30),
    e4=E4Config(
        d_grid=(1000,),
        r=1.5,
        n_max=40,
        trials=50,
        contaminations=(Contamination("C0"), Contamination("C3", 0.0, 50.0), Contamination("C3", 0.2, 50.0)),
    ),
    e5=replace(
        FULL.e5,
        calibration_trials=50,
        heldout_e1=replace(FULL.e5.heldout_e1, d_grid=(10,), r_grid=(1.0, 2.0), n_max=20, trials=50),
        heldout_e4=replace(
            FULL.e5.heldout_e4,
            trials=50,
            contaminations=(Contamination("C3", 0.1, 50.0), Contamination("C3", 0.3, 50.0)),
        ),
    ),
)

PRESETS = {preset.name: preset for preset in (QUICK, FULL)}


def parse_experiments(text: str) -> tuple[str, ...]:
    chosen = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    unknown = [name for name in chosen if name not in EXPERIMENT_NAMES]
    if not chosen or unknown or len(set(chosen)) != len(chosen):
        raise ValueError(f"experiments must be a unique subset of {','.join(EXPERIMENT_NAMES)}; got {text!r}")
    return tuple(name for name in EXPERIMENT_NAMES if name in chosen)


def preset_config(preset: Preset) -> dict:
    """The resolved preset as JSON-ready data, including derived grids."""
    payload = asdict(preset)
    payload["e2"]["n_grid"] = list(preset.e2.n_grid)
    payload["e4"]["contamination_keys"] = [c.key for c in preset.e4.contaminations]
    payload["e5"]["heldout_e4"]["contamination_keys"] = [c.key for c in preset.e5.heldout_e4.contaminations]
    payload["e5"]["candidate_count"] = candidate_count(preset.e5)
    return payload


def candidate_count(config: E5Config) -> int:
    return sum(math.comb(len(config.edge_grid), k - 1) for k in config.k_grid)


def plan_summary(preset: Preset, experiments: tuple[str, ...]) -> dict:
    """Cells, trials, and posterior computations per experiment. No draws are made."""
    plan: dict[str, dict] = {}
    if "e0" in experiments:
        c0 = preset.e0
        plan["e0"] = {
            "entropy_cells": len(c0.d_grid) * len(c0.sigma2_grid) * len(c0.sample_sizes),
            "moment_cells": len(c0.d_grid) * len(c0.sigma2_grid),
            "moment_draws_per_cell": c0.moment_draws,
            "bin_mass_tests": len(c0.d_grid) * len(c0.schemes),
        }
    if "e1" in experiments:
        c1 = preset.e1
        cells = len(c1.d_grid) * len(c1.r_grid)
        plan["e1"] = {
            "cells": cells,
            "trials": c1.trials,
            "n_max": c1.n_max,
            "scorers": list(c1.scorers),
            "posteriors": cells * c1.trials * len(c1.scorers),
        }
    if "e2" in experiments:
        c2 = preset.e2
        plan["e2"] = {
            "cells": len(c2.d_grid),
            "trials": c2.trials,
            "n_max": c2.n_max,
            "n_grid": list(c2.n_grid),
            "scorers": list(c2.scorers),
            "posteriors": len(c2.d_grid) * c2.trials * len(c2.scorers),
        }
    if "e3" in experiments:
        c3 = preset.e3
        cells = len(c3.spacing_grid) * len(c3.d_grid)
        plan["e3"] = {
            "cells": cells,
            "trials": c3.trials,
            "n_max": c3.n_max,
            "scorers": list(c3.scorers),
            "posteriors": cells * c3.trials * len(c3.scorers),
        }
    if "e4" in experiments:
        c4 = preset.e4
        cells = len(c4.d_grid) * len(c4.contaminations)
        plan["e4"] = {
            "cells": cells,
            "contaminations": [c.key for c in c4.contaminations],
            "trials": c4.trials,
            "n_max": c4.n_max,
            "scorers": list(c4.scorers),
            "posteriors": cells * c4.trials * len(c4.scorers),
        }
    if "e5" in experiments:
        c5 = preset.e5
        heldout_cells = len(c5.heldout_e1.d_grid) * len(c5.heldout_e1.r_grid)
        heldout_cells += len(c5.heldout_e4.d_grid) * len(c5.heldout_e4.contaminations)
        plan["e5"] = {
            "candidates": candidate_count(c5),
            "calibration_trials": c5.calibration_trials,
            "heldout_cells": heldout_cells,
            "scorers": list(c5.heldout_e1.scorers),
        }
    return {"preset": preset.name, "base_seed": preset.base_seed, "experiments": list(experiments), "plan": plan}


Progress = Callable[[str], None]
EQUAL_MASS_THRESHOLD = 1e-12
EXACT_ACROSS_D_THRESHOLD = 1e-9
ORACLE_THRESHOLD = 1e-12


def print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class ExperimentOutput:
    """Rows for metrics.jsonl and cells.jsonl, and the E3 heatmap posteriors."""

    metrics: list[dict] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)
    heatmaps: list[dict] = field(default_factory=list)

    def extend(self, other: ExperimentOutput) -> None:
        self.metrics.extend(other.metrics)
        self.cells.extend(other.cells)
        self.heatmaps.extend(other.heatmaps)


def _number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _log_posterior(
    scorer: str,
    q: np.ndarray,
    sigma2: np.ndarray,
    d: int,
    *,
    q_plane: np.ndarray | None = None,
    schemes: Mapping[str, BinScheme] | None = None,
) -> np.ndarray:
    """Normalized log posteriors (trials, n, M) of one scorer under a uniform prior."""
    base = content_oracle_base(scorer)
    if base is None:
        scores = score_matrix(scorer, q, sigma2, d, schemes=schemes)
    elif q_plane is None:
        raise ValueError(f"{scorer} needs in-plane q")
    else:
        scores = score_matrix(base, q_plane, sigma2, CONTENT_DIMENSIONS, schemes=schemes)
    return cumulative_log_posterior(scores, uniform_log_prior(q.shape[-1]))


def _scorer_rows(
    key: dict,
    log_post: np.ndarray,
    true_index: np.ndarray,
    n_grid: tuple[int, ...],
    *,
    exact_log_post: np.ndarray | None,
    locations: np.ndarray | None = None,
    subsets: Mapping[str, np.ndarray] | None = None,
    wrong_mass: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Metric rows at each n of n_grid and one n90 row, for every subset of trials.

    With wrong_mass, the truth is the same hypothesis in every trial, and each row also
    records the largest trial-mean posterior of any wrong hypothesis.
    """
    per_trial = metrics(
        log_post,
        true_index,
        uniform_log_prior(log_post.shape[-1]),
        exact_log_post=exact_log_post,
        locations=locations,
    )
    if subsets is None:
        subsets = {"all": np.ones(true_index.shape[0], dtype=bool)}
    metric_rows: list[dict] = []
    cell_rows: list[dict] = []
    for subset, mask in subsets.items():
        trials = int(mask.sum())
        if trials == 0:
            continue
        summaries = {name: mean_and_se(values[mask]) for name, values in per_trial.items()}
        wrong = None
        if wrong_mass:
            mean_posterior = np.exp(log_post[mask]).mean(axis=0)
            mean_posterior[:, int(true_index[mask][0])] = -np.inf
            wrong = (mean_posterior.max(axis=1), mean_posterior.argmax(axis=1))
        for n in n_grid:
            row = {**key, "subset": subset, "n": n, "trials": trials}
            for name, (mean, se) in summaries.items():
                row[name] = {"mean": _number(mean[n - 1]), "se": None if se is None else _number(se[n - 1])}
            if wrong is not None:
                row["max_wrong_mean_posterior"] = _number(wrong[0][n - 1])
                row["max_wrong_hypothesis"] = int(wrong[1][n - 1])
            metric_rows.append(row)
        cell_rows.append(
            {
                **key,
                "kind": "n90",
                "subset": subset,
                "n90": first_reaching(summaries["p_true"][0]),
                "n_max": int(log_post.shape[-2]),
                "trials": trials,
            }
        )
    return metric_rows, cell_rows


def _check_row(key: dict, name: str, difference: float, threshold: float, **extra) -> dict:
    return {
        **key,
        "kind": "check",
        "name": name,
        "max_abs_diff": float(difference),
        "threshold": threshold,
        "passed": bool(difference < threshold),
        **extra,
    }


def _q_draws(rng: np.random.Generator, mu: np.ndarray, variance: np.ndarray, count: int) -> np.ndarray:
    """q = ||x - mu||^2 / sigma2 for count draws x ~ N(mu, sigma2 I_d), drawn in chunks of about 4e6 values."""
    d = mu.shape[1]
    rows = max(1, 4_000_000 // d)
    values = np.empty(count)
    for start in range(0, count, rows):
        size = min(rows, count - start)
        x = mu + math.sqrt(float(variance[0])) * rng.standard_normal((size, d))
        values[start : start + size] = quadratic_forms(x, mu, variance)[:, 0]
    return values


def run_e0(config: E0Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Closed forms against Monte Carlo: H and V from S draws, moments of T, and bin masses."""
    output = ExperimentOutput()
    shifts: dict[int, list[float]] = {size: [] for size in config.sample_sizes}
    for d in config.d_grid:
        started = time.perf_counter()
        moments = typicality_central_moments(d)
        v = surprisal_variance(d)
        surprisal_fourth = v * v * moments[4]
        gamma = moments[3]
        # Delta-method N * Var of the sample skewness, in the standardized central moments of T.
        skew_variance = (
            moments[6] - 6.0 * moments[4] + 9.0 + gamma**2 * (2.25 * moments[4] + 8.75) - 3.0 * gamma * moments[5]
        )
        pooled = []
        for sigma2 in config.sigma2_grid:
            variance = np.array([sigma2])
            cell = f"d={d}|sigma2={sigma2:g}"
            mu = trial_rng(base_seed, "e0", f"{cell}|mean", 0).standard_normal((1, d))
            h = float(entropy(variance, d)[0])
            for size in config.sample_sizes:
                rng = trial_rng(base_seed, "e0", f"{cell}|S={size}", 0)
                values = surprisal(_q_draws(rng, mu, variance, size), variance, d)
                h_hat = float(values.mean())
                v_hat = float(values.var(ddof=1))
                se_v = math.sqrt((surprisal_fourth - (size - 3.0) / (size - 1.0) * v * v) / size)
                shift = (h - h_hat) / math.sqrt(v)
                shifts[size].append(shift)
                output.cells.append(
                    {
                        "experiment": "e0",
                        "kind": "entropy",
                        "d": d,
                        "sigma2": sigma2,
                        "S": size,
                        "H": h,
                        "H_hat": h_hat,
                        "z_H": (h_hat - h) / math.sqrt(v / size),
                        "V": v,
                        "V_hat": v_hat,
                        "z_V": (v_hat - v) / se_v,
                        "t_center_shift": shift,
                    }
                )
            rng = trial_rng(base_seed, "e0", f"{cell}|moments", 0)
            t = typicality(_q_draws(rng, mu, variance, config.moment_draws), d)
            pooled.append(t)
            count = t.size
            mean_t = float(t.mean())
            var_t = float(t.var(ddof=1))
            skew_t = float(stats.skew(t))
            output.cells.append(
                {
                    "experiment": "e0",
                    "kind": "moments",
                    "d": d,
                    "sigma2": sigma2,
                    "draws": count,
                    "mean": mean_t,
                    "var": var_t,
                    "skew": skew_t,
                    "target_skew": gamma,
                    "z_mean": mean_t * math.sqrt(count),
                    "z_var": (var_t - 1.0) / math.sqrt((moments[4] - 1.0) / count),
                    "z_skew": (skew_t - gamma) / math.sqrt(skew_variance / count),
                }
            )
        t_all = np.concatenate(pooled)
        for name in config.schemes:
            scheme = scheme_by_name(name, d)
            masses = np.exp(log_masses(scheme, d))
            observed = np.bincount(bin_index(scheme, t_all, d), minlength=masses.size)
            expected = t_all.size * masses
            output.cells.append(
                {
                    "experiment": "e0",
                    "kind": "bin_mass",
                    "scheme": name,
                    "d": d,
                    "edges": list(scheme.edges),
                    "draws": int(t_all.size),
                    "masses": masses.tolist(),
                    "observed": observed.tolist(),
                    "expected": expected.tolist(),
                    "p_value": float(stats.chisquare(observed, expected).pvalue),
                    "dropped_bins": sorted(set(range(scheme.bin_count)) - set(retained_bins(scheme, d).tolist())),
                }
            )
        progress(f"e0 d={d}: {time.perf_counter() - started:.1f}s")
    for size, values in shifts.items():
        output.cells.append(
            {
                "experiment": "e0",
                "kind": "plugin_shift",
                "S": size,
                "cells": len(values),
                "rms_shift": math.sqrt(float(np.mean(np.square(values)))),
                "predicted_sd": 1.0 / math.sqrt(size),
            }
        )
    return output


def _mean_shift_q(
    base_seed: int,
    experiment: str,
    cell_key: str,
    trials: int,
    d: int,
    delta: float,
    n: int,
    contaminations: tuple[Contamination, ...],
) -> dict[str, np.ndarray]:
    """q arrays (trials, n, 20) of a mean-shift cell, keyed by contamination key."""
    q = {c.key: np.empty((trials, n, MEAN_SHIFT_HYPOTHESES)) for c in contaminations}
    for trial in range(trials):
        family, observations = mean_shift_trial(base_seed, experiment, cell_key, trial, d, delta, n, contaminations)
        for key, x in observations.items():
            q[key][trial] = quadratic_forms(x, family.mus, family.sigma2)
    return q


def _run_mean_shift(
    label: str,
    seed_experiment: str,
    base_seed: int,
    cells: list[tuple[int, float]],
    contaminations: tuple[Contamination, ...],
    *,
    trials: int,
    n_max: int,
    scorers: tuple[str, ...],
    progress: Progress,
    schemes: Mapping[str, BinScheme] | None = None,
) -> ExperimentOutput:
    """Mean-shift cells (d, r) under each contamination. Draws come from seed_experiment; rows are labeled label."""
    output = ExperimentOutput()
    sigma2 = np.ones(MEAN_SHIFT_HYPOTHESES)
    truth = np.zeros(trials, dtype=np.int64)
    n_grid = tuple(range(1, n_max + 1))
    for d, r in cells:
        started = time.perf_counter()
        delta = r * separation_threshold(d)
        cell_key = f"d={d}|r={r:g}"
        cell = {"d": d, "r": r, "delta": delta}
        q_by_key = _mean_shift_q(base_seed, seed_experiment, cell_key, trials, d, delta, n_max, contaminations)
        for contamination in contaminations:
            q = q_by_key[contamination.key]
            exact = _log_posterior("exact", q, sigma2, d) if contamination.clean else None
            for scorer in scorers:
                if scorer == "exact" and exact is not None:
                    log_post = exact
                else:
                    log_post = _log_posterior(scorer, q, sigma2, d, schemes=schemes)
                key = {
                    "experiment": label,
                    "cell_key": cell_key,
                    "cell": cell,
                    "contamination": contamination.key,
                    "scorer": scorer,
                }
                metric_rows, cell_rows = _scorer_rows(
                    key, log_post, truth, n_grid, exact_log_post=exact, wrong_mass=True
                )
                output.metrics.extend(metric_rows)
                output.cells.extend(cell_rows)
                if scorer.startswith("bins_equalmass"):
                    deviation = np.max(np.abs(np.exp(log_post) - 1.0 / MEAN_SHIFT_HYPOTHESES))
                    output.cells.append(
                        _check_row(key, "equal_mass_posterior_is_prior", deviation, EQUAL_MASS_THRESHOLD)
                    )
        progress(f"{label} {cell_key} ({len(contaminations)} contamination(s)): {time.perf_counter() - started:.1f}s")
    return output


def run_e1(config: E1Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Clean mean-shift cells over (d, r): convergence, information, and calibration of every scorer."""
    cells = [(d, r) for d in config.d_grid for r in config.r_grid]
    return _run_mean_shift(
        "e1",
        "e1",
        base_seed,
        cells,
        (CLEAN,),
        trials=config.trials,
        n_max=config.n_max,
        scorers=config.scorers,
        progress=progress,
    )


def run_e4(config: E4Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Mean-shift cells at fixed r under every contamination. Contaminations of a trial share its clean draws."""
    return _run_mean_shift(
        "e4",
        "e4",
        base_seed,
        [(d, config.r) for d in config.d_grid],
        config.contaminations,
        trials=config.trials,
        n_max=config.n_max,
        scorers=config.scorers,
        progress=progress,
    )


def _scale_limit(
    scorer: str,
    sigma2_true: float,
    sigma2: np.ndarray,
    d: int,
    base_seed: int,
    draws: int,
) -> tuple[np.ndarray, str]:
    """E_{x ~ N(0, sigma2_true I_d)}[l_z(x)] for every z: closed form for exact and typ_normal, else Monte Carlo.

    The Monte Carlo draws are shared by every z and every scorer of one (d, truth).
    """
    if scorer in ("exact", "typ_normal"):
        return scale_objective(scorer, sigma2_true, sigma2, d), "closed_form"
    rng = trial_rng(base_seed, "e2", f"d={d}|limit|truth={sigma2_true:g}", 0)
    r = sigma2_true * rng.chisquare(d, size=draws)
    return score_matrix(scorer, r[:, None] / sigma2, sigma2, d).mean(axis=0), "monte_carlo"


def run_e2(config: E2Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Scale family: convergence and the limit each scorer's posterior concentrates on."""
    output = ExperimentOutput()
    sigma2 = scale_sigma2()
    for d in config.d_grid:
        started = time.perf_counter()
        truth = np.empty(config.trials, dtype=np.int64)
        r = np.empty((config.trials, config.n_max))
        for trial in range(config.trials):
            truth[trial], r[trial] = scale_trial(base_seed, "e2", d, trial, config.n_max)
        q = r[..., None] / sigma2
        subsets = {"all": np.ones(config.trials, dtype=bool)}
        subsets.update({f"truth={value:g}": truth == z for z, value in enumerate(sigma2)})
        cell_key = f"d={d}"
        exact = _log_posterior("exact", q, sigma2, d)
        for scorer in config.scorers:
            log_post = exact if scorer == "exact" else _log_posterior(scorer, q, sigma2, d)
            key = {"experiment": "e2", "cell_key": cell_key, "cell": {"d": d}, "contamination": CLEAN.key, "scorer": scorer}
            metric_rows, cell_rows = _scorer_rows(key, log_post, truth, config.n_grid, exact_log_post=exact, subsets=subsets)
            output.metrics.extend(metric_rows)
            output.cells.extend(cell_rows)
            modes = posterior_mode(log_post[:, -1, :])
            for z, value in enumerate(sigma2):
                objective, method = _scale_limit(scorer, float(value), sigma2, d, base_seed, config.limit_draws)
                predicted = unique_argmax(objective)
                ranked = np.sort(objective)
                chosen = modes[truth == z]
                counts = {f"{s:g}": int(np.sum(chosen == j)) for j, s in enumerate(sigma2)}
                counts["tie"] = int(np.sum(chosen == -1))
                continuous = {"exact": float(value), "typ_normal": float(value) * (1.0 + 2.0 / d)}.get(scorer)
                output.cells.append(
                    {
                        **key,
                        "kind": "limit",
                        "subset": f"truth={value:g}",
                        "truth_sigma2": float(value),
                        "objective": {f"{s:g}": float(o) for s, o in zip(sigma2, objective)},
                        "objective_method": method,
                        "limit_draws": None if method == "closed_form" else config.limit_draws,
                        "objective_margin": float(ranked[-1] - ranked[-2]),
                        "predicted_sigma2": None if predicted < 0 else float(sigma2[predicted]),
                        "continuous_limit_sigma2": continuous,
                        "mode_counts": counts,
                        "mode_match": None if predicted < 0 else float(np.mean(chosen == predicted)),
                        "truth_match": float(np.mean(chosen == z)),
                        "n": config.n_max,
                        "trials": int(chosen.size),
                    }
                )
        progress(f"e2 d={d}: {time.perf_counter() - started:.1f}s")
    return output


def run_e3(config: E3Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Localization with nuisance dimensions, the identity checks of P5, and the heatmap trial."""
    output = ExperimentOutput()
    locations = grid_locations()
    m = locations.shape[0]
    sigma2 = np.ones(m)
    n_grid = tuple(range(1, config.n_max + 1))
    reference_d = 2 if 2 in config.d_grid else min(config.d_grid)
    order = [reference_d] + sorted(d for d in config.d_grid if d != reference_d)
    for spacing in config.spacing_grid:
        reference: dict[str, np.ndarray] = {}
        for d in order:
            started = time.perf_counter()
            mus = localization_means(spacing, d)
            plane_mus = np.ascontiguousarray(mus[:, :2])
            q = np.empty((config.trials, config.n_max, m))
            q_plane = np.empty_like(q)
            truth = np.empty(config.trials, dtype=np.int64)
            for trial in range(config.trials):
                truth[trial], x = localization_trial(base_seed, "e3", trial, spacing, d, config.n_max)
                q[trial] = quadratic_forms(x, mus, sigma2)
                q_plane[trial] = quadratic_forms(np.ascontiguousarray(x[:, :2]), plane_mus, sigma2)
            cell_key = f"s={spacing:g}|d={d}"
            cell = {"spacing": spacing, "d": d}
            base_key = {"experiment": "e3", "cell_key": cell_key, "cell": cell, "contamination": CLEAN.key}
            exact = _log_posterior("exact", q, sigma2, d)
            if d == reference_d:
                reference["exact"] = np.exp(exact)
                reference["typ_normal"] = np.exp(_log_posterior("typ_normal", q, sigma2, d))
            else:
                difference = np.max(np.abs(np.exp(exact) - reference["exact"]))
                output.cells.append(
                    _check_row(
                        {**base_key, "scorer": "exact"},
                        "exact_identical_across_d",
                        difference,
                        EXACT_ACROSS_D_THRESHOLD,
                        reference_d=reference_d,
                    )
                )
            for scorer in config.scorers:
                log_post = exact if scorer == "exact" else _log_posterior(scorer, q, sigma2, d, q_plane=q_plane)
                key = {**base_key, "scorer": scorer}
                metric_rows, cell_rows = _scorer_rows(
                    key, log_post, truth, n_grid, exact_log_post=exact, locations=locations
                )
                output.metrics.extend(metric_rows)
                output.cells.extend(cell_rows)
                if content_oracle_base(scorer) == "typ_normal" and reference_d == 2:
                    difference = np.max(np.abs(np.exp(log_post) - reference["typ_normal"]))
                    output.cells.append(
                        _check_row(key, "content_oracle_matches_typ_normal_at_d2", difference, ORACLE_THRESHOLD)
                    )
                if (
                    spacing == config.heatmap_spacing
                    and d in config.heatmap_d
                    and scorer in config.heatmap_scorers
                    and config.heatmap_trial < config.trials
                ):
                    trial = config.heatmap_trial
                    for n in config.heatmap_n:
                        if n > config.n_max:
                            continue
                        output.heatmaps.append(
                            {
                                "spacing": spacing,
                                "d": d,
                                "trial": trial,
                                "scorer": scorer,
                                "n": n,
                                "true_index": int(truth[trial]),
                                "true_cell": [int(v) for v in locations[truth[trial]]],
                                "posterior": np.exp(log_post[trial, n - 1]).reshape(21, 21).tolist(),
                            }
                        )
            progress(f"e3 s={spacing:g} d={d}: {time.perf_counter() - started:.1f}s")
    return output


def _candidate_metrics(
    scheme: BinScheme,
    abs_t: dict[str, np.ndarray],
    d: int,
    config: E5Config,
) -> tuple[dict, dict]:
    """Mean and SE of clean bits(objective_n) and contaminated acc(constraint_n) for one scheme."""
    masses = log_masses(scheme, d)
    results = []
    for kind, n, metric in (("clean", config.objective_n, "bits"), ("contaminated", config.constraint_n, "acc")):
        scores = masses[bin_index(scheme, abs_t[kind], d)]
        log_post = cumulative_log_posterior(scores[:, :n, :], uniform_log_prior(MEAN_SHIFT_HYPOTHESES))
        values = metrics(log_post, 0, uniform_log_prior(MEAN_SHIFT_HYPOTHESES))[metric][:, n - 1]
        mean, se = mean_and_se(values)
        results.append({"mean": _number(mean), "se": None if se is None else _number(se)})
    return results[0], results[1]


def run_e5(config: E5Config, base_seed: int, progress: Progress = print_progress) -> ExperimentOutput:
    """Search absolute bin edges on the calibration setting, then evaluate the chosen schemes on held-out cells."""
    output = ExperimentOutput()
    started = time.perf_counter()
    d = config.calibration_d
    delta = config.calibration_r * separation_threshold(d)
    cell_key = f"d={d}|r={config.calibration_r:g}"
    contaminations = (CLEAN, config.calibration_contamination)
    q = _mean_shift_q(base_seed, "e5_calibration", cell_key, config.calibration_trials, d, delta,
                      config.calibration_n_max, contaminations)
    abs_t = {
        "clean": typicality(q[CLEAN.key], d),
        "contaminated": typicality(q[config.calibration_contamination.key], d),
    }
    calibration = {
        "experiment": "e5",
        "cell_key": cell_key,
        "cell": {"d": d, "r": config.calibration_r, "delta": delta},
        "contamination": config.calibration_contamination.key,
        "trials": config.calibration_trials,
    }
    best: dict[int, tuple[float, bool, BinScheme, dict, dict]] = {}
    for k in config.k_grid:
        for edges in itertools.combinations(config.edge_grid, k - 1):
            scheme = absolute_scheme(edges)
            clean, contaminated = _candidate_metrics(scheme, abs_t, d, config)
            feasible = contaminated["mean"] >= config.min_accuracy
            output.cells.append(
                {
                    **calibration,
                    "kind": "candidate",
                    "scheme": scheme.name,
                    "K": k,
                    "edges": list(edges),
                    "clean_bits": clean,
                    "contaminated_acc": contaminated,
                    "objective_n": config.objective_n,
                    "constraint_n": config.constraint_n,
                    "feasible": feasible,
                }
            )
            rank = (feasible, clean["mean"])
            if k not in best or rank > (best[k][1], best[k][0]):
                best[k] = (clean["mean"], feasible, scheme, clean, contaminated)
    for name in ("abs2", "abs3", "signed5", "equalmass2", "equalmass3", "equalmass4"):
        scheme = scheme_by_name(name, d)
        clean, contaminated = _candidate_metrics(scheme, abs_t, d, config)
        output.cells.append(
            {
                **calibration,
                "kind": "named_scheme",
                "scheme": name,
                "K": scheme.bin_count,
                "edges": list(scheme.edges),
                "clean_bits": clean,
                "contaminated_acc": contaminated,
                "objective_n": config.objective_n,
                "constraint_n": config.constraint_n,
                "feasible": contaminated["mean"] >= config.min_accuracy,
            }
        )
    chosen: dict[str, BinScheme] = {}
    for k, (_, feasible, scheme, clean, contaminated) in sorted(best.items()):
        chosen[f"opt{k}"] = BinScheme(f"opt{k}", "abs", scheme.edges)
        output.cells.append(
            {
                **calibration,
                "kind": "choice",
                "scheme": f"opt{k}",
                "K": k,
                "edges": list(scheme.edges),
                "clean_bits": clean,
                "contaminated_acc": contaminated,
                "objective_n": config.objective_n,
                "constraint_n": config.constraint_n,
                "feasible": feasible,
                "flag": None if feasible else f"no K={k} candidate reached acc({config.constraint_n}) >= "
                f"{config.min_accuracy}; this is the maximum of the objective",
            }
        )
    progress(f"e5 calibration ({len(output.cells)} schemes): {time.perf_counter() - started:.1f}s")
    heldout_e1 = config.heldout_e1
    output.extend(
        _run_mean_shift(
            "e5_heldout",
            "e1",
            base_seed,
            [(d1, r) for d1 in heldout_e1.d_grid for r in heldout_e1.r_grid],
            (CLEAN,),
            trials=heldout_e1.trials,
            n_max=heldout_e1.n_max,
            scorers=heldout_e1.scorers,
            progress=progress,
            schemes=chosen,
        )
    )
    heldout_e4 = config.heldout_e4
    output.extend(
        _run_mean_shift(
            "e5_heldout",
            "e4",
            base_seed,
            [(d4, heldout_e4.r) for d4 in heldout_e4.d_grid],
            heldout_e4.contaminations,
            trials=heldout_e4.trials,
            n_max=heldout_e4.n_max,
            scorers=heldout_e4.scorers,
            progress=progress,
            schemes=chosen,
        )
    )
    return output


RUNNERS: dict[str, Callable[..., ExperimentOutput]] = {
    "e0": run_e0,
    "e1": run_e1,
    "e2": run_e2,
    "e3": run_e3,
    "e4": run_e4,
    "e5": run_e5,
}


def _git_state(root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit.strip() if commit else None,
        "dirty": None if status is None else bool(status.strip()),
        "dirty_paths": [] if not status else [line[3:] for line in status.splitlines() if line.strip()],
    }


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "scipy", "matplotlib", "robust-likelihood"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["numpy_runtime"] = np.__version__
    return versions


def _utc(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def run(
    preset: Preset,
    experiments: tuple[str, ...],
    *,
    results_root: Path | None = None,
    now: datetime | None = None,
    progress: Progress = print_progress,
    write_report: bool = True,
) -> Path:
    """Run the chosen experiments of a preset into results/gaussian/<UTC timestamp>-<preset>/ and return that folder."""
    root = repo_root()
    started_at = now or datetime.now(timezone.utc)
    parent = results_root or (root / "results" / "gaussian")
    run_dir = parent / f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{preset.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {"preset": preset_config(preset), "experiments": list(experiments)})
    manifest = {
        "preset": preset.name,
        "base_seed": preset.base_seed,
        "experiments": list(experiments),
        "git": _git_state(root),
        "versions": _versions(),
        "thread_limits": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "started_at": _utc(started_at),
        "finished_at": None,
        "wall_clock_seconds": {},
        "rows": {},
        "quantity": "Exact Gaussian computations with Monte Carlo observations. No language-model scores.",
    }
    write_json(run_dir / "manifest.json", manifest)
    clock = time.perf_counter()
    for name in experiments:
        progress(f"{name}: started")
        began = time.perf_counter()
        output = RUNNERS[name](getattr(preset, name), preset.base_seed, progress)
        append_jsonl(run_dir / "metrics.jsonl", output.metrics)
        append_jsonl(run_dir / "cells.jsonl", output.cells)
        if output.heatmaps:
            write_json(run_dir / "e3_heatmaps.json", {"heatmaps": output.heatmaps})
        manifest["wall_clock_seconds"][name] = round(time.perf_counter() - began, 3)
        manifest["rows"][name] = {"metrics": len(output.metrics), "cells": len(output.cells)}
        write_json(run_dir / "manifest.json", manifest)
        progress(f"{name}: {manifest['wall_clock_seconds'][name]:.1f}s")
    manifest["wall_clock_seconds"]["total"] = round(time.perf_counter() - clock, 3)
    manifest["finished_at"] = _utc(datetime.now(timezone.utc))
    write_json(run_dir / "manifest.json", manifest)
    if write_report:
        from robust_likelihood.gaussian.report import write_report as build_report

        build_report(run_dir)
    return run_dir
