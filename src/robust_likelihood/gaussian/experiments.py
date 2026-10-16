"""Experiments E0-E5 of the Gaussian typicality test: configuration and presets.

A preset fixes every grid, trial count, and the base seed. ``quick`` is a smoke run
for a laptop. ``full`` is the reportable run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from robust_likelihood.gaussian.sampling import DEFAULT_BASE_SEED, Contamination

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
