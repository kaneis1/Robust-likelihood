"""Hypothesis families of the Gaussian test. Every hypothesis is isotropic: x | z ~ N(mu_z, sigma2_z I_d)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEAN_SHIFT_HYPOTHESES = 20
SCALE_SIGMA2 = (0.25, 0.5, 1.0, 2.0, 4.0)
GRID_SIDE = 21
GRID_MARGIN = 2


@dataclass(frozen=True, eq=False)
class Family:
    """Means (M, d), variances (M,), and the index of the true hypothesis."""

    mus: np.ndarray
    sigma2: np.ndarray
    true_index: int


def separation_threshold(d: int) -> float:
    """delta*(d) = (2 d)^(1/4): at this separation a wrong hypothesis shifts T by one unit on average."""
    return (2.0 * d) ** 0.25


def random_unit_vectors(rng: np.random.Generator, count: int, d: int) -> np.ndarray:
    """count independent directions, uniform on the unit sphere of R^d. Returns (count, d)."""
    draws = rng.standard_normal((count, d))
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def mean_shift_family(rng: np.random.Generator, d: int, delta: float, m: int = MEAN_SHIFT_HYPOTHESES) -> Family:
    """Hypothesis 0 is the truth, mu_0 = 0. Hypotheses 1 to m - 1 sit at delta * u_j for uniform unit u_j. sigma2 = 1."""
    mus = np.vstack([np.zeros((1, d)), delta * random_unit_vectors(rng, m - 1, d)])
    return Family(mus=mus, sigma2=np.ones(m), true_index=0)


def scale_sigma2() -> np.ndarray:
    """Variances of the scale family, which shares mean 0: (0.25, 0.5, 1, 2, 4)."""
    return np.array(SCALE_SIGMA2)


def scale_truth_index(trial: int) -> int:
    """The truth of a scale-family trial. It cycles over the hypotheses, so each is the truth equally often."""
    return trial % len(SCALE_SIGMA2)


def grid_locations(side: int = GRID_SIDE) -> np.ndarray:
    """(side^2, 2) grid positions (row, column) in grid units. Hypothesis index = row * side + column."""
    rows, columns = np.divmod(np.arange(side * side), side)
    return np.column_stack([rows, columns]).astype(np.float64)


def grid_plane_means(spacing: float, side: int = GRID_SIDE) -> np.ndarray:
    """(side^2, 2) in-plane means: the grid centered at the origin, with the given spacing between neighbors."""
    return spacing * (grid_locations(side) - (side - 1) / 2.0)


def localization_means(spacing: float, d: int, side: int = GRID_SIDE) -> np.ndarray:
    """(side^2, d) means: the grid in coordinates 1-2 of R^d and 0 in every other coordinate."""
    if d < 2:
        raise ValueError("the localization family needs d >= 2")
    mus = np.zeros((side * side, d))
    mus[:, :2] = grid_plane_means(spacing, side)
    return mus


def draw_true_cell(rng: np.random.Generator, side: int = GRID_SIDE, margin: int = GRID_MARGIN) -> int:
    """A grid index drawn uniformly from the cells at least margin cells from the border."""
    row, column = rng.integers(margin, side - margin, size=2)
    return int(row) * side + int(column)
