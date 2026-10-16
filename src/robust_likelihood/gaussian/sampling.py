"""Seeding and the contaminations C0-C5 of the Gaussian test."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BASE_SEED = 20261019
CONTAMINATION_CODES = ("C0", "C1", "C2", "C3", "C4", "C5")


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
