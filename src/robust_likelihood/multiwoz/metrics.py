"""Size-principle metrics for elicited scores. The definitions are frozen in the plan.

Jev's yes-probability s is read on the logit scale, l = logit(clip(s, eps, 1 - eps)).
When the narrower hypothesis scores higher, logit gaps are at least as large as log
gaps, so the logit reading is the generous one for Jev and any shortfall is conservative.
"""

from __future__ import annotations

import math

EPSILON = 1e-3


def clip(score: float, epsilon: float = EPSILON) -> float:
    return min(1.0 - epsilon, max(epsilon, score))


def is_clipped(score: float, epsilon: float = EPSILON) -> bool:
    return score < epsilon or score > 1.0 - epsilon


def logit_score(score: float, epsilon: float = EPSILON) -> float:
    value = clip(score, epsilon)
    return math.log(value / (1.0 - value))


def log_score(score: float, epsilon: float = EPSILON) -> float:
    return math.log(clip(score, epsilon))


def empirical_log_ratio(removed: list[str], status: dict[str, str], terms: dict[str, dict]) -> float | None:
    """L_E: the sum over removed slots of ln(a/b) if the turn names the slot, else ln((1-a)/(1-b))."""
    total = 0.0
    for slot in removed:
        kind = status[slot]
        if kind not in ("named", "omitted") or terms[slot][kind] is None:
            return None
        total += terms[slot][kind]
    return total


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
