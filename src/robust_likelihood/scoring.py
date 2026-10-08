import math

from robust_likelihood.constants import INTENTS, NLL_CLIP_FLOOR, PROBABILITY_SUM_TOLERANCE, YES_THRESHOLD


def is_probability(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return False
    return 0.0 <= number <= 1.0


def sums_to_one(total: float, tolerance: float = PROBABILITY_SUM_TOLERANCE) -> bool:
    return abs(total - 1.0) <= tolerance


def validate_distribution(raw: object) -> dict[str, float] | None:
    """Return the three intent probabilities unchanged, or None when malformed.

    Probabilities must lie in [0, 1] and sum to 1 within absolute tolerance.
    This function does not renormalize.
    """
    if not isinstance(raw, dict) or set(raw) != set(INTENTS):
        return None
    parsed: dict[str, float] = {}
    for intent in INTENTS:
        value = raw[intent]
        if not is_probability(value):
            return None
        parsed[intent] = float(value)
    if not sums_to_one(sum(parsed.values())):
        return None
    return parsed


def predict(scores: dict[str, float]) -> str:
    top = max(scores.values())
    winners = [key for key, value in scores.items() if value == top]
    if len(winners) == 1:
        return winners[0]
    return "tie"


def correct_rank(scores: dict[str, float], correct: str) -> int:
    truth = scores[correct]
    return 1 + sum(1 for value in scores.values() if value > truth)


def margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] == ordered[1]:
        return 0.0
    return ordered[0] - ordered[1]


def negative_log_likelihood(probability: float, floor: float = NLL_CLIP_FLOOR) -> float:
    clipped = min(1.0, max(floor, probability))
    return -math.log(clipped)


def multiclass_brier(scores: dict[str, float], correct: str) -> float:
    total = 0.0
    for key, probability in scores.items():
        outcome = 1.0 if key == correct else 0.0
        total += (probability - outcome) ** 2
    return total


def yes_label(probability: float, threshold: float = YES_THRESHOLD) -> str:
    if probability > threshold:
        return "yes"
    if probability < threshold:
        return "no"
    return "tie"
