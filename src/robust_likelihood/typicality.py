"""Formal typicality for a known distribution, kept apart from elicited wording.

Typicality of an observation x under hypothesis H is the signed gap

    delta(x|H) = log p(x|H) - E[log p(X|H)].

The expectation is over X drawn from H. A point can have higher probability
than that average and still sit outside the typical set. For a Gaussian the
gap depends only on the standardized residual, so the mode is 0.5 nats above
the average and the points at one standard deviation sit on the average.

A bin is usable as Bayesian evidence only when the same subset of outcomes is
scored under every hypothesis. Bins whose boundaries move with the hypothesis
being scored are a different event for each hypothesis. Their probabilities
are not a likelihood for one shared observation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

LOG_TWO_PI = math.log(2.0 * math.pi)
FIXED_BIN_EDGES = (float("-inf"), -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, float("inf"))
PRIMARY_EPSILON_NATS = 0.25

OPEN_SPECIFICATION = (
    "The reference distribution over utterances is not specified.",
    "The output scale is not chosen: signed log-probability gap, absolute distance, or a yes/no within epsilon.",
    "Bin thresholds are not agreed. The Gaussian checks use a stated epsilon and report neighboring values.",
    "Jev returns a yes-probability for one question. That number is not a typicality regression.",
)


class TypicalitySpecificationError(ValueError):
    """Raised when a formal typicality question is missing a required choice."""


@dataclass(frozen=True)
class GaussianHypothesis:
    name: str
    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0:
            raise ValueError("sd must be positive")


def gaussian_log_density(x: float, hypothesis: GaussianHypothesis) -> float:
    residual = (x - hypothesis.mean) / hypothesis.sd
    return -0.5 * LOG_TWO_PI - math.log(hypothesis.sd) - 0.5 * residual * residual


def expected_log_density(hypothesis: GaussianHypothesis) -> float:
    """E[log p(X|H)] for X ~ H. Equals -0.5 * log(2 * pi * sd^2) - 0.5."""
    return -0.5 * LOG_TWO_PI - math.log(hypothesis.sd) - 0.5


def signed_deviation(x: float, hypothesis: GaussianHypothesis) -> float:
    """Log-density minus its expectation under the same hypothesis, in nats."""
    residual = (x - hypothesis.mean) / hypothesis.sd
    return 0.5 * (1.0 - residual * residual)


def absolute_deviation(x: float, hypothesis: GaussianHypothesis) -> float:
    return abs(signed_deviation(x, hypothesis))


def normal_cdf(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    z = (x - mean) / sd
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def deviation_bin(delta: float, epsilon: float) -> str:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta > epsilon:
        return "above"
    if delta < -epsilon:
        return "below"
    return "near"


def pivotal_bin_probability(kind: str, epsilon: float) -> float:
    """P(bin | H) for every Gaussian H.

    The standardized residual is standard normal under every Gaussian, and the
    signed gap is a function of that residual alone. The three bin probabilities
    therefore do not depend on the mean or the variance.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if kind not in {"above", "near", "below"}:
        raise ValueError(kind)
    above_radius_sq = 1.0 - 2.0 * epsilon
    below_radius = math.sqrt(1.0 + 2.0 * epsilon)
    if above_radius_sq <= 0:
        above = 0.0
    else:
        above = 2.0 * normal_cdf(math.sqrt(above_radius_sq)) - 1.0
    below = 2.0 * (1.0 - normal_cdf(below_radius))
    near = 1.0 - above - below
    return {"above": above, "near": near, "below": below}[kind]


def fixed_bin_index(x: float, edges: tuple[float, ...] = FIXED_BIN_EDGES) -> int:
    if len(edges) < 2:
        raise ValueError("edges must contain at least one interval")
    for index in range(len(edges) - 1):
        if x <= edges[index + 1]:
            return index
    return len(edges) - 2


def fixed_bin_probability(
    hypothesis: GaussianHypothesis,
    index: int,
    edges: tuple[float, ...] = FIXED_BIN_EDGES,
) -> float:
    low = edges[index]
    high = edges[index + 1]
    upper = 1.0 if math.isinf(high) else normal_cdf(high, hypothesis.mean, hypothesis.sd)
    lower = 0.0 if math.isinf(low) else normal_cdf(low, hypothesis.mean, hypothesis.sd)
    return max(0.0, upper - lower)


def _normalize(log_weights: list[float]) -> list[float]:
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    total = sum(weights)
    return [weight / total for weight in weights]


def posterior_from_log_likelihoods(log_likelihoods: list[float], log_prior: list[float] | None = None) -> list[float]:
    if log_prior is None:
        log_prior = [0.0] * len(log_likelihoods)
    if len(log_prior) != len(log_likelihoods):
        raise ValueError("prior and likelihoods must have the same length")
    return _normalize([lik + prior for lik, prior in zip(log_likelihoods, log_prior, strict=True)])


def raw_posterior(x: float, hypotheses: list[GaussianHypothesis]) -> list[float]:
    return posterior_from_log_likelihoods([gaussian_log_density(x, hypothesis) for hypothesis in hypotheses])


def fixed_bin_posterior(
    x: float,
    hypotheses: list[GaussianHypothesis],
    edges: tuple[float, ...] = FIXED_BIN_EDGES,
) -> list[float]:
    """Likelihood of one shared bin. The bin edges do not depend on the hypothesis."""
    index = fixed_bin_index(x, edges)
    return posterior_from_log_likelihoods(
        [math.log(fixed_bin_probability(hypothesis, index, edges)) for hypothesis in hypotheses]
    )


def hypothesis_dependent_bin_posterior(
    x: float,
    hypotheses: list[GaussianHypothesis],
    epsilon: float = PRIMARY_EPSILON_NATS,
) -> list[float]:
    """Probability of whichever typicality bin contains x under that same hypothesis.

    This is not a likelihood for a shared observation. Each hypothesis contributes
    the probability of its own event.
    """
    log_likelihoods = []
    for hypothesis in hypotheses:
        kind = deviation_bin(signed_deviation(x, hypothesis), epsilon)
        log_likelihoods.append(math.log(pivotal_bin_probability(kind, epsilon)))
    return posterior_from_log_likelihoods(log_likelihoods)


def map_label(names: list[str], probabilities: list[float]) -> str:
    best = max(probabilities)
    winners = [name for name, probability in zip(names, probabilities, strict=True) if abs(probability - best) <= 1e-12]
    if len(winners) != 1:
        return "tie"
    return winners[0]


def total_variation(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _kl_from_prior(posterior: list[float]) -> float:
    prior = 1.0 / len(posterior)
    total = 0.0
    for probability in posterior:
        if probability > 0:
            total += probability * math.log(probability / prior)
    return total


def _sample(hypothesis: GaussianHypothesis, generator: random.Random) -> float:
    return generator.gauss(hypothesis.mean, hypothesis.sd)


def point_comparison(
    x: float,
    hypotheses: list[GaussianHypothesis],
    epsilon: float = PRIMARY_EPSILON_NATS,
) -> dict:
    names = [hypothesis.name for hypothesis in hypotheses]
    raw = raw_posterior(x, hypotheses)
    fixed = fixed_bin_posterior(x, hypotheses)
    dependent = hypothesis_dependent_bin_posterior(x, hypotheses, epsilon)
    return {
        "x": x,
        "epsilon_nats": epsilon,
        "signed_deviation": {hypothesis.name: signed_deviation(x, hypothesis) for hypothesis in hypotheses},
        "deviation_bin": {
            hypothesis.name: deviation_bin(signed_deviation(x, hypothesis), epsilon) for hypothesis in hypotheses
        },
        "raw_posterior": dict(zip(names, raw, strict=True)),
        "fixed_bin_posterior": dict(zip(names, fixed, strict=True)),
        "hypothesis_dependent_bin_posterior": dict(zip(names, dependent, strict=True)),
        "raw_map": map_label(names, raw),
        "fixed_bin_map": map_label(names, fixed),
        "hypothesis_dependent_bin_map": map_label(names, dependent),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def simulate_known_class(
    hypotheses: list[GaussianHypothesis],
    *,
    draws: int,
    seed: int,
    epsilon: float = PRIMARY_EPSILON_NATS,
) -> dict:
    """Draw H, then X from H. Compare posteriors when the class contains the truth."""
    generator = random.Random(seed)
    names = [hypothesis.name for hypothesis in hypotheses]
    raw_kl = []
    fixed_kl = []
    fixed_tv = []
    dependent_tv = []
    fixed_agree = 0
    dependent_agree = 0
    by_source = {
        name: {"n": 0, "raw_correct": 0, "fixed_correct": 0, "dependent_correct": 0, "raw_mass": []}
        for name in names
    }
    for _ in range(draws):
        hypothesis = hypotheses[generator.randrange(len(hypotheses))]
        x = _sample(hypothesis, generator)
        raw = raw_posterior(x, hypotheses)
        fixed = fixed_bin_posterior(x, hypotheses)
        dependent = hypothesis_dependent_bin_posterior(x, hypotheses, epsilon)
        raw_map = map_label(names, raw)
        raw_kl.append(_kl_from_prior(raw))
        fixed_kl.append(_kl_from_prior(fixed))
        fixed_tv.append(total_variation(fixed, raw))
        dependent_tv.append(total_variation(dependent, raw))
        if map_label(names, fixed) == raw_map:
            fixed_agree += 1
        if map_label(names, dependent) == raw_map:
            dependent_agree += 1
        bucket = by_source[hypothesis.name]
        bucket["n"] += 1
        bucket["raw_mass"].append(raw[names.index(hypothesis.name)])
        if raw_map == hypothesis.name:
            bucket["raw_correct"] += 1
        if map_label(names, fixed) == hypothesis.name:
            bucket["fixed_correct"] += 1
        if map_label(names, dependent) == hypothesis.name:
            bucket["dependent_correct"] += 1
    sources = []
    for name in names:
        bucket = by_source[name]
        count = bucket["n"]
        sources.append(
            {
                "source": name,
                "n": count,
                "raw_map_rate": bucket["raw_correct"] / count,
                "fixed_bin_map_rate": bucket["fixed_correct"] / count,
                "dependent_bin_map_rate": bucket["dependent_correct"] / count,
                "mean_raw_posterior_on_source": _mean(bucket["raw_mass"]),
            }
        )
    return {
        "draws": draws,
        "seed": seed,
        "epsilon_nats": epsilon,
        "mutual_information_raw_nats": _mean(raw_kl),
        "mutual_information_fixed_bin_nats": _mean(fixed_kl),
        "mean_tv_fixed_bin_vs_raw": _mean(fixed_tv),
        "mean_tv_dependent_bin_vs_raw": _mean(dependent_tv),
        "map_agreement_fixed_bin": fixed_agree / draws,
        "map_agreement_dependent_bin": dependent_agree / draws,
        "by_source": sources,
    }


def simulate_contamination(
    *,
    draws: int,
    seed: int,
    epsilon: float = PRIMARY_EPSILON_NATS,
    contamination: float = 0.1,
    outlier_sd: float = 10.0,
    outlier_abs: float = 3.0,
) -> dict:
    """Data are a contaminated narrow Gaussian. The model class is narrow versus wide.

    The hypothesis-dependent bin probability is the same for every Gaussian, so
    an extreme point that both hypotheses call 'below' produces no update.
    """
    narrow = GaussianHypothesis("narrow", 0.0, 1.0)
    wide = GaussianHypothesis("wide", 0.0, 2.0)
    hypotheses = [narrow, wide]
    generator = random.Random(seed)
    outlier_raw = []
    outlier_fixed = []
    outlier_dependent = []
    outliers = 0
    for _ in range(draws):
        if generator.random() < contamination:
            x = generator.gauss(0.0, outlier_sd)
        else:
            x = generator.gauss(0.0, 1.0)
        if abs(x) <= outlier_abs:
            continue
        outliers += 1
        outlier_raw.append(raw_posterior(x, hypotheses)[0])
        outlier_fixed.append(fixed_bin_posterior(x, hypotheses)[0])
        outlier_dependent.append(hypothesis_dependent_bin_posterior(x, hypotheses, epsilon)[0])
    return {
        "draws": draws,
        "seed": seed,
        "epsilon_nats": epsilon,
        "contamination": contamination,
        "outlier_sd": outlier_sd,
        "outlier_rule": f"|x| > {outlier_abs}",
        "outlier_count": outliers,
        "mean_posterior_on_narrow_among_outliers": {
            "raw": _mean(outlier_raw),
            "fixed_bin": _mean(outlier_fixed),
            "hypothesis_dependent_bin": _mean(outlier_dependent),
        },
    }


def gaussian_summary(draws: int = 6000, seed: int = 20261007) -> dict:
    narrow = GaussianHypothesis("narrow", 0.0, 1.0)
    wide = GaussianHypothesis("wide", 0.0, 2.0)
    shift = GaussianHypothesis("shift", 2.0, 1.0)
    pair = [narrow, wide]
    triple = [narrow, wide, shift]
    points = [point_comparison(x, pair) for x in (0.0, 1.0, 10.0)]
    epsilon_maps = []
    for epsilon in (0.1, 0.25, 0.5):
        row = point_comparison(1.0, pair, epsilon)
        epsilon_maps.append(
            {
                "epsilon_nats": epsilon,
                "narrow_bin": row["deviation_bin"]["narrow"],
                "wide_bin": row["deviation_bin"]["wide"],
                "raw_map": row["raw_map"],
                "fixed_bin_map": row["fixed_bin_map"],
                "hypothesis_dependent_bin_map": row["hypothesis_dependent_bin_map"],
            }
        )
    return {
        "quantity": "Exact Gaussian probabilities. These are not elicited language-model scores.",
        "signed_deviation_definition": "log p(x|H) - E[log p(X|H)], in nats",
        "fixed_bin_edges": list(FIXED_BIN_EDGES),
        "primary_epsilon_nats": PRIMARY_EPSILON_NATS,
        "pivotal_bin_probabilities": {
            kind: pivotal_bin_probability(kind, PRIMARY_EPSILON_NATS) for kind in ("above", "near", "below")
        },
        "two_hypothesis_points": points,
        "epsilon_at_x_equals_1": epsilon_maps,
        "clean_class": simulate_known_class(triple, draws=draws, seed=seed),
        "contamination": simulate_contamination(draws=draws, seed=seed + 1),
        "open_specification": list(OPEN_SPECIFICATION),
    }


DIAGNOSTIC_PROBES = (
    {
        "example_id": "diag-empty",
        "role": "empty",
        "relation": "no utterance",
        "utt": "",
    },
    {
        "example_id": "diag-short-a",
        "role": "very_short",
        "relation": "one character",
        "utt": "a",
    },
    {
        "example_id": "diag-short-ok",
        "role": "very_short",
        "relation": "two characters",
        "utt": "ok",
    },
    {
        "example_id": "diag-ordinary",
        "role": "ordinary",
        "relation": "baseline sentence",
        "utt": "Set an alarm for seven.",
    },
    {
        "example_id": "diag-paraphrase",
        "role": "meaning_preserving",
        "relation": "rewording of the baseline sentence",
        "utt": "Please create an alarm at 7:00.",
    },
    {
        "example_id": "diag-repetition",
        "role": "repetition",
        "relation": "the baseline sentence repeated",
        "utt": "Set an alarm for seven. Set an alarm for seven.",
    },
    {
        "example_id": "diag-truncated",
        "role": "information_removed",
        "relation": "the time is removed from the baseline sentence",
        "utt": "Set an alarm.",
    },
    {
        "example_id": "diag-irrelevant",
        "role": "irrelevant_extension",
        "relation": "an unrelated clause is added to the baseline sentence",
        "utt": "Set an alarm for seven. The capital of France is Paris.",
    },
)


def diagnostic_items() -> list[dict]:
    """Probes of length and added or removed material.

    No reference distribution is specified, so these rows have no correct
    typicality label. Short and empty strings are included as behavior checks.
    """
    rows = []
    for probe in DIAGNOSTIC_PROBES:
        rows.append(
            {
                **probe,
                "dataset_version": "typicality_diagnostics",
                "reference_distribution": None,
                "typicality_label": None,
                "log_probability_label": None,
                "intended_use": "model behavior only",
            }
        )
    return rows


COMPARISON_SLICE = (
    ("direct_request", "synthetic_v2", "v2-op-delete"),
    ("quoted_command", "synthetic_v2", "v2-quote-delete"),
    ("negation", "synthetic_families", "negation-01__canonical"),
    ("ordered_plan", "synthetic_v2", "v2-plan-delete-then-create"),
)


def comparison_slice() -> list[dict]:
    """Fixed alarm utterances for a later likely-versus-formal-typicality run.

    The slice keeps the existing hypotheses and sentences. It does not assign
    typicality labels, and it is not an approval to send the formal question.
    """
    from robust_likelihood.synthetic_families_items import authored_items as family_items
    from robust_likelihood.synthetic_v2_items import authored_items as v2_items

    catalogs = {
        "synthetic_v2": {row["example_id"]: row for row in v2_items()},
        "synthetic_families": {row["example_id"]: row for row in family_items()},
    }
    rows = []
    for kind, source, example_id in COMPARISON_SLICE:
        item = catalogs[source][example_id]
        rows.append(
            {
                "kind": kind,
                "source": source,
                "example_id": example_id,
                "utt": item["utt"],
                "hypothesis_version": item["hypothesis_version"],
                "evaluation": item["evaluation"],
                "typicality_label": None,
                "status": "prepared_not_sent",
            }
        )
    return rows


def formal_typicality_question(
    *,
    scale: str,
    reference_distribution: str | None,
    epsilon_nats: float | None = None,
) -> str:
    """Build a question only after the reference distribution and scale are set.

    ``within_epsilon`` is the only scale that matches Jev's yes/no probability.
    The returned sentence is the question. The stored Jev number would still be
    the probability of yes, not the signed gap.
    """
    reference = (reference_distribution or "").strip()
    if not reference:
        raise TypicalitySpecificationError(
            "a reference distribution is required before a formal typicality question can be sent"
        )
    definition = (
        "Use this definition. Draw utterances from the reference distribution for the stated intention. "
        f"The reference distribution is: {reference} "
        "The signed typicality gap of an utterance is its log-probability under that distribution "
        "minus the average log-probability of draws from the same distribution, in nats. "
        "A positive gap means the utterance is more probable than an average draw. "
    )
    if scale == "signed_deviation":
        return definition + "Report the signed typicality gap of the stated utterance."
    if scale == "absolute_distance":
        return definition + "Report the absolute value of the signed typicality gap of the stated utterance."
    if scale == "within_epsilon":
        if epsilon_nats is None or epsilon_nats <= 0:
            raise TypicalitySpecificationError("within_epsilon requires a positive epsilon in nats")
        return (
            definition
            + f"Is the absolute signed typicality gap of the stated utterance at most {epsilon_nats} nats? "
            + "Answer about that yes-or-no question only."
        )
    raise TypicalitySpecificationError(
        "scale must be signed_deviation, absolute_distance, or within_epsilon"
    )
