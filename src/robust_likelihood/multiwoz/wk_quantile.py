"""Week of 24 Nov, Part C (docs/week_nov24_plan.md): typicality with quantile bins, from saved samples.

A quantile variant reads the statistic and the global normalization of typv0 (S or cent; T = (s - H)/sigma
over the pool of the 463 distinct clean hypothesis texts, uniform weights), and then:

- edges: the quantiles of T among the pool's own samples, each text weighted w(h) and its samples
  sharing that weight equally (the same mixture whose moments give H and sigma). One-sided.
  Quantiles 0.50, 0.70, 0.85, 0.95, 0.99, so the bins hold 50 / 20 / 15 / 10 / 4 / 1% of the samples.
- score (option B form): -(H + sigma * c_b), c_b the mean T of the pooled samples in bin b; in the top
  bin c_b is the pooled 0.995 quantile of T, so a turn beyond every sample gets a capped score.

A weighted quantile is the smallest value whose cumulative weight reaches q of the total. A value equal
to an edge goes to the bin above it, as typv0.bin_index does.
"""

from __future__ import annotations

from robust_likelihood.multiwoz.claim1 import ORIGINAL, ItemBoot
from robust_likelihood.multiwoz.typv0 import (
    Inputs,
    bin_index,
    moments,
    occupancy,
    pool_stats,
    score_map,
    size_and_information,
    variant_scores,
)

FIXED = ("glob1A", "glob1B", "cent1B", "cent1B:6bins")


def weighted_quantile(pairs: list[tuple[float, float]], q: float) -> float:
    """pairs of (value, weight); the smallest value with cumulative weight >= q * total."""
    ordered = sorted(pairs)
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= q * total - 1e-12:
            return value
    return ordered[-1][0]


def pooled_t(inputs: Inputs, statistic: str, weights: dict[str, float], pool: list[str] | None = None
             ) -> tuple[list[tuple[float, float]], float, float]:
    """(T, weight) of every pooled sample, with H and sigma."""
    pool = pool or inputs.pool
    found = {text: moments(inputs.samples[text][statistic]) for text in pool if text in inputs.samples}
    found = {text: moment for text, moment in found.items() if moment}
    mean, spread = pool_stats(found, weights)
    total = sum(weights[text] for text in found)
    pairs = []
    for text in found:
        values = inputs.samples[text][statistic]
        share = weights[text] / total / len(values)
        pairs += [((value - mean) / spread, share) for value in values]
    return pairs, mean, spread


def quantile_scheme(pairs: list[tuple[float, float]], quantiles: list[float], top: float) -> dict:
    edges = [weighted_quantile(pairs, q) for q in quantiles]
    bins = len(edges) + 1
    mass, total = [0.0] * bins, [0.0] * bins
    for value, weight in pairs:
        b = bin_index(value, tuple(edges), 1)
        mass[b] += weight
        total[b] += weight * value
    whole = sum(mass)
    centres = [total[b] / mass[b] if mass[b] > 0 else None for b in range(bins)]
    centres[-1] = weighted_quantile(pairs, top)
    return {"edges": edges, "centres": centres, "masses": [value / whole for value in mass]}


def quantile_scores(inputs: Inputs, spec: dict, weights: dict[str, float], cells: list[dict] | None = None,
                    scheme: dict | None = None) -> tuple[dict, dict, dict]:
    """log scores, bins, and the scheme (H, sigma, edges, centres, masses) of one quantile variant, in the
    shape typv0.variant_scores returns."""
    if scheme is None:
        pairs, mean, spread = pooled_t(inputs, spec["statistic"], weights)
        scheme = {"H": mean, "sigma": spread, **quantile_scheme(pairs, spec["quantiles"], spec["top_centre_quantile"])}
    edges = tuple(scheme["edges"])
    log, binned = {}, {}
    for cell in cells if cells is not None else inputs.cells:
        value = cell[spec["statistic"]]
        if value is None:
            continue
        b = bin_index((value - scheme["H"]) / scheme["sigma"], edges, 1)
        log[cell["key"]] = -(scheme["H"] + scheme["sigma"] * scheme["centres"][b])
        binned[cell["key"]] = b
    return log, binned, scheme


def specs(typ: dict) -> dict[str, dict]:
    """Every variant Part C compares: the fixed-edge ones of the simulator study and the quantile ones."""
    found = {name: dict(typ["variants"][name]) for name in ("glob1A", "glob1B", "cent1B")}
    found["cent1B:6bins"] = dict(typ["variants"]["cent1B"], edges=typ["sensitivity"]["edges"]["6bins"])
    block = typ["quantile"]
    for name, spec in block["variants"].items():
        found[name] = dict(spec, top_centre_quantile=block["top_centre_quantile"])
        for label, quantiles in block["sensitivity"].items():
            found[f"{name}:{label}"] = dict(found[name], quantiles=quantiles)
    return found


def scores_of(inputs: Inputs, spec: dict, weights: dict[str, float], cells: list[dict] | None = None):
    if spec.get("score") == "Q":
        return quantile_scores(inputs, spec, weights, cells)
    return variant_scores(inputs, spec, weights, cells=cells)


def agreement(bins: dict, pairs: list[tuple[tuple, tuple]], boot: ItemBoot) -> dict:
    """Share of (original key, changed key) pairs that keep their bin, item bootstrap."""
    ids, same = [], []
    for before, after in pairs:
        if before in bins and after in bins:
            ids.append(before[0])
            same.append(float(bins[before] == bins[after]))
    return boot.mean(ids, same)


def rewrite_pairs(bins: dict, kind: str, side: str) -> list[tuple[tuple, tuple]]:
    """(original, rewritten) cell keys of one claims rewrite kind, utterance or hypothesis side."""
    found = []
    for key in bins:
        item, hyp, hvar, uvar = key
        if side == "utterance" and hvar == ORIGINAL and uvar == kind:
            found.append(((item, hyp, ORIGINAL, ORIGINAL), key))
        if side == "hypothesis" and hvar == kind and uvar == ORIGINAL:
            found.append(((item, hyp, ORIGINAL, ORIGINAL), key))
    return found


UTTERANCE_REWRITES = ("para1", "para2", "typo", "pad", "verbose", "zh")
HYPOTHESIS_REWRITES = ("hpara1", "hpara2", "hpara3", "hzh")


def evaluate(inputs: Inputs, spec: dict, weights: dict[str, float], changed: dict, corpus_boot, boot: ItemBoot) -> dict:
    """The simulator study's frozen metrics for one variant with one reference model."""
    log, bins, pool = scores_of(inputs, spec, weights)
    count = len(spec["quantiles"]) + 1 if spec.get("score") == "Q" else len(spec["edges"]) + 1
    found = occupancy(bins, count)
    size = size_and_information(inputs.items, score_map(log), changed, corpus_boot)
    ratio = {}
    for regime in ("impl", "expl"):
        group = size["occam"].get(f"{regime}|slots_mean|named")
        ratio[regime] = None if not group or not group.get("L_E") else group["mean_L_J"] / group["L_E"]
    block = {"pool": {key: value for key, value in pool.items()}, "occupancy": found, "extreme_share": found[-1],
             "size_kept": ratio, "contra_log_bf": size["contra_mean_log_bf"], "size": size,
             "agreement": {}}
    for kind in UTTERANCE_REWRITES:
        block["agreement"][kind] = agreement(bins, rewrite_pairs(bins, kind, "utterance"), boot)
    para = rewrite_pairs(bins, "para1", "utterance") + rewrite_pairs(bins, "para2", "utterance")
    block["agreement"]["para"] = agreement(bins, para, boot)
    for kind in HYPOTHESIS_REWRITES:
        block["agreement"][kind] = agreement(bins, rewrite_pairs(bins, kind, "hypothesis"), boot)
    return block
