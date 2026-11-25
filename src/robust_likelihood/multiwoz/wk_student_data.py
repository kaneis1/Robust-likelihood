"""Week of 24 Nov, Part D (docs/week_nov24_plan.md 5.1-5.3): the teacher target and the student's data.

Teacher [Frozen at STOP 3]: typicality with qwen3_base·F as the reference, centered statistic
s(x | h) = S(x | h) - S(x | broad|impl), normalized globally, T = (s - H) / sigma, with H and sigma the
pool statistics of cent1B (463 clean texts, uniform weights). The regression target is T clipped to the
pooled 0.5% and 99.5% quantiles of T; the classification target is T's bin under the chosen scheme.

Training data is synthetic only: the model's own Phase 2 samples (matched pairs, all 64 per text) and
their cross pairs (xscore: the first 32 per text scored under every other text of the same item).
Samples that were cut off or are empty are left out. Split [Frozen]: 20% of the food values of the
clean gold goals, seeded, are held out; every text that names one and every sample whose regex slot
tuple names one leave training; 5% of the remaining samples (a sample's matched and cross pairs
together) are validation.
"""

from __future__ import annotations

import hashlib
import random

from robust_likelihood.multiwoz.typv0 import bin_index, centres

SEED = 20261124
HELD_OUT_SHARE = 0.2
VALIDATION_SHARE = 0.05
CLIP = (0.005, 0.995)


def held_out_foods(items: list[dict], seed: int = SEED, share: float = HELD_OUT_SHARE) -> list[str]:
    foods = sorted({item["goal"]["food"] for item in items if item["goal"].get("food")})
    return sorted(random.Random(f"{seed}:held-out-foods").sample(foods, round(share * len(foods))))


def scheme_for(choice: str, pooled: list[tuple[float, float]], mean: float, spread: float, typ: dict) -> dict:
    """Edges, bin centres (T units), clip bounds and pool statistics of the chosen bin scheme."""
    from robust_likelihood.multiwoz.wk_quantile import quantile_scheme, weighted_quantile

    if choice == "cent1Q":
        block = typ["quantile"]
        found = quantile_scheme(pooled, block["variants"]["cent1Q"]["quantiles"], block["top_centre_quantile"])
        edges, centre = found["edges"], found["centres"]
    elif choice == "cent1B:6bins":
        edges = list(typ["sensitivity"]["edges"]["6bins"])
        centre = centres(tuple(edges))
    else:
        raise ValueError(choice)
    return {"choice": choice, "H": mean, "sigma": spread, "edges": edges, "centres": centre,
            "clip": [weighted_quantile(pooled, CLIP[0]), weighted_quantile(pooled, CLIP[1])]}


def target(s: float, scheme: dict) -> dict:
    t = (s - scheme["H"]) / scheme["sigma"]
    low, high = scheme["clip"]
    return {"T": t, "T_clip": min(max(t, low), high), "bin": bin_index(t, tuple(scheme["edges"]), 1)}


def sample_split(key: str, seed: int = SEED) -> str:
    value = int(hashlib.sha256(f"{seed}:split:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < VALIDATION_SHARE else "train"


def build_pairs(samples: list[dict], xscore: list[dict], pool: set[str], held_out: set[str], extractor,
                scheme: dict, seed: int = SEED) -> tuple[list[dict], dict]:
    """Every matched and cross pair with its teacher target and split (train, val, or held_out_food)."""
    names_food: dict[str, bool] = {}

    def held(text: str) -> bool:
        if text not in names_food:
            names_food[text] = extractor.extract(text)["food"] in held_out
        return names_food[text]

    usable = {}
    counts = {"samples": 0, "cut_off": 0, "empty": 0}
    for row in samples:
        if row["text"] not in pool:
            continue
        counts["samples"] += 1
        if row.get("S") is None:
            counts["cut_off"] += 1
            continue
        if not row["reply"].strip():
            counts["empty"] += 1
            continue
        usable[(row["text"], row["sample"])] = row
    pairs = []

    def add(text_from: str, sample: int, text: str, s_value: float, kind: str) -> None:
        row = usable[(text_from, sample)]
        key = f"{text_from}\x1f{sample}"
        food_out = held(text_from) or held(text) or held(row["reply"])
        split = "held_out_food" if food_out else sample_split(key, seed)
        pairs.append({"h": text, "x": row["reply"], "kind": kind, "text_from": text_from, "sample": sample,
                      "split": split, **target(s_value - row["S_broad"], scheme)})

    for (text, sample), row in sorted(usable.items()):
        add(text, sample, text, row["S"], "matched")
    missing = 0
    for row in xscore:
        if (row["text_from"], row["sample"]) not in usable:
            missing += 1
            continue
        add(row["text_from"], row["sample"], row["text"], row["S"], "cross")
    counts.update({"usable_samples": len(usable), "xscore_without_sample": missing,
                   "texts_naming_held_out_food": sum(1 for text in pool if held(text))})
    for kind in ("matched", "cross"):
        for split in ("train", "val", "held_out_food"):
            counts[f"{kind}_{split}"] = sum(1 for pair in pairs if pair["kind"] == kind and pair["split"] == split)
    return pairs, counts


def teacher(root, run_dir, choice: str, derived=None):
    """The scheme of the chosen teacher from qwen3_base's Phase 2 run, and typv0 Inputs on it."""
    from robust_likelihood.multiwoz.typv0 import Inputs, load_typv0_config, pool_weights
    from robust_likelihood.multiwoz.wk_quantile import pooled_t

    inputs = Inputs(root, run_dir, [run_dir], derived, model_key="qwen3_base")
    pooled, mean, spread = pooled_t(inputs, "cent", pool_weights(inputs.probe, "uniform"))
    return scheme_for(choice, pooled, mean, spread, load_typv0_config(root)), inputs


def human_targets(inputs, scheme: dict) -> dict[tuple[str, str], dict]:
    """(item, hyp_key) -> the teacher's target for every clean human pair (cells of the original turn)."""
    found = {}
    for cell in inputs.cells:
        item, key, hvar, uvar = cell["key"]
        if hvar == uvar == "original":
            found[(item, key)] = target(cell["cent"], scheme)
    return found
