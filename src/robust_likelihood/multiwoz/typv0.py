"""Claims plan step 3: typicality v0, computed from the saved samples of step 2 (no new GPU work).

A variant reads one statistic s for an utterance under a hypothesis text h, normalizes it to T,
puts T in a bin, and scores the bin [Frozen definitions, configs/typicality_v0.json]:

- statistics: S = S(U|h) (Qwen surprisal, Phase 8 prompt); cent = S(U|h) - S(U|broad|impl);
  sem = -log P_sem(U|h), P_sem = (n + 0.5) / (K + 1) with n of h's K samples whose regex slot
  tuple matches U's 2.2 annotation; semf, the factorized sensitivity, multiplies per-slot
  (n_slot + 0.5) / (K + 1).
- normalization: per-hypothesis T = (s - mu_h) / sigma_h, or global T = (s - H) / sigma with
  H = sum_h w(h) mu_h and sigma^2 = sum_h w(h) [sigma_h^2 + (mu_h - H)^2] over the pool, the
  distinct hypothesis texts of the clean probe. mu_h and sigma_h (n - 1) are the mean and SD of
  the statistic over h's own samples.
- bins: two-sided on |T| or one-sided on T; a bin index counts the edges at or below the value.
- score A: log P_h(bin), the share of h's own samples in the bin with add-one smoothing,
  (count + 1) / (K + bins); score B: -(H + sigma * c_b), with c_b the bin's centre and the
  extreme bins capped half an interior width beyond the outer edges.

A sample's own statistic is held out of its P_sem: n counts the other K - 1 samples, so
P = (n + 0.5) / K, as for an utterance that is not among the samples. An utterance rewrite
keeps U's annotated slots (the rewrite validity check), so its sem statistic is U's.

Scores are keyed as in claim1.py, (item, hyp_key, hypothesis variant, utterance variant), and
the step 2 robustness metrics are claim1.py's own functions.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR
from robust_likelihood.multiwoz.claim1 import (
    H_KINDS,
    ORIGINAL,
    U_GROUPS,
    ItemBoot,
    Scorer,
    bf_pairs,
    clipped_log,
    item_meta,
    jev_scorers,
    posterior,
    read_lm_runs,
    rewording,
    token_log,
    utterance,
    verbal_scorers,
)
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.metrics import (
    Bootstrap,
    analyze_scorer,
    item_bootstrap,
    replicate_terms,
    term_vector,
)
from robust_likelihood.multiwoz.semantic import Extractor, annotated_tuple, load_vocab
from robust_likelihood.storage import read_json, read_jsonl, write_json

CONFIG_FILE = Path("configs") / "typicality_v0.json"
STATISTICS = ("S", "cent", "sem", "semf")
BROAD = "broad|impl"


def load_typv0_config(root: Path) -> dict:
    return read_json(root / CONFIG_FILE)


# ---------------------------------------------------------------- definitions


def bin_index(value: float, edges: tuple[float, ...], sides: int) -> int:
    probe = abs(value) if sides == 2 else value
    return sum(1 for edge in edges if probe >= edge)


def centres(edges: tuple[float, ...]) -> list[float]:
    """Option B's c_b: interior bin midpoints; extreme bins half an interior width beyond the outer edges."""
    if len(edges) == 1:
        return [edges[0] - 0.5, edges[0] + 0.5]
    inner = [(low + high) / 2 for low, high in zip(edges, edges[1:])]
    width = (edges[-1] - edges[0]) / (len(edges) - 1)
    return [edges[0] - width / 2] + inner + [edges[-1] + width / 2]


def moments(values: list[float]) -> tuple[float, float] | None:
    """Mean and sample SD (n - 1), as Phase 8 computes mu_h and sigma_h."""
    if len(values) < 2:
        return None
    return statistics.fmean(values), statistics.stdev(values)


def pool_stats(moments_by_text: dict[str, tuple[float, float]], weights: dict[str, float]) -> tuple[float, float]:
    total = sum(weights[text] for text in moments_by_text)
    share = {text: weights[text] / total for text in moments_by_text}
    mean = sum(share[text] * mu for text, (mu, _) in moments_by_text.items())
    spread = sum(share[text] * (sigma * sigma + (mu - mean) ** 2) for text, (mu, sigma) in moments_by_text.items())
    return mean, math.sqrt(spread)


def pool_weights(probe: list[dict], scheme: str) -> dict[str, float]:
    """uniform: each distinct text once; prior: each text as often as clean items carry it in their probe."""
    counts = Counter(row["text"] for row in probe)
    return {text: 1.0 if scheme == "uniform" else float(count) for text, count in counts.items()}


def matches(found: tuple, wanted: dict[str, list[str]]) -> bool:
    return all(value in wanted[slot] for slot, value in zip(SLOTS, found))


def p_sem(tuples: list[tuple], wanted: dict[str, list[str]]) -> float:
    return (sum(1 for found in tuples if matches(found, wanted)) + 0.5) / (len(tuples) + 1)


def p_sem_factorized(tuples: list[tuple], wanted: dict[str, list[str]]) -> float:
    found = 1.0
    for position, slot in enumerate(SLOTS):
        found *= (sum(1 for row in tuples if row[position] in wanted[slot]) + 0.5) / (len(tuples) + 1)
    return found


def held_out_sem(tuples: list[tuple]) -> tuple[list[float], list[float]]:
    """-log P_sem of every sample against the other K - 1: joint, and factorized over slots."""
    size = len(tuples)
    joint = Counter(tuples)
    per_slot = [Counter(row[position] for row in tuples) for position in range(len(SLOTS))]
    sem, semf = [], []
    for row in tuples:
        sem.append(-math.log((joint[row] - 1 + 0.5) / size))
        semf.append(-sum(math.log((per_slot[position][row[position]] - 1 + 0.5) / size)
                         for position in range(len(SLOTS))))
    return sem, semf


# ---------------------------------------------------------------- inputs


class Inputs:
    """Samples per hypothesis source (an original text, or a rewritten unit) and every scored cell."""

    def __init__(self, root: Path, phase8: Path, lm_runs: list[Path], derived: Path | None = None):
        derived = derived or root / DERIVED_DIR
        vocab = load_vocab(derived)
        extractor = Extractor(vocab)
        self.items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
        self.clean = {item["item_id"] for item in self.items}
        self.probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
        self.metas = item_meta(self.items, self.probe)
        self.subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
        self.pool = sorted({row["text"] for row in self.probe})
        lm = read_lm_runs(lm_runs).get("qwen", {})
        broad_of_sample = {(row["hyp_text"], row["sample"]): row["S"] for row in lm.get("center", [])
                           if row["source"] == "sample"}
        self.samples: dict[str, dict[str, list]] = {}
        pool = set(self.pool)
        for row in read_jsonl(phase8 / "lm_samples.jsonl"):
            if row["text"] in pool:
                found = self.samples.setdefault(row["text"], {"S": [], "cent": [], "tuples": []})
                found["S"].append(row["S"])
                found["cent"].append(row["S"] - broad_of_sample[(row["text"], row["sample"])])
                found["tuples"].append(tuple(extractor.extract(row["reply"]).values()))
        for row in lm.get("hsample", []):
            found = self.samples.setdefault(row["unit_id"], {"S": [], "cent": [], "tuples": []})
            found["S"].append(row["S"])
            found["cent"].append(row["S"] - row["S_broad"])
            found["tuples"].append(tuple(extractor.extract(row["reply"]).values()))
        for found in self.samples.values():
            found["sem"], found["semf"] = held_out_sem(found["tuples"])
        wanted = {item["item_id"]: annotated_tuple(item, vocab) for item in self.items}
        phase8_s = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(phase8 / "lm_scores.jsonl")}
        u_rows = {(row["item_id"], row["hyp_key"], row["kind"]): row["S"] for row in lm.get("urewrite", [])}
        self.cells: list[dict] = []
        sem_cache: dict[tuple[str, str], tuple[float | None, float | None]] = {}

        def semantic(item: str, source: str) -> tuple[float | None, float | None]:
            if (item, source) not in sem_cache:
                tuples = self.samples.get(source, {}).get("tuples")
                sem_cache[(item, source)] = (None, None) if not tuples else (
                    -math.log(p_sem(tuples, wanted[item])), -math.log(p_sem_factorized(tuples, wanted[item])))
            return sem_cache[(item, source)]

        for row in self.probe:
            item, key, text = row["item_id"], row["hyp_key"], row["text"]
            value = phase8_s[(item, key)]
            sem, semf = semantic(item, text)
            self.cells.append({"key": (item, key, ORIGINAL, ORIGINAL), "source": text, "S": value,
                               "cent": value - phase8_s[(item, BROAD)], "sem": sem, "semf": semf})
            for kind in {kind for kinds in U_GROUPS.values() for kind in kinds}:
                if (item, key, kind) in u_rows and (item, BROAD, kind) in u_rows:
                    value = u_rows[(item, key, kind)]
                    self.cells.append({"key": (item, key, ORIGINAL, kind), "source": text, "S": value,
                                       "cent": value - u_rows[(item, BROAD, kind)], "sem": sem, "semf": semf})
        for row in lm.get("hrewrite", []):
            if row["item_id"] not in self.clean:
                continue
            item = row["item_id"]
            sem, semf = semantic(item, row["unit_id"])
            self.cells.append({"key": (item, row["hyp_key"], row["kind"], ORIGINAL), "source": row["unit_id"],
                               "S": row["S"], "cent": row["S"] - phase8_s[(item, BROAD)], "sem": sem, "semf": semf})
        self.phase8_s = phase8_s
        self.lm = lm


# ---------------------------------------------------------------- variants


def variant_scores(inputs: Inputs, spec: dict, weights: dict[str, float], pool: list[str] | None = None,
                   cells: list[dict] | None = None) -> tuple[dict, dict, dict]:
    """log scores, bins, and the pool statistics of one variant."""
    statistic, edges, sides = spec["statistic"], tuple(spec["edges"]), int(spec["sides"])
    bins = len(edges) + 1
    per = spec["normalization"] == "per"
    found_moments = {source: moments(samples[statistic]) for source, samples in inputs.samples.items()}
    pool = pool or inputs.pool
    mean = spread = None
    if not per:
        mean, spread = pool_stats({text: found_moments[text] for text in pool if found_moments.get(text)}, weights)
    counts: dict[str, list[int]] = {}
    if spec["score"] == "A":
        for source, samples in inputs.samples.items():
            moment = found_moments.get(source)
            if per and (moment is None or moment[1] <= 0):
                continue
            centre, scale = moment if per else (mean, spread)
            tally = [0] * bins
            for value in samples[statistic]:
                tally[bin_index((value - centre) / scale, edges, sides)] += 1
            counts[source] = tally
    c_values = centres(edges)
    log, binned = {}, {}
    for cell in cells if cells is not None else inputs.cells:
        value = cell[statistic]
        if value is None:
            continue
        if per:
            moment = found_moments.get(cell["source"])
            if moment is None or moment[1] <= 0:
                continue
            t = (value - moment[0]) / moment[1]
        else:
            t = (value - mean) / spread
        b = bin_index(t, edges, sides)
        if spec["score"] == "A":
            tally = counts.get(cell["source"])
            if tally is None:
                continue
            log[cell["key"]] = math.log((tally[b] + 1) / (sum(tally) + bins))
        else:
            log[cell["key"]] = -(mean + spread * c_values[b])
        binned[cell["key"]] = b
    return log, binned, {"H": mean, "sigma": spread}


def bin_agreement(bins: dict, items: list[str], boot: ItemBoot) -> dict:
    found = {}
    for group, kinds in U_GROUPS.items():
        ids, same = [], []
        for (item, key, hvar, uvar), value in bins.items():
            if hvar == ORIGINAL and uvar in kinds and (item, key, ORIGINAL, ORIGINAL) in bins:
                ids.append(item)
                same.append(float(value == bins[(item, key, ORIGINAL, ORIGINAL)]))
        found[group] = boot.mean(ids, same)
    for kind in H_KINDS:
        ids, same = [], []
        for (item, key, hvar, uvar), value in bins.items():
            if hvar == kind and uvar == ORIGINAL and (item, key, ORIGINAL, ORIGINAL) in bins:
                ids.append(item)
                same.append(float(value == bins[(item, key, ORIGINAL, ORIGINAL)]))
        found[kind] = boot.mean(ids, same)
    return found


def occupancy(bins: dict, count: int) -> list[float]:
    originals = [value for (_, _, hvar, uvar), value in bins.items() if hvar == uvar == ORIGINAL]
    tally = Counter(originals)
    return [tally.get(index, 0) / len(originals) if originals else 0.0 for index in range(count)]


def score_map(log: dict) -> dict[str, dict[str, float]]:
    found: dict[str, dict[str, float]] = {}
    for (item, key, hvar, uvar), value in log.items():
        if hvar == uvar == ORIGINAL:
            found.setdefault(item, {})[key] = value
    return found


# ---------------------------------------------------------------- metrics 1 and 2


def corpus_bootstrap(derived: Path, config: dict, items: list[dict]) -> Bootstrap:
    """The clean-item resample metrics.analyze pairs with the reference replicates."""
    seed, replicates = int(config["bootstrap"]["seed"]), int(config["bootstrap"]["replicates"])
    smoothing = float(config["reference"]["b_smoothing"])
    table = read_json(derived / "reference_table.json")
    stored = read_json(derived / table["replicates_file"])
    variant = table["variants"]["primary"]
    full_set = sorted(item["item_id"] for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean")
    weights, _ = item_bootstrap(items, "clean", full_set, seed, replicates)
    terms = [term_vector(row) for row in replicate_terms(stored["populations"], variant["a_from"], variant["b_from"],
                                                         smoothing)]
    return Bootstrap(weights, term_vector(variant["slots"]), terms)


def size_and_information(items: list[dict], scores: dict, changed: dict, boot: Bootstrap) -> dict:
    block = analyze_scorer(items, scores, changed, boot)
    occam = {key: block["occam"].get(key) for key in (f"{regime}|slots_mean|{direction}" for regime in ("expl", "impl")
                                                      for direction in ("named", "omitted"))}
    contra = block["controls"]["ctrl:contra"]
    return {"occam": occam, "contra_auroc": contra.get("auroc"), "contra_mean_log_bf": contra["mean_gap"],
            "contra_mean_log_bf_ci95": contra["mean_gap_ci95"], "contra_n": contra["n"]}


# ---------------------------------------------------------------- metric 4


def pool_sensitivity(inputs: Inputs, spec: dict, weights: dict[str, float], base_log: dict, base_bins: dict,
                     draws: int, fraction: float, seed: int, label: str) -> dict:
    originals = [cell for cell in inputs.cells if cell["key"][2] == cell["key"][3] == ORIGINAL]
    rng = random.Random(f"{seed}:typv0:pool:{label}")
    shifts, flips = [], []
    for _ in range(draws):
        half = sorted(rng.sample(inputs.pool, int(len(inputs.pool) * fraction)))
        log, bins, _ = variant_scores(inputs, spec, weights, pool=half, cells=originals)
        deltas = []
        for item in inputs.metas:
            for goal, other, _ in bf_pairs(inputs.metas[item]):
                keys = [(item, goal, ORIGINAL, ORIGINAL), (item, other, ORIGINAL, ORIGINAL)]
                if all(key in log and key in base_log for key in keys):
                    deltas.append(abs((log[keys[0]] - log[keys[1]]) - (base_log[keys[0]] - base_log[keys[1]])))
        shifts.append(statistics.median(deltas) if deltas else None)
        common = [key for key in bins if key in base_bins]
        flips.append(sum(bins[key] != base_bins[key] for key in common) / len(common) if common else None)
    usable = [value for value in shifts if value is not None]
    rates = [value for value in flips if value is not None]
    return {"draws": draws, "fraction": fraction,
            "median_abs_delta_log_bf": statistics.median(usable) if usable else None,
            "max_median_abs_delta_log_bf": max(usable) if usable else None,
            "bin_flip_rate_mean": statistics.fmean(rates) if rates else None,
            "bin_flip_rate_max": max(rates) if rates else None}


# ---------------------------------------------------------------- metric 5: dice


def die_distribution(probabilities: dict[int, Fraction], k: int) -> list[tuple[float, float]]:
    """Exact (surprisal, probability) of a k-roll sequence, grouped by how many rolls hit each probability level."""
    levels: dict[Fraction, Fraction] = {}
    for p in probabilities.values():
        if p > 0:
            levels[p] = levels.get(p, Fraction(0)) + p
    values = sorted(levels)
    found: dict[float, float] = {}
    for counts in product(range(k + 1), repeat=len(values)):
        if sum(counts) != k:
            continue
        ways = math.factorial(k)
        for n in counts:
            ways //= math.factorial(n)
        mass = ways * math.prod(float(levels[v]) ** n for v, n in zip(values, counts))
        surprisal = sum(-n * math.log(v) for v, n in zip(values, counts))
        key = round(surprisal, 12)
        found[key] = found.get(key, 0.0) + mass
    return sorted(found.items())


def dice_check(dice_config: dict, pairs: list[dict], variants: dict[str, dict]) -> dict:
    """per2A, glob2A, glob1A, glob1B with exact moments; A uses the exact bin mass, without smoothing."""
    from robust_likelihood.dice.design import face_probabilities

    dice = face_probabilities(dice_config)
    results = {}
    for name in ("per2A", "glob2A", "glob1A", "glob1B"):
        spec = variants[name]
        edges, sides = tuple(spec["edges"]), int(spec["sides"])
        c_values = centres(edges)
        scored: dict[str, float] = {}
        for k in sorted({row["k"] for row in pairs}):
            dist = {die: die_distribution(probabilities, k) for die, probabilities in dice.items()}
            mom = {}
            for die, rows in dist.items():
                mu = sum(s * p for s, p in rows)
                mom[die] = (mu, math.sqrt(max(0.0, sum(p * (s - mu) ** 2 for s, p in rows))))
            mean = statistics.fmean(mu for mu, _ in mom.values())
            spread = math.sqrt(statistics.fmean(sigma * sigma + (mu - mean) ** 2 for mu, sigma in mom.values()))

            def t_value(value: float, die: str) -> float:
                if spec["normalization"] == "per":
                    mu, sigma = mom[die]
                    if sigma == 0:
                        return 0.0 if math.isclose(value, mu, abs_tol=1e-9) else math.copysign(math.inf, value - mu)
                    return (value - mu) / sigma
                return (value - mean) / spread

            mass = {die: [0.0] * (len(edges) + 1) for die in dice}
            for die, rows in dist.items():
                for s, p in rows:
                    mass[die][bin_index(t_value(s, die), edges, sides)] += p
            for row in pairs:
                if row["k"] != k:
                    continue
                value = math.inf if row["true_logp"] is None else -row["true_logp"]
                b = bin_index(t_value(value, row["hypothesis"]), edges, sides)
                if spec["score"] == "A":
                    scored[row["query_id"]] = math.log(mass[row["hypothesis"]][b]) if mass[row["hypothesis"]][b] > 0 \
                        else -math.inf
                else:
                    scored[row["query_id"]] = -(mean + spread * c_values[b])
        by_seq: dict[str, dict[str, dict]] = {}
        for row in pairs:
            by_seq.setdefault(row["seq_id"], {})[row["hypothesis"]] = row
        even_rows, impossible = [], []
        for seq_id, rows in by_seq.items():
            even, fair = rows.get("even"), rows.get("fair")
            if even is None or fair is None:
                continue
            predicted = scored[even["query_id"]] - scored[fair["query_id"]]
            if even["true_logp"] is None:
                impossible.append(predicted)
            elif all(face % 2 == 0 for face in even["faces"]):
                even_rows.append({"k": even["k"], "true": even["true_logp"] - fair["true_logp"], "predicted": predicted})
        per_k = {}
        for row in even_rows:
            per_k.setdefault(row["k"], []).append(row["predicted"])
        results[name] = {
            "all_even_sequences": len(even_rows),
            "tie_rate": sum(1 for row in even_rows if row["predicted"] == 0) / len(even_rows) if even_rows else None,
            "by_k": {str(k): {"true": k * math.log(2), "median_predicted": statistics.median(values), "n": len(values)}
                     for k, values in sorted(per_k.items())},
            "impossible_sequences": len(impossible),
            "impossible_ruled_out": sum(1 for value in impossible if value == -math.inf) / len(impossible)
            if impossible else None,
        }
    return results


# ---------------------------------------------------------------- predictions


def predictions(metrics: dict) -> list[dict]:
    found = []
    rows = metrics["variants"]

    def named(name: str) -> dict | None:
        block = rows.get(name, {}).get("uniform", {}).get("size")
        if not block:
            return None
        return {regime: block["occam"].get(f"{regime}|slots_mean|named") for regime in ("expl", "impl")}

    per = named("per2A")
    if per and all(per.values()):
        found.append({"id": "T1", "statement": "per2A erases the size principle: named-slot Occam gap ≤ −0.8·L_E",
                      "pass": all(block["OG"] <= -0.8 * block["L_E"] for block in per.values()),
                      "value": {regime: {"OG": block["OG"], "L_E": block["L_E"]} for regime, block in per.items()}})
    values, verdicts = {}, []
    for name in ("per2A", "glob2A", "glob1A", "glob1B", "cent1B", "sem1B"):
        block = named(name)
        if not block or not all(block.values()):
            continue
        ratio = {regime: found_block["mean_L_J"] / found_block["L_E"] for regime, found_block in block.items()}
        values[name] = ratio
        option_a = name.endswith("A")
        verdicts.append(all(value < 0.5 for value in ratio.values()) if option_a
                        else all(value >= 0.5 for value in ratio.values()))
    if values:
        found.append({"id": "T2", "statement": "Option A variants tie or erase the size principle (L_J < 0.5·L_E); "
                      "option B variants keep it (L_J ≥ 0.5·L_E), named single-slot pairs, both regimes",
                      "pass": all(verdicts), "value": {"L_J/L_E": values}})
    sem = rows.get("sem1B", {}).get("uniform", {})
    raw_icc = metrics["baselines"].get("qwen_tok_subset", {}).get("icc1")
    if sem:
        icc = sem["robustness_subset"]["rewording"]["icc1"]
        agree = {group: sem["bin_agreement"][group]["mean"] for group in ("para", "zh")}
        found.append({"id": "T3", "statement": "sem1B: hypothesis-rewording ICC > 0.9 and bin agreement > 0.9 under "
                      "para and zh, where raw log-prob ICC < 0.5",
                      "pass": icc is not None and icc > 0.9 and all(value is not None and value > 0.9
                                                                    for value in agree.values()),
                      "value": {"icc1": icc, "bin_agreement": agree, "raw_logprob_icc1_subset": raw_icc},
                      "note": None if raw_icc is None or raw_icc < 0.5 else
                      f"the premise does not hold: raw log-prob ICC is {raw_icc:.2f}"})
    cent = rows.get("cent1B", {}).get("uniform", {})
    glob = rows.get("glob1B", {}).get("uniform", {})
    if cent and glob:
        mine = {group: cent["bin_agreement"][group]["mean"] for group in ("pad", "typo")}
        other = {group: glob["bin_agreement"][group]["mean"] for group in ("pad", "typo")}
        found.append({"id": "T4", "statement": "cent1B bin agreement under pad and typo > 0.9, above glob1B",
                      "pass": all(mine[group] > 0.9 and mine[group] > other[group] for group in mine),
                      "value": {"cent1B": mine, "glob1B": other}})
    return found


# ---------------------------------------------------------------- run


def run_typv0(
    root: Path,
    *,
    phase8: Path,
    lm_runs: list[Path],
    verbal: list[Path],
    jev_full: Path,
    claim1_metrics: Path,
    out: Path,
    derived: Path | None = None,
) -> dict:
    from robust_likelihood.multiwoz import load_config

    clock = time.monotonic()
    derived = derived or root / DERIVED_DIR
    config = load_config(root)
    claims = read_json(root / "configs" / "claims.json")
    typ = load_typv0_config(root)
    seed, replicates = int(claims["bootstrap"]["seed"]), int(claims["bootstrap"]["replicates"])
    inputs = Inputs(root, phase8, lm_runs, derived)
    changed = {(row["item_id"], row["hyp_key"]): row["changed"]["slot"]
               for row in read_jsonl(derived / "probe.jsonl") if "changed" in row}
    corpus_boot = corpus_bootstrap(derived, config, inputs.items)
    clean_ids = sorted(inputs.clean)
    boot = ItemBoot(clean_ids, replicates, seed)
    subset_boot = ItemBoot(sorted(inputs.subset), replicates, seed)
    weights = {scheme: pool_weights(inputs.probe, scheme) for scheme in typ["weightings"]}

    def evaluate(name: str, spec: dict, scheme: str, full: bool) -> dict:
        log, bins, pool = variant_scores(inputs, spec, weights[scheme])
        scorer = Scorer(f"typ_{name}_{scheme}", "loglik", log, items=clean_ids)
        block = {"spec": spec, "weighting": scheme, "pool": pool, "scores": len(log),
                 "occupancy": occupancy(bins, len(spec["edges"]) + 1),
                 "size": size_and_information(inputs.items, score_map(log), changed, corpus_boot)}
        if full:
            subset = scorer.restricted(scorer.name + "_subset", inputs.subset)
            block.update({
                "rewording": rewording(scorer, inputs.metas, boot),
                "robustness_subset": {"rewording": rewording(subset, inputs.metas, subset_boot)},
                "utterance": utterance(scorer, inputs.metas, boot),
                "posterior": posterior(scorer, inputs.metas, boot, seed),
                "bin_agreement": bin_agreement(bins, clean_ids, boot),
            })
            if spec["normalization"] == "global":
                block["pool_sensitivity"] = pool_sensitivity(
                    inputs, spec, weights[scheme], log, bins, int(typ["pool_sensitivity"]["draws"]),
                    float(typ["pool_sensitivity"]["fraction"]), seed, f"{name}:{scheme}")
        return block

    metrics: dict = {"variants": {}, "sensitivity": {}}
    for name, spec in typ["variants"].items():
        metrics["variants"][name] = {scheme: evaluate(name, spec, scheme, full=True) for scheme in typ["weightings"]}
        if spec["normalization"] == "per":  # no pool: the weighting does not enter
            metrics["variants"][name] = {"uniform": metrics["variants"][name]["uniform"]}
    for label, edges in typ["sensitivity"]["edges"].items():
        for name, spec in typ["variants"].items():
            if spec["sides"] == 1:
                changed_spec = dict(spec, edges=edges)
                metrics["sensitivity"][f"{name}:{label}"] = evaluate(name, changed_spec, "uniform", full=True)
    if typ["sensitivity"].get("factorized_sem"):
        spec = dict(typ["variants"]["sem1B"], statistic="semf")
        metrics["sensitivity"]["sem1B:factorized"] = evaluate("sem1B", spec, "uniform", full=True)

    # Baselines: size principle and information from the same machinery; robustness from claim 1.
    qwen = token_log({key: -value for key, value in inputs.phase8_s.items()}, inputs.lm, inputs.clean)
    baseline_maps = {"qwen_tok": score_map(qwen), "qwen_clip": score_map(clipped_log(qwen, float(claims["clip_nats"])))}
    epsilon = float(config["score_mapping"]["clip_epsilon"])
    for scorer in jev_scorers(read_jsonl(jev_full / "scores.jsonl"), [], inputs.clean, epsilon):
        baseline_maps[scorer.name] = score_map(scorer.log)
    verbal_found, _ = verbal_scorers([row for run_dir in verbal for row in read_jsonl(run_dir / "scores.jsonl")])
    subset_items = [item for item in inputs.items if item["item_id"] in set(inputs.subset)]
    subset_corpus = corpus_bootstrap(derived, config, subset_items)
    claim1 = read_json(claim1_metrics)["scorers"]
    metrics["baselines"] = {}
    for name, scores in baseline_maps.items():
        metrics["baselines"][name] = {"size": size_and_information(inputs.items, scores, changed, corpus_boot)}
    for scorer in verbal_found:
        metrics["baselines"][scorer.name] = {
            "size": size_and_information(subset_items, score_map(scorer.log), changed, subset_corpus),
            "items": "60-item subset"}
    for name, block in metrics["baselines"].items():
        found = claim1.get(name)
        if found:
            block.update({"icc1": found["rewording"]["icc1"], "rewording": found["rewording"],
                          "utterance": found["utterance"], "posterior": found["posterior"]})
    subset_raw = claim1.get("qwen_tok_subset")
    if subset_raw:
        metrics["baselines"]["qwen_tok_subset"] = {"icc1": subset_raw["rewording"]["icc1"],
                                                   "rewording": subset_raw["rewording"]}
    from robust_likelihood.dice import load_config as load_dice_config

    metrics["dice"] = dice_check(load_dice_config(root), read_jsonl(root / "data" / "dice" / "pairs.jsonl"),
                                 typ["variants"])
    metrics["checks"] = {
        "pool_texts": len(inputs.pool),
        "pool_texts_with_samples": sum(1 for text in inputs.pool if text in inputs.samples),
        "rewritten_units_with_samples": sum(1 for source in inputs.samples if source.startswith("h:")),
        "cells": len(inputs.cells),
    }
    metrics["predictions"] = predictions(metrics)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    manifest = {
        "kind": "typicality-v0", "git": git_state(root), "config": typ,
        "inputs": {"phase8": phase8.as_posix(), "lm_runs": [path.as_posix() for path in lm_runs],
                   "verbal": [path.as_posix() for path in verbal], "jev_full": jev_full.as_posix(),
                   "claim1_metrics": claim1_metrics.as_posix()},
        "bootstrap": {"replicates": replicates, "seed": seed},
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1),
    }
    write_json(out / "manifest.json", manifest)
    from robust_likelihood.multiwoz.typv0_report import summary_markdown

    (out / "summary.md").write_text(summary_markdown(metrics, manifest), encoding="utf-8", newline="\n")
    return {"out": str(out), "predictions": metrics["predictions"], "elapsed_seconds": manifest["elapsed_seconds"]}
