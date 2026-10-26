"""Size-principle metrics for elicited and language-model scores. The definitions are frozen in the plan.

Jev's yes-probability s is read on the logit scale, l = logit(clip(s, eps, 1 - eps)).
When the narrower hypothesis scores higher, logit gaps are at least as large as log
gaps, so the logit reading is the generous one for Jev and any shortfall is conservative.
Language-model scores are already in nats and are used as they are.

For a narrow hypothesis and a broader one that drops slots, L_J is the score gap and
L_E the corpus log-ratio. The Occam gap of a group of pairs is mean(L_J) - mean(L_E).
Bootstrap replicate b resamples items for L_J and takes reference replicate b for L_E.
When the analysis covers every clean item, replicate b redraws the same clean-item
resample the reference used for a, so the pairing is exact.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict

from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.reference import bootstrap_rng, interval, replicate_terms, resample
from robust_likelihood.storage import read_json, read_jsonl

EPSILON = 1e-3
REGIMES = ("expl", "impl")
DIRECTIONS = ("named", "omitted")
TERMS = tuple((slot, kind) for slot in SLOTS for kind in DIRECTIONS)
TERM_INDEX = {term: position for position, term in enumerate(TERMS)}
SINGLE_GROUPS = tuple(f"{slot}|{direction}" for direction in DIRECTIONS for slot in SLOTS)


def clip(score: float, epsilon: float = EPSILON) -> float:
    return min(1.0 - epsilon, max(epsilon, score))


def is_clipped(score: float, epsilon: float = EPSILON) -> bool:
    return score < epsilon or score > 1.0 - epsilon


def logit_score(score: float, epsilon: float = EPSILON) -> float:
    value = clip(score, epsilon)
    return math.log(value / (1.0 - value))


def log_score(score: float, epsilon: float = EPSILON) -> float:
    return math.log(clip(score, epsilon))


MAPPINGS = {"logit": logit_score, "log": log_score}


def empirical_log_ratio(removed: list[str], status: dict[str, str], terms: dict[str, dict]) -> float | None:
    """L_E: the sum over removed slots of ln(a/b) if the turn names the slot, else ln((1-a)/(1-b))."""
    total = 0.0
    for slot in removed:
        kind = status[slot]
        if kind not in DIRECTIONS or terms[slot][kind] is None:
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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def goal_key(n_constrained: int, regime: str) -> str:
    """The |C| = 3 goal has one text, so its implicit score serves both regimes."""
    return "G|impl" if regime == "impl" or n_constrained == 3 else "G|expl"


def split_key(hyp_key: str) -> tuple[str, str]:
    """hyp_type and regime of a key; controls are rendered with base(), so they are implicit."""
    if "|" in hyp_key:
        hyp_type, regime = hyp_key.split("|")
        return hyp_type, regime
    return hyp_key, "impl"


def term_vector(slots: dict[str, dict]) -> list[float | None]:
    return [slots[slot][kind] for slot, kind in TERMS]


def score_maps(rows: list[dict], mapping, epsilon: float = EPSILON) -> dict[str, dict[str, dict[str, float]]]:
    """prompt version -> item -> hypothesis key -> mapped score, from the baseline rows."""
    found: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("repeat_id") is None:
            found[row["prompt_version"]][row["item_id"]][row["hyp_key"]] = mapping(row["noul"], epsilon)
    return found


def removal_pairs(items: list[dict], scores: dict[str, dict[str, float]]) -> list[dict]:
    """Every (narrow, broader) pair: single-slot removals and the full removal, within each regime."""
    pairs = []
    for index, item in enumerate(items):
        found = scores.get(item["item_id"], {})
        constrained = [slot for slot in SLOTS if slot in item["goal"]]
        status = item["slot_status"]
        named = sum(1 for slot in constrained if status[slot] == "named")
        full = "named" if named == len(constrained) else "omitted" if named == 0 else "mixed"
        for regime in REGIMES:
            narrow = found.get(goal_key(len(constrained), regime))
            if narrow is None:
                continue
            candidates = [([slot], f"minus:{slot}|{regime}", slot, status[slot]) for slot in constrained]
            candidates.append((constrained, f"broad|{regime}", "all", full))
            for removed, key, removal, direction in candidates:
                broader = found.get(key)
                if broader is None:
                    continue
                usable = all(status[slot] in DIRECTIONS for slot in removed)
                pairs.append(
                    {
                        "item_index": index,
                        "item_id": item["item_id"],
                        "regime": regime,
                        "removal": removal,
                        "direction": direction,
                        "n_constrained": len(constrained),
                        "k_named": named,
                        "narrow": narrow,
                        "broader": broader,
                        "L_J": narrow - broader,
                        "signature": tuple(TERM_INDEX[(slot, status[slot])] for slot in removed) if usable else None,
                    }
                )
    return pairs


def group_pairs(pairs: list[dict]) -> dict[str, list[dict]]:
    """regime|removal|direction for every pair, and regime|all|C{n}k{k} for full removals."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["signature"] is None:
            continue
        groups[f"{pair['regime']}|{pair['removal']}|{pair['direction']}"].append(pair)
        if pair["removal"] == "all":
            groups[f"{pair['regime']}|all|C{pair['n_constrained']}k{pair['k_named']}"].append(pair)
    return dict(groups)


def bootstrap_weights(size: int, replicates: int, seed: int, label: str) -> list[list[int]]:
    """Item multiplicities per replicate, from the same generator the reference uses."""
    rng = bootstrap_rng(seed, label)
    weights = []
    for _ in range(replicates):
        counts = [0] * size
        for index in resample(size, rng):
            counts[index] += 1
        weights.append(counts)
    return weights


class Bootstrap:
    """Item resamples and matching reference terms, shared by every scorer of one item set."""

    def __init__(self, weights: list[list[int]], term_point: list[float], term_replicates: list[list[float]]):
        self.weights = weights
        self.term_point = term_point
        self.term_replicates = term_replicates

    def weighted_mean(self, rows: list[tuple[int, float]]) -> list[float | None]:
        found = []
        for counts in self.weights:
            total = 0
            value = 0.0
            for index, number in rows:
                weight = counts[index]
                if weight:
                    total += weight
                    value += weight * number
            found.append(value / total if total else None)
        return found

    def occam(self, pairs: list[dict]) -> tuple[dict, list[float | None]]:
        size = len(pairs)
        point_le = [sum(self.term_point[k] for k in pair["signature"]) for pair in pairs]
        composition = [0.0] * len(TERMS)
        for pair in pairs:
            for k in pair["signature"]:
                composition[k] += 1.0 / size
        mean_lj = sum(pair["L_J"] for pair in pairs) / size
        mean_le = sum(point_le) / size
        lj_boot: list[float | None] = []
        og_boot: list[float | None] = []
        le_boot: list[float | None] = []
        for counts, terms in zip(self.weights, self.term_replicates):
            total = 0
            value = 0.0
            weighted = [0.0] * len(TERMS)
            for pair in pairs:
                weight = counts[pair["item_index"]]
                if weight:
                    total += weight
                    value += weight * pair["L_J"]
                    for k in pair["signature"]:
                        weighted[k] += weight
            le_boot.append(_dot(composition, terms))
            if not total:
                lj_boot.append(None)
                og_boot.append(None)
                continue
            lj = value / total
            le = _dot([number / total for number in weighted], terms)
            lj_boot.append(lj)
            og_boot.append(None if le is None else lj - le)
        summary = {
            "n": size,
            "mean_L_J": mean_lj,
            "mean_L_J_ci95": interval(lj_boot),
            "L_E": mean_le,
            "L_E_ci95": interval(le_boot),
            "OG": mean_lj - mean_le,
            "OG_ci95": interval(og_boot),
        }
        return summary, og_boot


def _dot(weights: list[float], terms: list[float | None]) -> float | None:
    total = 0.0
    for weight, term in zip(weights, terms):
        if weight:
            if term is None:
                return None
            total += weight * term
    return total


def full_removal_bins(items: list[dict], minimum: int = 15) -> list[tuple[str, list[tuple[int, int]]]]:
    """Full-removal groups (|C|, k_named) pooled with their neighbours until each bin has `minimum` items.

    Within each |C|, groups are taken in increasing k_named and added to the open bin until it
    is large enough. A short last bin joins the bin before it.
    """
    sizes = Counter(
        (item["n_constrained"], sum(1 for slot in item["goal"] if item["slot_status"][slot] == "named"))
        for item in items
    )
    bins = []
    for size in sorted({size for size, _ in sizes}):
        groups = [(size, named) for named in range(size + 1) if sizes[(size, named)]]
        current: list[tuple[int, int]] = []
        closed: list[list[tuple[int, int]]] = []
        for group in groups:
            current.append(group)
            if sum(sizes[member] for member in current) >= minimum:
                closed.append(current)
                current = []
        if current:
            if closed:
                closed[-1].extend(current)
            else:
                closed.append(current)
        for members in closed:
            low, high = members[0][1], members[-1][1]
            bins.append((f"C{size}k{low}" if low == high else f"C{size}k{low}-{high}", members))
    return bins


def occam_table(pairs: list[dict], boot: Bootstrap, bins=None) -> tuple[dict, dict[str, list]]:
    table = {}
    replicates = {}
    for key, rows in sorted(group_pairs(pairs).items()):
        table[key], replicates[key] = boot.occam(rows)
    for label, members in bins or []:
        for regime in REGIMES:
            rows = [
                pair for pair in pairs
                if pair["regime"] == regime and pair["removal"] == "all" and pair["signature"] is not None
                and (pair["n_constrained"], pair["k_named"]) in members
            ]
            if rows:
                key = f"{regime}|bin|{label}"
                table[key], replicates[key] = boot.occam(rows)
    for regime in REGIMES:
        for direction in DIRECTIONS:
            keys = [f"{regime}|{slot}|{direction}" for slot in SLOTS]
            if all(key in table for key in keys):
                values = [
                    None if any(replicates[key][b] is None for key in keys) else mean([replicates[key][b] for key in keys])
                    for b in range(len(boot.weights))
                ]
                table[f"{regime}|slots_mean|{direction}"] = {
                    "n": sum(table[key]["n"] for key in keys),
                    "OG": mean([table[key]["OG"] for key in keys]),
                    "OG_ci95": interval(values),
                    "mean_L_J": mean([table[key]["mean_L_J"] for key in keys]),
                    "L_E": mean([table[key]["L_E"] for key in keys]),
                }
    return table, replicates


def wording_gap(items: list[dict], scores: dict[str, dict[str, float]], boot: Bootstrap) -> dict:
    """Delta = l(explicit) - l(implicit) for the same breadth; the target is 0."""
    rows = []
    for index, item in enumerate(items):
        found = scores.get(item["item_id"], {})
        constrained = [slot for slot in SLOTS if slot in item["goal"]]
        breadths = (["G"] if len(constrained) < len(SLOTS) else []) + [f"minus:{slot}" for slot in constrained] + ["broad"]
        for breadth in breadths:
            explicit = found.get(f"{breadth}|expl")
            implicit = found.get(f"{breadth}|impl")
            if explicit is not None and implicit is not None:
                rows.append((index, breadth, explicit - implicit))
    return {
        "all": _delta_summary([(index, delta) for index, _, delta in rows], boot),
        **{
            breadth: _delta_summary([(index, delta) for index, found, delta in rows if found == breadth], boot)
            for breadth in sorted({breadth for _, breadth, _ in rows})
        },
    }


def _delta_summary(rows: list[tuple[int, float]], boot: Bootstrap) -> dict:
    deltas = [delta for _, delta in rows]
    if not deltas:
        return {"n": 0}
    return {
        "n": len(deltas),
        "mean": mean(deltas),
        "mean_ci95": interval(boot.weighted_mean(rows)),
        "share_abs_above_1": sum(1 for delta in deltas if abs(delta) > 1) / len(deltas),
        "median": median(deltas),
    }


def monotonicity(pairs: list[dict]) -> dict:
    """Share of pairs where the broader hypothesis strictly outscores the narrower one."""
    found = {}
    for regime in REGIMES:
        for label, chosen in (
            ("single_named", [p for p in pairs if p["regime"] == regime and p["removal"] != "all" and p["direction"] == "named"]),
            ("full_all_named", [p for p in pairs if p["regime"] == regime and p["removal"] == "all" and p["direction"] == "named"]),
        ):
            violations = sum(1 for pair in chosen if pair["broader"] > pair["narrow"])
            ties = sum(1 for pair in chosen if pair["broader"] == pair["narrow"])
            found[f"{regime}|{label}"] = {
                "n": len(chosen),
                "violations": violations,
                "violation_rate": violations / len(chosen) if chosen else None,
                "ties": ties,
                "tie_rate": ties / len(chosen) if chosen else None,
            }
    return found


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """P(positive > negative) + P(tie) / 2 over every pair."""
    if not positives or not negatives:
        return None
    ordered = sorted(negatives)
    wins = 0.0
    for value in positives:
        below = bisect.bisect_left(ordered, value)
        wins += below + 0.5 * (bisect.bisect_right(ordered, value) - below)
    return wins / (len(positives) * len(negatives))


def controls(items: list[dict], scores: dict[str, dict[str, float]], changed: dict, boot: Bootstrap) -> dict:
    """Contradiction: G|impl against one named slot changed. Tie: one omitted slot changed; target 0."""
    found = {}
    for kind in ("ctrl:contra", "ctrl:tie", "ctrl:tie2"):
        rows = []
        for index, item in enumerate(items):
            values = scores.get(item["item_id"], {})
            if "G|impl" in values and kind in values:
                rows.append((index, values["G|impl"], values[kind], changed[(item["item_id"], kind)]))
        gaps = [(index, goal - control) for index, goal, control, _ in rows]
        block = {
            "n": len(rows),
            "mean_gap": mean([gap for _, gap in gaps]),
            "mean_gap_ci95": interval(boot.weighted_mean(gaps)) if gaps else None,
            "median_gap": median([gap for _, gap in gaps]),
            "goal_higher": sum(1 for _, goal, control, _ in rows if goal > control),
            "ties": sum(1 for _, goal, control, _ in rows if goal == control),
            "by_changed_slot": {
                slot: {"n": len(chosen), "mean_gap": mean(chosen)}
                for slot in SLOTS
                if (chosen := [goal - control for _, goal, control, moved in rows if moved == slot])
            },
        }
        if kind == "ctrl:contra":
            block["auroc"] = auroc([goal for _, goal, _, _ in rows], [control for _, _, control, _ in rows])
        found[kind] = block
    ties = [
        (index, scores[item["item_id"]]["G|impl"] - scores[item["item_id"]][kind])
        for index, item in enumerate(items)
        for kind in ("ctrl:tie", "ctrl:tie2")
        if kind in scores.get(item["item_id"], {}) and "G|impl" in scores.get(item["item_id"], {})
    ]
    found["tie_offset_all"] = {
        "n": len(ties),
        "mean": mean([gap for _, gap in ties]),
        "mean_ci95": interval(boot.weighted_mean(ties)) if ties else None,
    }
    return found


def hypothesis_bias(items: list[dict], scores: dict[str, dict[str, float]]) -> dict:
    """Within-item centred score, averaged by hypothesis type and regime over every item."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in items:
        values = scores.get(item["item_id"], {})
        if not values:
            continue
        centre = mean(list(values.values()))
        for key, value in values.items():
            hyp_type, regime = split_key(key)
            buckets[f"{hyp_type}|{regime}"].append(value - centre)
    return {key: {"n": len(found), "mean_centred": mean(found)} for key, found in sorted(buckets.items())}


def repeat_noise(rows: list[dict], mapping, epsilon: float = EPSILON) -> dict:
    """Spread of repeated scores for the same item, hypothesis, and wording."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["prompt_version"], row["item_id"], row["hyp_key"])].append(row["noul"])
    found = {}
    for version in sorted({key[0] for key in groups}):
        chosen = [values for key, values in groups.items() if key[0] == version and len(values) > 1]
        spreads = []
        ranges = []
        for values in chosen:
            mapped = [mapping(value, epsilon) for value in values]
            centre = sum(mapped) / len(mapped)
            spreads.append(math.sqrt(sum((value - centre) ** 2 for value in mapped) / (len(mapped) - 1)))
            ranges.append(max(mapped) - min(mapped))
        identical = sum(1 for values in chosen if len(set(values)) == 1)
        found[version] = {
            "groups": len(chosen),
            "observations_per_group": dict(Counter(len(values) for values in chosen)),
            "identical_groups": identical,
            "deterministic": bool(chosen) and identical == len(chosen),
            "mean_sd": mean(spreads),
            "max_range": max(ranges) if ranges else None,
            "mean_range": mean(ranges),
        }
    return found


LM_SCORERS = ("lm_loglik", "lm_typ_normal", "lm_typ_bins")
QUANTITY = (
    "Jev scores are elicited yes-probabilities read on the logit scale; language-model scores are in nats. "
    "L_E comes from MultiWOZ naming rates of real users."
)


def item_bootstrap(items: list[dict], set_name: str, full_set: list[str], seed: int, replicates: int) -> tuple[list, str]:
    """Item weights per replicate. Every clean item: the reference's own clean-item resample."""
    ids = [item["item_id"] for item in items]
    label = "match_clean_turn0" if set_name == "clean" and ids == full_set else f"items:{set_name}:{len(ids)}"
    return bootstrap_weights(len(ids), replicates, seed, label), label


def pairing_holds(items: list[dict], weights: list[list[int]], replicates: dict, checked: int = 25) -> bool:
    """The a-counts of the first replicates, recomputed from the item weights, match the stored reference."""
    stored = replicates["populations"]["match_clean_turn0"]
    for b in range(min(checked, len(weights))):
        for slot in SLOTS:
            named = sum(w for w, item in zip(weights[b], items) if slot in item["goal"] and item["slot_status"][slot] == "named")
            constrained = sum(w for w, item in zip(weights[b], items) if slot in item["goal"])
            if [named, constrained] != stored[slot][b][:2]:
                return False
    return True


def analyze_scorer(items: list[dict], scores: dict[str, dict[str, float]], changed: dict, boot: Bootstrap) -> dict:
    pairs = removal_pairs(items, scores)
    occam, _ = occam_table(pairs, boot, full_removal_bins(items))
    return {
        "items_scored": sum(1 for item in items if scores.get(item["item_id"])),
        "pairs": len(pairs),
        "occam": occam,
        "wording_gap": wording_gap(items, scores, boot),
        "monotonicity": monotonicity(pairs),
        "controls": controls(items, scores, changed, boot),
        "bias": hypothesis_bias(items, scores),
    }


def lm_score_maps(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    found: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        for scorer in LM_SCORERS:
            if row.get(scorer) is not None:
                found[scorer][row["item_id"]][row["hyp_key"]] = row[scorer]
    return found


def analyze(
    run_dir,
    derived,
    config: dict,
    *,
    repeats_dir=None,
    lm_dir=None,
) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    rows = read_jsonl(run_dir / "scores.jsonl")
    epsilon = float(config["score_mapping"]["clip_epsilon"])
    seed = int(config["bootstrap"]["seed"])
    replicates_count = int(config["bootstrap"]["replicates"])
    smoothing = float(config["reference"]["b_smoothing"])
    every_item = read_jsonl(derived / "items.jsonl")
    changed = {
        (row["item_id"], row["hyp_key"]): row["changed"]["slot"] for row in read_jsonl(derived / "probe.jsonl") if "changed" in row
    }
    table = read_json(derived / "reference_table.json")
    replicates = read_json(derived / table["replicates_file"])
    selected = set(manifest["selection"]["item_ids"])
    baseline = [row for row in rows if row.get("repeat_id") is None]
    versions = sorted({row["prompt_version"] for row in baseline})
    lm_rows = read_jsonl(lm_dir / "lm_scores.jsonl") if lm_dir is not None else []
    metrics = {
        "run": run_dir.name,
        "quantity": QUANTITY,
        "requested_model_id": manifest.get("requested_model_id"),
        "returned_model_ids": manifest.get("returned_model_ids"),
        "prompt_versions": versions,
        "clip_epsilon": epsilon,
        "clipped": {
            version: sum(1 for row in baseline if row["prompt_version"] == version and is_clipped(row["noul"], epsilon))
            for version in versions
        },
        "scores": {version: sum(1 for row in baseline if row["prompt_version"] == version) for version in versions},
        "bootstrap": {"replicates": replicates_count, "seed": seed},
        "reference": {"a_from": table["variants"]["primary"]["a_from"], "b_from": table["variants"]["primary"]["b_from"]},
        "sets": {},
        "variants": {},
    }
    if lm_dir is not None:
        metrics["lm_run"] = lm_dir.name
    variant_terms = {
        name: (
            term_vector(variant["slots"]),
            [term_vector(row) for row in replicate_terms(replicates["populations"], variant["a_from"], variant["b_from"], smoothing)],
        )
        for name, variant in table["variants"].items()
    }
    for set_name in ("clean", "fail_info"):
        full_set = sorted(item["item_id"] for item in every_item if item["set"] == set_name)
        items = [item for item in every_item if item["set"] == set_name and item["item_id"] in selected]
        if not items:
            continue
        weights, label = item_bootstrap(items, set_name, full_set, seed, replicates_count)
        point, boot_terms = variant_terms["primary"]
        boot = Bootstrap(weights, point, boot_terms)
        block = {
            "items": len(items),
            "bootstrap_label": label,
            "paired_with_reference": label == "match_clean_turn0",
            "mappings": {},
        }
        if block["paired_with_reference"]:
            block["pairing_verified"] = pairing_holds(items, weights, replicates)
        for mapping in ("logit", "log"):
            maps = score_maps(baseline, MAPPINGS[mapping], epsilon)
            block["mappings"][mapping] = {version: analyze_scorer(items, maps[version], changed, boot) for version in versions}
        if lm_rows:
            lm_maps = lm_score_maps(lm_rows)
            block["lm"] = {scorer: analyze_scorer(items, lm_maps[scorer], changed, boot) for scorer in LM_SCORERS}
        metrics["sets"][set_name] = block
        if set_name == "clean":
            maps = score_maps(baseline, logit_score, epsilon)
            for name, (variant_point, variant_boot) in variant_terms.items():
                if name == "primary":
                    continue
                variant = Bootstrap(weights, variant_point, variant_boot)
                metrics["variants"][name] = {
                    "a_from": table["variants"][name]["a_from"],
                    "b_from": table["variants"][name]["b_from"],
                    "occam": {version: occam_table(removal_pairs(items, maps[version]), variant)[0] for version in versions},
                }
    if repeats_dir is not None:
        repeat_rows = read_jsonl(repeats_dir / "scores.jsonl")
        metrics["repeat_run"] = repeats_dir.name
        metrics["repeat_noise"] = repeat_noise(repeat_rows, logit_score, epsilon)
    metrics["predictions"] = predictions(metrics)
    return metrics


def _og(block: dict, key: str) -> tuple[float | None, list | None]:
    group = block["occam"].get(key)
    return (None, None) if group is None else (group["OG"], group.get("OG_ci95"))


def predictions(metrics: dict) -> list[dict]:
    """The pre-registered scorecard. A failed prediction is a result, not a bug."""
    clean = metrics["sets"].get("clean")
    found = []
    if clean is None or "multiwoz_likely" not in clean["mappings"]["logit"]:
        return found
    likely = clean["mappings"]["logit"]["multiwoz_likely"]
    value, ci = _og(likely, "expl|slots_mean|named")
    found.append(_prediction("P1", "Explicit, named, single-slot OG within ±1.5 nats (mean over slots, likely)",
                             value, ci, value is not None and -1.5 <= value <= 1.5))
    value, ci = _og(likely, "impl|slots_mean|named")
    found.append(_prediction("P2", "Implicit, named, single-slot OG ≤ −2 nats (mean over slots, likely)",
                             value, ci, value is not None and value <= -2))
    value, ci = _og(likely, "impl|all|named")
    found.append(_prediction("P3", "Implicit full removal with all constrained slots named: OG ≤ −6 nats (likely)",
                             value, ci, value is not None and value <= -6))
    gap = likely["wording_gap"]["all"]
    found.append(_prediction("P4", "Wording gap mean ≤ −1 logit (explicit penalized more than implicit; likely)",
                             gap.get("mean"), gap.get("mean_ci95"), gap.get("mean") is not None and gap["mean"] <= -1))
    tie = likely["controls"]["ctrl:tie"]
    found.append(_prediction("P5", "Tie offset |mean l(G|impl) − l(ctrl:tie)| > 0.5 logit (likely)",
                             tie["mean_gap"], tie["mean_gap_ci95"], tie["mean_gap"] is not None and abs(tie["mean_gap"]) > 0.5))
    lm = clean.get("lm")
    if lm:
        values = {
            scorer: mean([
                lm[scorer]["occam"][key]["mean_L_J"]
                for regime in REGIMES
                if (key := f"{regime}|slots_mean|named") in lm[scorer]["occam"]
            ])
            for scorer in LM_SCORERS
        }
        if any(value is None for value in values.values()):
            found.append({"id": "P6", "statement": "Phase 8 language-model check", "value": None, "pass": None,
                          "status": "incomplete: the language-model run does not cover every named single-slot group",
                          "detail": values})
            return found
        passed = values["lm_loglik"] >= 2 and values["lm_typ_normal"] <= 0.5 and values["lm_typ_bins"] <= 0.5
        found.append(_prediction(
            "P6",
            "LM log-likelihood named single-slot L ≥ +2 nats; both typicality scores have L ≤ 0.5 nats "
            "(mean over slots and regimes)",
            values["lm_loglik"], None, passed, detail=values,
        ))
    else:
        found.append({"id": "P6", "statement": "Phase 8 language-model check", "value": None, "pass": None,
                      "status": "pending: no language-model scores yet"})
    return found


def _prediction(identifier: str, statement: str, value, ci, passed: bool, detail=None) -> dict:
    row = {"id": identifier, "statement": statement, "value": value, "ci95": ci, "pass": passed}
    if detail is not None:
        row["detail"] = detail
    return row
