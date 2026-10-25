"""Corpus rates for the size principle and the empirical log-ratio per slot.

For slot X, a_X is the rate at which the turn names X when the goal constrains X,
and b_X is the rate when the goal has no preference about X (absent or dontcare).
Removing a slot the turn names contributes ln(a_X/b_X) to the empirical log-ratio;
removing one it omits contributes ln((1-a_X)/(1-b_X)).

The ratio ignores which value was named. Under a no-preference goal a specific
value is less likely still, so the true likelihood ratio is at least a/b: the
reference is conservative.

The bootstrap resamples dialogues within each population, with a generator seeded
from the population name. Rows are in dialogue-id order, so Phase 6 can redraw the
same resample of the clean items and pair Jev's scores with the a-rates computed
from that resample.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter

from robust_likelihood.multiwoz.corpus import SLOTS, differences

Z95 = 1.959963984540054
POPULATIONS = ("match_clean_turn0", "match_all_turn0", "pooled_first_restaurant_turn", "pooled_no_woz")
VARIANTS = {
    "V2": ("match_all_turn0", "match_all_turn0"),
    "V3": ("pooled_first_restaurant_turn", "pooled_first_restaurant_turn"),
    "V4": ("pooled_no_woz", "pooled_no_woz"),
}
REPLICATE_FIELDS = ("named", "constrained", "no_pref_named", "no_pref")

APPENDIX_A2 = {
    "pooled_first_restaurant_turn": {
        "size": 3927,
        "food": {"named": 1809, "constrained": 3108, "no_pref_named": 5, "no_pref": 819},
        "area": {"named": 1730, "constrained": 3104, "no_pref_named": 12, "no_pref": 823},
        "pricerange": {"named": 1758, "constrained": 3167, "no_pref_named": 10, "no_pref": 760},
    },
    "pooled_no_woz": {
        "food": {"no_pref_named": 3, "no_pref": 567},
        "area": {"no_pref_named": 8, "no_pref": 559},
        "pricerange": {"no_pref_named": 8, "no_pref": 498},
    },
}
APPENDIX_A3 = {
    "food": {"a": 0.728, "b": 0.00671, "named": 4.69, "omitted": -1.30},
    "area": {"a": 0.742, "b": 0.01517, "named": 3.89, "omitted": -1.34},
    "pricerange": {"a": 0.750, "b": 0.01380, "named": 4.00, "omitted": -1.37},
    "all_named": 12.57,
}
A3_TOLERANCE = 0.01


def slot_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts = {}
    for slot in SLOTS:
        constrained = [row for row in rows if slot in row["constrained"]]
        no_pref = [row for row in rows if slot not in row["constrained"]]
        named = [row for row in constrained if row["status"][slot] == "named"]
        counts[slot] = {
            "constrained": len(constrained),
            "named": len(named),
            "named_mismatch": sum(1 for row in named if row["match"][slot] is False),
            "no_pref": len(no_pref),
            "no_pref_named": sum(1 for row in no_pref if row["status"][slot] == "named"),
        }
    return counts


def a_rate(named: int, constrained: int) -> float | None:
    return named / constrained if constrained else None


def b_rate(no_pref_named: int, no_pref: int, smoothing: float) -> float:
    """(#named + s) / (n + 2s); with s = 0.5 this is (#named + 0.5) / (n + 1)."""
    return (no_pref_named + smoothing) / (no_pref + 2 * smoothing)


def wilson(successes: int, trials: int, z: float = Z95) -> list[float] | None:
    if not trials:
        return None
    p = successes / trials
    scale = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / scale
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / scale
    return [centre - half, centre + half]


def log_ratios(a: float | None, b: float | None) -> dict[str, float | None]:
    """Reference term for a removed slot: named -> ln(a/b), omitted -> ln((1-a)/(1-b))."""
    named = math.log(a / b) if a and b else None
    omitted = math.log((1 - a) / (1 - b)) if a is not None and b is not None and a < 1 and b < 1 else None
    return {"named": named, "omitted": omitted}


def bootstrap_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def resample(size: int, rng: random.Random) -> list[int]:
    return rng.choices(range(size), k=size)


def _pattern(row: dict) -> tuple:
    return tuple((slot in row["constrained"], row["status"][slot] == "named") for slot in SLOTS)


def replicate_counts(rows: list[dict], replicates: int, seed: int, label: str) -> dict[str, list[list[int]]]:
    """Per slot and replicate: [named, constrained, no_pref_named, no_pref] in a resample of rows."""
    rows = sorted(rows, key=lambda row: row["dialogue_id"])
    patterns = [_pattern(row) for row in rows]
    rng = bootstrap_rng(seed, label)
    found = {slot: [] for slot in SLOTS}
    for _ in range(replicates):
        tally = Counter(patterns[index] for index in resample(len(rows), rng))
        for position, slot in enumerate(SLOTS):
            named = constrained = no_pref_named = no_pref = 0
            for pattern, count in tally.items():
                is_constrained, is_named = pattern[position]
                if is_constrained:
                    constrained += count
                    named += count if is_named else 0
                else:
                    no_pref += count
                    no_pref_named += count if is_named else 0
            found[slot].append([named, constrained, no_pref_named, no_pref])
    return found


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval(values: list[float | None]) -> list[float] | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return [percentile(usable, 0.025), percentile(usable, 0.975)]


def replicate_terms(replicates: dict, a_from: str, b_from: str, smoothing: float) -> list[dict[str, dict]]:
    """Per replicate: the a, b, and reference terms of every slot for one variant."""
    count = len(replicates[a_from][SLOTS[0]])
    found = []
    for index in range(count):
        slots = {}
        for slot in SLOTS:
            named, constrained, _, _ = replicates[a_from][slot][index]
            _, _, no_pref_named, no_pref = replicates[b_from][slot][index]
            a = a_rate(named, constrained)
            b = b_rate(no_pref_named, no_pref, smoothing)
            slots[slot] = {"a": a, "b": b, **log_ratios(a, b)}
        found.append(slots)
    return found


def _variant(name: str, a_from: str, b_from: str, counts: dict, replicates: dict, smoothing: float) -> dict:
    boot = replicate_terms(replicates, a_from, b_from, smoothing)
    slots = {}
    for slot in SLOTS:
        a = a_rate(counts[a_from][slot]["named"], counts[a_from][slot]["constrained"])
        b = b_rate(counts[b_from][slot]["no_pref_named"], counts[b_from][slot]["no_pref"], smoothing)
        terms = log_ratios(a, b)
        slots[slot] = {
            "a": a,
            "a_wilson95": wilson(counts[a_from][slot]["named"], counts[a_from][slot]["constrained"]),
            "a_boot95": interval([row[slot]["a"] for row in boot]),
            "b": b,
            "b_boot95": interval([row[slot]["b"] for row in boot]),
            "named": terms["named"],
            "named_boot95": interval([row[slot]["named"] for row in boot]),
            "omitted": terms["omitted"],
            "omitted_boot95": interval([row[slot]["omitted"] for row in boot]),
        }
    totals = [
        None if any(row[slot]["named"] is None for slot in SLOTS) else sum(row[slot]["named"] for slot in SLOTS)
        for row in boot
    ]
    all_named = None if any(slots[slot]["named"] is None for slot in SLOTS) else sum(slots[slot]["named"] for slot in SLOTS)
    return {
        "name": name,
        "a_from": a_from,
        "b_from": b_from,
        "slots": slots,
        "all_named": all_named,
        "all_named_boot95": interval(totals),
    }


def build_reference(populations: dict[str, list[dict]], config: dict) -> tuple[dict, dict]:
    """The reference table and the bootstrap replicate counts behind its intervals."""
    seed = int(config["bootstrap"]["seed"])
    replicates = int(config["bootstrap"]["replicates"])
    smoothing = float(config["reference"]["b_smoothing"])
    counts = {name: slot_counts(rows) for name, rows in populations.items()}
    boot = {name: replicate_counts(rows, replicates, seed, name) for name, rows in populations.items()}
    variants = {
        "primary": _variant(
            "primary", config["reference"]["a_population"], config["reference"]["b_population"], counts, boot, smoothing
        )
    }
    for name, (a_from, b_from) in VARIANTS.items():
        if a_from in populations and b_from in populations:
            variants[name] = _variant(name, a_from, b_from, counts, boot, smoothing)
    table = {
        "quantity": (
            "Corpus naming rates of real MultiWOZ users. a: the turn names the slot when the goal constrains it. "
            "b: the turn names it when the goal has no preference about it (absent or dontcare)."
        ),
        "conservative": (
            "The slot-level ratio ignores which value was named. Under a no-preference goal a specific value is "
            "less likely still, so the true likelihood ratio is at least a/b."
        ),
        "b_smoothing": smoothing,
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "unit": "dialogue, resampled within each population",
            "rng": "random.Random(int(sha256(f'{seed}:{population}').hexdigest()[:16], 16)); "
            "rng.choices(range(n), k=n) per replicate over rows sorted by dialogue id",
        },
        "populations": {
            name: {
                "size": len(rows),
                "counts": counts[name],
                "mismatch_rate": {
                    slot: counts[name][slot]["named_mismatch"] / counts[name][slot]["named"]
                    if counts[name][slot]["named"]
                    else None
                    for slot in SLOTS
                },
            }
            for name, rows in populations.items()
        },
        "variants": variants,
    }
    return table, {"fields": list(REPLICATE_FIELDS), "populations": boot, "replicates": replicates, "seed": seed}


def appendix_checks(table: dict) -> dict[str, list[str]]:
    """Exact counts for Appendix A2 and rates within 0.01 for A3."""
    found = {}
    for name, expected in APPENDIX_A2.items():
        population = table["populations"].get(name, {})
        actual = {"size": population.get("size"), **population.get("counts", {})}
        found.setdefault("A2", []).extend(differences(expected, actual, name))
    primary = table["variants"]["primary"]
    a3 = []
    for slot in SLOTS:
        for key in ("a", "b", "named", "omitted"):
            value = primary["slots"][slot][key]
            if value is None or abs(value - APPENDIX_A3[slot][key]) > A3_TOLERANCE:
                a3.append(f"{slot}.{key}: expected {APPENDIX_A3[slot][key]}, got {value}")
    if primary["all_named"] is None or abs(primary["all_named"] - APPENDIX_A3["all_named"]) > A3_TOLERANCE:
        a3.append(f"all_named: expected {APPENDIX_A3['all_named']}, got {primary['all_named']}")
    found["A3"] = a3
    return found


def _number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(bounds: list[float] | None, digits: int) -> str:
    return "n/a" if bounds is None else f"[{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


SLOT_LABELS = {"food": "food", "area": "area", "pricerange": "price"}


def reference_markdown(table: dict, selection: dict, pooled_counts: dict) -> str:
    primary = table["variants"]["primary"]
    lines = [
        "# MultiWOZ reference rates",
        "",
        "Written by `python -m robust_likelihood.multiwoz prepare` from the pinned MultiWOZ 2.1 goals and "
        "2.2 turn annotations. These are rates of real users, not model scores.",
        "",
        "## Selection",
        "",
        "| Quantity | Count |",
        "| --- | --- |",
        f"| Dialogues in 2.1 / in 2.2 | {selection['dialogues']['v21']:,} / {selection['dialogues']['v22']:,} |",
        f"| With a restaurant goal / restaurant-only / SNG restaurant-only | {selection['restaurant_goal']:,} / "
        f"{selection['restaurant_only']:,} / {selection['sng_restaurant_only']:,} |",
        f"| Name-based (excluded): clean / fail_info | {selection['name_based']['clean']} / "
        f"{selection['name_based']['fail_info']} |",
        f"| Slot-based: clean / fail_info | {selection['slot_based']['clean']} / {selection['slot_based']['fail_info']} |",
        f"| Clean items by constrained slots: 2 / 3 | {selection['clean_by_n_constrained'].get('2', 0)} / "
        f"{selection['clean_by_n_constrained'].get('3', 0)} |",
        f"| Items with no named constrained slot | {selection['no_named_constrained']} |",
        "",
        "Clean set, turn-0 slot status. `dontcare` is its own status, not a named value.",
        "",
        "| Slot | Constrained | Named (mismatch) | Omitted | No-pref | No-pref named | No-pref dontcare |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for slot in SLOTS:
        row = selection["clean_turn0"][slot]
        lines.append(
            f"| {SLOT_LABELS[slot]} | {row['constrained']} | {row['named']} ({row['named_mismatch']}) | {row['omitted']} | "
            f"{row['no_pref']} | {row['no_pref_named']} | {row['no_pref_dontcare']} |"
        )
    lines += [
        "",
        "## Populations",
        "",
        f"The pooled population is the first restaurant turn of every dialogue, any prefix, with a slot-based "
        f"restaurant goal: {pooled_counts.get('slot_based_goal', 0):,} such goals, of which "
        f"{pooled_counts.get('no_restaurant_turn', 0)} never reach a restaurant turn and "
        f"{pooled_counts.get('missing_in_v22', 0)} are missing from 2.2.",
        "",
        "| Population | Dialogues | Slot | Named / constrained | a | No-pref named / n | Value mismatch among named |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, population in table["populations"].items():
        for slot in SLOTS:
            row = population["counts"][slot]
            a = a_rate(row["named"], row["constrained"])
            lines.append(
                f"| `{name}` | {population['size']:,} | {SLOT_LABELS[slot]} | {row['named']:,} / {row['constrained']:,} | "
                f"{_number(a, 3)} | {row['no_pref_named']} / {row['no_pref']:,} | "
                f"{_number(population['mismatch_rate'][slot], 3)} |"
            )
    lines += [
        "",
        f"## Primary reference: a from `{primary['a_from']}`, b from `{primary['b_from']}` with +{table['b_smoothing']} smoothing",
        "",
        "| Slot | a (Wilson 95%) | b (bootstrap 95%) | Named: ln(a/b) (95%) | Omitted: ln((1−a)/(1−b)) (95%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for slot in SLOTS:
        row = primary["slots"][slot]
        lines.append(
            f"| {SLOT_LABELS[slot]} | {_number(row['a'], 3)} {_ci(row['a_wilson95'], 3)} | "
            f"{_number(row['b'], 5)} {_ci(row['b_boot95'], 5)} | {_number(row['named'], 2)} {_ci(row['named_boot95'], 2)} | "
            f"{_number(row['omitted'], 2)} {_ci(row['omitted_boot95'], 2)} |"
        )
    lines += [
        "",
        f"All three slots named and removed: L_E = {_number(primary['all_named'], 2)} nats "
        f"{_ci(primary['all_named_boot95'], 2)}.",
        "",
        "## Sensitivity variants",
        "",
        "| Variant | a from | b from | Slot | a | b | Named | Omitted |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, variant in table["variants"].items():
        for slot in SLOTS:
            row = variant["slots"][slot]
            lines.append(
                f"| {name} | `{variant['a_from']}` | `{variant['b_from']}` | {SLOT_LABELS[slot]} | {_number(row['a'], 3)} | "
                f"{_number(row['b'], 5)} | {_number(row['named'], 2)} | {_number(row['omitted'], 2)} |"
            )
    lines += [
        "",
        "## Reading",
        "",
        "a must match the probe population: turn 0 of single-domain restaurant dialogues. b is a rare-event rate, "
        "so it comes from the pooled corpus, where the matched population would have only 0 or 1 events per slot.",
        "",
        table["conservative"],
        "",
        f"Intervals are percentile bootstrap intervals over {table['bootstrap']['replicates']:,} replicates that "
        "resample dialogues within each population; the a interval in the primary table is Wilson's.",
        "",
    ]
    return "\n".join(lines)
