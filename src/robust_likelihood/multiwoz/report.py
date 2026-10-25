"""Written reports for MultiWOZ runs."""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.metrics import EPSILON, empirical_log_ratio, is_clipped, logit_score, median
from robust_likelihood.storage import read_json, read_jsonl

SLOT_LABELS = {"food": "food", "area": "area", "pricerange": "price"}
VERSION_LABELS = {"multiwoz_likely": "likely", "multiwoz_typical": "typical"}
REGIME_LABELS = {"expl": "explicit", "impl": "implicit"}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def goal_key(n_constrained: int, regime: str) -> str:
    """The |C| = 3 goal has one text, so its implicit score serves both regimes."""
    return "G|impl" if regime == "impl" or n_constrained == 3 else "G|expl"


def pilot_report(run_dir: Path, derived: Path, item_count: int = 5) -> str:
    manifest = read_json(run_dir / "manifest.json")
    scores = read_jsonl(run_dir / "scores.jsonl")
    items = {item["item_id"]: item for item in read_jsonl(derived / "items.jsonl")}
    texts = {(row["item_id"], row["hyp_key"]): row["text"] for row in read_jsonl(derived / "probe.jsonl")}
    terms = read_json(derived / "reference_table.json")["variants"]["primary"]["slots"]
    versions = [version for version in VERSION_LABELS if version in manifest["prompts"]]
    noul = {
        (row["item_id"], row["hyp_key"], row["prompt_version"]): row["noul"]
        for row in scores
        if row["repeat_id"] is None
    }
    planned = manifest["planned_calls"]
    scored = sum(1 for row in scores if row["repeat_id"] is None)
    returned = manifest["returned_model_ids"]
    constant = len(returned) == 1 and "none" not in returned
    clipped = sum(1 for row in scores if is_clipped(row["noul"]))
    item_ids = manifest["selection"]["item_ids"]
    lines = [
        "# MultiWOZ pilot: Jev on seeded clean items",
        "",
        f"Run `{run_dir.name}`. Requested `{manifest['requested_model_id']}`; prompt versions "
        f"{', '.join(f'`{version}`' for version in versions)}. {len(item_ids)} clean items drawn with seed "
        f"{manifest['selection']['seed']}. Scores are Jev's elicited yes-probabilities, read as "
        f"l = logit(clip(s, {EPSILON}, 1 - {EPSILON})); they are not likelihoods.",
        "",
        "## Checks",
        "",
        "| Check | Result | Target | Pass |",
        "| --- | --- | --- | --- |",
        f"| Success rate | {scored}/{planned} ({scored / planned:.1%}) | at least 99% | "
        f"{'yes' if planned and scored / planned >= 0.99 else 'no'} |",
        f"| Returned model id | {', '.join(f'{model} ({count})' for model, count in returned.items())} | constant | "
        f"{'yes' if constant else 'no'} |",
    ]
    for version in versions:
        gaps = [
            logit_score(noul[(item_id, "G|impl", version)]) - logit_score(noul[(item_id, "ctrl:contra", version)])
            for item_id in item_ids
            if (item_id, "G|impl", version) in noul and (item_id, "ctrl:contra", version) in noul
        ]
        value = median(gaps)
        lines.append(
            f"| Median l(G\\|impl) − l(ctrl:contra), {VERSION_LABELS[version]} ({len(gaps)} items) | {_fmt(value)} | "
            f"above 2 | {'yes' if value is not None and value > 2 else 'no'} |"
        )
    lines += [
        "",
        f"Scores at or beyond the clip bounds: {clipped} of {len(scores)}. HTTP requests: {manifest['http_requests']}; "
        f"cache hits: {manifest['cache_hits']}; failures: {len(manifest['failures'])}.",
        "",
        "## Five items",
        "",
        "For each item, the first table lists every hypothesis with Jev's score s and its logit l. The second "
        "compares Jev's log-ratio L_J = l(narrow) − l(broader) with the corpus log-ratio L_E for the same removal, "
        "by wording, as in the 25 October pilot. Positive values favour the narrower goal. For a three-slot goal "
        "the explicit and implicit goal texts are the same, so one score serves both.",
        "",
    ]
    for item_id in item_ids[:item_count]:
        lines += _item_section(item_id, items[item_id], texts, noul, versions, terms)
    return "\n".join(lines)


def _item_section(item_id, item, texts, noul, versions, terms) -> list[str]:
    status = item["slot_status"]
    constrained = [slot for slot in SLOTS if slot in item["goal"]]
    goal = ", ".join(f"{SLOT_LABELS[slot]} {item['goal'][slot]}" for slot in constrained)
    turn = ", ".join(f"{SLOT_LABELS[slot]} {status[slot]}" for slot in constrained)
    lines = [
        f"### {item_id}",
        "",
        f"> {item['utterance']}",
        "",
        f"Goal: {goal}. Turn 0: {turn}.",
        "",
        "| Hypothesis | Text | " + " | ".join(f"{VERSION_LABELS[v]} s | {VERSION_LABELS[v]} l" for v in versions) + " |",
        "| --- | --- | " + " | ".join("--- | ---" for _ in versions) + " |",
    ]
    keys = [key for (found, key) in texts if found == item_id]
    for key in keys:
        cells = []
        for version in versions:
            value = noul.get((item_id, key, version))
            cells.append(f"{_fmt(value, 3)} | {_fmt(None if value is None else logit_score(value))}")
        lines.append(f"| `{key}` | {texts[(item_id, key)]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Removed | Turn 0 | Wording | " + " | ".join(f"L_J {VERSION_LABELS[v]}" for v in versions) + " | L_E |",
        "| --- | --- | --- | " + " | ".join("---" for _ in versions) + " | --- |",
    ]
    removals = [([slot], f"minus:{slot}", SLOT_LABELS[slot], status[slot]) for slot in constrained]
    named = sum(1 for slot in constrained if status[slot] == "named")
    removals.append((constrained, "broad", "all", f"{named} of {len(constrained)} named"))
    for removed, prefix, label, turn_label in removals:
        reference = empirical_log_ratio(removed, status, terms)
        for regime in ("expl", "impl"):
            cells = []
            for version in versions:
                narrow = noul.get((item_id, goal_key(len(constrained), regime), version))
                broad = noul.get((item_id, f"{prefix}|{regime}", version))
                gap = None if narrow is None or broad is None else logit_score(narrow) - logit_score(broad)
                cells.append(_fmt(gap))
            lines.append(f"| {label} | {turn_label} | {REGIME_LABELS[regime]} | " + " | ".join(cells) + f" | {_fmt(reference)} |")
    lines.append("")
    return lines
