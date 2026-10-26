"""Written reports and figures for MultiWOZ runs. matplotlib is imported only to draw."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.metrics import (
    EPSILON,
    REGIMES,
    empirical_log_ratio,
    goal_key,
    is_clipped,
    logit_score,
    median,
    removal_pairs,
    score_maps,
    split_key,
    term_vector,
)
from robust_likelihood.storage import read_json, read_jsonl

SLOT_LABELS = {"food": "food", "area": "area", "pricerange": "price"}
VERSION_LABELS = {"multiwoz_likely": "likely", "multiwoz_typical": "typical"}
REGIME_LABELS = {"expl": "explicit", "impl": "implicit"}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


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


GROUP_ROWS = (
    ("food|named", "single slot named: food"),
    ("area|named", "single slot named: area"),
    ("pricerange|named", "single slot named: price"),
    ("slots_mean|named", "single slot named: mean over slots"),
    ("food|omitted", "single slot omitted: food"),
    ("area|omitted", "single slot omitted: area"),
    ("pricerange|omitted", "single slot omitted: price"),
    ("slots_mean|omitted", "single slot omitted: mean over slots"),
    ("all|named", "full removal, every constrained slot named"),
    ("all|mixed", "full removal, some named"),
    ("all|omitted", "full removal, none named"),
)
SIZE_ROWS = tuple((f"all|C{size}k{named}", f"full removal, {size} slots, {named} named") for size in (2, 3) for named in range(size + 1))
BREADTHS = ("G", "minus:food", "minus:area", "minus:pricerange", "broad")
SCORER_LABELS = {
    **VERSION_LABELS,
    "lm_loglik": "LM log-likelihood",
    "lm_typ_normal": "LM typicality (normal)",
    "lm_typ_bins": "LM typicality (bins)",
}


def _ci(bounds, digits: int = 2) -> str:
    return "" if not bounds else f" [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def occam_markdown(block: dict, rows=GROUP_ROWS) -> list[str]:
    lines = [
        "| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for suffix, label in rows:
        expl = block["occam"].get(f"expl|{suffix}")
        impl = block["occam"].get(f"impl|{suffix}")
        if expl is None and impl is None:
            continue
        either = impl or expl
        lines.append(
            f"| {label} | {either['n']} | {_fmt(either['L_E'])}{_ci(either.get('L_E_ci95'))} | "
            f"{_fmt(None if expl is None else expl['mean_L_J'])} | "
            f"{_fmt(None if expl is None else expl['OG'])}{_ci(None if expl is None else expl.get('OG_ci95'))} | "
            f"{_fmt(None if impl is None else impl['mean_L_J'])} | "
            f"{_fmt(None if impl is None else impl['OG'])}{_ci(None if impl is None else impl.get('OG_ci95'))} |"
        )
    return lines


def wording_markdown(scorers: dict) -> list[str]:
    lines = [
        "| Scorer | n | Mean Δ (95%) | Share with abs(Δ) > 1 | " + " | ".join(f"Mean Δ, {name}" for name in BREADTHS) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in BREADTHS) + " |",
    ]
    for scorer, block in scorers.items():
        gap = block["wording_gap"]
        overall = gap["all"]
        cells = " | ".join(_fmt(gap.get(name, {}).get("mean")) for name in BREADTHS)
        lines.append(
            f"| {SCORER_LABELS.get(scorer, scorer)} | {overall.get('n', 0)} | {_fmt(overall.get('mean'))}"
            f"{_ci(overall.get('mean_ci95'))} | {_fmt(overall.get('share_abs_above_1'))} | {cells} |"
        )
    return lines


def monotonicity_markdown(scorers: dict) -> list[str]:
    lines = [
        "| Scorer | Wording | Single slot named: broader wins | Ties | Full removal, all named: broader wins | Ties |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scorer, block in scorers.items():
        for regime in ("expl", "impl"):
            single = block["monotonicity"][f"{regime}|single_named"]
            full = block["monotonicity"][f"{regime}|full_all_named"]
            lines.append(
                f"| {SCORER_LABELS.get(scorer, scorer)} | {REGIME_LABELS[regime]} | "
                f"{single['violations']}/{single['n']} ({_fmt(single['violation_rate'])}) | {single['ties']} | "
                f"{full['violations']}/{full['n']} ({_fmt(full['violation_rate'])}) | {full['ties']} |"
            )
    return lines


def controls_markdown(scorers: dict) -> list[str]:
    lines = [
        "| Scorer | Contradiction AUROC | G minus contra (95%) | by changed slot: food / area / price | "
        "G minus tie (95%) | G minus tie2 | All tie controls (95%) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scorer, block in scorers.items():
        found = block["controls"]
        contra = found["ctrl:contra"]
        by_slot = " / ".join(_fmt(contra["by_changed_slot"].get(slot, {}).get("mean_gap")) for slot in SLOTS)
        tie = found["ctrl:tie"]
        lines.append(
            f"| {SCORER_LABELS.get(scorer, scorer)} | {_fmt(contra.get('auroc'), 3)} | {_fmt(contra['mean_gap'])}"
            f"{_ci(contra['mean_gap_ci95'])} | {by_slot} | {_fmt(tie['mean_gap'])}{_ci(tie['mean_gap_ci95'])} | "
            f"{_fmt(found['ctrl:tie2']['mean_gap'])} | {_fmt(found['tie_offset_all']['mean'])}"
            f"{_ci(found['tie_offset_all']['mean_ci95'])} |"
        )
    return lines


def bias_markdown(scorers: dict) -> list[str]:
    keys = sorted({key for block in scorers.values() for key in block["bias"]})
    lines = [
        "| Hypothesis type | " + " | ".join(SCORER_LABELS.get(scorer, scorer) for scorer in scorers) + " |",
        "| --- | " + " | ".join("---" for _ in scorers) + " |",
    ]
    for key in keys:
        cells = " | ".join(_fmt(block["bias"].get(key, {}).get("mean_centred")) for block in scorers.values())
        hyp_type, regime = key.split("|")
        lines.append(f"| `{hyp_type}`, {REGIME_LABELS[regime]} | {cells} |")
    return lines


def scorecard_markdown(predictions: list[dict]) -> list[str]:
    lines = ["| ID | Prediction | Value (95%) | Result |", "| --- | --- | --- | --- |"]
    for row in predictions:
        result = "pending" if row["pass"] is None else ("pass" if row["pass"] else "fail")
        lines.append(f"| {row['id']} | {row['statement']} | {_fmt(row.get('value'))}{_ci(row.get('ci95'))} | {result} |")
    return lines


def metrics_markdown(metrics: dict) -> str:
    clean = metrics["sets"]["clean"]
    logit = clean["mappings"]["logit"]
    lines = [
        f"# MultiWOZ size-principle metrics: `{metrics['run']}`",
        "",
        metrics["quantity"],
        "",
        f"Scores per prompt version: {metrics['scores']}. Clipped at {metrics['clip_epsilon']}: {metrics['clipped']}. "
        f"Returned model ids: {metrics['returned_model_ids']}. Bootstrap: {metrics['bootstrap']['replicates']} "
        f"replicates; clean items {'paired with' if clean['paired_with_reference'] else 'resampled apart from'} "
        f"the reference resample{' (verified)' if clean.get('pairing_verified') else ''}.",
        "",
        "## Occam gap, clean set, logit",
        "",
        "L_J = l(narrow) − l(broader); L_E is the corpus log-ratio; OG = mean(L_J) − mean(L_E). The target is 0.",
        "",
    ]
    for version, block in logit.items():
        lines += [f"### {SCORER_LABELS.get(version, version)}", "", *occam_markdown(block), "", *occam_markdown(block, SIZE_ROWS), ""]
    lines += ["## Wording gap, clean set, logit", "", "Δ = l(explicit) − l(implicit) for the same breadth. The target is 0.", ""]
    lines += [*wording_markdown(logit), ""]
    lines += ["## Monotonicity, clean set, logit", "", *monotonicity_markdown(logit), ""]
    lines += ["## Controls, clean set, logit", "", *controls_markdown(logit), ""]
    lines += ["## Per-hypothesis bias, clean set, logit", "", "Mean within-item centred score.", "", *bias_markdown(logit), ""]
    lines += ["## Secondary analyses", ""]
    for label, block in (
        ("Log mapping, clean set", clean["mappings"]["log"]),
        ("fail_info set, logit", metrics["sets"].get("fail_info", {}).get("mappings", {}).get("logit", {})),
    ):
        for version, found in block.items():
            lines += [f"### {label}: {SCORER_LABELS.get(version, version)}", "", *occam_markdown(found), ""]
    for name, variant in metrics["variants"].items():
        for version, occam in variant["occam"].items():
            lines += [
                f"### Reference {name} (a from `{variant['a_from']}`, b from `{variant['b_from']}`): "
                f"{SCORER_LABELS.get(version, version)}",
                "",
                *occam_markdown({"occam": occam}),
                "",
            ]
    if "repeat_noise" in metrics:
        lines += [
            f"## Repeat noise: `{metrics['repeat_run']}`",
            "",
            "| Prompt version | Groups | Identical groups | Deterministic | Mean SD of l | Max range of l |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for version, found in metrics["repeat_noise"].items():
            lines.append(
                f"| {version} | {found['groups']} | {found['identical_groups']} | {found['deterministic']} | "
                f"{_fmt(found['mean_sd'], 3)} | {_fmt(found['max_range'], 3)} |"
            )
        lines.append("")
    if "lm" in clean:
        lines += [f"## Language model, clean set: `{metrics.get('lm_run')}`", ""]
        for scorer, block in clean["lm"].items():
            lines += [f"### {SCORER_LABELS[scorer]}", "", *occam_markdown(block), ""]
        lines += [*wording_markdown(clean["lm"]), ""]
    lines += ["## Prediction scorecard", "", *scorecard_markdown(metrics["predictions"]), ""]
    return "\n".join(lines)


# Figures. Colours are the reference palette's first three categorical slots, the
# largest set that stays distinguishable for every pair under colour-vision
# deficiency; marker shape, legend, and the CSV beside each figure carry identity too.

SERIES = {"single_named": "#2a78d6", "single_omitted": "#eb6834", "full": "#1baf7a"}
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
KIND_LABELS = {
    "single_named": "One slot removed; turn 0 names it",
    "single_omitted": "One slot removed; turn 0 omits it",
    "full": "Every constrained slot removed",
}
LABEL_OFFSETS = {"food": (7, -3, "left"), "area": (-7, -3, "right"), "price": (0, 7, "center")}
# Full-removal bins sit close together on x when few slots are named, so two-slot and
# three-slot labels go on opposite sides of their points.
BIN_OFFSETS = {"2": (6, -11, "left"), "3": (6, 5, "left")}
HYP_TYPES = ("G", "minus:food", "minus:area", "minus:pricerange", "broad", "ctrl:contra", "ctrl:tie", "ctrl:tie2")
DASHED = (0, (4, 3))


def bin_label(label: str) -> str:
    size, named = label[1], label.split("k", 1)[1]
    return f"{size} slots, {named.replace('-', '–')} named"


def _point(scorer: str, regime: str, kind: str, group_label: str, group: dict) -> dict:
    l_e = group.get("L_E_ci95") or [None, None]
    l_j = group.get("mean_L_J_ci95") or [None, None]
    og = group.get("OG_ci95") or [None, None]
    return {
        "scorer": scorer,
        "regime": REGIME_LABELS[regime],
        "kind": kind,
        "group": group_label,
        "n": group["n"],
        "L_E": group["L_E"],
        "L_E_low": l_e[0],
        "L_E_high": l_e[1],
        "mean_L_J": group["mean_L_J"],
        "L_J_low": l_j[0],
        "L_J_high": l_j[1],
        "OG": group["OG"],
        "OG_low": og[0],
        "OG_high": og[1],
    }


def headline_rows(block: dict, scorer: str) -> list[dict]:
    """One point per group: single-slot named and omitted removals, then the merged full-removal bins."""
    bins = sorted({key.split("|")[2] for key in block["occam"] if "|bin|" in key})
    rows = []
    for regime in REGIMES:
        for direction, kind in (("named", "single_named"), ("omitted", "single_omitted")):
            for slot in SLOTS:
                group = block["occam"].get(f"{regime}|{slot}|{direction}")
                if group:
                    rows.append(_point(scorer, regime, kind, SLOT_LABELS[slot], group))
        for label in bins:
            group = block["occam"].get(f"{regime}|bin|{label}")
            if group:
                rows.append(_point(scorer, regime, "full", bin_label(label), group))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.titlesize": 10,
            "axes.facecolor": INK["surface"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK["axis"],
            "ytick.color": INK["axis"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "figure.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "legend.labelcolor": INK["secondary"],
        }
    )
    return plt


def _limits(values: list[float | None]) -> tuple[float, float]:
    found = [value for value in values if value is not None] + [0.0]
    return float(int(min(found)) - 1), float(int(max(found)) + 1)


def _reference_lines(ax, limits: tuple[float, float]) -> None:
    ax.axhline(0, color=INK["axis"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=INK["axis"], linewidth=0.8, zorder=1)
    ax.plot(limits, limits, color=INK["muted"], linewidth=1.0, linestyle=DASHED, zorder=1)


def _draw_occam(ax, rows: list[dict], title: str, limits: tuple[float, float], y_label: str) -> None:
    _reference_lines(ax, limits)
    for row in rows:
        color = SERIES[row["kind"]]
        x, y = row["L_E"], row["mean_L_J"]
        xerr = None if row["L_E_low"] is None else [[x - row["L_E_low"]], [row["L_E_high"] - x]]
        yerr = None if row["L_J_low"] is None else [[y - row["L_J_low"]], [row["L_J_high"] - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0, alpha=0.6, zorder=2)
        if row["regime"] == "explicit":
            ax.plot(x, y, marker="o", markersize=7.5, markerfacecolor=color, markeredgecolor=INK["surface"],
                    markeredgewidth=1.5, linestyle="none", zorder=4)
            if row["kind"] != "single_omitted":
                if row["kind"] == "full":
                    dx, dy, align = BIN_OFFSETS.get(row["group"][0], (6, 3, "left"))
                else:
                    dx, dy, align = LABEL_OFFSETS.get(row["group"], (6, 3, "left"))
                ax.annotate(row["group"], (x, y), xytext=(dx, dy), textcoords="offset points", ha=align,
                            fontsize=7.5, color=INK["secondary"], zorder=5)
        else:
            ax.plot(x, y, marker="s", markersize=7, markerfacecolor=INK["surface"], markeredgecolor=color,
                    markeredgewidth=1.8, linestyle="none", zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Corpus log-ratio L_E (nats)")
    ax.set_ylabel(y_label)


def _occam_legend(ax) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=SERIES[kind],
               markeredgecolor=INK["surface"])
        for kind in SERIES
    ]
    labels = [KIND_LABELS[kind] for kind in SERIES]
    handles += [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=INK["muted"],
               markeredgecolor=INK["surface"]),
        Line2D([0], [0], marker="s", linestyle="none", markersize=6.5, markerfacecolor=INK["surface"],
               markeredgecolor=INK["muted"], markeredgewidth=1.8),
        Line2D([0], [0], color=INK["muted"], linewidth=1.0, linestyle=DASHED),
    ]
    labels += ["Explicit wording", "Implicit wording", "y = x: the corpus ratio"]
    ax.legend(handles, labels, loc="upper left")


def plot_headline(stem: Path, panels: list[tuple[str, list[dict]]], y_label: str) -> list[Path]:
    """Mean score log-ratio against corpus log-ratio, one point per group, 95% intervals both ways."""
    plt = _pyplot()
    rows = [row for _, found in panels for row in found]
    limits = _limits([row[key] for row in rows for key in ("L_E_low", "L_E_high", "L_J_low", "L_J_high", "L_E", "mean_L_J")])
    figure, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.4), squeeze=False)
    for ax, (title, found) in zip(axes[0], panels):
        _draw_occam(ax, found, title, limits, y_label)
    _occam_legend(axes[0][0])
    figure.tight_layout()
    written = []
    for suffix in (".png", ".pdf"):
        target = stem.parent / f"{stem.name}{suffix}"
        figure.savefig(target, dpi=200)
        written.append(target)
    plt.close(figure)
    return written


def item_pairs(scores: dict[str, dict[str, float]], items: list[dict], terms: dict) -> list[dict]:
    """Per-item (L_E, L_J) for every removal pair, from point estimates."""
    status = {item["item_id"]: item["slot_status"] for item in items}
    vector = term_vector(terms)
    rows = []
    for pair in removal_pairs(items, scores):
        if pair["signature"] is None:
            continue
        rows.append({"regime": pair["regime"], "L_E": sum(vector[k] for k in pair["signature"]), "L_J": pair["L_J"],
                     "item_id": pair["item_id"], "status": status[pair["item_id"]]})
    return rows


def plot_item_hexbin(target: Path, panels: list[tuple[str, list[float], list[float]]], y_label: str) -> Path:
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    limits = _limits([value for _, xs, ys in panels for value in xs + ys])
    figure, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6), squeeze=False)
    for ax, (title, xs, ys) in zip(axes[0], panels):
        _reference_lines(ax, limits)
        cells = ax.hexbin(xs, ys, gridsize=30, extent=(*limits, *limits), cmap=ramp, mincnt=1,
                          linewidths=0.3, edgecolors=INK["surface"], zorder=2)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Corpus log-ratio L_E (nats)")
        ax.set_ylabel(y_label)
        bar = figure.colorbar(cells, ax=ax, shrink=0.8)
        bar.set_label("pairs", color=INK["secondary"])
        bar.outline.set_visible(False)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target


def plot_score_histograms(
    target: Path, values: dict[str, dict[str, list[float]]], title: str, x_label: str, span=None
) -> Path:
    """Small multiples by hypothesis type; explicit and implicit wordings as two step histograms."""
    plt = _pyplot()
    found = [value for by_regime in values.values() for numbers in by_regime.values() for value in numbers]
    low, high = span or (min(found), max(found))
    edges = [low + (high - low) * index / 24 for index in range(25)]
    types = [hyp_type for hyp_type in HYP_TYPES if hyp_type in values]
    columns = 4
    rows = (len(types) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.5 * rows), squeeze=False, sharex=True)
    for ax in axes.flat[len(types):]:
        ax.set_visible(False)
    for ax, hyp_type in zip(axes.flat, types):
        for regime, color in (("expl", SERIES["single_named"]), ("impl", SERIES["single_omitted"])):
            numbers = values[hyp_type].get(regime, [])
            if numbers:
                ax.hist(numbers, bins=edges, histtype="step", linewidth=2.0, color=color,
                        label=f"{REGIME_LABELS[regime]} (n = {len(numbers)})")
        ax.set_title(hyp_type, loc="left", fontsize=9)
        ax.legend(loc="best", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    figure.suptitle(title, x=0.01, ha="left", color=INK["primary"], fontsize=10)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target


def plot_mu_sigma(target: Path, stats: list[dict]) -> Path:
    """Reference-generator mean and spread of surprisal against the number of slots a hypothesis states."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
    for ax, field, label in ((axes[0][0], "mu", "μ_h: mean surprisal of samples (nats)"),
                             (axes[0][1], "sigma", "σ_h: spread of sample surprisal (nats)")):
        for offset, (regime, color, marker) in enumerate((("expl", SERIES["single_named"], "o"),
                                                          ("impl", SERIES["single_omitted"], "s"))):
            chosen = [row for row in stats if row["regime"] == regime and row.get(field) is not None]
            if not chosen:
                continue
            xs = [row["n_constrained"] + (offset - 0.5) * 0.18 for row in chosen]
            ax.plot(xs, [row[field] for row in chosen], marker=marker, linestyle="none", markersize=4,
                    color=color, alpha=0.35, zorder=2)
            sizes = sorted({row["n_constrained"] for row in chosen})
            means = [median([row[field] for row in chosen if row["n_constrained"] == size]) for size in sizes]
            ax.plot(sizes, means, color=color, linewidth=2.0, marker=marker, markersize=7,
                    markeredgecolor=INK["surface"], label=f"{REGIME_LABELS[regime]} wording: median", zorder=3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Slots the hypothesis constrains")
        ax.set_ylabel(label)
        ax.legend(loc="best")
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return target


def write_figures(run_dir: Path, derived: Path, metrics: dict, lm_dir: Path | None = None) -> list[Path]:
    """Headline Occam figures per wording and side by side, plus the supplementary figures."""
    written: list[Path] = []
    clean = metrics["sets"]["clean"]
    logit = clean["mappings"]["logit"]
    y_label = "Jev log-ratio L_J = l(narrow) − l(broader) (logit)"
    panels = []
    for version, block in logit.items():
        name = VERSION_LABELS.get(version, version)
        rows = headline_rows(block, version)
        write_csv(run_dir / f"fig_occam_{name}.csv", rows)
        written.append(run_dir / f"fig_occam_{name}.csv")
        written += plot_headline(run_dir / f"fig_occam_{name}", [(f"Jev, {name} wording", rows)], y_label)
        panels.append((f"Jev, {name} wording", rows))
    if len(panels) > 1:
        write_csv(run_dir / "fig_occam_both.csv", [row for _, rows in panels for row in rows])
        written.append(run_dir / "fig_occam_both.csv")
        written += plot_headline(run_dir / "fig_occam_both", panels, y_label)
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    terms = read_json(derived / "reference_table.json")["variants"]["primary"]["slots"]
    rows = read_jsonl(run_dir / "scores.jsonl")
    maps = score_maps(rows, logit_score, metrics["clip_epsilon"])
    selected = {item["item_id"] for item in items}
    for version in logit:
        name = VERSION_LABELS.get(version, version)
        pairs = item_pairs(maps[version], items, terms)
        hexbin = [
            (f"{REGIME_LABELS[regime]} wording", [row["L_E"] for row in pairs if row["regime"] == regime],
             [row["L_J"] for row in pairs if row["regime"] == regime])
            for regime in REGIMES
        ]
        written.append(plot_item_hexbin(run_dir / f"fig_items_hexbin_{name}.png", hexbin, y_label))
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["prompt_version"] == version and row.get("repeat_id") is None and row["item_id"] in selected:
                hyp_type, regime = split_key(row["hyp_key"])
                by_type[hyp_type][regime].append(row["noul"])
        written.append(plot_score_histograms(run_dir / f"fig_scores_by_type_{name}.png", by_type,
                                             f"Jev scores by hypothesis type, {name} wording, clean items",
                                             "Jev yes-probability s", span=(0.0, 1.0)))
    if lm_dir is not None and "lm" in clean:
        lm_panels = [(SCORER_LABELS[scorer], headline_rows(block, scorer)) for scorer, block in clean["lm"].items()]
        lm_rows = [row for _, found in lm_panels for row in found]
        if lm_rows:
            write_csv(run_dir / "fig_occam_lm.csv", lm_rows)
            written.append(run_dir / "fig_occam_lm.csv")
            written += plot_headline(
                run_dir / "fig_occam_lm", lm_panels, "LM log-ratio L = score(narrow) − score(broader) (nats)"
            )
        stats = read_jsonl(lm_dir / "lm_hypotheses.jsonl")
        if stats:
            written.append(plot_mu_sigma(run_dir / "fig_lm_mu_sigma.png", stats))
    return written


def _og_sentence(block: dict, key: str) -> str:
    group = block["occam"].get(key)
    if group is None:
        return "n/a"
    return f"{_fmt(group['OG'])}{_ci(group.get('OG_ci95'))}"


def _headline_table(metrics: dict) -> list[str]:
    logit = metrics["sets"]["clean"]["mappings"]["logit"]
    rows = (
        ("slots_mean|named", "One named slot removed (mean over slots)"),
        ("slots_mean|omitted", "One omitted slot removed (mean over slots)"),
        ("all|named", "Every constrained slot removed, all named"),
        ("all|mixed", "Every constrained slot removed, some named"),
    )
    header = "| Pairs | L_E (nats) | " + " | ".join(
        f"{VERSION_LABELS.get(version, version)} {wording} mean L_J | OG (95%)"
        for version in logit
        for wording in ("explicit", "implicit")
    ) + " |"
    lines = [header, "| --- | --- | " + " | ".join("--- | ---" for _ in logit for _ in range(2)) + " |"]
    for suffix, label in rows:
        cells = []
        reference = None
        for block in logit.values():
            for regime in REGIMES:
                group = block["occam"].get(f"{regime}|{suffix}")
                reference = reference if group is None else group["L_E"]
                cells.append(
                    "n/a | n/a" if group is None
                    else f"{_fmt(group['mean_L_J'])} | {_fmt(group['OG'])}{_ci(group.get('OG_ci95'))}"
                )
        lines.append(f"| {label} | {_fmt(reference)} | " + " | ".join(cells) + " |")
    return lines


def summary_markdown(
    metrics: dict, reference: dict, manifest: dict, commit: str, lm_manifest: dict | None = None
) -> str:
    clean = metrics["sets"]["clean"]
    logit = clean["mappings"]["logit"]
    likely = logit.get("multiwoz_likely")
    typical = logit.get("multiwoz_typical")
    primary = reference["variants"]["primary"]
    selection = reference["selection"]
    named_expl = likely["occam"]["expl|slots_mean|named"]
    named_impl = likely["occam"]["impl|slots_mean|named"]
    full_impl = likely["occam"].get("impl|all|named")
    full_expl = likely["occam"].get("expl|all|named")
    lines = [
        "# MultiWOZ size-principle test: Jev's scores against corpus likelihood ratios",
        "",
        "Claim under test: a likelihood obeys the size principle. If the user's first turn names a slot value, "
        "a goal that constrains that slot must beat a goal with no preference about it by the ratio real users "
        "show, ln(a/b). Jev's scores are elicited yes-probabilities read on the logit scale. The corpus ratios "
        "come from MultiWOZ users.",
        "",
        "![Jev log-ratio against corpus log-ratio](fig_occam_both.png)",
        "",
        "## 1. Setup and counts",
        "",
        f"- Corpus: MultiWOZ 2.1 goals and 2.2 turn annotations at commit `{commit[:12]}`, "
        f"{selection['dialogues']['v21']:,} and {selection['dialogues']['v22']:,} dialogues.",
        f"- Items: first user turns of single-domain restaurant dialogues (SNG) with a slot-based goal: "
        f"{selection['slot_based']['clean']} clean and {selection['slot_based']['fail_info']} with fail_info "
        f"({selection['name_based']['clean'] + selection['name_based']['fail_info']} name-based goals excluded). "
        "The clean set is the primary analysis.",
        f"- Hypotheses: the goal, the goal minus each constrained slot, and the broad goal, each in an implicit "
        "wording and an explicit wording that adds \"The user has no preference about ...\", plus contradiction "
        f"and tie controls. One hypothesis per call. Run `{metrics['run']}`: {sum(metrics['scores'].values()):,} "
        f"scores from `{metrics['requested_model_id']}` ({_returned(metrics['returned_model_ids'])}), "
        f"{manifest.get('http_requests', 0):,} HTTP requests, {len(manifest.get('failures', []))} failures, "
        f"{sum(metrics['clipped'].values())} scores clipped at {metrics['clip_epsilon']}.",
        *_repeat_lines(metrics),
        f"- Prompt versions: {', '.join(f'`{version}`' for version in metrics['prompt_versions'])}; "
        "`stated_intention` is the constant `find_restaurant` and the hypothesis text is the intention description.",
        f"- Intervals: {metrics['bootstrap']['replicates']:,} paired bootstrap replicates. Replicate b resamples "
        "items for L_J and takes reference replicate b for L_E"
        + ("; for the clean set it is the very resample that produced a." if clean.get("pairing_verified") else "."),
        "",
        "## 2. Reference table",
        "",
        f"a from `{primary['a_from']}`, b from `{primary['b_from']}` with +{reference['b_smoothing']} smoothing.",
        "",
        "| Slot | a | b | Named: ln(a/b) (95%) | Omitted: ln((1−a)/(1−b)) (95%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for slot in SLOTS:
        row = primary["slots"][slot]
        lines.append(
            f"| {SLOT_LABELS[slot]} | {_fmt(row['a'], 3)} | {_fmt(row['b'], 5)} | {_fmt(row['named'])}"
            f"{_ci(row['named_boot95'])} | {_fmt(row['omitted'])}{_ci(row['omitted_boot95'])} |"
        )
    lines += [
        "",
        f"All three slots named and removed: {_fmt(primary['all_named'])} nats. {reference['conservative']} "
        "Full tables and sensitivity variants: `data/multiwoz/derived/reference_table.md`.",
        "",
        "## 3. Headline: Occam gap",
        "",
        "OG = mean(L_J) − L_E. Zero means Jev's breadth penalty matches the corpus; negative means Jev "
        "penalizes breadth less than real users' first turns warrant.",
        "",
        *_headline_table(metrics),
        "",
        f"With the `likely` question, removing one named slot costs the broader goal {_fmt(named_expl['mean_L_J'])} "
        f"logits under the explicit wording and {_fmt(named_impl['mean_L_J'])} under the implicit wording, against "
        f"{_fmt(named_expl['L_E'])} nats in the corpus. Removing every constrained slot when the turn names all of "
        f"them costs {_fmt(None if full_expl is None else full_expl['mean_L_J'])} (explicit) and "
        f"{_fmt(None if full_impl is None else full_impl['mean_L_J'])} (implicit) against "
        f"{_fmt(None if full_impl is None else full_impl['L_E'])} nats.",
        "",
        f"Jev's penalty is {_share(named_expl['mean_L_J'], named_expl['L_E'])} of the corpus ratio for one named "
        f"slot under the explicit wording and {_share(named_impl['mean_L_J'], named_impl['L_E'])} under the implicit "
        f"wording; for the full removal it is {_share(None if full_expl is None else full_expl['mean_L_J'], None if full_expl is None else full_expl['L_E'])} "
        f"and {_share(None if full_impl is None else full_impl['mean_L_J'], None if full_impl is None else full_impl['L_E'])}.",
        "",
        *_robustness_lines(metrics),
        "",
        "Per-slot and per-bin values with intervals: `metrics.md`. Figure data: `fig_occam_likely.csv`, "
        "`fig_occam_typical.csv`.",
        "",
        "## 4. Wording gap",
        "",
        "Δ = l(explicit) − l(implicit) for the same breadth. The two texts share their first sentence word for "
        "word, so any gap comes from stating the absence of a preference. The target is 0.",
        "",
        *wording_markdown(logit),
        "",
        "## 5. Monotonicity",
        "",
        "Counts of pairs where the broader goal strictly outscores the narrower one although the turn names "
        "every removed slot.",
        "",
        *monotonicity_markdown(logit),
        "",
        "## 6. Controls and tie offset",
        "",
        "Contradiction: one named slot changed to another value; the goal should win. Tie: one omitted slot "
        "changed; the utterance cannot tell the two apart, so the target gap is 0.",
        "",
        *controls_markdown(logit),
        "",
        _bias_sentence(likely),
        "",
        "## 7. Reverse direction",
        "",
        "When the turn omits a constrained slot, the broader goal should win by ln((1−b)/(1−a)), about 1.3 nats "
        "per slot.",
        "",
        "| Prompt version | Wording | n | L_E | Mean L_J | OG (95%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for version, block in logit.items():
        for regime in REGIMES:
            group = block["occam"].get(f"{regime}|slots_mean|omitted")
            if group:
                lines.append(
                    f"| {VERSION_LABELS.get(version, version)} | {REGIME_LABELS[regime]} | {group['n']} | "
                    f"{_fmt(group['L_E'])} | {_fmt(group['mean_L_J'])} | {_fmt(group['OG'])}{_ci(group.get('OG_ci95'))} |"
                )
    lines += ["", *_reverse_lines(logit)]
    lines += [
        "",
        "## 8. Typicality and the language model",
        "",
    ]
    if typical is not None:
        lines.append(
            f"The `typical` question gives a named single-slot OG of "
            f"{_og_sentence(typical, 'expl|slots_mean|named')} (explicit) and "
            f"{_og_sentence(typical, 'impl|slots_mean|named')} (implicit), against "
            f"{_og_sentence(likely, 'expl|slots_mean|named')} and {_og_sentence(likely, 'impl|slots_mean|named')} "
            "for `likely`."
        )
        lines.append("")
    if "lm" in clean:
        lm = clean["lm"]
        lines += [
            f"Language model run `{metrics.get('lm_run')}`: `{(lm_manifest or {}).get('model')}` at revision "
            f"`{(lm_manifest or {}).get('revision')}`, {(lm_manifest or {}).get('samples_per_hypothesis')} samples "
            f"per hypothesis, {_fmt((lm_manifest or {}).get('drop_rate'), 3)} of samples dropped at max_new_tokens.",
            "",
            "| Scorer | Wording | Named single-slot mean L (nats) | OG (95%) | Full removal, all named: mean L |",
            "| --- | --- | --- | --- | --- |",
        ]
        for scorer, block in lm.items():
            for regime in REGIMES:
                single = block["occam"].get(f"{regime}|slots_mean|named")
                full = block["occam"].get(f"{regime}|all|named")
                lines.append(
                    f"| {SCORER_LABELS[scorer]} | {REGIME_LABELS[regime]} | {_fmt(None if single is None else single['mean_L_J'])} | "
                    f"{_og_sentence(block, f'{regime}|slots_mean|named')} | {_fmt(None if full is None else full['mean_L_J'])} |"
                )
        lines += ["", "![Language-model scores against corpus log-ratio](fig_occam_lm.png)", "",
                  "![Reference-generator surprisal against breadth](fig_lm_mu_sigma.png)", ""]
    else:
        lines += [
            "The language-model check (Phase 8) has not run yet. It needs a GPU and the user's approval; "
            "`scripts/multiwoz_lm_job.sh` holds the job.",
            "",
        ]
    lines += ["## 9. Prediction scorecard", "", *scorecard_markdown(metrics["predictions"]), ""]
    lines += ["## 10. Notes for Xuan", ""]
    lines += notes_for_xuan(metrics, lm_manifest)
    return "\n".join(lines) + "\n"


def notes_for_xuan(metrics: dict, lm_manifest: dict | None) -> list[str]:
    clean = metrics["sets"]["clean"]
    likely = clean["mappings"]["logit"]["multiwoz_likely"]
    named_expl = likely["occam"]["expl|slots_mean|named"]
    named_impl = likely["occam"]["impl|slots_mean|named"]
    gap = likely["wording_gap"]["all"]
    notes = [
        f"- Removing one slot the turn names costs the broader goal {_fmt(named_expl['mean_L_J'])} logits under "
        f"the explicit wording and {_fmt(named_impl['mean_L_J'])} under the implicit wording; real users' first "
        f"turns give {_fmt(named_expl['L_E'])} nats. {_verdict(named_expl, named_impl)} The logit reading is the "
        "generous one for Jev, so a shortfall is not an artefact of the score mapping.",
        f"- Stating the absence of a preference moves the score by {_fmt(gap.get('mean'))} logits on average at "
        f"the same breadth ({_share(gap.get('share_abs_above_1'), 1.0)} of pairs move by more than 1), while removing "
        f"a named slot under the implicit wording moves it by {_fmt(named_impl['mean_L_J'])}. Jev's breadth penalty "
        "follows how breadth is worded more than the breadth itself.",
        *_scorecard_notes(metrics),
    ]
    if "lm" not in clean:
        notes.append(
            "- Whether normalization cancels Occam's razor is still open: it needs the Phase 8 language-model run. "
            "The diagnostic is μ_h, the mean surprisal of the model's own samples for hypothesis h, against the "
            "number of slots h constrains. If μ_h rises as hypotheses broaden, T = (S − μ_h)/σ_h subtracts exactly "
            "the size penalty that the raw log-likelihood −S keeps, and the typicality scores will show L near 0 "
            "where the log-likelihood shows L ≥ 2."
        )
        return notes
    lm = clean["lm"]
    loglik = mean_regimes(lm["lm_loglik"])
    normal = mean_regimes(lm["lm_typ_normal"])
    bins = mean_regimes(lm["lm_typ_bins"])
    notes.append(
        f"- Normalization and Occam's razor: for named single-slot removals the language model's log-likelihood "
        f"gives L = {_fmt(loglik)} nats, the normal typicality score {_fmt(normal)}, and the binned score "
        f"{_fmt(bins)}. See `fig_lm_mu_sigma.png` for μ_h against the number of constrained slots; a μ_h that "
        "rises as hypotheses broaden is the mechanism by which T removes the size penalty."
    )
    return notes


def _returned(found: dict) -> str:
    if len(found) == 1:
        return f"every response reported `{next(iter(found))}`"
    return "responses reported " + ", ".join(f"`{model}` ({count})" for model, count in found.items())


def _share(part: float | None, whole: float | None) -> str:
    if part is None or not whole:
        return "n/a"
    return f"{part / whole:.0%}"


def _repeat_lines(metrics: dict) -> list[str]:
    noise = metrics.get("repeat_noise")
    if not noise:
        return []
    lines = []
    for version, found in noise.items():
        if found["deterministic"]:
            verdict = "Jev returned identical scores in every group: it is deterministic here."
        else:
            verdict = (
                f"Jev is not deterministic: {found['identical_groups']} of {found['groups']} groups were identical. "
                f"The mean within-group SD is {_fmt(found['mean_sd'], 3)} logits and the largest range "
                f"{_fmt(found['max_range'], 2)}."
            )
        lines.append(
            f"- Repeat check, run `{metrics['repeat_run']}`, `{version}`: each (item, hypothesis) scored "
            f"{'/'.join(str(size) for size in found['observations_per_group'])} times. {verdict}"
        )
    return lines


def _robustness_lines(metrics: dict) -> list[str]:
    key = "slots_mean|named"
    clean = metrics["sets"]["clean"]
    parts = []
    log = clean["mappings"]["log"].get("multiwoz_likely")
    if log:
        parts.append(
            f"the log mapping ({_og_sentence(log, f'expl|{key}')} explicit, {_og_sentence(log, f'impl|{key}')} implicit)"
        )
    fail = metrics["sets"].get("fail_info", {}).get("mappings", {}).get("logit", {}).get("multiwoz_likely")
    if fail:
        parts.append(
            f"the fail_info set ({_og_sentence(fail, f'expl|{key}')} explicit, {_og_sentence(fail, f'impl|{key}')} implicit)"
        )
    variants = []
    for name, variant in metrics.get("variants", {}).items():
        found = variant["occam"].get("multiwoz_likely", {})
        if f"expl|{key}" in found and f"impl|{key}" in found:
            variants.append(f"{name} {_fmt(found[f'expl|{key}']['OG'])} / {_fmt(found[f'impl|{key}']['OG'])}")
    if variants:
        parts.append("the reference variants (explicit / implicit: " + ", ".join(variants) + ")")
    if not parts:
        return []
    return ["Named single-slot OG under `likely` in the secondary analyses: " + "; ".join(parts) + "."]


def _bias_sentence(block: dict) -> str:
    found = {key: row["mean_centred"] for key, row in block["bias"].items() if not key.startswith("ctrl:contra")}
    if not found:
        return ""
    high = max(found, key=found.get)
    low = min(found, key=found.get)

    def name(key: str) -> str:
        hyp_type, regime = key.split("|")
        return f"`{hyp_type}` ({REGIME_LABELS[regime]})"

    return (
        f"Within each item, `likely` favours {name(high)} most (mean centred score {_fmt(found[high])}) and "
        f"{name(low)} least ({_fmt(found[low])}), leaving the contradiction controls aside. The full table is in "
        "`metrics.md`."
    )


def _reverse_lines(logit: dict) -> list[str]:
    parts = []
    for version, block in logit.items():
        expl = block["occam"].get("expl|slots_mean|omitted")
        impl = block["occam"].get("impl|slots_mean|omitted")
        if expl and impl:
            parts.append(
                f"with `{VERSION_LABELS.get(version, version)}`, the broader goal wins by {_fmt(-expl['mean_L_J'])} "
                f"(explicit) and {_fmt(-impl['mean_L_J'])} (implicit) logits against {_fmt(-expl['L_E'])} nats: "
                + _verdict(expl, impl, reverse=True, subject="the win")[:-1].replace(" (95% intervals)", "")
            )
    if not parts:
        return []
    sentence = "; ".join(parts)
    return [sentence[0].upper() + sentence[1:] + " (95% intervals)."]


def _verdict(explicit: dict, implicit: dict, reverse: bool = False, subject: str = "Jev's penalty") -> str:
    first, second = _versus(explicit, reverse), _versus(implicit, reverse)
    if first == second:
        return f"{subject} {first} the corpus ratio under both wordings (95% intervals)."
    return (
        f"{subject} {first} the corpus ratio under the explicit wording and {second} it under the "
        "implicit wording (95% intervals)."
    )


def _scorecard_notes(metrics: dict) -> list[str]:
    failed = [row for row in metrics["predictions"] if row["pass"] is False]
    if not failed:
        return []
    names = ", ".join(f"{row['id']} ({_fmt(row['value'])})" for row in failed)
    return [f"- Pre-registered predictions that failed: {names}. A failed prediction is a result; see section 9."]


def _versus(group: dict, reverse: bool = False) -> str:
    """How Jev's gap compares with the corpus ratio. When the turn omits the removed slots (reverse), L_E
    is negative, so a positive OG means the broader goal wins by less than the corpus says."""
    bounds = group.get("OG_ci95")
    if not bounds:
        return "is compared with"
    short, over = ("falls short of", "exceeds") if not reverse else ("exceeds", "falls short of")
    if bounds[1] < 0:
        return short
    if bounds[0] > 0:
        return over
    return "is consistent with"


def mean_regimes(block: dict) -> float | None:
    values = [block["occam"][f"{regime}|slots_mean|named"]["mean_L_J"] for regime in REGIMES
              if f"{regime}|slots_mean|named" in block["occam"]]
    return sum(values) / len(values) if values else None
