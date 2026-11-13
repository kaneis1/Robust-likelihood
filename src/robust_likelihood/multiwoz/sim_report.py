"""Simulator study, Phase 4: summary.md and figures from docs/simulator_results/metrics.json.

Models are rows. Figures with one quantity per model are single-hue dot plots (point and 95%
item-bootstrap interval), as in claim1_report.py; the two-series figure (human turns against the
model's own samples) uses the first two categorical slots of the shared palette, with a legend.
Reference values (the corpus, prediction thresholds) are dashed ink lines. The recommendation and
the notes for Xuan are written by hand in discussion.md and included verbatim.
"""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.multiwoz.report import DASHED, INK, _pyplot, write_csv
from robust_likelihood.storage import read_json

MARK = "#2a78d6"
HUMAN, MODEL = "#2a78d6", "#eb6834"
ORDER = ("qwen25_inst|P0", "qwen25_inst|S", "qwen25_broad|B", "qwen3_inst|S", "qwen3_base|F", "osim_mid|S",
         "osim_final|S", "userlm|S", "osim_mid|N", "osim_final|N", "userlm|N")
LABELS = {
    "qwen25_inst|P0": "Qwen2.5-7B-Inst · P0 (Phase 8)", "qwen25_inst|S": "Qwen2.5-7B-Inst · S",
    "qwen25_broad|B": "Qwen2.5-7B-Inst · B (broadened)", "qwen3_inst|S": "Qwen3-8B · S",
    "qwen3_base|F": "Qwen3-8B-Base · F (few-shot)", "osim_mid|S": "OSim-8B-Mid · S", "osim_final|S": "OSim-8B · S",
    "userlm|S": "UserLM-8B · S", "osim_mid|N": "OSim-8B-Mid · N", "osim_final|N": "OSim-8B · N",
    "userlm|N": "UserLM-8B · N",
}
U_GROUPS = ("typo", "pad", "para", "verbose", "zh")
VARIANTS = ("per2A", "glob2A", "glob1A", "glob1B", "cent1B", "sem1B", "cent1B:6bins")
CONCENTRATION = ("qwen3_base|F", "osim_mid|S", "qwen3_inst|S", "osim_final|S")


def ordered(runs: dict) -> list[str]:
    return [label for label in ORDER if label in runs] + sorted(label for label in runs if label not in ORDER)


def _get(block, *path):
    for key in path:
        if block is None:
            return None
        block = block.get(key) if isinstance(block, dict) else None
    return block


def num(value, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}"


def pct(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def with_ci(block: dict | None, field: str, digits: int = 2, percent: bool = False) -> str:
    if not block or block.get(field) is None:
        return "–"
    show = pct if percent else (lambda value: num(value, digits))
    interval = block.get("ci95")
    return show(block[field]) + (f" [{show(interval[0])}, {show(interval[1])}]" if interval else "")


def _save(figure, stem: Path) -> list[Path]:
    paths = []
    for suffix, options in ((".png", {"dpi": 200}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    return paths


def _dots(ax, labels, points, intervals, *, scale="linear"):
    positions = list(range(len(labels)))[::-1]
    for y, point, interval in zip(positions, points, intervals):
        if point is None:
            ax.text(0.02, y, "not run", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK["muted"])
            continue
        if interval:
            ax.hlines(y, interval[0], interval[1], color=MARK, linewidth=2, alpha=0.45)
        ax.plot([point], [y], marker="o", markersize=6, color=MARK, markeredgecolor=INK["surface"], markeredgewidth=1,
                zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    if scale == "symlog":
        ax.set_xscale("symlog", linthresh=0.001)
    elif scale != "linear":
        ax.set_xscale(scale)
    ax.grid(axis="x", color=INK["grid"], linewidth=0.6)


# ---------------------------------------------------------------- figures


def figure_surprisal(plt, runs: dict, labels: list[str], out: Path) -> list[Path]:
    rows = []
    figure, ax = plt.subplots(figsize=(7.5, 0.55 * len(labels) + 1.2))
    positions = list(range(len(labels)))[::-1]
    for y, label in zip(positions, labels):
        for who, color, offset in (("human", HUMAN, 0.17), ("samples", MODEL, -0.17)):
            found = _get(runs[label], "q1", "gold_surprisal", who)
            if not found:
                continue
            rows.append({"run": label, "who": who, **found})
            stats = [{"whislo": found["q05"], "q1": found["q25"], "med": found["q50"], "q3": found["q75"],
                      "whishi": found["q95"], "fliers": []}]
            ax.bxp(stats, positions=[y + offset], vert=False, widths=0.28, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.8},
                   medianprops={"color": INK["surface"], "linewidth": 2},
                   whiskerprops={"color": color}, capprops={"color": color})
    ax.set_yticks(positions)
    ax.set_yticklabels([LABELS.get(label, label) for label in labels])
    ax.set_xscale("log")
    ax.set_xlabel("surprisal S under the gold hypothesis (nats, log scale; box 25-75%, whiskers 5-95%)")
    ax.grid(axis="x", color=INK["grid"], linewidth=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUMAN), plt.Rectangle((0, 0), 1, 1, color=MODEL)]
    ax.legend(handles, ["human first turn", "model's own samples"], loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Human first turns against each model's own samples", fontsize=10, color=INK["primary"])
    write_csv(out / "fig1_surprisal.csv", rows)
    paths = _save(figure, out / "fig1_surprisal")
    plt.close(figure)
    return paths


def figure_naming(plt, runs: dict, labels: list[str], out: Path) -> list[Path]:
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(9, 0.4 * len(labels) + 1.4), sharey=True)
    corpus = None
    for name, field, ax, scale in (("a: names a constrained slot", "a_model", axes[0], "linear"),
                                   ("b: names a slot the goal leaves out (minus:X|impl)", "b_model_impl", axes[1],
                                    "symlog")):
        points, intervals = [], []
        for label in labels:
            block = _get(runs[label], "q2", "naming", "slots_mean")
            found = _get(block, field) or {}
            points.append(found.get("mean"))
            intervals.append(found.get("ci95"))
            corpus = block or corpus
            rows.append({"run": label, "rate": field, "value": found.get("mean"),
                         "low": (found.get("ci95") or [None, None])[0], "high": (found.get("ci95") or [None, None])[1]})
        _dots(ax, [LABELS.get(label, label) for label in labels], points, intervals, scale=scale)
        reference = (corpus or {}).get("a_corpus" if field == "a_model" else "b_corpus")
        if reference is not None:
            ax.axvline(reference, color=INK["primary"], linestyle=DASHED, linewidth=1)
            ax.text(reference, 1.01, "corpus", transform=ax.get_xaxis_transform(), ha="center", fontsize=7,
                    color=INK["primary"])
        ax.set_xlabel(name, fontsize=8)
    figure.suptitle("Slot naming in the models' samples (mean over slots)", fontsize=10, color=INK["primary"])
    write_csv(out / "fig2_naming.csv", rows)
    paths = _save(figure, out / "fig2_naming")
    plt.close(figure)
    return paths


def figure_occam(plt, runs: dict, labels: list[str], out: Path) -> list[Path]:
    rows = []
    keys = ("impl|slots_mean|named", "impl|slots_mean|omitted", "expl|slots_mean|named", "expl|slots_mean|omitted")
    figure, axes = plt.subplots(1, 4, figsize=(13, 0.4 * len(labels) + 1.4), sharey=True)
    for ax, key in zip(axes, keys):
        points, intervals, reference = [], [], None
        for label in labels:
            group = _get(runs[label], "q2", "raw_occam", "occam", key)
            if group is None:
                points.append(None)
                intervals.append(None)
                continue
            reference = group["L_E"]
            interval = group.get("OG_ci95")
            points.append(group["mean_L_J"])
            intervals.append([group["L_E"] + interval[0], group["L_E"] + interval[1]] if interval else None)
            rows.append({"run": label, "group": key, "mean_L_J": group["mean_L_J"], "L_E": group["L_E"],
                         "OG": group["OG"]})
        _dots(ax, [LABELS.get(label, label) for label in labels], points, intervals)
        if reference is not None:
            ax.axvline(reference, color=INK["primary"], linestyle=DASHED, linewidth=1)
        regime, _, direction = key.split("|")
        ax.set_xlabel(f"{regime}, slot {direction}: mean L_J (nats)", fontsize=8)
    figure.suptitle("Raw log-likelihood Occam gap: model L_J against corpus L_E (dashed)", fontsize=10,
                    color=INK["primary"])
    write_csv(out / "fig3_occam.csv", rows)
    paths = _save(figure, out / "fig3_occam")
    plt.close(figure)
    return paths


def figure_rewrites(plt, runs: dict, labels: list[str], out: Path) -> list[Path]:
    rows = []
    shown = [label for label in labels if _get(runs[label], "q3")]
    if not shown:
        return []
    figure, axes = plt.subplots(1, len(U_GROUPS), figsize=(13, 0.4 * len(shown) + 1.4), sharey=True)
    for ax, group in zip(axes, U_GROUPS):
        points, intervals = [], []
        for label in shown:
            found = _get(runs[label], "q3", "utterance", group, "abs_delta") or {}
            points.append(found.get("median"))
            intervals.append(found.get("ci95"))
            rows.append({"run": label, "kind": group, "median_abs_delta_log_bf": found.get("median"),
                         "low": (found.get("ci95") or [None, None])[0], "high": (found.get("ci95") or [None, None])[1]})
        _dots(ax, [LABELS.get(label, label) for label in shown], points, intervals)
        ax.axvline(2.0, color=INK["primary"], linestyle=DASHED, linewidth=0.8)
        ax.set_xlabel(f"{group}: median |Δ log BF| (nats)", fontsize=8)
        ax.set_xlim(left=0)
    figure.suptitle("Utterance rewrites move Bayes factors (dashed: S4's 2 nats)", fontsize=10, color=INK["primary"])
    write_csv(out / "fig4_rewrites.csv", rows)
    paths = _save(figure, out / "fig4_rewrites")
    plt.close(figure)
    return paths


def figure_typicality(plt, runs: dict, labels: list[str], out: Path) -> list[Path]:
    from matplotlib.colors import LinearSegmentedColormap

    rows, matrix = [], []
    for label in labels:
        line = []
        for variant in VARIANTS:
            value = _get(runs[label], "q4", variant, "uniform", "extreme_share")
            line.append(float("nan") if value is None else value)
            rows.append({"run": label, "variant": variant, "extreme_share": value})
        matrix.append(line)
    ramp = LinearSegmentedColormap.from_list("blue", ["#eaf2fc", "#2a78d6", "#123a6b"])
    figure, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.3))
    image = ax.imshow(matrix, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for row, line in enumerate(matrix):
        for column, value in enumerate(line):
            if value == value:
                ax.text(column, row, f"{100 * value:.0f}%", ha="center", va="center", fontsize=7,
                        color=INK["surface"] if value > 0.55 else INK["primary"])
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([LABELS.get(label, label) for label in labels], fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.03)
    colorbar.set_label("extreme-bin share", fontsize=8)
    ax.set_title("Typicality v0 with each model as the reference (uniform weighting)", fontsize=10,
                 color=INK["primary"])
    write_csv(out / "fig5_typicality.csv", rows)
    paths = _save(figure, out / "fig5_typicality")
    plt.close(figure)
    return paths


def figure_concentration(plt, runs: dict, out: Path) -> list[Path]:
    labels = [label for label in CONCENTRATION if label in runs]
    if not labels:
        return []
    rows = []
    fields = (("mean_sample_surprisal", "mean sample surprisal (nats)"),
              ("distinct_per_text", "distinct samples out of 64"), ("top_share", "share of the most frequent sample"))
    figure, axes = plt.subplots(1, 3, figsize=(11, 0.45 * len(labels) + 1.4), sharey=True)
    for ax, (field, name) in zip(axes, fields):
        values = [_get(runs[label], "q5", field) for label in labels]
        _dots(ax, [LABELS.get(label, label) for label in labels], values, [None] * len(labels))
        ax.set_xlabel(name, fontsize=8)
        ax.set_xlim(left=0)
    for label in labels:
        rows.append({"run": label, **{field: _get(runs[label], "q5", field) for field, _ in fields}})
    figure.suptitle("Concentration of the models' own samples (over the 463 clean hypothesis texts)", fontsize=10,
                    color=INK["primary"])
    write_csv(out / "fig6_concentration.csv", rows)
    paths = _save(figure, out / "fig6_concentration")
    plt.close(figure)
    return paths


def write_figures(out: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    runs = metrics["runs"]
    labels = ordered(runs)
    paths = []
    paths += figure_surprisal(plt, runs, labels, out)
    paths += figure_naming(plt, runs, labels, out)
    paths += figure_occam(plt, runs, labels, out)
    paths += figure_rewrites(plt, runs, labels, out)
    paths += figure_typicality(plt, runs, labels, out)
    paths += figure_concentration(plt, runs, out)
    return paths


# ---------------------------------------------------------------- summary


def table(header: list[str], rows: list[list[str]]) -> str:
    escape = lambda cell: str(cell).replace("|", "\\|")
    lines = ["| " + " | ".join(map(escape, header)) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(escape, row)) + " |" for row in rows]
    return "\n".join(lines)


def summary_markdown(metrics: dict, manifest: dict, discussion: str | None) -> str:
    runs = metrics["runs"]
    labels = ordered(runs)
    name = lambda label: LABELS.get(label, label) + ("" if runs[label]["primary"] or label.endswith("P0") else " (sensitivity)")
    parts = ["# Simulator study: which model gives human first turns a reasonable probability?", "",
             "Plan: `docs/simulator_study_plan.md`; decisions and interpretations: `docs/simulator_decisions.md`; "
             "format check: `format_check.md`. Every number is on the 235 clean MultiWOZ items; brackets are 95% "
             f"item-bootstrap intervals ({manifest['bootstrap']['replicates']} replicates, seed "
             f"{manifest['bootstrap']['seed']}). S is surprisal in nats under each model's own prompt format.", "",
             "Caveats: format F shows the base model 8 real human turns, an advantage the chat models do not get. "
             "`osim_final` starts from Qwen3-8B (instruct) and `osim_mid` from Qwen3-8B-Base, so mid vs final mixes "
             "the starting point with RL; the clean contrasts are base → mid and instruct → final. Rescoring noise "
             "between runs is up to about 1.4 nats (bfloat16 kernels).", ""]

    parts += ["## Prediction scorecard", ""]
    parts.append(table(["ID", "Prediction", "Result"],
                       [[row["id"], row["statement"], "**pass**" if row["pass"] else "fail"]
                        for row in metrics["predictions"]]))
    parts += ["", "Values behind each verdict are in `metrics.json` under `predictions`.", ""]

    parts += ["## Q1 Coverage", "",
              "Human percentile: the share of a hypothesis's own samples at least as surprising as the human turn "
              "(0 = beyond all samples). z: the human turn's standardized gap (S − μ_H)/σ_H.", ""]
    rows = []
    for label in labels:
        q1 = runs[label]["q1"]
        rows.append([name(label), with_ci(q1["all"]["median_percentile"], "median", 3),
                     with_ci(q1["all"]["share_beyond_all"], "mean", percent=True),
                     with_ci(q1["all"]["share_below_005"], "mean", percent=True),
                     with_ci(q1["gold"]["share_beyond_all"], "mean", percent=True),
                     with_ci(q1["all"]["z_median"], "median", 1), num(q1["all"]["z_p90"], 1)])
    parts.append(table(["Model · format", "median percentile", "beyond all samples", "percentile < 0.05",
                        "gold: beyond all", "median z", "z p90"], rows))
    parts += ["", "Surprisal per word under the gold hypothesis, and semantic coverage (does any of the gold "
                  "hypothesis's samples carry the human turn's annotated slot tuple?):", ""]
    rows = []
    for label in labels:
        q1 = runs[label]["q1"]
        rows.append([name(label), with_ci(q1["per_word"]["human"], "median"), with_ci(q1["per_word"]["samples"], "median"),
                     with_ci(q1["semantic"]["covered"], "mean", percent=True),
                     with_ci(q1["semantic"]["median_p_sem"], "median", 3)])
    parts.append(table(["Model · format", "S/word, human", "S/word, samples", "slot tuple covered", "median P_sem"],
                       rows))
    parts += ["", "Style: each human turn against one sample under its gold hypothesis (235 each). Typo proxy: a "
                  "word outside the MultiWOZ 2.2 train vocabulary (words in at least 5 dialogues).", ""]
    first = runs[labels[0]]["q1"]["style"]["human"] if labels else {}
    rows = [["**human turns**", num(first.get("median_words"), 0), pct(first.get("greeting")), pct(first.get("question")),
             pct(first.get("non_dictionary")), num(first.get("distinct_2"), 3)]]
    for label in labels:
        found = runs[label]["q1"]["style"]["samples"]
        rows.append([name(label), num(found.get("median_words"), 0), pct(found.get("greeting")),
                     pct(found.get("question")), pct(found.get("non_dictionary")), num(found.get("distinct_2"), 3)])
    parts.append(table(["Text", "median words", "greeting", "question", "non-dictionary word", "distinct-2"], rows))
    parts += ["", "![](fig1_surprisal.png)", ""]

    parts += ["## Q2 Fidelity", "", "Slot naming in the samples, mean over the three slots (a: under the gold "
              "hypothesis, a constrained slot is named; b: under minus:X, the left-out slot X is named).", ""]
    rows = []
    corpus = None
    for label in labels:
        block = runs[label]["q2"]["naming"]["slots_mean"]
        corpus = block
        rows.append([name(label), with_ci(block["a_model"], "mean", 3), with_ci(block["b_model_impl"], "mean", 4),
                     with_ci(block["b_model_expl"], "mean", 4)])
    if corpus:
        rows.insert(0, ["**corpus**", num(corpus["a_corpus"], 3), num(corpus["b_corpus"], 4), "–"])
    parts.append(table(["Model · format", "a", "b (impl)", "b (expl)"], rows))
    parts += ["", "Per slot ln(a/b) (implicit wording) against the corpus:", ""]
    rows = []
    for label in labels:
        slots = runs[label]["q2"]["naming"]["slots"]
        rows.append([name(label)] + [f"{num(slots[slot]['ln_a_over_b_impl'])} ({num(slots[slot]['ln_a_over_b_corpus'])})"
                                     for slot in ("area", "food", "pricerange")])
    parts.append(table(["Model · format", "area (corpus)", "food (corpus)", "pricerange (corpus)"], rows))
    parts += ["", "Raw log-likelihood Occam gap (scores −S; `metrics.py`): the model's mean L_J against the corpus "
                  "L_E, mean over slots.", ""]
    rows = []
    for label in labels:
        occam = runs[label]["q2"]["raw_occam"]["occam"]
        cells = []
        for key in ("impl|slots_mean|named", "impl|slots_mean|omitted", "expl|slots_mean|named",
                    "expl|slots_mean|omitted"):
            group = occam.get(key)
            cells.append("–" if group is None else f"{num(group['mean_L_J'])} vs {num(group['L_E'])}")
        gap = runs[label]["q2"]["raw_occam"]["wording_gap"] or {}
        cells.append(num(gap.get("median") if "median" in gap else gap.get("mean")))
        rows.append([name(label)] + cells)
    parts.append(table(["Model · format", "impl named", "impl omitted", "expl named", "expl omitted",
                        "expl − impl wording gap"], rows))
    parts += ["", "![](fig2_naming.png)", "", "![](fig3_occam.png)", ""]

    parts += ["## Q3 Robustness (claim 1 with each model)", ""]
    rows = []
    for label in labels:
        q3 = runs[label].get("q3")
        if not q3:
            continue
        utterance = q3["utterance"]
        rows.append([name(label), with_ci(q3["rewording"], "icc1"), with_ci(q3["rewording"]["sd_ratio"], "median")]
                    + [num(_get(utterance, group, "abs_delta", "median"), 1) for group in U_GROUPS]
                    + [pct(_get(utterance, "para", "flip_rate", "mean")),
                       with_ci(q3["posterior"]["sd_p_goal"], "mean", 3),
                       pct(_get(q3, "posterior", "argmax_flip_rate", "mean"))])
    parts.append(table(["Model · format", "ICC(1)", "within/between SD"] + [f"|ΔBF| {group}" for group in U_GROUPS]
                       + ["flip (para)", "SD of P(G)", "argmax flip"], rows))
    parts += ["", "![](fig4_rewrites.png)", ""]

    parts += ["## Q4 Typicality", "", "Share of human pairs in the extreme bin, per typicality v0 variant, with this "
              "model as the reference (uniform weighting; `cent1B:6bins` is cent1B with the 6-bin edges).", ""]
    rows = []
    for label in labels:
        rows.append([name(label)] + [pct(_get(runs[label], "q4", variant, "uniform", "extreme_share"))
                                     for variant in VARIANTS])
    parts.append(table(["Model · format"] + list(VARIANTS), rows))
    parts += ["", "Size principle kept (L_J ÷ L_E, implicit, slot named), contradiction log Bayes factor, and bin "
                  "agreement under the para rewrites:", ""]
    rows = []
    for label in labels:
        cells = []
        for variant in ("glob1B", "cent1B", "sem1B"):
            size = _get(runs[label], "q4", variant, "uniform", "size")
            group = _get(size, "occam", "impl|slots_mean|named")
            ratio = None if not group or not group.get("L_E") else group["mean_L_J"] / group["L_E"]
            cells.append(f"{num(ratio)} / {num(_get(size, 'contra_mean_log_bf'), 1)} / "
                         f"{pct(_get(runs[label], 'q4', variant, 'uniform', 'bin_agreement', 'para', 'mean'))}")
        rows.append([name(label)] + cells)
    parts.append(table(["Model · format", "glob1B", "cent1B", "sem1B"], rows))
    parts += ["", "![](fig5_typicality.png)", ""]

    parts += ["## Q5 What RL does", "", "Concentration over the 463 clean hypothesis texts:", ""]
    rows = []
    for label in labels:
        q5 = runs[label]["q5"]
        rows.append([name(label), num(q5["mean_sample_surprisal"], 1), num(q5["distinct_per_text"], 1),
                     pct(q5["top_share"])])
    parts.append(table(["Model · format", "mean sample S", "distinct of 64", "top sample share"], rows))
    parts += ["", "Paired differences (second − first, the same item draws):", ""]
    rows = []
    for pair, found in metrics.get("paired", {}).items():
        rows.append([pair] + [with_ci(found[field], "difference", 3) for field in
                              ("median_percentile", "share_beyond_all", "mean_mu", "gold_z_median")])
    if rows:
        parts.append(table(["Pair", "median percentile", "share beyond all", "mean μ_H", "gold median z"], rows))
    examples = metrics.get("examples", {})
    shown = [label for label in ("osim_mid|S", "osim_final|S") if label in examples.get("samples", {})]
    if shown:
        parts += ["", f"Ten samples for the same goal ({examples['item_id']}):", ""]
        for label in shown:
            parts += [f"**{LABELS.get(label, label)}**", ""]
            parts += [f"{index + 1}. {text}" for index, text in enumerate(examples["samples"][label])] + [""]
    parts += ["![](fig6_concentration.png)", ""]
    parts += [discussion.strip() if discussion else "## Recommendation\n\n_Pending: written after reading the results._"]
    return "\n".join(parts) + "\n"


def run_sim_report(root: Path, out: Path, figures: bool = True) -> dict:
    metrics = read_json(out / "metrics.json")
    manifest = read_json(out / "manifest.json")
    paths = write_figures(out, metrics) if figures else []
    discussion_path = out / "discussion.md"
    discussion = discussion_path.read_text(encoding="utf-8") if discussion_path.exists() else None
    (out / "summary.md").write_text(summary_markdown(metrics, manifest, discussion), encoding="utf-8", newline="\n")
    return {"summary": str(out / "summary.md"), "figures": [str(path) for path in paths]}
