"""Week of 24 Nov: docs/week_nov24_results/summary.md and figures, from metrics_abc.json (and Part D's).

Six models are six series: the reference palette's first six categorical slots in a fixed order by
model, each with its own marker as well, and a legend. Single-quantity figures are dot plots in the
first slot; the bin-change matrix is a one-hue sequential ramp. Every figure is written as .png,
.pdf and .csv. The notes for Xuan are hand-written in discussion.md and included verbatim.
"""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.multiwoz.report import DASHED, INK, _pyplot, write_csv
from robust_likelihood.multiwoz.sim_report import LABELS, _get, _save, num, pct, table, with_ci
from robust_likelihood.storage import read_json

MODELS = ("qwen3_base|F", "osim_mid|S", "osim_final|S", "userlm|S", "qwen25_inst|S", "qwen3_inst|S")
COLORS = dict(zip(MODELS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "v", "P")))
SHORT = {"qwen3_base|F": "Qwen3-8B-Base · F", "osim_mid|S": "OSim-8B-Mid", "osim_final|S": "OSim-8B",
         "userlm|S": "UserLM-8B", "qwen25_inst|S": "Qwen2.5-7B-Inst", "qwen3_inst|S": "Qwen3-8B",
         "qwen25_broad|B": "Qwen2.5-7B-Inst · B"}
KINDS = ("typo1", "typo2", "typo4", "slottypo1", "lower", "nopunct", "noapos", "fill1", "fill2", "fill4", "terse",
         "terse_rx", "fixed")
VARIANTS = ("glob1A", "glob1B", "cent1B", "cent1B:6bins", "glob1Q", "cent1Q", "glob1Q:3q", "cent1Q:3q")
TEACHER = "qwen3_base|F"
FIXED_PAIRS = (("glob1B", "glob1Q"), ("cent1B:6bins", "cent1Q"))


def shown(models: dict) -> list[str]:
    return [label for label in MODELS if label in models]


# ---------------------------------------------------------------- figures


def figure_dose(plt, models: dict, out: Path) -> list[Path]:
    labels = [label for label in shown(models) if "raw" in models[label]]
    if not labels:
        return []
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    rows = []
    panels = ((0, 0, ("typo1", "typo2", "typo4"), "abs_delta_log_bf", "typos in the turn", "median |Δ log BF| (nats)"),
              (0, 1, ("typo1", "typo2", "typo4"), "delta_log_p", "typos in the turn", "median Δ log p(U | h) (nats)"),
              (1, 0, ("fill1", "fill2", "fill4"), "abs_delta_log_bf", "filler sentences added",
               "median |Δ log BF| (nats)"),
              (1, 1, ("fill1", "fill2", "fill4"), "delta_log_p", "filler sentences added",
               "median Δ log p(U | h) (nats)"))
    for row, column, kinds, field, xlabel, ylabel in panels:
        ax = axes[row][column]
        for label in labels:
            xs, ys, lows, highs = [], [], [], []
            for dose, kind in zip((1, 2, 4), kinds):
                found = _get(models[label], "raw", kind, field) or {}
                if found.get("median") is None:
                    continue
                xs.append(dose)
                ys.append(found["median"])
                interval = found.get("ci95") or [found["median"], found["median"]]
                lows.append(found["median"] - interval[0])
                highs.append(interval[1] - found["median"])
                rows.append({"run": label, "kind": kind, "dose": dose, "metric": field, "median": found["median"],
                             "low": interval[0], "high": interval[1]})
            ax.errorbar(xs, ys, yerr=[lows, highs], color=COLORS[label], marker=MARKERS[label], markersize=6,
                        linewidth=2, capsize=0, elinewidth=1.2, markeredgecolor=INK["surface"],
                        markeredgewidth=1, label=SHORT[label])
        ax.set_xticks((1, 2, 4))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(0, color=INK["axis"], linewidth=0.8)
        ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Typos and filler: cost in log p against effect on Bayes factors (item-bootstrap 95% intervals)",
                    fontsize=10, color=INK["primary"])
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    write_csv(out / "fig1_dose.csv", rows)
    paths = _save(figure, out / "fig1_dose")
    plt.close(figure)
    return paths


def figure_bin_change(plt, models: dict, out: Path) -> list[Path]:
    from matplotlib.colors import LinearSegmentedColormap

    typ = _get(models.get(TEACHER), "typicality")
    if not typ:
        return []
    kinds = [kind for kind in KINDS if any(_get(typ, variant, kind) for variant in VARIANTS)]
    rows, matrix = [], []
    for variant in VARIANTS:
        line = []
        for kind in kinds:
            value = _get(typ, variant, kind, "bin_change", "mean")
            line.append(float("nan") if value is None else value)
            rows.append({"variant": variant, "kind": kind, "bin_change": value})
        matrix.append(line)
    ramp = LinearSegmentedColormap.from_list("blue", ["#eaf2fc", "#2a78d6", "#123a6b"])
    figure, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(matrix, cmap=ramp, vmin=0, vmax=1, aspect="auto")
    for row, line in enumerate(matrix):
        for column, value in enumerate(line):
            if value == value:
                ax.text(column, row, f"{100 * value:.0f}%", ha="center", va="center", fontsize=7,
                        color=INK["surface"] if value > 0.55 else INK["primary"])
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(VARIANTS, fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.03)
    colorbar.set_label("share of human pairs whose bin changes", fontsize=8)
    ax.set_title("Bin changes under Part A's edits, Qwen3-8B-Base · F as the reference", fontsize=10,
                 color=INK["primary"])
    write_csv(out / "fig2_bin_change.csv", rows)
    paths = _save(figure, out / "fig2_bin_change")
    plt.close(figure)
    return paths


def figure_fixed_quantile(plt, models: dict, out: Path) -> list[Path]:
    labels = [label for label in MODELS + ("qwen25_broad|B",) if _get(models.get(label), "part_c")]
    if not labels:
        return []
    metrics = (("extreme_share", "extreme-bin share of human pairs"), ("size_kept", "size principle kept (L_J ÷ L_E)"),
               ("para1", "bin agreement under para1"), ("typo1", "bin agreement under typo1"))
    figure, axes = plt.subplots(len(FIXED_PAIRS), len(metrics), figsize=(13, 2 * (0.32 * len(labels) + 1)),
                                sharey=True)
    rows = []
    positions = list(range(len(labels)))[::-1]
    for row_index, (fixed, quantile) in enumerate(FIXED_PAIRS):
        for column, (metric, title) in enumerate(metrics):
            ax = axes[row_index][column]
            for offset, (variant, color, marker) in enumerate(((fixed, "#2a78d6", "o"), (quantile, "#eb6834", "s"))):
                for y, label in zip(positions, labels):
                    block = _get(models[label], "part_c", variant) or {}
                    if metric == "extreme_share":
                        value = block.get("extreme_share")
                    elif metric == "size_kept":
                        value = _get(block, "size_kept", "impl")
                    elif metric == "para1":
                        value = _get(block, "agreement", "para1", "mean")
                    else:
                        change = _get(models[label], "typicality", variant, "typo1", "bin_change", "mean")
                        value = None if change is None else 1 - change
                    rows.append({"run": label, "variant": variant, "metric": metric, "value": value})
                    if value is not None:
                        ax.plot([value], [y + (0.15 if offset == 0 else -0.15)], marker=marker, color=color,
                                markersize=6, linestyle="none", markeredgecolor=INK["surface"], markeredgewidth=1,
                                label=variant if y == positions[0] else None)
            if metric == "size_kept":
                ax.axvline(1.0, color=INK["primary"], linestyle=DASHED, linewidth=0.8)
            ax.set_yticks(positions)
            ax.set_yticklabels([SHORT.get(label, label) for label in labels], fontsize=8)
            ax.set_xlabel(title, fontsize=8)
            ax.grid(axis="x", color=INK["grid"], linewidth=0.6)
            if column == 0:
                ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    figure.suptitle("Fixed edges against quantile bins (top: raw S; bottom: centered), each model as the reference",
                    fontsize=10, color=INK["primary"])
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    write_csv(out / "fig3_fixed_vs_quantile.csv", rows)
    paths = _save(figure, out / "fig3_fixed_vs_quantile")
    plt.close(figure)
    return paths


def figure_student(plt, part_d: dict, out: Path) -> list[Path]:
    runs = [run for run in ("R-lin", "R-mlp", "C-mlp") if run in part_d.get("students", {})]
    if not runs:
        return []
    figure, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), sharex=True, sharey=True)
    axes = axes if len(runs) > 1 else [axes]
    rows = []
    for ax, run in zip(axes, runs):
        pairs = part_d["students"][run]["human_pairs"]
        for unseen, color, marker, label in ((0, "#2a78d6", "o", "seen food"), (1, "#eb6834", "s", "unseen food")):
            mine = [pair for pair in pairs if pair[2] == unseen]
            ax.scatter([pair[0] for pair in mine], [pair[1] for pair in mine], s=16 if unseen else 9, color=color,
                       marker=marker, alpha=0.9 if unseen else 0.45, linewidths=0, label=label)
            rows += [{"run": run, "teacher_T": pair[0], "student_T": pair[1], "unseen_food": unseen} for pair in mine]
        low = min(min(pair[0] for pair in pairs), min(pair[1] for pair in pairs))
        high = max(max(pair[0] for pair in pairs), max(pair[1] for pair in pairs))
        ax.plot([low, high], [low, high], color=INK["primary"], linestyle=DASHED, linewidth=0.8)
        rho = _get(part_d, "students", run, "agreement", "all", "spearman")
        ax.set_title(f"{run}: Spearman {num(rho)}", fontsize=9, color=INK["primary"])
        ax.set_xlabel("teacher T (64 samples, 8B)", fontsize=8)
        ax.grid(color=INK["grid"], linewidth=0.6)
    axes[0].set_ylabel("student T-hat (0.6B)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    figure.suptitle("Student against teacher on the 2,355 human pairs (dashed: equality)", fontsize=10,
                    color=INK["primary"])
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    write_csv(out / "fig4_student_teacher.csv", rows)
    paths = _save(figure, out / "fig4_student_teacher")
    plt.close(figure)
    return paths


def figure_robustness(plt, part_d: dict, out: Path) -> list[Path]:
    found = part_d.get("figure5")
    if not found:
        return []
    kinds = ("para1", "typo1", "fill2", "hypothesis rewording")
    order = ("raw", "teacher", "student R-lin", "student R-mlp", "student C-mlp")
    scorers = sorted(found, key=lambda name: next((i for i, prefix in enumerate(order) if name.startswith(prefix)),
                                                  len(order)))
    colors = ("#2a78d6", "#e87ba4", "#1baf7a", "#eda100", "#eb6834")
    figure, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(scorers)
    rows = []
    for index, scorer in enumerate(scorers):
        for position, kind in enumerate(kinds):
            block = found[scorer].get(kind) or {}
            value, interval = block.get("mean"), block.get("ci95")
            rows.append({"scorer": scorer, "kind": kind, "argmax_flip": value,
                         "low": (interval or [None, None])[0], "high": (interval or [None, None])[1]})
            if value is None:
                continue
            x = position + (index - (len(scorers) - 1) / 2) * width
            ax.bar([x], [value], width=width * 0.9, color=colors[index % len(colors)],
                   label=scorer if position == 0 else None)
            if interval:
                ax.vlines(x, interval[0], interval[1], color=INK["secondary"], linewidth=1)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, fontsize=8)
    ax.set_ylabel("posterior argmax flip rate", fontsize=8)
    ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("How often the top hypothesis changes: raw log-likelihood, the teacher and the students",
                 fontsize=10, color=INK["primary"])
    write_csv(out / "fig5_robustness.csv", rows)
    paths = _save(figure, out / "fig5_robustness")
    plt.close(figure)
    return paths


def write_figures(out: Path, metrics: dict, part_d: dict | None = None) -> list[Path]:
    plt = _pyplot()
    models = metrics["models"]
    paths = (figure_dose(plt, models, out) + figure_bin_change(plt, models, out)
             + figure_fixed_quantile(plt, models, out))
    if part_d:
        paths += figure_student(plt, part_d, out) + figure_robustness(plt, part_d, out)
    return paths


# ---------------------------------------------------------------- markdown


def part_a_tables(models: dict, simulator: dict | None) -> list[str]:
    labels = [label for label in shown(models) if "raw" in models[label]]
    parts = ["## Part A: typos, style and length", ""]
    if not labels:
        return parts + ["The `uperturb` runs are not in yet.", ""]
    header = ["Kind"] + [SHORT[label] for label in labels]
    for field, title, percent in (("delta_log_p", "Median Δ log p(U | h) per unit, nats", False),
                                  ("abs_delta_log_bf", "Median |Δ log BF|, nats", False),
                                  ("relative_delta_log_bf", "Median |Δ log BF| ÷ |log BF|", False),
                                  ("sign_change", "Share of Bayes factors that change sign", True),
                                  ("argmax_flip", "Posterior argmax flip rate", True)):
        rows = [[kind] + [with_ci(_get(models[label], "raw", kind, field), "mean" if percent else "median",
                                  percent=percent) for label in labels] for kind in KINDS
                if any(_get(models[label], "raw", kind) for label in labels)]
        parts += [f"**{title}** (95% item-bootstrap interval)", "", table(header, rows), ""]
    rows = []
    for name, kind, field in (("per typo, typo1", "typo1", "delta_log_p_per_unit_dose"),
                              ("per typo, typo4", "typo4", "delta_log_p_per_unit_dose"),
                              ("per sentence, fill1", "fill1", "delta_log_p_per_unit_dose"),
                              ("per sentence, fill4", "fill4", "delta_log_p_per_unit_dose"),
                              ("per added token, fill4", "fill4", "delta_log_p_per_token"),
                              ("per removed token, terse", "terse", "delta_log_p_per_token"),
                              ("named slot's log BF kept, slottypo1", "slottypo1", "named_slot_bf_kept")):
        rows.append([name] + [with_ci(_get(models[label], "raw", kind, field), "median") for label in labels])
    parts += ["**Unit costs** (median, nats; the slot row is a ratio)", "", table(header, rows), ""]
    ops = sorted({op for label in labels for op in (_get(models[label], "raw", "typo1", "by_op") or {})})
    rows = [[op] + [num(_get(models[label], "raw", "typo1", "by_op", op, "median")) for label in labels] for op in ops]
    parts += ["**typo1 by operation**: median Δ log p, nats", "", table(header, rows), ""]
    rows = []
    for family in ("typo", "fill"):
        for field in ("abs_delta_log_bf", "delta_log_p"):
            rows.append([f"{family}: {field}"] + [
                with_ci(_get(models[label], "raw", "dose", family, field), "slope") for label in labels])
    parts += ["**Dose slopes** (least squares on unit values, nats per typo or per sentence)", "",
              table(header, rows), ""]
    if simulator:
        rows = [["verbose (Phase 2, Claude rewrites)"] + [
            with_ci(_get(simulator, "runs", label, "q3", "utterance", "verbose", "abs_delta"), "median")
            for label in labels]]
        rows += [[kind] + [with_ci(_get(models[label], "raw", kind, "abs_delta_log_bf"), "median") for label in labels]
                 for kind in ("terse", "terse_rx")]
        parts += ["**Length both ways**: median |Δ log BF|, `terse` next to the simulator study's `verbose`", "",
                  table(header, rows), ""]
    typ = _get(models.get(TEACHER), "typicality")
    if typ:
        kinds = [kind for kind in KINDS if any(_get(typ, variant, kind) for variant in VARIANTS)]
        rows = [[variant] + [pct(_get(typ, variant, kind, "bin_change", "mean")) for kind in kinds] for variant in VARIANTS]
        parts += ["**Typicality, Qwen3-8B-Base · F as the reference**: share of human pairs whose bin changes", "",
                  table(["Variant"] + kinds, rows), ""]
        rows = [[variant] + [pct(_get(typ, variant, kind, "into_extreme", "mean")) for kind in kinds]
                for variant in VARIANTS]
        parts += ["Share moved into the extreme bin", "", table(["Variant"] + kinds, rows), ""]
        rows = [[variant] + [num(_get(typ, variant, kind, "abs_delta_score", "median")) for kind in kinds]
                for variant in VARIANTS]
        parts += ["Median |Δ score|, nats", "", table(["Variant"] + kinds, rows), ""]
    fixed = _get(models.get(TEACHER), "fixed")
    if fixed:
        rows = [[row["variant"], row["item_id"], str(row["bin_typo"]), str(row["bin_fixed"]), num(row["score_typo"]),
                 num(row["score_fixed"])] for row in fixed]
        parts += ["**A3: the six natural typos of \"restaurant\", corrected** (teacher bins under the gold hypothesis, "
                  "n = 6, descriptive)", "",
                  table(["Variant", "Item", "Bin with typo", "Bin corrected", "Score with typo", "Score corrected"], rows),
                  ""]
    spread = [[SHORT[label], str(_get(models[label], "layout_spread", "originals")),
               num(_get(models[label], "layout_spread", "median"), 3), num(_get(models[label], "layout_spread", "p90"), 3)]
              for label in labels]
    parts += ["**Batch layout.** The same original scored in different units' batches (max − min per pair, nats); "
              "co-batching keeps this out of every Δ:", "", table(["Model", "Pairs", "Median", "p90"], spread), ""]
    return parts


def part_b_table(models: dict) -> list[str]:
    labels = [label for label in MODELS + ("qwen25_broad|B",) if _get(models.get(label), "part_b")]
    parts = ["## Part B: hypothesis rewording", "",
             "Hypothesis rewording was already measured on raw log-likelihood in the simulator study "
             "(`docs/simulator_results/`). This part adds the typicality side. Bin agreement and the "
             "global-normalization variants need no samples under the reworded text; glob1A does, so it has "
             "rewording rows only for the models with an `hsample` run.", ""]
    if not labels:
        return parts
    rows = []
    for label in labels:
        for name in ("raw",) + VARIANTS:
            block = _get(models[label], "part_b", name)
            if not block or block.get("measured") is False:
                continue
            kept = [value for value in (block.get("size_kept") or {}).values() if value is not None]
            agree = [] if name == "raw" else [
                _get(models[label], "part_c", name, "agreement", kind, "mean") for kind in ("hpara1", "hpara2", "hpara3",
                                                                                           "hzh")]
            agree = [value for value in agree if value is not None]
            rows.append([SHORT.get(label, label), name,
                         with_ci(_get(block, "posterior", "argmax_flip_rate"), "mean", percent=True),
                         with_ci(_get(block, "posterior", "sd_p_goal"), "mean", digits=3),
                         pct(sum(agree) / len(agree)) if agree else "–",
                         num(sum(kept) / len(kept)) if kept else "–"])
    parts += [table(["Model", "Score", "Argmax flip rate", "SD of P(G)", "Bin agreement (hpara1–3, hzh mean)",
                     "Size principle kept, reworded (mean of kinds)"], rows), ""]
    return parts


def part_c_table(models: dict) -> list[str]:
    labels = [label for label in MODELS + ("qwen25_broad|B",) if _get(models.get(label), "part_c")]
    parts = ["## Part C: quantile bins", ""]
    rows = []
    for label in labels:
        for name in VARIANTS:
            block = _get(models[label], "part_c", name)
            if not block:
                continue
            typo1 = _get(models[label], "typicality", name, "typo1", "bin_change", "mean")
            fill2 = _get(models[label], "typicality", name, "fill2", "bin_change", "mean")
            rows.append([SHORT.get(label, label), name, pct(block["extreme_share"]), num(_get(block, "size_kept", "impl")),
                         num(block["contra_log_bf"])]
                        + [pct(_get(block, "agreement", kind, "mean")) for kind in ("para1", "typo", "pad", "verbose",
                                                                                     "zh")]
                        + [pct(None if typo1 is None else 1 - typo1), pct(None if fill2 is None else 1 - fill2)])
    parts += [table(["Model", "Variant", "Extreme bin", "Size kept", "Contra log BF", "para1", "typo", "pad", "verbose",
                     "zh", "typo1", "fill2"], rows), "",
              "Bin agreement columns: para1 to zh are the claims rewrites (Phase 2 scores), typo1 and fill2 are Part A's "
              "(co-batched).", ""]
    return parts


def part_d_tables(part_d: dict | None) -> list[str]:
    parts = ["## Part D: a first trained typicality estimator", ""]
    if not part_d:
        return parts + ["The student runs are not in yet.", ""]
    scheme = part_d["scheme"]
    parts += [f"Teacher: {scheme['choice']} with Qwen3-8B-Base · F as the reference (H {num(scheme['H'])}, "
              f"sigma {num(scheme['sigma'])}). Held-out foods: {', '.join(part_d['held_out_foods'])} "
              f"({len(part_d['unseen_food_items'])} items). Student: Qwen3-0.6B-Base with LoRA r 16.", ""]
    students = part_d["students"]
    rows = [[run, f"{block['train_pairs']:,}", str(block.get("best_step")), num(block.get("best_val_loss"), 3),
             num((block.get("train_seconds") or 0) / 60, 1), num(block.get("pairs_per_second"), 0)]
            for run, block in students.items()]
    parts += ["**Training**", "", table(["Run", "Training pairs", "Best step", "Best val loss", "Minutes",
                                          "Pairs/s (prediction)"], rows), ""]
    rows = []
    for run, block in students.items():
        agree = block["agreement"]
        rows.append([run, num(_get(agree, "all", "spearman")), num(_get(agree, "seen_food", "spearman")),
                     num(_get(agree, "unseen_food", "spearman")),
                     with_ci(_get(agree, "all", "exact_bin"), "mean", percent=True),
                     with_ci(_get(agree, "all", "within_one_bin"), "mean", percent=True),
                     num(_get(agree, "synthetic_val", "spearman")),
                     num(_get(agree, "synthetic_held_out_food", "spearman"))])
    parts += ["**Agreement with the teacher** (human pairs; the synthetic columns are held-out samples)", "",
              table(["Run", "Spearman", "Seen food", "Unseen food", "Exact bin", "Within one bin",
                     "Synthetic val", "Synthetic, held-out food"], rows), ""]
    rows = []
    forms = {"R-lin": ("cont", "bin"), "R-mlp": ("cont", "bin"), "C-mlp": ("expected", "argmax")}

    def inference_row(label: str, found: dict) -> list[str]:
        return [label, num(_get(found, "size_kept", "named")), num(_get(found, "size_kept", "omitted")),
                num(found.get("contra_log_bf")), with_ci(found.get("mean_log_p_goal"), "mean"),
                with_ci(found.get("top1"), "mean", percent=True)]

    for run, block in students.items():
        for form in forms[run]:
            rows.append(inference_row(f"{run} · {form}", _get(block, "forms", form, "all") or {}))
    for name, label in (("teacher", "teacher (64 samples, 8B)"), ("raw_8b", "raw log-lik, Qwen3-8B-Base · F"),
                        ("raw_06b", "raw log-lik, Qwen3-0.6B-Base · F, untrained"), ("jev_likely", "Jev (likely)"),
                        ("jev_typical", "Jev (typical)")):
        found = _get(part_d, "baselines", name, "all")
        if found:
            rows.append(inference_row(label, found))
    parts += ["**Inference on the human pairs** (size principle kept: L_J ÷ L_E, implicit regime; "
              "top-1 gives 1/k credit when the true goal ties k-way at the top, which binned scores often do)", "",
              table(["Score", "Size kept, named", "Size kept, omitted", "Contra log BF", "Mean log P(G)", "Top-1"],
                    rows), ""]
    cost = part_d.get("teacher_cost")
    if cost:
        speeds = ", ".join(f"{run} {num(block.get('pairs_per_second'), 0)}" for run, block in students.items())
        parts += [f"**Cost.** The teacher needs {num(cost['seconds_per_new_hypothesis'], 1)} s of 8B sampling and "
                  f"scoring per new hypothesis text; the students score {speeds} pairs per second.", ""]
    verdict = part_d.get("verdict")
    if verdict:
        parts += [f"**Verdict (frozen rule): {verdict['verdict']}.**", ""]
    return parts


def scorecard(predictions: list[dict]) -> list[str]:
    rows = [[item["id"], item["statement"], "pass" if item["pass"] else "fail"] for item in predictions]
    return ["## Prediction scorecard", "", table(["ID", "Prediction", "Result"], rows), ""]


def summary_markdown(metrics: dict, manifest: dict, discussion: str | None, simulator: dict | None,
                     headline: list[str] | None = None) -> str:
    models = metrics["models"]
    parts = ["# Week of 24 Nov: typos and length, hypothesis rewording, quantile bins, and a trained estimator", ""]
    if headline:
        parts += headline + [""]
    parts += [f"Inputs and commit: `manifest_abc.json` (git `{manifest['git']['commit'][:7]}`). Intervals: item "
              f"bootstrap, {manifest['bootstrap']['replicates']:,} replicates, seed {manifest['bootstrap']['seed']}. "
              "Plan: `docs/week_nov24_plan.md`; decisions and deviations: `docs/week_nov24_decisions.md`.", ""]
    parts += part_a_tables(models, simulator)
    parts += part_b_table(models)
    parts += part_c_table(models)
    choice = metrics.get("teacher_choice")
    part_d = metrics.get("part_d")
    if choice:
        parts += ["**Training target (plan 5.1 rule).** " + choice["rule"] + f". cent1Q: para1 "
                  f"{pct(choice['cent1Q']['para1'])}, typo1 {pct(choice['cent1Q']['typo1'])}, size kept "
                  f"{num(choice['cent1Q']['size_kept'])}; cent1B:6bins: para1 {pct(choice['cent1B:6bins']['para1'])}, "
                  f"typo1 {pct(choice['cent1B:6bins']['typo1'])}. **Rule's choice: {choice['choice']}.**", ""]
    parts += part_d_tables(part_d)
    parts += scorecard(metrics.get("predictions", []) + (part_d or {}).get("predictions", []))
    if discussion:
        parts += [discussion.strip(), ""]
    return "\n".join(parts)


def run_wk_report(root: Path, out: Path, figures: bool = True) -> dict:
    metrics = read_json(out / "metrics_abc.json")
    manifest = read_json(out / "manifest_abc.json")
    part_d = read_json(out / "metrics_d.json") if (out / "metrics_d.json").exists() else None
    metrics["part_d"] = part_d
    simulator_path = root / "docs" / "simulator_results" / "metrics.json"
    simulator = read_json(simulator_path) if simulator_path.exists() else None
    discussion_path = out / "discussion.md"
    discussion = discussion_path.read_text(encoding="utf-8") if discussion_path.exists() else None
    headline_path = out / "headline.md"
    headline = headline_path.read_text(encoding="utf-8").strip().splitlines() if headline_path.exists() else None
    paths = write_figures(out, metrics, part_d) if figures else []
    (out / "summary.md").write_text(summary_markdown(metrics, manifest, discussion, simulator, headline),
                                    encoding="utf-8", newline="\n")
    return {"summary": str(out / "summary.md"), "figures": [str(path) for path in paths]}
