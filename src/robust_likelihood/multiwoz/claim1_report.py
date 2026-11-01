"""Figures and summary.md for claims plan 2e. matplotlib is imported only to draw.

There are up to eleven scorers, more than a categorical palette can carry, so every figure is a
single-hue dot plot with scorers as rows (point estimate and 95% bootstrap interval) or a small
multiple per scorer; reference thresholds of the predictions are dashed ink lines.
"""

from __future__ import annotations

import math
from pathlib import Path

from robust_likelihood.multiwoz.claim1 import SCORER_LABELS, U_GROUPS, _variant
from robust_likelihood.multiwoz.report import DASHED, INK, _pyplot, write_csv

MARK = "#2a78d6"
GROUP_LABELS = {"typo": "typo", "pad": "pad", "para": "para1 + para2", "verbose": "verbose", "zh": "zh (Chinese)"}
VERBAL_ORDER = ("qwen_verbal_exact", "qwen_verbal_meaning", "claude_verbal", "gpt_verbal")


def _save(figure, stem: Path) -> list[Path]:
    paths = []
    for suffix, options in ((".png", {"dpi": 200}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    return paths


def _dots(ax, labels: list[str], points: list, intervals: list, *, scale: str = "linear"):
    """scale "symlog" is linear below 0.1 nats, so a median of exactly 0 is drawn at 0."""
    positions = list(range(len(labels)))[::-1]
    for y, point, interval in zip(positions, points, intervals):
        if point is None:
            ax.text(0.02, y, "not run", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                    color=INK["muted"])
            continue
        if interval:
            ax.hlines(y, interval[0], interval[1], color=MARK, linewidth=2, alpha=0.45)
        ax.plot([point], [y], marker="o", markersize=6, color=MARK, markeredgecolor=INK["surface"],
                markeredgewidth=1, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    if scale == "symlog":
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(left=-0.02)
    elif scale != "linear":
        ax.set_xscale(scale)
    ax.grid(axis="x", color=INK["grid"], linewidth=0.6)


def rewording_rows(metrics: dict) -> list[dict]:
    rows = []
    for name, block in metrics["scorers"].items():
        found = block["rewording"]
        rows.append({"scorer": name, "label": block["label"], "icc1": found["icc1"],
                     "icc1_low": (found["icc1_ci95"] or [None, None])[0],
                     "icc1_high": (found["icc1_ci95"] or [None, None])[1],
                     "sd_ratio": found["sd_ratio"]["median"],
                     "sd_ratio_low": (found["sd_ratio"]["ci95"] or [None, None])[0],
                     "sd_ratio_high": (found["sd_ratio"]["ci95"] or [None, None])[1]})
    return rows


def plot_rewording(stem: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    rows = rewording_rows(metrics)
    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.4), sharey=True)
    _dots(axes[0], labels, [row["icc1"] for row in rows],
          [None if row["icc1_low"] is None else [row["icc1_low"], row["icc1_high"]] for row in rows])
    axes[0].axvline(0.5, color=INK["primary"], linestyle=DASHED, linewidth=1)
    axes[0].set_xlabel("ICC(1) across rewordings of a hypothesis (R2 threshold 0.5)")
    _dots(axes[1], labels, [row["sd_ratio"] for row in rows],
          [None if row["sd_ratio_low"] is None else [row["sd_ratio_low"], row["sd_ratio_high"]] for row in rows],
          scale="log")
    axes[1].axvline(1.0, color=INK["primary"], linestyle=DASHED, linewidth=1)
    axes[1].set_xlabel("SD within ÷ SD between hypotheses (median over items)")
    figure.suptitle("Hypothesis rewording: original, three paraphrases, and a Chinese translation", fontsize=10,
                    color=INK["primary"])
    figure.tight_layout()
    write_csv(stem.with_suffix(".csv"), rows)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def utterance_figure_rows(metrics: dict) -> list[dict]:
    rows = []
    for name, block in metrics["scorers"].items():
        for group in U_GROUPS:
            found = block["utterance"][group]["abs_delta"]
            interval = found["ci95"] or [None, None]
            rows.append({"scorer": name, "label": block["label"], "rewrite": group, "pairs": found["n"],
                         "median_abs_delta": found["median"], "low": interval[0], "high": interval[1]})
    return rows


def plot_utterance(stem: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    rows = utterance_figure_rows(metrics)
    names = list(metrics["scorers"])
    labels = [metrics["scorers"][name]["label"] for name in names]
    figure, axes = plt.subplots(1, len(U_GROUPS), figsize=(3.0 * len(U_GROUPS) + 2.5, 0.42 * len(names) + 1.6),
                                sharey=True, sharex=True)
    largest = max([row["high"] or row["median_abs_delta"] or 0 for row in rows] + [1.0])
    for ax, group in zip(axes, U_GROUPS):
        mine = {row["scorer"]: row for row in rows if row["rewrite"] == group}
        _dots(ax, labels, [mine[name]["median_abs_delta"] for name in names],
              [None if mine[name]["low"] is None else [mine[name]["low"], mine[name]["high"]] for name in names],
              scale="symlog")
        for threshold in (0.5, 1.0):
            ax.axvline(threshold, color=INK["primary"], linestyle=DASHED, linewidth=0.8)
        ax.set_xlim(-0.02, largest * 1.6)
        ax.set_title(GROUP_LABELS[group], fontsize=9)
    axes[len(axes) // 2].set_xlabel("median |Δ log Bayes factor| between the original and the rewritten utterance "
                                    "(nats; Jev in logits; R3 lines at 0.5 and 1)")
    figure.suptitle("Utterance rewrites: G against minus:X, broad, and contra", fontsize=10, color=INK["primary"])
    figure.tight_layout()
    write_csv(stem.with_suffix(".csv"), rows)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def verbal_points(verbal_rows: list[dict], qwen_log: dict, truth) -> list[dict]:
    points = []
    for row in verbal_rows:
        if row["source"] == "norm" or row.get("status", "success") != "success" or not row["p"]:
            continue
        name = f"qwen_verbal_{row['framing']}" if row["model"] == "qwen" else f"{row['model']}_verbal"
        hvar, uvar = _variant(row["source"], row["side"])
        exact = qwen_log.get((row["item_id"], row["hyp_key"], hvar, uvar))
        semantic = truth(row["item_id"], row["hyp_key"], hvar, row.get("unit_id"))
        points.append({"scorer": name, "item_id": row["item_id"], "hyp_key": row["hyp_key"], "source": row["source"],
                       "log10_verbal": math.log10(row["p"]),
                       "log10_exact": None if exact is None else exact / math.log(10),
                       "log10_semantic": None if semantic is None else math.log10(semantic)})
    return points


def plot_verbal(stem: Path, metrics: dict, points: list[dict]) -> list[Path]:
    plt = _pyplot()
    names = [name for name in VERBAL_ORDER if any(point["scorer"] == name for point in points)]
    figure, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4), squeeze=False)
    for column, name in enumerate(names):
        mine = [point for point in points if point["scorer"] == name]
        block = metrics["claim2"][name]
        for row_index, (truth, label, key) in enumerate((("log10_exact", "log10 exp(−S), exact string", "exact_all"),
                                                         ("log10_semantic", "log10 P_sem, meaning", "semantic_all"))):
            ax = axes[row_index][column]
            shown = [point for point in mine if point[truth] is not None]
            ax.scatter([point[truth] for point in shown], [point["log10_verbal"] for point in shown], s=6, color=MARK,
                       alpha=0.25, linewidths=0)
            low = min([point[truth] for point in shown] + [point["log10_verbal"] for point in shown] + [-1])
            ax.plot([low, 0], [low, 0], color=INK["primary"], linestyle=DASHED, linewidth=1)
            stats = block[key]
            rho = "n/a" if stats["spearman"] is None else f"{stats['spearman']:.2f}"
            gap = stats["gap_orders"]["median"]
            # Top row: the answers sit near 0, far above the truth, so the lower right is empty.
            corner = (0.97, 0.04, "right", "bottom") if row_index == 0 else (0.03, 0.04, "left", "bottom")
            ax.text(corner[0], corner[1], f"ρ = {rho}\nmedian gap {gap:.1f} orders" if gap is not None else f"ρ = {rho}",
                    transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=8, color=INK["primary"],
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": INK["surface"], "edgecolor": INK["grid"],
                          "alpha": 0.9}, zorder=5)
            ax.grid(color=INK["grid"], linewidth=0.6)
            if row_index == 0:
                zeros = block["zero_answers"]
                ax.set_title(f"{SCORER_LABELS[name]}\n({zeros} answers of 0 not shown)", fontsize=9)
            ax.set_xlabel(label)
            if column == 0:
                ax.set_ylabel("log10 verbalized probability")
    figure.suptitle("Verbalized probability against generation probability (dashed: equality)", fontsize=10,
                    color=INK["primary"])
    figure.tight_layout()
    write_csv(stem.with_suffix(".csv"), points)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def write_figures(out: Path, metrics: dict, verbal_rows: list[dict], qwen_log: dict, truth) -> list[Path]:
    return (plot_rewording(out / "fig_claim1_rewording", metrics) + plot_utterance(out / "fig_claim1_utterance", metrics)
            + plot_verbal(out / "fig_claim2_verbal", metrics, verbal_points(verbal_rows, qwen_log, truth)))


# ---------------------------------------------------------------- summary.md


def _f(value, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(bounds, digits: int = 2) -> str:
    return "" if not bounds else f" [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def _m(block: dict, key: str = "median", digits: int = 2) -> str:
    return _f(block.get(key), digits) + _ci(block.get("ci95"), digits)


def summary_markdown(metrics: dict, manifest: dict) -> str:
    scorers = metrics["scorers"]
    lines = ["# Claims 1 and 2 on MultiWOZ: rewrites, rewording, and verbalized probabilities", "",
             "Step 2e of `docs/claims_plan.md`, from GPU jobs A and B, the Claude and GPT runs, and Jev. Robustness is "
             "measured on log Bayes factors and posteriors, never on raw log p. Brackets are 95% bootstrap intervals "
             f"over items ({manifest['bootstrap']['replicates']} replicates, seed {manifest['bootstrap']['seed']}). "
             "Language-model and verbalized scores are in nats; Jev is in logits.", "",
             "## 1. Scorers", "", "| Scorer | Items | Scores | Notes |", "|---|---|---|---|"]
    for name, block in scorers.items():
        notes = block["notes"]
        note = (f"{notes.get('zero', 0)} answers of 0 (no log score), {notes.get('unparsed', 0)} unparsed, "
                f"{notes.get('failed_calls', 0)} failed calls") if notes else ""
        lines.append(f"| {block['label']} | {block['items']} | {block['scores']} | {note} |")
    for name, reason in metrics["pending"].items():
        lines.append(f"| {SCORER_LABELS.get(name, name)} | pending | – | {reason} |")
    noise = metrics["checks"]["rescoring_noise_nats"]
    if noise:
        lines += ["", f"Noise floor. Job A rescored S(U | broad|impl) for all {noise['pairs']} utterances with the "
                      f"prompt Phase 8 used. {noise['identical']} scores are identical; the rest differ because batches "
                      "of another shape take other bfloat16 kernels: |difference| median "
                      f"{_f(noise['median'])}, 90th percentile {_f(noise['p90'])}, max {_f(noise['max'])} nats. A log "
                      "Bayes factor shift that compares an original with a rewrite spans two such runs, so shifts "
                      "below about 1 nat are within this floor.", ""]

    lines += ["## 2. R1: the raw log-likelihood is correct in a narrow sense", "",
              "Named single-slot pairs, mean over slots, against the MultiWOZ corpus log-ratio L_E.", "",
              "| Regime | L_J (Qwen) | L_E (corpus) | Occam gap L_J − L_E |", "|---|---|---|---|"]
    for regime, block in metrics["r1_occam"].items():
        if block:
            lines.append(f"| {regime} | {_f(block['mean_L_J'])}{_ci(block.get('mean_L_J_ci95'))} | "
                         f"{_f(block['L_E'])}{_ci(block.get('L_E_ci95'))} | {_f(block['OG'])}{_ci(block.get('OG_ci95'))} |")

    lines += ["", "## 3. Hypothesis rewording (R2)", "",
              "ICC(1) of item-centered scores, groups = (item, hypothesis), members = the original text, three "
              "paraphrases, and a Chinese translation. SD ratio: the pooled SD across a hypothesis's rewordings over the "
              "SD across the item's hypotheses, for the same utterance (median over items). Shift: |score(H′) − "
              "score(H)|, the change in every log Bayes factor against H (median).", "",
              "| Scorer | ICC(1) | SD ratio | hpara1 | hpara2 | hpara3 | hzh |", "|---|---|---|---|---|---|---|"]
    for name, block in scorers.items():
        found = block["rewording"]
        shifts = " | ".join(_m(found["shift_abs"][kind]) for kind in ("hpara1", "hpara2", "hpara3", "hzh"))
        lines.append(f"| {block['label']} | {_f(found['icc1'])}{_ci(found['icc1_ci95'])} | {_m(found['sd_ratio'])} | "
                     f"{shifts} |")

    lines += ["", "## 4. Utterance rewrites (R3)", "",
              "|Δ log Bayes factor| between the original and the rewritten utterance, for G against minus:X, broad, and "
              "contra, pooled (median). Flip: the preferred hypothesis of the pair changes.", "",
              "| Scorer | " + " | ".join(GROUP_LABELS[group] for group in U_GROUPS) + " |",
              "|---|" + "---|" * len(U_GROUPS)]
    for name, block in scorers.items():
        lines.append(f"| {block['label']} | " + " | ".join(_m(block["utterance"][group]["abs_delta"])
                                                        for group in U_GROUPS) + " |")
    lines += ["", "Flip rate of the preferred hypothesis:", "",
              "| Scorer | " + " | ".join(GROUP_LABELS[group] for group in U_GROUPS) + " |",
              "|---|" + "---|" * len(U_GROUPS)]
    for name, block in scorers.items():
        lines.append(f"| {block['label']} | " + " | ".join(_m(block["utterance"][group]["flip_rate"], "mean")
                                                        for group in U_GROUPS) + " |")
    if "qwen_tok" in scorers:
        raw = scorers["qwen_tok"]["utterance"]
        lines += ["", "For contrast, the raw log p of the utterance under G moves by a median of "
                  + ", ".join(f"{_f(raw[group]['raw_shift_abs']['median'], 1)} nats ({group})" for group in U_GROUPS)
                  + " for Qwen."]

    lines += ["", "## 5. Posterior instability", "",
              f"Per item, one textual variant per class of {{G, minus:X, broad, contra}} (implicit), uniform prior, "
              f"{manifest['posterior_draws']} seeded draws.", "",
              "| Scorer | Items | SD of P(G) | Argmax flip rate |", "|---|---|---|---|"]
    for name, block in scorers.items():
        found = block["posterior"]
        lines.append(f"| {block['label']} | {found['items']} | {_m(found['sd_p_goal'], 'mean', 3)} | "
                     f"{_m(found['argmax_flip_rate'], 'mean')} |")

    lines += ["", "## 6. Claim 2: verbalized against generation probabilities (R4, R5)", "",
              "log10 of the verbalized answer against log10 exp(−S) (Qwen's exact-string probability) and log10 P_sem "
              "(share of Qwen's 64 samples under H with U's annotated slots, (n + 0.5)/(K + 1)). Gap: median of "
              "log10 verbal − log10 truth, in orders of magnitude. Answers of 0 have no log and are left out.", "",
              "| Verbalizer | Answers of 0 | ρ vs exp(−S) | gap vs exp(−S) | ρ vs P_sem | gap vs P_sem |",
              "|---|---|---|---|---|---|"]
    for name in VERBAL_ORDER:
        block = metrics["claim2"].get(name)
        if not block or "exact_all" not in block:
            continue
        exact, semantic = block["exact_all"], block["semantic_all"]
        lines.append(f"| {SCORER_LABELS[name]} | {block['zero_answers']} of {block['queries']} | "
                     f"{_f(exact['spearman'])}{_ci(exact['spearman_ci95'])} | {_m(exact['gap_orders'], digits=1)} | "
                     f"{_f(semantic['spearman'])}{_ci(semantic['spearman_ci95'])} | {_m(semantic['gap_orders'], digits=1)} |")
    lines += ["", "Normalization: the verbalized probabilities of 20 distinct samples drawn under G|impl, summed per item. "
              "For comparison, Qwen's exact probabilities of the same samples, summed per item (median and largest "
              "over items); a sum above 1 is incoherent either way.", "",
              "| Verbalizer | Items | Median sum | Share of items above 1 | Exact sum, median | Exact sum, largest |",
              "|---|---|---|---|---|---|"]
    for name in VERBAL_ORDER:
        block = (metrics["claim2"].get(name) or {}).get("normalization")
        if block:
            lines.append(f"| {SCORER_LABELS[name]} | {block['items']} | {_m(block['sum'])} | "
                         f"{_m(block['share_above_1'], 'mean')} | {block['true_sum_median']:.1e} | "
                         f"{block['true_sum_max']:.2f} |")

    lines += ["", "## 7. Prediction scorecard", "", "Failing a prediction is a result.", "",
              "| ID | Prediction | Value | Result |", "|---|---|---|---|"]
    for row in metrics["predictions"]:
        result = row.get("status", "pending") if row["pass"] is None else ("pass" if row["pass"] else "fail")
        value = row.get("value")
        if isinstance(value, dict):
            value = "; ".join(f"{key} {_f(item) if not isinstance(item, dict) else ', '.join(f'{k} {_f(v)}' for k, v in item.items())}"
                              for key, item in value.items())
        else:
            value = _f(value)
        lines.append(f"| {row['id']} | {row['statement']} | {value} | {result} |")
    lines += ["", "Figures: `fig_claim1_rewording`, `fig_claim1_utterance`, `fig_claim2_verbal` (.png, .pdf, and .csv of "
              "the plotted values). Interpretations of points the plan leaves open are in `docs/claims_decisions.md`.", ""]
    return "\n".join(lines)
