"""Figures and summary.md for the dice sweep. matplotlib is imported only to draw.

Scorers keep fixed categorical slots (S1 blue, S2 orange, S3 aqua) and fixed markers,
so a scorer looks the same in every panel; the exact likelihood is drawn in ink.
"""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.dice.analyze import D2_LIMIT, D3_LIMIT, D4_LIMIT, TRUE_SLOPES
from robust_likelihood.multiwoz.report import DASHED, INK, _pyplot, write_csv

SCORER_COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S1-nospace": "#eda100"}
SCORER_MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S1-nospace": "D"}
SCORER_LABELS = {"S1": "S1 token log-prob", "S2": "S2 verbalized, judgment", "S3": "S3 verbalized, computation",
                 "S1-nospace": "S1 token log-prob, lists as 3,5,2 (sensitivity)"}
MODEL_ORDER = ("qwen", "olmo", "claude", "gpt")
MODEL_LABELS = {"qwen": "Qwen2.5-7B-Instruct", "olmo": "OLMo-2-7B-Instruct", "claude": "Claude (API)", "gpt": "GPT (API)"}
HYPOTHESES = ("fair", "even", "loaded")
NORM_SETS = ("k1_fair", "k1_even", "k1_loaded", "k2_fair")
NORM_LABELS = {"k1_fair": "k=1 fair", "k1_even": "k=1 even", "k1_loaded": "k=1 loaded", "k2_fair": "k=2 fair (36)"}


def _models(metrics: dict) -> list[str]:
    present = {block["model"] for block in metrics["sources"].values()}
    return [model for model in MODEL_ORDER if model in present] + sorted(present - set(MODEL_ORDER))


def _blocks(metrics: dict, model: str) -> list[dict]:
    return [block for block in metrics["sources"].values() if block["model"] == model]


def _save(figure, stem: Path) -> list[Path]:
    paths = []
    for suffix, options in ((".png", {"dpi": 200}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    return paths


def slope_rows(metrics: dict) -> list[dict]:
    rows = []
    for block in metrics["sources"].values():
        for hypothesis, found in block["slopes"].items():
            for point in found.get("per_k", []):
                rows.append({"model": block["model"], "scorer": block["scorer"], "hypothesis": hypothesis, **point})
    return sorted(rows, key=lambda row: (row["model"], row["hypothesis"], row["scorer"], row["k"]))


def plot_slopes(stem: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    rows = slope_rows(metrics)
    models = _models(metrics)
    figure, axes = plt.subplots(len(HYPOTHESES), len(models), figsize=(3.3 * len(models), 2.7 * len(HYPOTHESES)),
                                squeeze=False, sharex=True)
    for column, model in enumerate(models):
        for row_index, hypothesis in enumerate(HYPOTHESES):
            ax = axes[row_index][column]
            mine = [row for row in rows if row["model"] == model and row["hypothesis"] == hypothesis]
            truth = {}
            for row in mine:
                truth.setdefault(row["k"], row["mean_true_log10"])
            if truth:
                ks = sorted(truth)
                ax.plot(ks, [truth[k] for k in ks], color=INK["primary"], linewidth=1.2, linestyle=DASHED,
                        label="exact", zorder=5)
            for scorer in SCORER_COLORS:
                points = [row for row in mine if row["scorer"] == scorer]
                if not points:
                    continue
                ks = [row["k"] for row in points]
                ax.vlines(ks, [row["min_log10"] for row in points], [row["max_log10"] for row in points],
                          color=SCORER_COLORS[scorer], linewidth=1, alpha=0.45, zorder=2)
                ax.plot(ks, [row["mean_log10"] for row in points], color=SCORER_COLORS[scorer], linewidth=2,
                        marker=SCORER_MARKERS[scorer], markersize=5, markeredgecolor=INK["surface"],
                        markeredgewidth=1, label=SCORER_LABELS[scorer], zorder=4)
            if row_index == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            if column == 0:
                ax.set_ylabel(f"{hypothesis} die\nlog10 P(sequence)")
            if row_index == len(HYPOTHESES) - 1:
                ax.set_xlabel("rolls k")
    handles, labels = [], []
    for ax in figure.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Sequences scored under their own die: mean per k (line) and range (bar)", y=1.0,
                    color=INK["primary"], fontsize=10)
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    write_csv(stem.with_suffix(".csv"), rows)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def bayes_rows(metrics: dict) -> list[dict]:
    rows = []
    for block in metrics["sources"].values():
        for point in block["bayes"]["points"]:
            if point["predicted"] is not None:
                rows.append({"model": block["model"], "scorer": block["scorer"], **point})
    return sorted(rows, key=lambda row: (row["model"], row["scorer"], row["seq_id"], row["pair"]))


def plot_bayes(stem: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    rows = bayes_rows(metrics)
    models = _models(metrics)
    values = [row["true"] for row in rows] + [row["predicted"] for row in rows] + [0.0]
    limits = (min(values) - 1, max(values) + 1)
    figure, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.7), squeeze=False)
    for column, model in enumerate(models):
        ax = axes[0][column]
        ax.axhline(0, color=INK["axis"], linewidth=0.8, zorder=1)
        ax.axvline(0, color=INK["axis"], linewidth=0.8, zorder=1)
        ax.plot(limits, limits, color=INK["muted"], linewidth=1, linestyle=DASHED, zorder=1, label="predicted = exact")
        for scorer in SCORER_COLORS:
            points = [row for row in rows if row["model"] == model and row["scorer"] == scorer]
            if points:
                ax.scatter([row["true"] for row in points], [row["predicted"] for row in points], s=18,
                           color=SCORER_COLORS[scorer], marker=SCORER_MARKERS[scorer], alpha=0.7,
                           edgecolors=INK["surface"], linewidths=0.5, label=SCORER_LABELS[scorer], zorder=3)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("exact log Bayes factor (nats)")
        if column == 0:
            ax.set_ylabel("scorer's log Bayes factor (nats)")
    handles, labels = [], []
    for ax in figure.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    write_csv(stem.with_suffix(".csv"), rows)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def normalization_rows(metrics: dict) -> list[dict]:
    rows = []
    for block in metrics["sources"].values():
        for name, found in block["normalization"].items():
            rows.append({"model": block["model"], "scorer": block["scorer"], "set": name, **found})
    return sorted(rows, key=lambda row: (row["model"], row["scorer"], row["set"]))


def plot_normalization(stem: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    rows = normalization_rows(metrics)
    models = _models(metrics)
    sums = [row["sum"] for row in rows if row["sum"] > 0] + [1.0]
    low, high = min(sums) / 2, max(sums) * 2
    figure, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.4), squeeze=False, sharey=True)
    for column, model in enumerate(models):
        ax = axes[0][column]
        scorers = [scorer for scorer in SCORER_COLORS if any(r["model"] == model and r["scorer"] == scorer for r in rows)]
        width = 0.8 / max(1, len(scorers))
        for offset, scorer in enumerate(scorers):
            found = {row["set"]: row for row in rows if row["model"] == model and row["scorer"] == scorer}
            positions = [index - 0.4 + width * (offset + 0.5) for index in range(len(NORM_SETS))]
            heights = [max(found[name]["sum"], low) if name in found else 0 for name in NORM_SETS]
            ax.bar(positions, heights, width=width - 0.04, color=SCORER_COLORS[scorer], label=SCORER_LABELS[scorer],
                   zorder=3)
        ax.axhline(1.0, color=INK["primary"], linewidth=1.2, linestyle=DASHED, zorder=4, label="true sum = 1")
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_xticks(range(len(NORM_SETS)))
        ax.set_xticklabels([NORM_LABELS[name] for name in NORM_SETS], rotation=30, ha="right")
        ax.set_title(MODEL_LABELS.get(model, model))
        if column == 0:
            ax.set_ylabel("sum of scores over the outcome set")
    handles, labels = [], []
    for ax in figure.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    write_csv(stem.with_suffix(".csv"), rows)
    paths = _save(figure, stem) + [stem.with_suffix(".csv")]
    plt.close(figure)
    return paths


def write_figures(out_dir: Path, metrics: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return (plot_slopes(out_dir / "fig_dice_slopes", metrics) + plot_bayes(out_dir / "fig_dice_bayes", metrics)
            + plot_normalization(out_dir / "fig_dice_normalization", metrics))


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(bounds, digits: int = 3) -> str:
    return "–" if not bounds else f"[{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def _result(row: dict) -> str:
    if row["pass"] is None:
        return row.get("status", "pending")
    return "pass" if row["pass"] else "fail"


def _value(row: dict) -> str:
    value = row["value"]
    if isinstance(value, dict):
        return ", ".join(f"{key} {_f(found)}" for key, found in value.items())
    return _f(value)


def summary_markdown(metrics: dict, manifest: dict) -> str:
    models = _models(metrics)
    lines = ["# Dice sweep: verbalized probabilities against exact likelihoods", ""]
    lines += [
        "Step 1 of `docs/claims_plan.md`. Three dice with known face probabilities generate roll sequences of",
        "length k ∈ {1, 2, 3, 4, 6, 8, 10}; every likelihood is exact, and the observation space grows from 6",
        "to 6¹⁰ outcomes. S1 is the token log-prob of the list as a simulated roll; S2 asks how likely a",
        "reported sequence is (the LLM-as-likelihood framing); S3 asks for the probability as an arithmetic",
        "question (control).",
        "",
        "## 1. Setup and counts",
        "",
        f"- Sequences: {metrics['counts']['sequences']}; pairs (sequence × die): {metrics['counts']['pairs']};"
        f" normalization queries: {metrics['counts']['normalization_queries']}.",
        "- API scorers see the 630-pair subset (10 sequences per cell) plus the normalization queries.",
    ]
    for run in manifest["runs"]:
        lines.append(f"- {run['kind']} run `{run['run_dir']}`: {run['summary']}")
    lines += ["", "| Model | Scorer | Queries | Parse failures | Zero answers |", "|---|---|---|---|---|"]
    for model in models:
        for block in sorted(_blocks(metrics, model), key=lambda block: block["scorer"]):
            rate = block["parse_failure_rate"]
            lines.append(f"| {MODEL_LABELS.get(model, model)} | {block['scorer']} | {block['queries']} | "
                         f"{block['parse_failures']} ({_f(None if rate is None else 100 * rate, 1)}%) | {block['zeros']} |")
    lines += ["", "## 2. Slope of log10 score in k (own die)", "",
              f"True slopes: fair {TRUE_SLOPES['fair']:.3f}, even {TRUE_SLOPES['even']:.3f} per roll; `loaded` varies by"
              " sequence, so its truth is fitted on the same sequences.", "",
              "| Model | Scorer | Die | Slope [95% CI] | Truth | Ratio [95% CI] | R² | n |", "|---|---|---|---|---|---|---|---|"]
    for model in models:
        for block in sorted(_blocks(metrics, model), key=lambda block: block["scorer"]):
            for hypothesis in HYPOTHESES:
                found = block["slopes"].get(hypothesis)
                if not found or not found.get("fit"):
                    continue
                lines.append(
                    f"| {MODEL_LABELS.get(model, model)} | {block['scorer']} | {hypothesis} | {_f(found['fit']['slope'])} "
                    f"{_ci(found.get('slope_ci95'))} | {_f(found['truth']['slope'])} | {_f(found.get('ratio_to_truth'), 2)} "
                    f"{_ci(found.get('ratio_ci95'), 2)} | {_f(found['fit']['r2'], 2)} | {found['n_used']}/{found['n']} |")
    lines += ["", "## 3. Accuracy at k = 1", "", "Each face under each die (18 queries).", "",
              "| Model | Scorer | Mean abs error | Max abs error | Share within ±0.10 |", "|---|---|---|---|---|"]
    for model in models:
        for block in sorted(_blocks(metrics, model), key=lambda block: block["scorer"]):
            found = block["k1_accuracy"]
            lines.append(f"| {MODEL_LABELS.get(model, model)} | {block['scorer']} | {_f(found['mean_abs_error'])} | "
                         f"{_f(found['max_abs_error'])} | {_f(found['share_within_0_10'], 2)} |")
    lines += ["", "## 4. Bayes factors", "",
              "Predicted against exact log Bayes factor over every sequence and die pair with both likelihoods",
              "finite. The even-vs-fair column uses all-even sequences only, where the truth is k ln 2.",
              "Impossible pairs: an odd face under the even die, whose exact probability is 0. Verbalized",
              "scorers are scored by the share answered exactly 0; token log-probs by the median log p.", "",
              "| Model | Scorer | Slope, all pairs [95% CI] | R² | Slope, even vs fair [95% CI] | Impossible pairs |",
              "|---|---|---|---|---|---|"]
    for model in models:
        for block in sorted(_blocks(metrics, model), key=lambda block: block["scorer"]):
            found = block["bayes"]
            fit = found["fit"] or {}
            even = found["even_vs_fair_all_even"] or {}
            impossible = block["impossible"]
            if "share_answered_zero" in impossible:
                cell = f"{_f(impossible['share_answered_zero'], 2)} answered 0 (n = {impossible['n']})"
            else:
                cell = f"median log p {_f(impossible['median_logp'], 1)} (n = {impossible['n']})"
            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {block['scorer']} | {_f(fit.get('slope'), 2)} "
                f"{_ci(found.get('slope_ci95'), 2)} | {_f(fit.get('r2'), 2)} | {_f(even.get('slope'), 2)} "
                f"{_ci(found.get('even_vs_fair_slope_ci95'), 2)} | {cell} |")
    lines += ["", "## 5. Normalization sums", "",
              "True sum = 1 for every set. Token log-probs should sum to at most 1.", "",
              "| Model | Scorer | " + " | ".join(NORM_LABELS[name] for name in NORM_SETS) + " |",
              "|---|---|" + "---|" * len(NORM_SETS)]
    for model in models:
        for block in sorted(_blocks(metrics, model), key=lambda block: block["scorer"]):
            cells = []
            for name in NORM_SETS:
                found = block["normalization"].get(name)
                if not found:
                    cells.append("–")
                    continue
                missing = f" ({found['missing']} missing)" if found["missing"] else ""
                cells.append(f"{found['sum']:.3g}{missing}")
            lines.append(f"| {MODEL_LABELS.get(model, model)} | {block['scorer']} | " + " | ".join(cells) + " |")
    lines += ["", "## 6. Prediction scorecard", "", "Failing a prediction is a result.", "",
              "| ID | Model | Claim | Value | Result |", "|---|---|---|---|---|"]
    for row in metrics["predictions"]:
        lines.append(f"| {row['id']} | {MODEL_LABELS.get(row['model'], row['model'])} | {row['claim']} | "
                     f"{_value(row)} | {_result(row)} |")
    accepted = metrics["acceptance"]
    lines += ["", "## 7. Acceptance", "",
              "- Sequences regenerate from seed 20261027 (`python -m robust_likelihood.dice build` is deterministic).",
              f"- Local parse failures under 2%: {_f(accepted['local_parse_failures_under_2pct'])}.",
              "- Every raw answer is kept in the run folders under `results/dice/`.", "",
              "## Figures", "",
              "- `fig_dice_slopes`: log10 P against k, exact and each scorer, one column per model.",
              "- `fig_dice_bayes`: predicted against exact log Bayes factor.",
              "- `fig_dice_normalization`: sums over the k = 1 and k = 2 outcome sets.",
              "", f"Operational definitions: D2 uses the fair die, |slope| < {D2_LIMIT}; D3 is the mean absolute error over",
              f"the 18 k = 1 queries, ≤ {D3_LIMIT}; D4 is the sum over the 36 k = 2 fair sequences, > {D4_LIMIT}.",
              "Answers of exactly 0 and unparsed answers have no log score and are left out of the fits.", ""]
    return "\n".join(lines)
