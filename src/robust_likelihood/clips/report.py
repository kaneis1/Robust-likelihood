"""Phase 5 of docs/clips_small_lm_plan.md: docs/clips_results/summary.md, figures 1-4 and the scorecard.

Inputs: metrics_phase3.json (python -m robust_likelihood.clips evaluate), metrics_phase4.json and
effects/<model>.jsonl (python -m robust_likelihood.clips robustness), and the hand-written headline.md
(one sentence per question, K1-K3) and discussion.md (notes for Xuan), included verbatim.

Eight models are too many series for one legend, so models sit on the x axis in a fixed order and the
colour says what changes within a figure: the method (figure 1: CLIPS, language-only IP, action-only IP,
the reference palette's first three slots) or the model type (figures 2-4: base, instruct, simulator).
Every figure is written as .png, .pdf and .csv.
"""

from __future__ import annotations

import random
from pathlib import Path

from robust_likelihood.clips.evaluate import PROBLEMS
from robust_likelihood.clips.perturb import KINDS
from robust_likelihood.multiwoz.report import DASHED, INK, _pyplot, write_csv
from robust_likelihood.multiwoz.sim_report import _get, _save, num, pct, table
from robust_likelihood.storage import read_json, read_jsonl

ORDER = ("qwen3_0p6b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b", "smollm2_1p7b", "qwen3_8b_inst", "qwen25_7b_inst",
         "osim_mid")
LABELS = {"qwen3_0p6b": "Qwen3-0.6B-Base", "qwen3_1p7b": "Qwen3-1.7B-Base", "qwen3_4b": "Qwen3-4B-Base",
          "qwen3_8b": "Qwen3-8B-Base", "smollm2_1p7b": "SmolLM2-1.7B", "qwen3_8b_inst": "Qwen3-8B (instruct)",
          "qwen25_7b_inst": "Qwen2.5-7B-Instruct", "osim_mid": "OSim-8B-Mid"}
TYPE_COLORS = {"base": "#2a78d6", "instruct": "#eb6834", "simulator": "#1baf7a"}
METHOD_COLORS = {"clips": "#2a78d6", "language_only": "#eb6834", "action_only": "#1baf7a"}
METHOD_LABELS = {"clips": "CLIPS", "language_only": "Language-only IP", "action_only": "Action-only IP",
                 "literal_naive": "Literal naive", "literal_efficient": "Literal efficient", "literal": "Literal listener"}
DOMAIN_LABELS = {"mdkg": "Doors, Keys & Gems (mDKG)", "vh": "VirtualHome (VH)"}
BASE_AT_LEAST_1P7B = ("qwen3_1p7b", "qwen3_4b", "qwen3_8b", "smollm2_1p7b")
EXAMPLE_SEED = 20261126


def models_in(metrics: dict) -> list[str]:
    return [model for model in ORDER if model in metrics.get("models", {})]


def mean_se(block: dict | None) -> str:
    if not block or block.get("mean") is None:
        return "–"
    return num(block["mean"]) + (f" ({num(block['se'])})" if block.get("se") is not None else "")


def corr(block: dict | None) -> str:
    if not block or block.get("r") is None:
        return "–"
    interval = block.get("ci95")
    return num(block["r"]) + (f" [{num(interval[0])}, {num(interval[1])}]" if interval else "")


def table1_row(label: str, block: dict) -> list[str]:
    return [label, mean_se(block.get("p_true")), mean_se(block.get("assist_precision")),
            mean_se(block.get("assist_recall")), mean_se(block.get("rel_plan_length")),
            mean_se(block.get("rel_human_cost")), corr(block.get("goal_correlation")),
            corr(block.get("assist_correlation"))]


TABLE1_HEAD = ["", "P(g_true)", "Assist. prec.", "Assist. rec.", "Rel. plan length", "Rel. human cost", "Goal r",
               "Assist. r"]


def table1_sections(phase3: dict, models: list[str]) -> list[str]:
    parts = []
    for domain in ("mdkg", "vh"):
        rows = []
        if domain == "mdkg" and phase3.get("humans"):
            humans = phase3["humans"]
            rows.append(["Humans (paper)", mean_se(humans["p_true"]), mean_se(humans["assist_precision"]),
                         mean_se(humans["assist_recall"]), "–", "–", "–", "–"])
        paper = phase3.get("paper_files", {}).get(domain, {})
        model_name = "curie" if domain == "mdkg" else "davinci-002"
        for method in ("clips", "language_only", "action_only"):
            if method in paper:
                rows.append(table1_row(f"{METHOD_LABELS[method]}, paper ({model_name})", paper[method]))
        for model in models:
            block = _get(phase3, "models", model, domain, "clips")
            if block:
                rows.append(table1_row(f"**CLIPS · {LABELS[model]}**", block))
        parts += [f"**{DOMAIN_LABELS[domain]}: CLIPS**", "", table(TABLE1_HEAD, rows), ""]
        rows = []
        literal = ("literal_naive", "literal_efficient") if domain == "mdkg" else ("literal",)
        for method in ("language_only",) + literal:
            if method in paper:
                rows.append(table1_row(f"{METHOD_LABELS[method]}, paper", paper[method]))
            for model in models:
                block = _get(phase3, "models", model, domain, method)
                if block:
                    rows.append(table1_row(f"{METHOD_LABELS[method]} · {LABELS[model]}", block))
        parts += [f"**{DOMAIN_LABELS[domain]}: language-only IP and the literal listeners**", "",
                  table(TABLE1_HEAD, rows), ""]
    return parts


def check_section(phase3: dict, models: list[str]) -> list[str]:
    rows = []
    for model in models:
        for domain in ("mdkg", "vh"):
            found = _get(phase3, "action_only_check", model, domain)
            if found:
                rows.append([LABELS[model], domain, f"{found['problems']}", num(found["p_true"], 3),
                             num(found["published"]), f"{found['max_abs_difference_per_problem_vs_paper_file']:.1e}",
                             "pass" if found["passes"] else "FAIL"])
    return ["**Setup check: action-only IP uses no language model** (plan: stop if it misses by more than "
            "0.02; the last column is the largest per-problem gap against the paper's own result file)", "",
            table(["Model", "Domain", "Problems", "P(g_true)", "Published", "Largest gap", "Check"], rows), ""]


def robustness_sections(phase4: dict, models: list[str]) -> list[str]:
    parts = ["## K3: robustness to the phrasing of the instruction", "",
             "Both domains pooled (65 instructions; the n row gives the valid units per kind). Each Δ is between the "
             "original and the perturbed instruction scored in the same batch. P(g_true) is CLIPS's at the final "
             "step; log BF is between the true goal and its strongest rival, from the language term alone.", ""]
    kinds = [kind for kind in KINDS if any(_get(phase4, "models", model, "both", kind) for model in models)]
    for title, field, show in (("Median |Δ P(g_true)|", "abs_delta_p_true", lambda block: num(block.get("median"), 3)),
                               ("Posterior argmax flips", "argmax_flip", lambda block: pct(block.get("rate"))),
                               ("Median |Δ log BF|, language term (nats)", "abs_delta_log_bf_language",
                                lambda block: num(block.get("median")))):
        rows = []
        for model in models:
            rows.append([LABELS[model]] + [show(_get(phase4, "models", model, "both", kind, field) or {})
                                           for kind in kinds])
        units = [str(_get(phase4, "models", models[0], "both", kind, "units") or "–") for kind in kinds] if models else []
        rows.append(["n (units)"] + units)
        parts += [f"**{title}**", "", table(["Model"] + kinds, rows), ""]
    rows = []
    for model in models:
        rows.append([LABELS[model]] + [num(_get(phase4, "models", model, "mdkg", kind, "goal_correlation", "change"), 3)
                                       for kind in kinds])
    parts += ["**mDKG: change in goal correlation with humans** (perturbed minus original, over the problems with a "
              "valid unit of the kind)", "", table(["Model"] + kinds, rows), ""]
    return parts


def predictions(phase3: dict, phase4: dict) -> list[dict]:
    def replayed(model, domain):
        """A domain counts only when every problem's replay is in (partial replays score nothing)."""
        return _get(phase3, "models", model, domain, "clips", "problems") == PROBLEMS[domain]

    def p_true(model, domain, method="clips"):
        return _get(phase3, "models", model, domain, method, "p_true", "mean") if replayed(model, domain) else None

    def goal_r(model):
        return _get(phase3, "models", model, "mdkg", "clips", "goal_correlation", "r") if replayed(model, "mdkg") \
            else None

    def pooled(model, kind, field, key):
        if not (replayed(model, "mdkg") and replayed(model, "vh")):
            return None  # the robustness numbers pool only the problems whose replay is in
        return _get(phase4, "models", model, "both", kind, field, key)

    found = []

    def add(pid, statement, value, passed):
        found.append({"id": pid, "statement": statement, "value": value,
                      "pass": None if passed is None else bool(passed)})

    p, r = p_true("qwen3_8b", "mdkg"), goal_r("qwen3_8b")
    add("C1", "qwen3_8b: mDKG P(g_true) within 0.05 of 0.74 and goal correlation >= 0.85", {"p_true": p, "goal_r": r},
        None if p is None or r is None else abs(p - 0.74) <= 0.05 and r >= 0.85)
    values = {model: {domain: p_true(model, domain) for domain in ("mdkg", "vh")}
              for model in ("qwen3_0p6b", "qwen3_1p7b", "qwen3_8b")}
    complete = all(value is not None for found_model in values.values() for value in found_model.values())

    def close(model):
        return all(abs(values[model][domain] - values["qwen3_8b"][domain]) <= 0.05 for domain in ("mdkg", "vh"))

    add("C2", "qwen3_1p7b within 0.05 of qwen3_8b on P(g_true) in both domains; qwen3_0p6b is not", values,
        close("qwen3_1p7b") and not close("qwen3_0p6b") if complete else None)
    inst = {"goal_r": goal_r("qwen3_8b_inst"), "mdkg": p_true("qwen3_8b_inst", "mdkg"), "vh": p_true("qwen3_8b_inst", "vh")}
    base = {"goal_r": goal_r("qwen3_8b"), "mdkg": p_true("qwen3_8b", "mdkg"), "vh": p_true("qwen3_8b", "vh")}
    add("C3", "qwen3_8b_inst below qwen3_8b on mDKG goal correlation and on P(g_true) in both domains",
        {"qwen3_8b_inst": inst, "qwen3_8b": base},
        None if None in inst.values() or None in base.values() else all(inst[key] < base[key] for key in inst))
    order = {}
    for model in BASE_AT_LEAST_1P7B:
        for domain in ("mdkg", "vh"):
            trio = [p_true(model, domain, method) for method in ("clips", "language_only", "action_only")]
            order[f"{model}:{domain}"] = None if None in trio else trio[0] > trio[1] > trio[2]
    add("C4", "CLIPS > language-only IP > action-only IP on P(g_true), every base model of 1.7B or more, both domains",
        order, None if None in order.values() else all(order.values()))
    robust = {}
    for model in BASE_AT_LEAST_1P7B:
        values5 = {"para1_abs_delta": pooled(model, "para1", "abs_delta_p_true", "median"),
                   "typo1_abs_delta": pooled(model, "typo1", "abs_delta_p_true", "median"),
                   "para1_flip": pooled(model, "para1", "argmax_flip", "rate"),
                   "typo1_flip": pooled(model, "typo1", "argmax_flip", "rate")}
        robust[model] = values5
    complete = all(value is not None for block in robust.values() for value in block.values())
    add("C5", "base models of 1.7B or more: median |Δ P(g_true)| <= 0.10 under para1 and <= 0.05 under typo1, argmax "
              "flips <= 10% (both domains pooled)", robust,
        all(block["para1_abs_delta"] <= 0.10 and block["typo1_abs_delta"] <= 0.05 and block["para1_flip"] <= 0.10
            and block["typo1_flip"] <= 0.10 for block in robust.values()) if complete else None)
    flips = {model: pooled(model, "para1", "argmax_flip", "rate") for model in ("qwen3_8b_inst", "qwen3_8b")}
    add("C6", "qwen3_8b_inst flips the argmax under para1 more often than qwen3_8b (both domains pooled)", flips,
        None if None in flips.values() else flips["qwen3_8b_inst"] > flips["qwen3_8b"])
    return found


def scorecard(found: list[dict]) -> list[str]:
    rows = [[row["id"], row["statement"], "–" if row["pass"] is None else ("pass" if row["pass"] else "fail")]
            for row in found]
    return ["## Prediction scorecard", "", table(["ID", "Prediction", "Result"], rows), ""]


def figure_p_true(plt, phase3: dict, models: list[str], out: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    rows = []
    for ax, domain in zip(axes, ("mdkg", "vh")):
        positions = list(range(len(models)))
        for offset, method in zip((-0.2, 0.0, 0.2), ("clips", "language_only", "action_only")):
            points = [_get(phase3, "models", model, domain, method, "p_true") or {} for model in models]
            xs = [x + offset for x, block in zip(positions, points) if block.get("mean") is not None]
            ys = [block["mean"] for block in points if block.get("mean") is not None]
            lows = [block["mean"] - block["ci95"][0] for block in points if block.get("mean") is not None]
            highs = [block["ci95"][1] - block["mean"] for block in points if block.get("mean") is not None]
            ax.errorbar(xs, ys, yerr=[lows, highs], fmt="o", color=METHOD_COLORS[method], ecolor=INK["secondary"],
                        elinewidth=0.8, markersize=6, label=METHOD_LABELS[method])
            published = _get(phase3, "published", domain, f"{'clips' if method == 'clips' else method}_p_true")
            if published is not None:
                ax.axhline(published, color=METHOD_COLORS[method], linestyle=DASHED, linewidth=0.9)
            for model, block in zip(models, points):
                rows.append({"domain": domain, "model": model, "method": method, "p_true": block.get("mean"),
                             "low": (block.get("ci95") or [None, None])[0],
                             "high": (block.get("ci95") or [None, None])[1], "published": published})
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[model] for model in models], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(DOMAIN_LABELS[domain], fontsize=9, color=INK["primary"])
        ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(g_true) at the final step (mean over problems, 95% CI)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    figure.suptitle("Goal accuracy by model and method (dashed: the paper's values with curie / davinci-002)",
                    fontsize=10, color=INK["primary"])
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    write_csv(out / "fig1_p_true.csv", rows)
    paths = _save(figure, out / "fig1_p_true")
    plt.close(figure)
    return paths


def figure_correlation(plt, phase3: dict, config: dict, models: list[str], out: Path) -> list[Path]:
    figure, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = []
    for model in models:
        block = _get(phase3, "models", model, "mdkg", "clips", "goal_correlation")
        if not block or block.get("r") is None:
            continue
        spec = config["models"][model]
        size, kind = spec["params_b"], spec["kind"]
        interval = block.get("ci95") or [block["r"], block["r"]]
        ax.errorbar([size], [block["r"]], yerr=[[block["r"] - interval[0]], [interval[1] - block["r"]]], fmt="o",
                    color=TYPE_COLORS[kind], ecolor=INK["secondary"], elinewidth=0.8, markersize=7)
        ax.annotate(LABELS[model], (size, block["r"]), textcoords="offset points", xytext=(6, -3), fontsize=7,
                    color=INK["secondary"])
        rows.append({"model": model, "params_b": size, "kind": kind, "goal_r": block["r"], "low": interval[0],
                     "high": interval[1]})
    published = _get(phase3, "paper_files", "mdkg", "clips", "goal_correlation", "r")
    if published is not None:
        ax.axhline(published, color=INK["muted"], linestyle=DASHED, linewidth=0.9)
        ax.text(0.55, published + 0.005, f"paper (curie): {published:.2f}", fontsize=7, color=INK["muted"])
    for kind, color in TYPE_COLORS.items():
        ax.plot([], [], "o", color=color, label=kind)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)", fontsize=8)
    ax.set_ylabel("Pearson r with mean human goal ratings (mDKG, 95% CI)", fontsize=8)
    ax.grid(color=INK["grid"], linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("Agreement with human goal judgments by model size", fontsize=10, color=INK["primary"])
    if rows:
        write_csv(out / "fig2_goal_correlation.csv", rows)
    paths = _save(figure, out / "fig2_goal_correlation")
    plt.close(figure)
    return paths


def figure_robustness(plt, phase4: dict, config: dict, models: list[str], out: Path) -> list[Path]:
    kinds = [kind for kind in KINDS if any(_get(phase4, "models", model, "both", kind) for model in models)]
    columns = 4
    lines = -(-len(kinds) // columns)
    figure, axes = plt.subplots(lines, columns, figsize=(12, 3.2 * lines), sharey=True, squeeze=False)
    rows = []
    for index, kind in enumerate(kinds):
        ax = axes[index // columns][index % columns]
        for x, model in enumerate(models):
            block = _get(phase4, "models", model, "both", kind, "abs_delta_p_true") or {}
            if block.get("median") is None:
                continue
            interval = block.get("ci95") or [block["median"]] * 2
            ax.errorbar([x], [block["median"]], yerr=[[block["median"] - interval[0]], [interval[1] - block["median"]]],
                        fmt="o", color=TYPE_COLORS[config["models"][model]["kind"]], ecolor=INK["secondary"],
                        elinewidth=0.8, markersize=5)
            rows.append({"kind": kind, "model": model, "median_abs_delta_p_true": block["median"],
                         "low": interval[0], "high": interval[1], "units": block.get("n")})
        ax.set_title(kind, fontsize=9, color=INK["primary"])
        ax.set_xticks(range(len(models)))
        bottom = index + columns >= len(kinds)
        ax.set_xticklabels([LABELS[model] for model in models] if bottom else [], rotation=55, ha="right",
                           fontsize=6.5)
        ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    for index in range(len(kinds), lines * columns):
        axes[index // columns][index % columns].axis("off")
    for line in axes:
        line[0].set_ylabel("median |Δ P(g_true)|", fontsize=8)
    for kind, color in TYPE_COLORS.items():
        axes[0][0].plot([], [], "o", color=color, label=kind)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    figure.suptitle("How far the true goal's posterior moves when the instruction is perturbed (both domains, 95% CI)",
                    fontsize=10, color=INK["primary"])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if rows:
        write_csv(out / "fig3_robustness.csv", rows)
    paths = _save(figure, out / "fig3_robustness")
    plt.close(figure)
    return paths


def example_problems(effects: dict[str, list[dict]], count: int = 6) -> list[str]:
    """Seeded choice of mDKG instructions with a valid para1 unit under every model."""
    shared = None
    for rows in effects.values():
        found = {row["utterance"] for row in rows if row["kind"] == "para1" and row["domain"] == "mdkg"}
        shared = found if shared is None else shared & found
    chosen = sorted(shared or [])
    random.Random(EXAMPLE_SEED).shuffle(chosen)
    return sorted(chosen[:count])


def figure_examples(plt, effects: dict[str, list[dict]], units: list[dict], config: dict, models: list[str],
                    out: Path) -> list[Path]:
    chosen = example_problems({model: effects[model] for model in models if model in effects})
    if not chosen:
        return []
    texts = {unit["utterance"]: (unit["original"], unit["rewrite"]) for unit in units if unit["kind"] == "para1"}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    rows = []
    for ax, key in zip([ax for line in axes for ax in line], chosen):
        for x, model in enumerate(models):
            row = next((row for row in effects.get(model, []) if row["utterance"] == key and row["kind"] == "para1"),
                       None)
            if row is None:
                continue
            color = TYPE_COLORS[config["models"][model]["kind"]]
            ax.plot([x, x], [row["p_true"], row["p_true_new"]], color=INK["axis"], linewidth=1.2, zorder=1)
            ax.scatter([x], [row["p_true"]], color=color, s=28, zorder=2)
            ax.scatter([x], [row["p_true_new"]], facecolor="none", edgecolor=color, marker="s", s=28, zorder=3)
            rows.append({"instruction": key, "model": model, "p_true_original": row["p_true"],
                         "p_true_paraphrase": row["p_true_new"]})
        original, rewrite = texts.get(key, ("", ""))
        ax.set_title(f"{key.split(':')[1]}\n“{original}”\n→ “{rewrite}”", fontsize=7,
                     color=INK["primary"])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS[model] for model in models], rotation=55, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    for line in axes:
        line[0].set_ylabel("P(g_true), CLIPS", fontsize=8)
    axes[0][0].scatter([], [], color=INK["secondary"], s=28, label="original")
    axes[0][0].scatter([], [], facecolor="none", edgecolor=INK["secondary"], marker="s", s=28, label="para1")
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    figure.suptitle("Six mDKG problems: the true goal's posterior under the original instruction and its paraphrase "
                    "(colour: model type)", fontsize=10, color=INK["primary"])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    write_csv(out / "fig4_examples.csv", rows)
    paths = _save(figure, out / "fig4_examples")
    plt.close(figure)
    return paths


def summary_markdown(phase3: dict, phase4: dict, found: list[dict], models: list[str], headline: list[str] | None,
                     discussion: str | None, phase4_units: dict | None) -> str:
    parts = ["# CLIPS with small local language models as the utterance likelihood", ""]
    if headline:
        parts += headline + [""]
    parts += ["Plan: `docs/clips_small_lm_plan.md`; decisions and deviations: `docs/clips_decisions.md`. Intervals: "
              "bootstrap over problems (Table 1) or instructions (robustness), 2,000 replicates, seed 20261126. "
              "Table 1's definitions reproduce every cell of the paper's table from its own result files.", ""]
    parts += ["## K1 and K2: goal accuracy, assistance and agreement with humans", ""]
    parts += table1_sections(phase3, models)
    parts += check_section(phase3, models)
    parts += robustness_sections(phase4, models)
    if phase4_units:
        rows = [[domain] + [str(_get(phase4_units, domain, kind, "valid") or 0) for kind in KINDS]
                for domain in ("mdkg", "vh")]
        parts += ["**Valid perturbed instructions per kind**", "", table(["Domain"] + list(KINDS), rows), ""]
    parts += scorecard(found)
    if discussion:
        parts += [discussion.strip(), ""]
    return "\n".join(parts)


def run_report(root: Path, out: Path, figures: bool = True) -> dict:
    config = read_json(root / "configs" / "clips.json")
    phase3 = read_json(out / "metrics_phase3.json")
    phase4 = read_json(out / "metrics_phase4.json") if (out / "metrics_phase4.json").exists() else {"models": {}}
    models = models_in(phase3)
    found = predictions(phase3, phase4)
    counts_path = root / "data" / "clips" / "derived" / "perturbations_counts.json"
    phase4_units = read_json(counts_path)["units"] if counts_path.exists() else None
    headline_path, discussion_path = out / "headline.md", out / "discussion.md"
    headline = headline_path.read_text(encoding="utf-8").strip().splitlines() if headline_path.exists() else None
    discussion = discussion_path.read_text(encoding="utf-8") if discussion_path.exists() else None
    paths = []
    if figures:
        plt = _pyplot()
        effects = {model: read_jsonl(out / "effects" / f"{model}.jsonl") for model in models
                   if (out / "effects" / f"{model}.jsonl").exists()}
        units = read_jsonl(root / "data" / "clips" / "derived" / "perturbations.jsonl")
        paths += figure_p_true(plt, phase3, models, out)
        paths += figure_correlation(plt, phase3, config, models, out)
        if phase4["models"]:
            paths += figure_robustness(plt, phase4, config, models, out)
            paths += figure_examples(plt, effects, units, config, models, out)
    from robust_likelihood.storage import write_json

    write_json(out / "predictions.json", found)
    (out / "summary.md").write_text(summary_markdown(phase3, phase4, found, models, headline, discussion, phase4_units),
                                    encoding="utf-8", newline="\n")
    return {"summary": str(out / "summary.md"), "figures": [str(path) for path in paths],
            "predictions": {row["id"]: row["pass"] for row in found}}
