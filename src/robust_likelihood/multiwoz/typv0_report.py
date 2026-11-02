"""summary.md for typicality v0 (claims plan step 3)."""

from __future__ import annotations

ORDER = ("per2A", "glob2A", "glob1A", "glob1B", "cent1B", "sem1B")
BASELINES = {
    "qwen_tok": "Qwen token log-prob",
    "qwen_clip": "Qwen clipped log-prob",
    "jev_likely": "Jev, likely",
    "jev_typical": "Jev, typical",
    "qwen_verbal_exact": "Qwen verbalized, exact (60 items)",
    "qwen_verbal_meaning": "Qwen verbalized, meaning (60 items)",
    "claude_verbal": "Claude verbalized (60 items)",
    "gpt_verbal": "GPT verbalized (60 items)",
}
GROUPS = ("pad", "typo", "para", "zh")


def _f(value, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if value in (float("inf"), float("-inf")):
            return "∞" if value > 0 else "−∞"
        return f"{value:.{digits}f}"
    return str(value)


def _get(block, *path):
    for key in path:
        if block is None:
            return None
        block = block.get(key) if isinstance(block, dict) else None
    return block


def _ratio(size: dict | None, regime: str) -> float | None:
    occam = _get(size, "occam", f"{regime}|slots_mean|named")
    if not occam or not occam["L_E"]:
        return None
    return occam["mean_L_J"] / occam["L_E"]


def size_cells(size: dict | None) -> list[str]:
    return [_f(_ratio(size, "expl")), _f(_ratio(size, "impl")),
            _f(_get(size, "occam", "expl|slots_mean|named", "OG")), _f(_get(size, "occam", "impl|slots_mean|named", "OG")),
            _f(_get(size, "contra_auroc")), _f(_get(size, "contra_mean_log_bf"))]


def robust_cells(block: dict | None, icc_path=("robustness_subset", "rewording", "icc1")) -> list[str]:
    return ([_f(_get(block, *icc_path)), _f(_get(block, "rewording", "shift_abs", "hzh", "median"))]
            + [_f(_get(block, "utterance", group, "abs_delta", "median")) for group in GROUPS]
            + [_f(_get(block, "posterior", "argmax_flip_rate", "mean"))])


def summary_markdown(metrics: dict, manifest: dict) -> str:
    variants = metrics["variants"]
    checks = metrics["checks"]
    lines = ["# Typicality v0 from saved samples", "",
             "Step 3 of `docs/claims_plan.md`, computed from step 2's Qwen samples and scores with no new GPU work. "
             "Scores are in nats. Brackets in `metrics.json` are 95% bootstrap intervals over items "
             f"({manifest['bootstrap']['replicates']} replicates, seed {manifest['bootstrap']['seed']}).", "",
             f"Pool: {checks['pool_texts']} distinct hypothesis texts of the clean probe, all with 64 Phase 8 samples; "
             f"{checks['rewritten_units_with_samples']} rewritten texts of the 60-item subset with 64 samples each; "
             f"{checks['cells']} scored cells.", "",
             "## 1. Where human utterances fall", "",
             "Share of the 2,355 original (utterance, hypothesis) pairs in each bin, lowest T first (uniform weighting). "
             "Qwen's own samples under a hypothesis have S of a few nats; the human utterances have S near 100, so "
             "they sit in the top bin of every variant built on S. A variant whose pairs share one bin scores them "
             "alike, so its bin agreement and its stability say nothing about meaning.", "",
             "| Variant | Bins (low → high) | Pool H, σ |", "|---|---|---|"]
    for name in ORDER:
        block = variants[name]["uniform"]
        pool = block["pool"]
        lines.append(f"| {name} | " + " / ".join(f"{100 * share:.1f}%" for share in block["occupancy"])
                     + f" | {_f(pool['H'])}, {_f(pool['sigma'])} |")
    lines += ["", "## 2. Variants against the metrics (uniform weighting)", "",
              "Size principle: L_J/L_E and the Occam gap OG = L_J − L_E on named single-slot pairs (mean over slots); "
              "information: contradiction AUROC (G|impl vs ctrl:contra) and mean log Bayes factor G vs contra; "
              "robustness: hypothesis-rewording ICC(1) on the 60-item subset, median |shift| under a Chinese "
              "hypothesis, median |Δ log BF| under utterance rewrites, posterior argmax flip rate; bin agreement under "
              "utterance rewrites; pool sensitivity: bin-flip rate when the pool is halved (20 draws).", "",
              "| Variant | L_J/L_E expl | L_J/L_E impl | OG expl | OG impl | contra AUROC | contra log BF | ICC (subset) "
              "| hzh shift | Δ pad | Δ typo | Δ para | Δ zh | flips | agree pad | agree typo | agree para | agree zh "
              "| pool flips |",
              "|" + "---|" * 19]
    for name in ORDER:
        block = variants[name]["uniform"]
        lines.append(f"| {name} | " + " | ".join(
            size_cells(block["size"]) + robust_cells(block)
            + [_f(_get(block, "bin_agreement", group, "mean")) for group in GROUPS]
            + [_f(_get(block, "pool_sensitivity", "bin_flip_rate_mean"))]) + " |")
    lines += ["", "## 3. Baselines on the same metrics", "",
              "Robustness columns are claim 1's (`docs/claim1_results/metrics.json`); the ICC is over all items "
              "the scorer covers. Verbalized scorers cover the 60-item subset, so their size-principle rows use it.", "",
              "| Scorer | L_J/L_E expl | L_J/L_E impl | OG expl | OG impl | contra AUROC | contra log BF | ICC "
              "| hzh shift | Δ pad | Δ typo | Δ para | Δ zh | flips |", "|" + "---|" * 14]
    for name, label in BASELINES.items():
        block = metrics["baselines"].get(name)
        if block:
            lines.append(f"| {label} | " + " | ".join(size_cells(block.get("size"))
                                                       + robust_cells(block, ("rewording", "icc1"))) + " |")
    lines += ["", "## 4. Prior weighting", "",
              "Each pool text weighted by how many clean items carry it in their probe, against uniform.", "",
              "| Variant | Weighting | Pool H, σ | L_J/L_E expl | L_J/L_E impl | contra AUROC | pool flips |",
              "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        for scheme, block in variants[name].items():
            lines.append(f"| {name} | {scheme} | {_f(block['pool']['H'])}, {_f(block['pool']['sigma'])} | "
                         f"{_f(_ratio(block['size'], 'expl'))} | {_f(_ratio(block['size'], 'impl'))} | "
                         f"{_f(block['size']['contra_auroc'])} | "
                         f"{_f(_get(block, 'pool_sensitivity', 'bin_flip_rate_mean'))} |")
    lines += ["", "## 5. Sensitivity: bin edges and factorized P_sem", "",
              "| Variant | Bins (low → high) | L_J/L_E expl | L_J/L_E impl | contra AUROC | ICC (subset) | agree para "
              "| agree zh |", "|---|---|---|---|---|---|---|---|"]
    for name, block in metrics["sensitivity"].items():
        lines.append(f"| {name} | " + " / ".join(f"{100 * share:.0f}%" for share in block["occupancy"]) + " | "
                     + " | ".join([_f(_ratio(block["size"], "expl")), _f(_ratio(block["size"], "impl")),
                                   _f(block["size"]["contra_auroc"]),
                                   _f(_get(block, "robustness_subset", "rewording", "icc1")),
                                   _f(_get(block, "bin_agreement", "para", "mean")),
                                   _f(_get(block, "bin_agreement", "zh", "mean"))]) + " |")
    lines += ["", "## 6. Dice check", "",
              "Step 1's sequences, exact moments of each die (pool: the three dice, uniform), and option A's exact bin "
              "mass. On all-even sequences the truth favours `even` over `fair` by k ln 2. Impossible: a sequence with "
              "an odd face under `even`.", "",
              "| Variant | Ties on all-even sequences | log BF at k = 2 (truth 1.39) | k = 6 (4.16) | k = 10 (6.93) "
              "| Impossible ruled out |", "|---|---|---|---|---|---|"]
    for name, block in metrics["dice"].items():
        by_k = block["by_k"]
        lines.append(f"| {name} | {_f(block['tie_rate'])} | "
                     + " | ".join(_f(_get(by_k, str(k), "median_predicted")) for k in (2, 6, 10))
                     + f" | {_f(block['impossible_ruled_out'])} |")
    lines += ["", "## 7. Prediction scorecard", "", "Failing a prediction is a result.", "",
              "| ID | Prediction | Value | Result |", "|---|---|---|---|"]
    for row in metrics["predictions"]:
        result = "pass" if row["pass"] else "fail"
        note = f" ({row['note']})" if row.get("note") else ""
        lines.append(f"| {row['id']} | {row['statement']} | {_value(row['value'])} | {result}{note} |")
    lines += ["", "The recommendation is in `docs/typicality_v0_decision.md`.", ""]
    return "\n".join(lines)


def _value(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key} {_value(found)}" for key, found in value.items())
    return _f(value)
