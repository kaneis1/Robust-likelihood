"""Phase 3 of docs/clips_small_lm_plan.md: the paper's Table 1 from CLIPS's own result files.

Each definition below reproduces the published Table 1 exactly from the paper's own result files on OSF
(`data/clips/raw/results/`; tests/test_clips.py checks it when they are present):

- Methods, read from the experiments CSV the loops write:
  CLIPS = pragmatic inference with modalities "action, utterance"; language-only IP = "utterance";
  action-only IP = "action". All three use offline QMDP assistance (assist_method qmdp_act, estim_type
  mode), the paper's Algorithm 2 with the posterior fixed. The literal listeners use estim_type mean:
  literal naive and literal efficient in mDKG, literal in VirtualHome.
- P(g_true): true_goal_probs at the final step, one row per problem, mean over problems.
- Assistance precision and recall: assist_precision and assist_recall, mean over problems.
- Relative plan length and relative human cost: per problem, the method's plan_length (speaker_cost)
  divided by the reference CLIPS's, mean over problems. The paper's reference is its own CLIPS; ours is
  CLIPS with qwen3_8b.
- Goal correlation (mDKG): Pearson r between goal_probs_1..4 and the mean human goal ratings, over the
  30 x 4 (problem, goal) cells.
- Assistance correlation (mDKG): Pearson r between assist_probs_k and the mean human assistance ratings,
  over each problem's real options (the first option_count).

Brackets in the paper are standard errors over problems (sd / sqrt(n)); they are reported as `se`.
Intervals (`ci95`) are a bootstrap over problems, 2,000 replicates, seed 20261126 (the paper bootstraps
participants for its correlation intervals).
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.storage import read_json, write_json

DOMAIN_DIRS = {"mdkg": "doors-keys-gems", "vh": "virtual-home"}
PROBLEMS = {"mdkg": 30, "vh": 25}
IP = {"clips": "action, utterance", "language_only": "utterance", "action_only": "action"}
LITERAL = {"mdkg": {"literal_naive": "naive", "literal_efficient": "efficient"}, "vh": {"literal": "literal"}}
LABELS = {"clips": "CLIPS", "language_only": "Language-only IP", "action_only": "Action-only IP",
          "literal_naive": "Literal naive", "literal_efficient": "Literal efficient", "literal": "Literal listener"}
PUBLISHED_KEYS = {"clips": "clips_p_true", "language_only": "language_only_p_true", "action_only": "action_only_p_true"}
ACTION_ONLY_TOLERANCE = 0.02


def read_rows(paths: list[Path]) -> list[dict]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows += list(csv.DictReader(handle))
    return rows


def fnum(value) -> float | None:
    if value in (None, "", "missing", "NaN", "nan"):
        return None
    found = float(value)
    return None if math.isnan(found) else found


def method_rows(rows: list[dict], domain: str) -> dict[str, dict[str, dict]]:
    """{method: {plan_id: row}}; refuses two rows for one plan."""
    def pick(test) -> dict[str, dict]:
        chosen: dict[str, dict] = {}
        for row in rows:
            if test(row):
                if row["plan_id"] in chosen:
                    raise ValueError(f"two rows for plan {row['plan_id']}")
                chosen[row["plan_id"]] = row
        return chosen

    found = {name: pick(lambda row, modality=modality: row["infer_method"] == "pragmatic"
                        and row.get("modalities") == modality and row["assist_method"] == "qmdp_act"
                        and row["estim_type"] == "mode")
             for name, modality in IP.items()}
    for name, method in LITERAL[domain].items():
        found[name] = pick(lambda row, method=method: row["infer_method"] == "literal"
                           and row["assist_method"] == method and row["estim_type"] == "mean")
    return {name: chosen for name, chosen in found.items() if chosen}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def read_humans(path: Path) -> dict[str, dict]:
    return {row["plan_id"]: row for row in read_rows([path])}


def goal_cells(rows: dict[str, dict], humans: dict[str, dict], plans: list[str]) -> tuple[list[float], list[float]]:
    x, y = [], []
    for plan in plans:
        for goal in range(1, 5):
            x.append(fnum(rows[plan][f"goal_probs_{goal}"]))
            y.append(fnum(humans[plan][f"goal_probs_{goal}"]))
    return x, y


def assist_cells(rows: dict[str, dict], humans: dict[str, dict], plans: list[str]) -> tuple[list[float], list[float]]:
    x, y = [], []
    for plan in plans:
        for option in range(1, int(float(rows[plan]["option_count"])) + 1):
            x.append(fnum(rows[plan][f"assist_probs_{option}"]))
            y.append(fnum(humans[plan][f"assist_option_{option}"]))
    return x, y


class ProblemBoot:
    def __init__(self, plans: list[str], replicates: int, seed: int):
        rng = random.Random(seed)
        self.samples = [[rng.choice(plans) for _ in plans] for _ in range(replicates)]

    def interval(self, stat) -> list[float] | None:
        values = sorted(value for value in (stat(sample) for sample in self.samples) if value is not None)
        if not values:
            return None
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def mean_block(values: dict[str, float | None], plans: list[str], boot: ProblemBoot) -> dict:
    found = [values[plan] for plan in plans if values.get(plan) is not None]
    if not found:
        return {"mean": None, "se": None, "ci95": None, "n": 0}

    def stat(sample):
        chosen = [values[plan] for plan in sample if values.get(plan) is not None]
        return statistics.fmean(chosen) if chosen else None

    return {"mean": statistics.fmean(found),
            "se": statistics.stdev(found) / math.sqrt(len(found)) if len(found) > 1 else None,
            "ci95": boot.interval(stat), "n": len(found)}


def method_metrics(rows: dict[str, dict], reference: dict[str, dict] | None, humans: dict[str, dict] | None,
                   boot: ProblemBoot) -> dict:
    plans = sorted(rows)
    block = {"problems": len(plans)}
    for key, column in (("p_true", "true_goal_probs"), ("assist_precision", "assist_precision"),
                        ("assist_recall", "assist_recall")):
        block[key] = mean_block({plan: fnum(rows[plan][column]) for plan in plans}, plans, boot)
    if reference:
        for key, column in (("rel_plan_length", "plan_length"), ("rel_human_cost", "speaker_cost")):
            ratios = {}
            for plan in plans:
                mine, theirs = fnum(rows[plan][column]), fnum(reference.get(plan, {}).get(column))
                ratios[plan] = mine / theirs if mine is not None and theirs else None
            block[key] = mean_block(ratios, plans, boot)
    if humans:
        shared = [plan for plan in plans if plan in humans]
        for key, cells in (("goal_correlation", goal_cells), ("assist_correlation", assist_cells)):
            if key == "goal_correlation" and not rows[shared[0]].get("goal_probs_1"):
                continue
            try:
                value = pearson(*cells(rows, humans, shared))
            except (TypeError, KeyError):
                continue
            block[key] = {"r": value, "ci95": boot.interval(lambda sample: pearson(*cells(rows, humans, sample)))}
    return block


def table1(rows: list[dict], domain: str, *, reference: list[dict] | None = None, humans: dict[str, dict] | None = None,
           replicates: int = 2000, seed: int = 20261126) -> dict:
    """Table 1's rows for one result file (or several of one run). reference: the rows whose CLIPS the
    relative metrics divide by (default: these rows' own CLIPS)."""
    methods = method_rows(rows, domain)
    reference_clips = method_rows(reference, domain)["clips"] if reference is not None else methods.get("clips")
    plans = sorted({plan for chosen in methods.values() for plan in chosen})
    boot = ProblemBoot(plans, replicates, seed)
    with_humans = humans if domain == "mdkg" and not (set(methods.get("clips", {})) - set(humans or {})) else None
    return {name: method_metrics(chosen, reference_clips, with_humans if name != "literal" else None, boot)
            for name, chosen in methods.items()}


def humans_row(humans: dict[str, dict], replicates: int = 2000, seed: int = 20261126) -> dict:
    plans = sorted(humans)
    boot = ProblemBoot(plans, replicates, seed)
    return {key: mean_block({plan: fnum(humans[plan][column]) for plan in plans}, plans, boot)
            for key, column in (("p_true", "true_goal_probs"), ("assist_precision", "assist_precision"),
                                ("assist_recall", "assist_recall"))}


def run_files(folder: Path, prefix: str) -> list[Path]:
    """A replay's result files of one kind, in its folder and its plan-chunk subfolders, oldest first (the
    file names carry the start time)."""
    return sorted(folder.rglob(f"{prefix}_*.csv"), key=lambda path: path.name)


def merged_rows(paths: list[Path]) -> list[dict]:
    """Rows of several result files of one model and domain; a plan in more than one file keeps the rows of
    the newest file only."""
    by_plan: dict[str, list[dict]] = {}
    for path in paths:
        mine: dict[str, list[dict]] = {}
        for row in read_rows([path]):
            mine.setdefault(row["plan_id"], []).append(row)
        by_plan.update(mine)
    return [row for plan in sorted(by_plan) for row in by_plan[plan]]


def run_rows(root: Path, model: str, domain: str, prefix: str = "experiments") -> tuple[list[dict], list[Path]]:
    paths = run_files(root / "results" / "clips" / model / domain, prefix)
    return merged_rows(paths), paths


def action_only_check(ours: list[dict], paper: list[dict], domain: str, published: float) -> dict:
    """Action-only IP uses no language model: its P(g_true) must match the paper's (the plan's setup check)."""
    mine, theirs = method_rows(ours, domain)["action_only"], method_rows(paper, domain)["action_only"]
    shared = sorted(set(mine) & set(theirs))
    gaps = [abs(fnum(mine[plan]["true_goal_probs"]) - fnum(theirs[plan]["true_goal_probs"])) for plan in shared]
    mean = statistics.fmean(fnum(mine[plan]["true_goal_probs"]) for plan in mine)
    return {"p_true": mean, "published": published, "difference": mean - published,
            "max_abs_difference_per_problem_vs_paper_file": max(gaps) if gaps else None,
            "problems": len(mine), "passes": len(mine) == PROBLEMS[domain] and abs(mean - published) <= ACTION_ONLY_TOLERANCE}


def run_evaluate(root: Path, models: list[str], out: Path, reference_model: str = "qwen3_8b") -> dict:
    clock = time.monotonic()
    config = read_json(root / "configs" / "clips.json")
    replicates, seed = config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    raw = root / "data" / "clips" / "raw" / "results"
    humans_path = raw / "doors-keys-gems" / "mean_human_responses.csv"
    humans = read_humans(humans_path) if humans_path.exists() else None
    result: dict = {"published": config["published"], "paper_files": {}, "models": {}, "action_only_check": {},
                    "reference_model": reference_model, "runs": {}}
    if humans:
        result["humans"] = humans_row(humans, replicates, seed)
    paper_rows = {}
    for domain, folder in DOMAIN_DIRS.items():
        paths = [raw / folder / f"{kind}_experiments.csv" for kind in ("literal", "pragmatic")]
        if all(path.exists() for path in paths):
            paper_rows[domain] = read_rows(paths)
            result["paper_files"][domain] = table1(paper_rows[domain], domain, humans=humans, replicates=replicates,
                                                   seed=seed)
    for domain in DOMAIN_DIRS:
        reference, _ = run_rows(root, reference_model, domain)
        reference = reference or None
        for model in models:
            rows, paths = run_rows(root, model, domain)
            if not rows:
                continue
            result["runs"].setdefault(model, {})[domain] = [path.relative_to(root).as_posix() for path in paths]
            result["models"].setdefault(model, {})[domain] = table1(rows, domain, reference=reference, humans=humans,
                                                                    replicates=replicates, seed=seed)
            if domain in paper_rows:
                result["action_only_check"].setdefault(model, {})[domain] = action_only_check(
                    rows, paper_rows[domain], domain, config["published"][domain]["action_only_p_true"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_phase3.json", result)
    write_json(out / "manifest_phase3.json", {
        "kind": "clips-evaluate", "models": models, "runs": result["runs"], "reference_model": reference_model,
        "bootstrap": {"replicates": replicates, "seed": seed},
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1)})
    return result
