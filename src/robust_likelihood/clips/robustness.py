"""Phase 4 of docs/clips_small_lm_plan.md: scores of the perturbed instructions and the posteriors they give.

The posterior under a swapped instruction is computed here, not by another Julia replay. Pragmatic
inference keeps one particle per (goal, cost profile) with no resampling, and an instruction enters a
particle's trace score only through its TableMixture term, logsumexp_c(score(prompt_c, u) + log w_c)
(temperature 1 for both pragmatic roles). So from a replay's final per-particle trace scores
(inferences_per_timestep_*.csv), the particle log of the record pass and the scores:

    trace'(i) = trace(i) - mix_table(i, u) + mix(i, u')

and P(g) is the softmax over particles summed by goal; the language-only posterior uses the utterance
trace scores the same way. On the unperturbed instruction this reproduces the replay's logged
posteriors to 1e-14 (docs/clips_decisions.md).

Scores come from score_batches: one batch per (instruction, prompt), holding the original output and
every valid perturbed one, so each delta is read against the original's score from the same batch
(batch layout alone moves a score by up to about 0.5 nats). Both sides of every delta use these
co-batched scores; mix_table is only what is subtracted from the replay's trace.

Metrics per model, domain and kind (plan Phase 4): median |delta P(g_true)| at the final step (CLIPS),
argmax flip rate (CLIPS), median |delta log BF| between the true goal and its strongest rival under the
original instruction, from the language term alone, and for mDKG the change in goal correlation with
humans over the problems that have a valid unit of the kind. Intervals: bootstrap over instructions,
2,000 replicates, seed 20261126.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.clips.evaluate import pearson, read_humans, run_rows
from robust_likelihood.clips.perturb import KINDS
from robust_likelihood.clips.record import prompt_id
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl


def lse(values: list[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(value - top) for value in values))


def score_batches(root: Path, key: str, *, batches_file: Path, prompts_file: Path, out: Path, device: str | None = None,
                  tokenizer=None, model=None) -> dict:
    """GPU: every batch row of batches.jsonl scored as one batch."""
    import torch
    import transformers

    from robust_likelihood.clips.score import encode, load_clips_config
    from robust_likelihood.lmcore import load_model, pick_device, surprisals
    from robust_likelihood.multiwoz.jev_run import git_state

    config = load_clips_config(root)
    spec, settings = config["models"][key], config["score"]
    clock = time.monotonic()
    started = datetime.now(timezone.utc)
    prompts = {row["prompt_id"]: row["prompt"] for row in read_jsonl(prompts_file)}
    rows = read_jsonl(batches_file)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if model is None:
        model = load_model(spec["id"], spec["revision"], settings["dtype"], pick_device(device))
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad = pad if pad is not None else 0
    found = []
    over = 0
    for row in rows:
        encoded = [encode(tokenizer, prompts[row["prompt_id"]], output, int(settings["max_tokens"]))
                   for output in row["outputs"]]
        usable = [index for index, item in enumerate(encoded) if item is not None]
        over += len(encoded) - len(usable)
        values = surprisals(model, [encoded[index] for index in usable], len(usable), pad) if usable else []
        scores: list[float | None] = [None] * len(encoded)
        for index, value in zip(usable, values):
            scores[index] = -value
        found.append({"utterance": row["utterance"], "prompt_id": row["prompt_id"], "outputs": row["outputs"],
                      "scores": scores})
    write_jsonl(out / f"{key}.jsonl", found)
    manifest = {"kind": "clips-score-perturbed", "model_key": key, "model": spec["id"], "revision": spec["revision"],
                "dtype": settings["dtype"], "device": str(next(model.parameters()).device), "git": git_state(root),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                "batches_file": batches_file.as_posix(), "batches": len(rows),
                "sequences": sum(len(row["outputs"]) for row in rows), "over_max_tokens": over,
                "layout": "one batch per (instruction, prompt): the original output and every perturbed one",
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_seconds": round(time.monotonic() - clock, 1)}
    write_json(out / f"{key}.manifest.json", manifest)
    return {"model": key, "batches": len(rows), "sequences": manifest["sequences"],
            "elapsed_seconds": manifest["elapsed_seconds"]}


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def final_rows(inference: list[dict]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for row in inference:
        if row["plan_id"] not in found or int(row["timestep"]) > int(found[row["plan_id"]]["timestep"]):
            found[row["plan_id"]] = row
    return found


MDKG_GOALS = ("gem1", "gem2", "gem3", "gem4")  # GOALS in doors-keys-gems/experiments.jl, in order
VH_GOAL_SETS = Path("external/CLIPS.jl/virtual-home/dataset/goals/sets")


def goal_labels(root: Path, domain: str, problem_id: str) -> list[str]:
    """The goals in the order of goal_probs_1, goal_probs_2, ...: GOALS for mDKG, the problem's goal set for VH."""
    if domain == "mdkg":
        return list(MDKG_GOALS)
    return (root / VH_GOAL_SETS / f"{problem_id}.txt").read_text(encoding="utf-8").splitlines()


def true_goal_index(root: Path, domain: str, experiments: list[dict], plan: str) -> int:
    """The k of the plan's true goal (the loop logs its label, not k), checked against true_goal_probs.

    The label's position in goal_labels gives k; goal_probs_k must then equal true_goal_probs in every
    pragmatic qmdp_act row. Matching on the probabilities alone is ambiguous when two goals tie.
    """
    rows = [row for row in experiments if row["plan_id"] == plan and row["infer_method"] == "pragmatic"
            and row["assist_method"] == "qmdp_act"]
    if not rows or len({row["true_goal"] for row in rows}) != 1:
        raise ValueError(f"cannot identify the true goal of {plan}: {sorted({row['true_goal'] for row in rows})}")
    labels = goal_labels(root, domain, rows[0]["problem_id"])
    if rows[0]["true_goal"] not in labels:
        raise ValueError(f"the true goal {rows[0]['true_goal']} of {plan} is not among {labels}")
    k = labels.index(rows[0]["true_goal"]) + 1
    if any(row[f"goal_probs_{k}"] != row["true_goal_probs"] for row in rows):
        raise ValueError(f"goal_probs_{k} of {plan} differs from true_goal_probs")
    return k


def goal_marginal(log_weights: dict[int, float], goals: dict[int, int]) -> dict[int, float]:
    total = lse(list(log_weights.values()))
    found: dict[int, float] = defaultdict(float)
    for particle, value in log_weights.items():
        found[goals[particle]] += math.exp(value - total)
    return dict(found)


def goal_log_scores(log_weights: dict[int, float], goals: dict[int, int]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for particle, value in log_weights.items():
        grouped[goals[particle]].append(value)
    return {goal: lse(values) for goal, values in grouped.items()}


class Problem:
    """One plan's particles, final trace scores and the language terms of its instructions."""

    def __init__(self, plan: str, final: dict, particles: list[dict], true_goal: int):
        self.plan, self.true_goal = plan, true_goal
        self.goals = {row["particle"]: row["goal"] for row in particles}
        self.trace = {i: float(final[f"trace_scores_{i}"]) for i in self.goals}
        self.utt = {i: float(final[f"utt_trace_scores_{i}"]) for i in self.goals}
        self.terms = {(row["particle"], row["t"]): row for row in particles}

    def mixture(self, particle: int, t: int, score_of) -> float:
        row = self.terms[(particle, t)]
        values = [score_of(prompt_id(prompt)) for prompt in row["prompts"]]
        if any(value is None for value in values):
            return -math.inf
        return lse([value + math.log(weight) for value, weight in zip(values, row["probs"])])

    def swapped(self, t: int, logged, new) -> tuple[dict[int, float], dict[int, float]]:
        """(trace, utterance trace) per particle with instruction t's term replaced."""
        trace, utt = {}, {}
        for i in self.goals:
            delta = self.mixture(i, t, new) - self.mixture(i, t, logged)
            trace[i], utt[i] = self.trace[i] + delta, self.utt[i] + delta
        return trace, utt


def unit_effect(problem: Problem, t: int, logged, original, perturbed) -> dict:
    trace0, utt0 = problem.swapped(t, logged, original)
    trace1, utt1 = problem.swapped(t, logged, perturbed)
    post0, post1 = goal_marginal(trace0, problem.goals), goal_marginal(trace1, problem.goals)
    lang0, lang1 = goal_log_scores(utt0, problem.goals), goal_log_scores(utt1, problem.goals)
    true = problem.true_goal
    rival = max((goal for goal in lang0 if goal != true), key=lambda goal: lang0[goal])
    return {"p_true": post0.get(true, 0.0), "p_true_new": post1.get(true, 0.0),
            "delta_p_true": post1.get(true, 0.0) - post0.get(true, 0.0),
            "flip": max(post0, key=post0.get) != max(post1, key=post1.get),
            "delta_log_bf": (lang1[true] - lang1[rival]) - (lang0[true] - lang0[rival]),
            "goal_probs": post0, "goal_probs_new": post1}


def median_block(values: list[float], ids: list[str], replicates: int, seed: int) -> dict:
    if not values:
        return {"median": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    pairs = list(zip(ids, values))
    medians = sorted(statistics.median(rng.choice(pairs)[1] for _ in pairs) for _ in range(replicates))
    return {"median": statistics.median(values), "ci95": [medians[int(0.025 * (replicates - 1))],
                                                           medians[int(0.975 * (replicates - 1))]], "n": len(values)}


def rate_block(values: list[bool], replicates: int, seed: int) -> dict:
    if not values:
        return {"rate": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    rates = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(replicates))
    return {"rate": sum(values) / len(values), "ci95": [rates[int(0.025 * (replicates - 1))],
                                                         rates[int(0.975 * (replicates - 1))]], "n": len(values)}


def model_effects(root: Path, model: str, units: list[dict], cobatched: list[dict], utterances: list[dict],
                  particles: dict[str, list[dict]], table: dict[tuple[str, str], float]) -> list[dict]:
    """One row per valid unit with a replay: its effect on the posterior."""
    scores: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in cobatched:
        scores[(row["utterance"], row["prompt_id"])] = dict(zip(row["outputs"], row["scores"]))
    by_key = {f"{row['domain']}:{row['plan_id']}:{row['index']}": row for row in utterances}
    problems: dict[tuple[str, str], Problem] = {}
    for domain in ("mdkg", "vh"):
        inference, _ = run_rows(root, model, domain, "inferences_per_timestep")
        rows, _ = run_rows(root, model, domain, "experiments")
        if not inference or not rows:
            continue
        finals = final_rows(inference)
        done = {row["plan_id"] for row in rows if row["infer_method"] == "pragmatic" and row["assist_method"] == "qmdp_act"}
        mine: dict[str, list[dict]] = defaultdict(list)
        for row in particles[domain]:
            mine[row["plan_id"]].append(row)
        for plan, final in finals.items():
            if plan in done:  # a plan still being replayed, or cut short, has inference rows but no pragmatic rows
                problems[(domain, plan)] = Problem(plan, final, mine[plan],
                                                   true_goal_index(root, domain, rows, plan))
    found = []
    for unit in units:
        if not unit["valid"] or (unit["domain"], unit["plan_id"]) not in problems:
            continue
        problem = problems[(unit["domain"], unit["plan_id"])]
        utterance = by_key[unit["utterance"]]
        original, perturbed = " " + utterance["text"].strip(), " " + unit["rewrite"].strip()
        key = unit["utterance"]

        def logged(found_id, output=original):
            return table.get((found_id, output))

        def cobatched_score(output):
            return lambda found_id: scores.get((key, found_id), {}).get(output)

        effect = unit_effect(problem, utterance["timestep"], logged, cobatched_score(original),
                             cobatched_score(perturbed))
        found.append({"unit_id": unit["unit_id"], "utterance": key, "domain": unit["domain"],
                      "plan_id": unit["plan_id"], "kind": unit["kind"], **effect})
    return found


def summarize(effects: list[dict], humans: dict[str, dict] | None, replicates: int, seed: int) -> dict:
    found: dict = {}
    for scope in ("mdkg", "vh", "both"):
        mine = [row for row in effects if scope == "both" or row["domain"] == scope]
        block = {}
        for kind in KINDS:
            rows = [row for row in mine if row["kind"] == kind]
            if not rows:
                continue
            ids = [row["utterance"] for row in rows]
            block[kind] = {
                "units": len(rows),
                "abs_delta_p_true": median_block([abs(row["delta_p_true"]) for row in rows], ids, replicates, seed),
                "argmax_flip": rate_block([row["flip"] for row in rows], replicates, seed),
                "abs_delta_log_bf_language": median_block([abs(row["delta_log_bf"]) for row in rows], ids,
                                                          replicates, seed)}
            if scope == "mdkg" and humans:
                plans = [row for row in rows if row["plan_id"] in humans]
                if len(plans) >= 3:
                    def corr(column):
                        x = [row[column].get(goal, 0.0) for row in plans for goal in range(1, 5)]
                        y = [float(humans[row["plan_id"]][f"goal_probs_{goal}"]) for row in plans for goal in range(1, 5)]
                        return pearson(x, y)
                    before, after = corr("goal_probs"), corr("goal_probs_new")
                    block[kind]["goal_correlation"] = {"original": before, "perturbed": after,
                                                       "change": None if before is None or after is None
                                                       else after - before, "problems": len(plans)}
        found[scope] = block
    return found


def run_robustness(root: Path, models: list[str], out: Path) -> dict:
    clock = time.monotonic()
    config = read_json(root / "configs" / "clips.json")
    replicates, seed = config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    derived = root / "data" / "clips" / "derived"
    units = read_jsonl(derived / "perturbations.jsonl")
    utterances = read_jsonl(derived / "utterances.jsonl")
    particles = {domain: read_jsonl(root / "results" / "clips" / "record" / f"{domain}.particles.jsonl")
                 for domain in ("mdkg", "vh")}
    humans_path = root / "data" / "clips" / "raw" / "results" / "doors-keys-gems" / "mean_human_responses.csv"
    humans = read_humans(humans_path) if humans_path.exists() else None
    result: dict = {"models": {}}
    for model in models:
        perturbed = derived / "scores_perturbed" / f"{model}.jsonl"
        table_path = derived / "scores" / f"{model}.jsonl"
        if not perturbed.exists() or not table_path.exists():
            continue
        table = {(row["prompt_id"], row["output"]): row["score"] for row in read_jsonl(table_path)}
        effects = model_effects(root, model, units, read_jsonl(perturbed), utterances, particles, table)
        write_jsonl(out / "effects" / f"{model}.jsonl",
                    [{key: value for key, value in row.items() if key not in ("goal_probs", "goal_probs_new")}
                     for row in effects])
        result["models"][model] = summarize(effects, humans, replicates, seed)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_phase4.json", result)
    write_json(out / "manifest_phase4.json", {
        "kind": "clips-robustness", "models": list(result["models"]), "bootstrap": {"replicates": replicates,
                                                                                    "seed": seed},
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1)})
    return result
