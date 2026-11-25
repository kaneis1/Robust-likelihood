"""Week of 24 Nov, Part D (docs/week_nov24_plan.md 5.5-5.7): the students against their teacher and baselines.

Scores of a student [fixed before any result]:
- R-lin, R-mlp: "cont" -(H + sigma * T_hat) is the primary score; "bin" -(H + sigma * c_b) of T_hat's bin.
- C-mlp: "expected" sum_b p_b * -(H + sigma * c_b) is the primary score; "argmax" the argmax bin's score.
A student's bin is T_hat's bin (regression) or the argmax bin (classification); its T for Spearman is
T_hat, or sum_b p_b c_b for the classifier.

Baselines: the teacher (the chosen scheme from 64 samples per text with the 8B model, wk_analyze's
numbers for its robustness), raw log-likelihood of qwen3_base·F (Phase 2), raw log-likelihood of
Qwen3-0.6B-Base with the F prompt and no training (the qwen3_06_base run), and Jev where its pairs match.
Every inference metric is the simulator study's (typv0.size_and_information, claim1.utterance,
claim1.posterior). Held-out-food items are those whose gold goal names a held-out food.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, load_config
from robust_likelihood.multiwoz.claim1 import (
    H_KINDS,
    ORIGINAL,
    U_KINDS,
    ItemBoot,
    Scorer,
    item_meta,
    jev_scorers,
    posterior,
    posterior_classes,
    spearman,
    token_log,
    utterance,
)
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.sim_analyze import discover
from robust_likelihood.multiwoz.typv0 import bin_index, corpus_bootstrap, score_map, size_and_information
from robust_likelihood.multiwoz.wk_analyze import KINDS as PART_A_KINDS
from robust_likelihood.multiwoz.wk_analyze import REPLICATES, SEED, discover_week
from robust_likelihood.storage import read_json, read_jsonl, write_json

PRIMARY = {"R-lin": "cont", "R-mlp": "cont", "C-mlp": "expected"}
FORMS = {"R-lin": ("cont", "bin"), "R-mlp": ("cont", "bin"), "C-mlp": ("expected", "argmax")}


def key_of(row: dict) -> tuple:
    if row["set"] == "human":
        return (row["item_id"], row["hyp_key"], ORIGINAL, ORIGINAL)
    if row["set"] in ("urewrite", "uperturb"):
        return (row["item_id"], row["hyp_key"], ORIGINAL, row["kind"])
    if row["set"] == "hrewrite":
        return (row["item_id"], row["hyp_key"], row["kind"], ORIGINAL)
    return None


def student_values(row_output, scheme: dict) -> dict:
    """T, bin, and every score form of one prediction."""
    level = lambda c: -(scheme["H"] + scheme["sigma"] * c)
    if isinstance(row_output, list):
        t = sum(p * c for p, c in zip(row_output, scheme["centres"]))
        top = max(range(len(row_output)), key=row_output.__getitem__)
        return {"T": t, "bin": top, "expected": sum(p * level(c) for p, c in zip(row_output, scheme["centres"])),
                "argmax": level(scheme["centres"][top])}
    b = bin_index(row_output, tuple(scheme["edges"]), 1)
    return {"T": row_output, "bin": b, "cont": level(row_output), "bin_score": level(scheme["centres"][b])}


def form_key(form: str) -> str:
    return "bin_score" if form == "bin" else form


def inference(log: dict, items: list[dict], metas: dict, changed: dict, corpus_boot, boot: ItemBoot,
              item_ids: list[str]) -> dict:
    """Size principle kept, contradiction log BF, mean log P(G) and top-1 accuracy, on the original pairs."""
    chosen = set(item_ids)
    mine = [item for item in items if item["item_id"] in chosen]
    scores = {item: found for item, found in score_map(log).items() if item in chosen}
    size = size_and_information(mine, scores, changed, corpus_boot)
    kept = {}
    for direction in ("named", "omitted"):
        group = size["occam"].get(f"impl|slots_mean|{direction}")
        kept[direction] = None if not group or not group.get("L_E") else group["mean_L_J"] / group["L_E"]
    log_p, top1, ids = [], [], []
    for item in sorted(chosen):
        classes = posterior_classes(metas[item])
        values = [log.get((item, key, ORIGINAL, ORIGINAL)) for key in classes]
        if len(classes) < 2 or None in values:
            continue
        top = max(values)
        weights = [math.exp(value - top) for value in values]
        p_goal = weights[0] / sum(weights)
        ids.append(item)
        log_p.append(math.log(max(p_goal, 1e-300)))
        top1.append(float(max(range(len(values)), key=values.__getitem__) == 0))
    return {"size_kept": kept, "contra_log_bf": size["contra_mean_log_bf"], "contra_ci95": size["contra_mean_log_bf_ci95"],
            "mean_log_p_goal": boot.mean(ids, log_p), "top1": boot.mean(ids, top1), "items": len(ids)}


def robustness(log: dict, bins: dict | None, metas: dict, boot: ItemBoot, item_ids: list[str], seed: int) -> dict:
    """Bin agreement, |delta score| and argmax flips under every utterance kind, and hypothesis rewording."""
    scorer = Scorer("x", "loglik", log, items=item_ids)
    found = {"utterance": utterance(scorer, metas, boot), "posterior_hrewrite": posterior(scorer, metas, boot, seed),
             "kinds": {}}
    for kind in U_KINDS + PART_A_KINDS + H_KINDS:
        ids, same, delta, flips, flip_ids = [], [], [], [], []
        for item in item_ids:
            meta = metas[item]
            for key in meta["keys"]:
                before = (item, key, ORIGINAL, ORIGINAL)
                after = (item, key, kind, ORIGINAL) if kind in H_KINDS else (item, key, ORIGINAL, kind)
                if before in log and after in log:
                    ids.append(item)
                    delta.append(abs(log[after] - log[before]))
                    if bins is not None and before in bins and after in bins:
                        same.append(float(bins[before] == bins[after]))
            if kind not in H_KINDS:
                classes = posterior_classes(meta)
                before = [log.get((item, key, ORIGINAL, ORIGINAL)) for key in classes]
                after = [log.get((item, key, ORIGINAL, kind)) for key in classes]
                if len(classes) >= 2 and None not in before and None not in after:
                    flip_ids.append(item)
                    flips.append(float(max(range(len(classes)), key=before.__getitem__)
                                       != max(range(len(classes)), key=after.__getitem__)))
        if ids:
            found["kinds"][kind] = {"abs_delta_score": boot.median(ids, delta),
                                    "bin_agreement": boot.mean(ids, same) if same else None,
                                    "argmax_flip": boot.mean(flip_ids, flips) if flips else None}
    return found


def part_a_flips(log: dict, metas: dict, boot: ItemBoot, item_ids: list[str], kind: str) -> dict:
    """Posterior argmax flips between a Part A unit and its co-batched original (keys orig:kind and kind)."""
    from robust_likelihood.multiwoz.wk_analyze import orig

    ids, flips = [], []
    for item in item_ids:
        classes = posterior_classes(metas[item])
        before = [log.get((item, key, ORIGINAL, orig(kind))) for key in classes]
        after = [log.get((item, key, ORIGINAL, kind)) for key in classes]
        if len(classes) >= 2 and None not in before and None not in after:
            ids.append(item)
            flips.append(float(max(range(len(classes)), key=before.__getitem__)
                               != max(range(len(classes)), key=after.__getitem__)))
    return boot.mean(ids, flips)


FIGURE_KINDS = ("para1", "typo1", "fill2", "hypothesis rewording")


def agreement_with_teacher(rows: list[dict], boot: ItemBoot) -> dict:
    if not rows:
        return {"pairs": 0}
    t_hat = [row["T_hat"] for row in rows]
    teacher = [row["T"] for row in rows]
    ids = [row["item_id"] for row in rows]
    return {"pairs": len(rows), "spearman": spearman(t_hat, teacher),
            "exact_bin": boot.mean(ids, [float(row["bin_hat"] == row["bin"]) for row in rows]),
            "within_one_bin": boot.mean(ids, [float(abs(row["bin_hat"] - row["bin"]) <= 1) for row in rows])}


def run_student_eval(root: Path, *, runs: list[Path], phase2: list[Path], jev: Path | None, out: Path,
                     derived: Path | None = None) -> dict:
    clock = time.monotonic()
    derived = derived or root / DERIVED_DIR
    config = load_config(root)
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    clean_ids = sorted(item["item_id"] for item in items)
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    metas = item_meta(items, probe)
    changed = {(row["item_id"], row["hyp_key"]): row["changed"]["slot"]
               for row in read_jsonl(derived / "probe.jsonl") if "changed" in row}
    corpus_boot = corpus_bootstrap(derived, config, items)
    boot = ItemBoot(clean_ids, REPLICATES, SEED)
    manifests = {path: read_json(path / "manifest.json") for path in runs}
    first = next(iter(manifests.values()))
    scheme, foods = first["scheme"], set(first["held_out_foods"])
    unseen = sorted(item["item_id"] for item in items if item["goal"].get("food") in foods)
    seen = sorted(set(clean_ids) - set(unseen))
    scopes = {"all": clean_ids, "seen_food": seen, "unseen_food": unseen}
    abc_path = out / "metrics_abc.json"
    abc = read_json(abc_path) if abc_path.exists() else {"models": {}}
    result: dict = {"scheme": scheme, "held_out_foods": sorted(foods), "unseen_food_items": unseen, "students": {},
                    "baselines": {}}

    for path, manifest in manifests.items():
        run = manifest["run"]
        rows = read_jsonl(path / "predictions.jsonl")
        values = [student_values(row["output"], scheme) for row in rows]
        block = {"run_dir": path.as_posix(), "train_pairs": manifest["train_pairs"], "best_step": manifest.get("best_step"),
                 "best_val_loss": manifest.get("best_val_loss"), "train_seconds": manifest.get("train_seconds"),
                 "pairs_per_second": manifest.get("pairs_per_second"), "curve": manifest.get("curve"), "forms": {}}
        human = [{"item_id": row["item_id"], "T_hat": value["T"], "bin_hat": value["bin"], "T": row["teacher"]["T"],
                  "bin": row["teacher"]["bin"]} for row, value in zip(rows, values)
                 if row["set"] == "human" and row.get("teacher")]
        block["agreement"] = {scope: agreement_with_teacher([row for row in human if row["item_id"] in set(ids)], boot)
                              for scope, ids in scopes.items()}
        unseen_set = set(unseen)
        block["human_pairs"] = [[row["T"], row["T_hat"], int(row["item_id"] in unseen_set)] for row in human]
        for split in ("synthetic_val", "synthetic_held_out_food"):
            synthetic = [{"item_id": clean_ids[0], "T_hat": value["T"], "bin_hat": value["bin"], "T": row["T"],
                          "bin": row["bin"]} for row, value in zip(rows, values) if row["set"] == split]
            block["agreement"][split] = {"pairs": len(synthetic), "spearman": spearman(
                [row["T_hat"] for row in synthetic], [row["T"] for row in synthetic]) if synthetic else None,
                "exact_bin": (sum(row["bin_hat"] == row["bin"] for row in synthetic) / len(synthetic)) if synthetic else None}
        bins = {}
        for row, value in zip(rows, values):
            key = key_of(row)
            if key is not None:
                bins[key] = value["bin"]
        for form in FORMS[run]:
            log = {}
            for row, value in zip(rows, values):
                key = key_of(row)
                if key is not None:
                    log[key] = value[form_key(form)]
            block["forms"][form] = {scope: inference(log, items, metas, changed, corpus_boot, boot, ids)
                                    for scope, ids in scopes.items()}
            block["forms"][form]["robustness"] = robustness(log, bins, metas, boot, clean_ids, SEED)
        result["students"][run] = block

    phase2_runs, week = discover(phase2), discover_week(phase2)
    teacher_run = phase2_runs.get("qwen3_base|F")
    if teacher_run is not None:
        scores = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(teacher_run / "lm_scores.jsonl")
                  if row["set"] == "clean"}
        lm = read_jsonl(teacher_run / "urewrite.jsonl"), read_jsonl(teacher_run / "hrewrite.jsonl")
        raw = token_log({pair: -value for pair, value in scores.items()}, {"urewrite": lm[0], "hrewrite": lm[1]},
                        set(clean_ids))
        result["baselines"]["raw_8b"] = {scope: inference(raw, items, metas, changed, corpus_boot, boot, ids)
                                         for scope, ids in scopes.items()}
        result["baselines"]["raw_8b"]["robustness"] = robustness(raw, None, metas, boot, clean_ids, SEED)
        # The teacher from 64 samples: its scores on the original pairs, and wk_analyze's robustness numbers.
        from robust_likelihood.multiwoz.wk_student_data import human_targets, teacher

        _, inputs = teacher(root, teacher_run, scheme["choice"], derived)
        targets = human_targets(inputs, scheme)
        level = {b: -(scheme["H"] + scheme["sigma"] * c) for b, c in enumerate(scheme["centres"])}
        teacher_log = {(item, key, ORIGINAL, ORIGINAL): level[found["bin"]] for (item, key), found in targets.items()}
        result["baselines"]["teacher"] = {scope: inference(teacher_log, items, metas, changed, corpus_boot, boot, ids)
                                          for scope, ids in scopes.items()}
        model = abc["models"].get("qwen3_base|F", {})
        result["baselines"]["teacher"]["robustness_from_wk_analyze"] = {
            "claims": model.get("part_c", {}).get(scheme["choice"], {}).get("agreement"),
            "part_a": model.get("typicality", {}).get(scheme["choice"]),
            "rewording": model.get("part_b", {}).get(scheme["choice"])}
    if teacher_run is not None:
        result["figure5"] = figure_five(root, derived, result, scheme, inputs, week, metas, boot, clean_ids)
    small = phase2_runs.get("qwen3_06_base|F") or week.get("qwen3_06_base|F", {}).get("human")
    if small is not None:
        scores = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(small / "lm_scores.jsonl")
                  if row["set"] == "clean"}
        parts = {part: read_jsonl(small / f"{part}.jsonl") for part in ("urewrite", "hrewrite")
                 if (small / f"{part}.jsonl").exists()}
        raw = token_log({pair: -value for pair, value in scores.items()}, parts, set(clean_ids))
        result["baselines"]["raw_06b"] = {scope: inference(raw, items, metas, changed, corpus_boot, boot, ids)
                                          for scope, ids in scopes.items()}
        result["baselines"]["raw_06b"]["robustness"] = robustness(raw, None, metas, boot, clean_ids, SEED)
    if jev is not None:
        epsilon = float(config["score_mapping"]["clip_epsilon"])
        for scorer in jev_scorers(read_jsonl(jev / "scores.jsonl"), [], set(clean_ids), epsilon):
            result["baselines"][scorer.name] = {"all": inference(scorer.log, items, metas, changed, corpus_boot, boot,
                                                                 clean_ids)}
    hsample = week.get("qwen3_base|F", {}).get("hsample")
    if hsample is not None:
        found = read_json(hsample / "manifest.json")
        seconds = found["stage_seconds"]["hsample"] / max(1, found["planned"]["hsample"])
        result["teacher_cost"] = {"seconds_per_new_hypothesis": seconds, "pairs_per_second_new_hypothesis": 1 / seconds,
                                  "source": hsample.as_posix()}
    result["predictions"] = predictions(result, abc)
    result["verdict"] = verdict(result)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_d.json", result)
    manifest = {"kind": "student-eval", "git": git_state(root), "runs": [path.as_posix() for path in runs],
                "phase2": [path.as_posix() for path in phase2], "jev": None if jev is None else jev.as_posix(),
                "bootstrap": {"replicates": REPLICATES, "seed": SEED},
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_seconds": round(time.monotonic() - clock, 1)}
    write_json(out / "manifest_d.json", manifest)
    return {"out": str(out), "predictions": result["predictions"], "verdict": result["verdict"]}


def figure_five(root: Path, derived: Path, result: dict, scheme: dict, inputs, week: dict, metas: dict, boot: ItemBoot,
                clean_ids: list[str]) -> dict:
    """Posterior argmax flip rates of raw log-likelihood (8B), the teacher and each student's primary score
    under para1, typo1, fill2 and hypothesis rewording. Part A's kinds are read against the co-batched
    original for the 8B scores, and against the student's own prediction on the original turn."""
    from robust_likelihood.multiwoz.typv0 import load_typv0_config, pool_weights
    from robust_likelihood.multiwoz.wk_analyze import perturb_cells, perturb_log
    from robust_likelihood.multiwoz.wk_quantile import scores_of, specs

    found: dict = {}
    raw = result["baselines"]["raw_8b"]["robustness"]
    uperturb = week.get("qwen3_base|F", {}).get("uperturb")
    rows = read_jsonl(uperturb / "uperturb.jsonl") if uperturb is not None else []
    raw_part_a = perturb_log(rows)
    found["raw log-likelihood (8B)"] = {
        "para1": _get(raw, "kinds", "para1", "argmax_flip"),
        "typo1": part_a_flips(raw_part_a, metas, boot, clean_ids, "typo1") if rows else None,
        "fill2": part_a_flips(raw_part_a, metas, boot, clean_ids, "fill2") if rows else None,
        "hypothesis rewording": _get(raw, "posterior_hrewrite", "argmax_flip_rate")}
    spec = specs(load_typv0_config(root))[scheme["choice"]]
    weights = pool_weights(inputs.probe, "uniform")
    log, bins, _ = scores_of(inputs, spec, weights)
    teacher = robustness(log, bins, metas, boot, clean_ids, SEED)
    part_a, _, _ = scores_of(inputs, spec, weights, perturb_cells(rows)) if rows else ({}, None, None)
    found[f"teacher ({scheme['choice']}, 64 samples, 8B)"] = {
        "para1": _get(teacher, "kinds", "para1", "argmax_flip"),
        "typo1": part_a_flips(part_a, metas, boot, clean_ids, "typo1") if rows else None,
        "fill2": part_a_flips(part_a, metas, boot, clean_ids, "fill2") if rows else None,
        "hypothesis rewording": _get(teacher, "posterior_hrewrite", "argmax_flip_rate")}
    for run, block in result["students"].items():
        robust = _get(block, "forms", PRIMARY[run], "robustness")
        found[f"student {run}"] = {kind: _get(robust, "kinds", kind, "argmax_flip") for kind in ("para1", "typo1", "fill2")}
        found[f"student {run}"]["hypothesis rewording"] = _get(robust, "posterior_hrewrite", "argmax_flip_rate")
    return found


def _get(block, *path):
    for key in path:
        if block is None:
            return None
        block = block.get(key) if isinstance(block, dict) else None
    return block


def predictions(result: dict, abc: dict) -> list[dict]:
    students = result["students"]
    found = []
    rho = {run: _get(block, "agreement", "all", "spearman") for run, block in students.items()}
    found.append({"id": "D1", "statement": "Spearman >= 0.7 between T_hat and T on human pairs, for at least one run",
                  "value": rho, "pass": any(value is not None and value >= 0.7 for value in rho.values())})
    kept = {run: _get(block, "forms", PRIMARY[run], "all", "size_kept", "named") for run, block in students.items()}
    found.append({"id": "D2", "statement": "size principle kept (implicit, slot named) between 0.5 and 2",
                  "value": kept, "per_run": {run: value is not None and 0.5 <= value <= 2 for run, value in kept.items()},
                  "pass": any(value is not None and 0.5 <= value <= 2 for value in kept.values())})
    contra = {run: _get(block, "forms", PRIMARY[run], "all", "contra_log_bf") for run, block in students.items()}
    found.append({"id": "D3", "statement": "contradiction log Bayes factor >= 3 nats", "value": contra,
                  "per_run": {run: value is not None and value >= 3 for run, value in contra.items()},
                  "pass": any(value is not None and value >= 3 for value in contra.values())})
    teacher_para1 = _get(abc, "models", "qwen3_base|F", "part_c", result["scheme"]["choice"], "agreement", "para1", "mean")
    para1 = {run: _get(block, "forms", PRIMARY[run], "robustness", "kinds", "para1", "bin_agreement", "mean")
             for run, block in students.items()}
    found.append({"id": "D4", "statement": "bin agreement under para1 >= the teacher's", "value": {
        "students": para1, "teacher": teacher_para1},
        "pass": teacher_para1 is not None and any(value is not None and value >= teacher_para1 for value in para1.values())})
    cost = _get(result, "teacher_cost", "pairs_per_second_new_hypothesis")
    speed = {run: block.get("pairs_per_second") for run, block in students.items()}
    found.append({"id": "D5", "statement": ">= 100 times more pairs per second than the teacher for a new hypothesis",
                  "value": {"students": speed, "teacher": cost},
                  "pass": cost is not None and any(value is not None and value >= 100 * cost for value in speed.values())})
    small_kept = _get(result, "baselines", "raw_06b", "all", "size_kept", "named")
    small_para = _get(result, "baselines", "raw_06b", "robustness", "utterance", "para", "abs_delta", "median")
    mine = {}
    for run, block in students.items():
        mine[run] = {"size_gap": None if kept[run] is None else abs(kept[run] - 1),
                     "para_abs_delta_log_bf": _get(block, "forms", PRIMARY[run], "robustness", "utterance", "para",
                                                   "abs_delta", "median")}
    ok = small_kept is not None and small_para is not None and any(
        found_run["size_gap"] is not None and found_run["para_abs_delta_log_bf"] is not None
        and found_run["size_gap"] < abs(small_kept - 1) and found_run["para_abs_delta_log_bf"] < small_para
        for found_run in mine.values())
    found.append({"id": "D6", "statement": "the 0.6B model's own log-prob is worse than the student on both |size kept - 1| "
                                           "and |delta log BF| under para", "value": {
        "raw_06b": {"size_gap": None if small_kept is None else abs(small_kept - 1), "para_abs_delta_log_bf": small_para},
        "students": mine}, "pass": ok})
    drops = {}
    for run, block in students.items():
        seen, unseen = _get(block, "agreement", "seen_food", "spearman"), _get(block, "agreement", "unseen_food", "spearman")
        drops[run] = None if seen is None or unseen is None else seen - unseen
    found.append({"id": "D7", "statement": "Spearman drops by <= 0.1 on unseen-food items", "value": drops,
                  "pass": any(value is not None and value <= 0.1 for value in drops.values())})
    return found


def verdict(result: dict) -> dict:
    """The frozen verdict of plan 5.7, run by run."""
    students = result["students"]
    per_run = {}
    for run, block in students.items():
        rho = _get(block, "agreement", "all", "spearman")
        kept = _get(block, "forms", PRIMARY[run], "all", "size_kept", "named")
        contra = _get(block, "forms", PRIMARY[run], "all", "contra_log_bf")
        d1 = rho is not None and rho >= 0.7
        d2 = kept is not None and 0.5 <= kept <= 2
        d3 = contra is not None and contra >= 3
        per_run[run] = {"D1": d1, "D2": d2, "D3": d3,
                        "synthetic_val_spearman": _get(block, "agreement", "synthetic_val", "spearman"),
                        "human_spearman": rho}
    if any(found["D1"] and found["D2"] and found["D3"] for found in per_run.values()):
        label = "The pipeline works"
    elif any(found["D1"] for found in per_run.values()):
        label = "Learns the teacher but not the task"
    else:
        label = "Does not learn the teacher"
    return {"verdict": label, "per_run": per_run}
