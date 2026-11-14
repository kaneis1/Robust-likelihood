"""Simulator study, noise floor of the utterance-rewrite shifts (Q3, prediction S4).

Claim 1's shift under an utterance rewrite U -> U' is Δ log BF = [l(U'|G) - l(U'|H)] - [l(U|G) - l(U|H)].
l(U|·) comes from Phase 2's human part and l(U'|·) from its urewrite part, scored in different
batches; bfloat16 logits depend on the batch layout (padding width and batch size), so Δ carries
rescoring noise even when nothing else changes.

sim_noise reruns the urewrite part of one Phase 2 run with every rewrite replaced by its original
utterance: the same rows in the same order, the same batch size, dtype and code path. Claim 1's
Δ log BF from those scores is then rescoring noise alone, aggregated exactly as the real shifts. It
also rescores the human part in Phase 2's own layout, to tell run-to-run nondeterminism from layout.

noise_analyze sets each run's noise beside its real shifts, per rewrite group.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.claim1 import U_GROUPS, ItemBoot, Scorer, item_meta, token_log, utterance, utterance_rows
from robust_likelihood.multiwoz.claim1_lm import load_claims_config
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.sim import SimFormat, load_sim_config
from robust_likelihood.multiwoz.sim_analyze import discover
from robust_likelihood.multiwoz.sim_report import LABELS, num, table, with_ci
from robust_likelihood.multiwoz.sim_score import score_plan
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

PHASE2_BATCH_SIZE = 32


def null_rows(plan_urewrite: list[dict], utterances: dict[str, str]) -> list[dict]:
    """The urewrite rows with every rewrite replaced by its item's original utterance."""
    return [row | {"utterance": utterances[row["item_id"]]} for row in plan_urewrite]


def sim_noise(
    root: Path,
    key: str,
    fmt: str,
    *,
    batch_size: int = PHASE2_BATCH_SIZE,
    dtype: str = "bfloat16",
    device: str | None = None,
    items: set[str] | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    tokenizer=None,
    model=None,
    now: datetime | None = None,
) -> dict:
    """One (model, format): the null urewrite part and the human part again, in Phase 2's layouts."""
    from robust_likelihood.lmcore import load_model, pick_device, surprisals

    config = load_sim_config(root)
    spec = config["models"][key]
    if fmt not in spec["formats"] or fmt == "P0":
        raise ValueError(f"{key} has no Phase 2 format {fmt}")
    derived = derived or root / DERIVED_DIR
    probe = read_jsonl(derived / "probe.jsonl")
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    plan = score_plan(probe, read_jsonl(derived / "rewrites.jsonl"), subset, load_claims_config(root)["centering_text"],
                      bool(config["phase2"]["include_fail"]), items)
    utterances = {row["item_id"]: row["utterance"] for row in probe}
    nulls = null_rows(plan["urewrite"], utterances)
    examples = json.loads((root / config["fewshot"]["file"]).read_text(encoding="utf-8"))["examples"] if fmt == "F" else []
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    import torch
    import transformers

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    sim_format = SimFormat(tokenizer, spec, fmt, config["prompts"], examples)
    if model is None:
        model = load_model(spec["id"], spec["revision"], dtype, pick_device(device))
    model.eval()
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else (sim_format.end if sim_format.end is not None else tokenizer.eos_token_id)
    run_dir = (results_root or root / RESULTS_DIR) / f"{started.strftime('%Y%m%dT%H%M%SZ')}-sim-noise-{key}-{fmt}"
    manifest = {"kind": "sim-noise", "model_key": key, "format": fmt, "model": spec["id"], "revision": spec["revision"],
                "dtype": dtype, "batch_size": batch_size, "git": git_state(root),
                "device": str(next(model.parameters()).device),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__},
                "items": None if items is None else sorted(items),
                "planned": {"urewrite_null": len(nulls), "human_repeat": len(plan["human"])},
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "stage_seconds": {}}
    write_json(run_dir / "manifest.json", manifest)
    for part, rows in (("urewrite_null", nulls), ("human_repeat", plan["human"])):
        stage = time.monotonic()
        values = surprisals(model, [sim_format.scored(row["text"], row["utterance"]) for row in rows], batch_size, pad)
        if part == "human_repeat":
            rows = [{"item_id": row["item_id"], "set": row["set"], "hyp_key": row["hyp_key"]} for row in rows]
        write_jsonl(run_dir / f"{part}.jsonl", [row | {"S": value} for row, value in zip(rows, values)])
        manifest["stage_seconds"][part] = round(time.monotonic() - stage, 1)
        write_json(run_dir / "manifest.json", manifest)
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "planned": manifest["planned"], "stage_seconds": manifest["stage_seconds"]}


def sim_noise_many(root: Path, labels: list[str], **options) -> list[dict]:
    """Several (model, format) labels in one job, loading each model once."""
    from robust_likelihood.lmcore import load_model, pick_device

    config = load_sim_config(root)
    pairs = [tuple(label.split(":")) for label in labels]
    found, loaded = [], {}
    for key, fmt in pairs:
        spec = config["models"][key]
        model_id = (spec["id"], spec["revision"])
        if model_id not in loaded:
            loaded.clear()
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded[model_id] = load_model(spec["id"], spec["revision"], options.get("dtype", "bfloat16"),
                                          pick_device(options.get("device")))
        found.append(sim_noise(root, key, fmt, model=loaded[model_id], **options))
    return found


# ---------------------------------------------------------------- analysis (CPU)


def discover_noise(run_dirs: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for run_dir in sorted(run_dirs):
        path = run_dir / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            if manifest.get("kind") == "sim-noise" and "finished_at" in manifest and manifest.get("items") is None:
                found[f"{manifest['model_key']}|{manifest['format']}"] = run_dir
    return found


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def compare(real_rows: list[dict], null_rows_: list[dict]) -> dict:
    """Per rewrite group: the noise's |Δ| quantiles, over all pairs and over the pairs whose null scores
    changed (a null row in the same batch shape as its original reproduces it bit for bit, which a real
    rewrite, of another length, never does), and the share of real |Δ| above the changed-layout p95."""
    found = {}
    for group, kinds in U_GROUPS.items():
        noise = [abs(row["delta"]) for row in null_rows_ if row["kind"] in kinds]
        changed = [value for value in noise if value != 0]
        real = [abs(row["delta"]) for row in real_rows if row["kind"] in kinds]
        p95 = quantile(changed, 0.95)
        found[group] = {"noise_p50": quantile(noise, 0.5), "noise_p90": quantile(noise, 0.9),
                        "noise_p95": quantile(noise, 0.95), "noise_zero_share": (1 - len(changed) / len(noise)
                                                                                 if noise else None),
                        "changed_p50": quantile(changed, 0.5), "changed_p95": p95, "real_p50": quantile(real, 0.5),
                        "real_above_changed_p95": (sum(1 for value in real if value > p95) / len(real)
                                                   if real and p95 is not None else None)}
    return found


def run_noise_analyze(root: Path, *, runs: list[Path], out: Path, derived: Path | None = None,
                      now: datetime | None = None) -> dict:
    config = load_sim_config(root)
    derived = derived or root / DERIVED_DIR
    noise_runs, phase2 = discover_noise(runs), discover(runs)
    missing = sorted(set(noise_runs) - set(phase2))
    if not noise_runs or missing:
        raise ValueError(f"need finished sim-noise runs and their Phase 2 runs (missing Phase 2: {missing})")
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    clean = {item["item_id"] for item in items}
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    metas = item_meta(items, probe)
    boot = ItemBoot(sorted(clean), int(config["bootstrap"]["replicates"]), int(config["bootstrap"]["seed"]))
    found = {}
    for label, noise_dir in sorted(noise_runs.items()):
        scores = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(phase2[label] / "lm_scores.jsonl")
                  if row["set"] == "clean"}
        originals = {pair: -value for pair, value in scores.items()}
        real = token_log(originals, {"urewrite": read_jsonl(phase2[label] / "urewrite.jsonl")}, clean)
        null = token_log(originals, {"urewrite": read_jsonl(noise_dir / "urewrite_null.jsonl")}, clean)
        real_scorer, null_scorer = (Scorer(label, "loglik", log, items=sorted(clean)) for log in (real, null))
        repeat = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(noise_dir / "human_repeat.jsonl")
                  if row["set"] == "clean"}
        gaps = [abs(repeat[pair] - scores[pair]) for pair in scores if pair in repeat]
        found[label] = {
            "noise_run": noise_dir.name, "phase2_run": phase2[label].name,
            "commit": (read_json(noise_dir / "manifest.json").get("git") or {}).get("commit"),
            "repeat": {"pairs": len(gaps), "max_abs_gap": max(gaps) if gaps else None,
                       "median_abs_gap": statistics.median(gaps) if gaps else None,
                       "identical": sum(1 for gap in gaps if gap == 0) / len(gaps) if gaps else None},
            "real": utterance(real_scorer, metas, boot),
            "noise": utterance(null_scorer, metas, boot),
            "compare": compare(utterance_rows(real_scorer, metas), utterance_rows(null_scorer, metas)),
        }
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"generated_at": stamp, "git": git_state(root), "runs": found}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "noise_floor.json", payload)
    (out / "noise_floor.md").write_text(noise_markdown(payload), encoding="utf-8", newline="\n")
    return {"out": str(out / "noise_floor.md"), "runs": sorted(found)}


def noise_markdown(payload: dict) -> str:
    runs = payload["runs"]
    labels = [label for label in LABELS if label in runs] + sorted(label for label in runs if label not in LABELS)
    name = {label: LABELS.get(label, label) for label in labels}
    commits = sorted({block.get("commit") or "?" for block in runs.values()})
    lines = ["# Simulator study: noise floor of the utterance-rewrite shifts", "",
             "Each primary run's urewrite part was scored again with every rewrite replaced by its original "
             "utterance: the same rows, order, batch size (32), bfloat16 and code path as Phase 2. Claim 1's "
             "|Δ log BF| from those scores is rescoring noise alone, aggregated exactly as the real shifts "
             "(median over the size-principle and contradiction pairs, 95% item-bootstrap intervals). "
             f"Noise runs at commit {', '.join(f'`{commit[:7]}`' for commit in commits)}; generated "
             f"{payload['generated_at']}.", "",
             "A null row that lands in a batch of the same shape as its original reproduces the original bit for "
             "bit (see the last table), so about half the null pairs have no noise at all, and the overall noise "
             "median understates what a real rewrite, which always changes length, carries. The columns marked "
             "*changed* use only the null pairs whose scores moved: the conservative noise floor.", ""]
    for group in ("para", "zh", "typo", "pad", "verbose"):
        lines += [f"## {group}", ""]
        body = []
        for label in labels:
            block = runs[label]
            versus = block["compare"][group]
            above = versus["real_above_changed_p95"]
            body.append([name[label], with_ci(block["real"][group]["abs_delta"], "median"),
                         with_ci(block["noise"][group]["abs_delta"], "median"), num(versus["changed_p50"]),
                         num(versus["changed_p95"]), "–" if above is None else f"{100 * above:.0f}%"])
        lines += [table(["Model", "Real median |Δ log BF|", "Noise median, all", "Noise median, changed",
                         "Noise p95, changed", "Real pairs above changed p95"], body), ""]
    lines += ["## Rescoring the human part in its own layout", "",
              "The human pairs scored again in Phase 2's order and batches, against Phase 2's scores.", ""]
    lines += [table(["Model", "Pairs", "Identical", "Median |gap| (nats)", "Max |gap| (nats)"],
                    [[name[label], str(runs[label]["repeat"]["pairs"]),
                      "–" if runs[label]["repeat"]["identical"] is None else f"{100 * runs[label]['repeat']['identical']:.1f}%",
                      num(runs[label]["repeat"]["median_abs_gap"], 3), num(runs[label]["repeat"]["max_abs_gap"], 3)]
                     for label in labels]), ""]
    return "\n".join(lines).rstrip("\n") + "\n"
