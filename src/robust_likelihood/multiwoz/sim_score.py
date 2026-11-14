"""Simulator study, Phase 2 (docs/simulator_study_plan.md): scoring and sampling for one (model, format).

Each part writes its own jsonl into results/multiwoz/<UTC>-sim-<key>-<fmt>/, in the layouts of Phase 8
(multiwoz/lm.py) and claims job A (multiwoz/claim1_lm.py), so the analysis reads every model alike:

- samples:  K samples per distinct clean hypothesis text, pure sampling. lm_samples.jsonl has each
            sample's text, S under its own prompt, and S_broad under broad|impl (for the centered
            variants); lm_hypotheses.jsonl has each text's mean and SD. A sample cut off at
            max_new_tokens has S = None and S_lower, the surprisal of its max_new_tokens tokens, a
            lower bound on its S.
- human:    S(U | H) for every clean probe row (and the fail_info rows with include_fail), with
            Phase 8's typicality fields where H has samples. lm_scores.jsonl.
- urewrite: S(U' | h) for every accepted utterance rewrite under every hypothesis of its item.
- hrewrite: S(U | h') for every clean probe row under every accepted rewrite of its text.
- hsample:  K samples per accepted rewritten hypothesis of the robustness subset, with S and S_broad
            (hsample.jsonl, hsample_hypotheses.jsonl); by the plan only for the models listed in
            configs/simulator.json phase2.hsample_models.

Before any of it the run checks the format's chat template, that sampling is exact, and the token
boundary of every pair it will score; boundary failures are logged, and more than 1% stop the run.
torch and transformers are imported only inside sim_score.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR, load_config
from robust_likelihood.multiwoz.claim1_lm import load_claims_config, plan_pairs
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.lm import hypothesis_shape, typicality_scores
from robust_likelihood.multiwoz.sim import (
    SimFormat,
    draw,
    flags,
    generation_config,
    load_sim_config,
    sample_surprisals,
    use_config,
)
from robust_likelihood.storage import read_jsonl, write_json, write_jsonl

PARTS = ("samples", "human", "urewrite", "hrewrite", "hsample")
FLAG_NAMES = ("empty", "truncated", "assistant_opening", "restaurant_voice", "think_block")


def score_plan(probe: list[dict], rewrites: list[dict], subset: list[str], centering_text: str,
               include_fail: bool, items: set[str] | None = None) -> dict:
    """The texts and pairs of every part; items restricts everything to those item ids (smoke tests)."""
    if items is not None:
        probe = [row for row in probe if row["item_id"] in items]
        texts = {row["text"] for row in probe}
        # A hypothesis rewrite is stored under one item but applies to every item sharing its text.
        rewrites = [row for row in rewrites if (row["item_id"] in items if row["side"] == "utterance"
                                                else row["original"] in texts)]
        subset = [item for item in subset if item in items]
    pairs = plan_pairs(probe, rewrites, subset, centering_text, [])
    clean = [row for row in probe if row["set"] == "clean"]
    return {"samples": sorted({row["text"] for row in clean}),
            "human": [row for row in probe if include_fail or row["set"] == "clean"],
            "urewrite": pairs["urewrite"], "hrewrite": pairs["hrewrite"], "hsample": pairs["hsample"]}


def default_parts(key: str, config: dict) -> list[str]:
    return [part for part in PARTS if part != "hsample" or key in config["phase2"]["hsample_models"]]


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and sample SD, as Phase 8's lm_hypotheses.jsonl reports them."""
    if not values:
        return None, None
    mu = sum(values) / len(values)
    if len(values) < 2:
        return mu, None
    return mu, (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def flag_names(found: dict) -> list[str]:
    return [name for name in FLAG_NAMES if found[name]] + list(found["flag_tokens"])


def tally(rows: list[dict]) -> dict:
    """Counts of drawn, ended, and flagged samples."""
    counts = {"drawn": len(rows), "ended": sum(1 for row in rows if row["ended"])}
    for row in rows:
        for name in row["flags"]:
            counts[name] = counts.get(name, 0) + 1
    return counts


def sample_block(model, sim_format: SimFormat, hypothesis, count: int, config, seed: int, batch_size: int,
                 pad: int, centering_text: str | None) -> list[dict]:
    """count samples under hypothesis, each with S, S_lower if cut off, S_broad if ended (None without a
    centering text), and its flags."""
    from robust_likelihood.lmcore import surprisals

    found = draw(model, sim_format, hypothesis, count, config, seed)
    own = sample_surprisals(model, sim_format, hypothesis, found, batch_size, pad)
    prefix = sim_format.prefix_ids(hypothesis)
    cut = [index for index, sample in enumerate(found) if not sample["ended"]]
    lower = dict(zip(cut, surprisals(model, [(prefix + found[index]["tokens"], len(prefix)) for index in cut],
                                     batch_size, pad))) if cut else {}
    ended = [index for index, sample in enumerate(found) if sample["ended"]]
    broad = {}
    if centering_text is not None and ended:
        broad_prefix = sim_format.prefix_ids(centering_text)
        broad = dict(zip(ended, surprisals(
            model, [sim_format.scored(centering_text, found[index]["text"], broad_prefix) for index in ended],
            batch_size, pad)))
    rows = []
    for index, sample in enumerate(found):
        row = {"sample": index, "reply": sample["text"], "S": own[index], "S_broad": broad.get(index),
               "S_lower": lower.get(index), "ended": sample["ended"],
               "flags": flag_names(flags(sample["text"], sample["raw"], not sample["ended"], sim_format.flag_tokens))}
        if sample["raw"] != sample["text"]:
            row["raw"] = sample["raw"]
        rows.append(row)
    return rows


def boundary_failures(sim_format: SimFormat, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(text, utterance) for text, utterance in pairs if not sim_format.boundary_holds(text, utterance)]


def sim_score(
    root: Path,
    key: str,
    fmt: str,
    *,
    parts: list[str] | None = None,
    include_fail: bool | None = None,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    device: str | None = None,
    label: str | None = None,
    items: set[str] | None = None,
    samples: int | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    tokenizer=None,
    model=None,
    now: datetime | None = None,
) -> dict:
    """One (model, format) run. items and samples shrink a run for smoke tests; tokenizer and model may
    be passed in (tests), otherwise they are loaded at the pinned revision."""
    from robust_likelihood.lmcore import load_model, pick_device, sampling_config, sampling_is_exact, surprisals, text_seed

    config = load_sim_config(root)
    spec = config["models"][key]
    if fmt not in spec["formats"] or fmt == "P0":
        raise ValueError(f"{key} has no Phase 2 format {fmt} (P0 is the Phase 8 run itself)")
    parts = list(parts) if parts else default_parts(key, config)
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}")
    if "hsample" in parts and key not in config["phase2"]["hsample_models"]:
        raise ValueError(f"the plan runs hsample only for {config['phase2']['hsample_models']}")
    include_fail = bool(config["phase2"]["include_fail"]) if include_fail is None else include_fail
    claims = load_claims_config(root)
    centering_text = claims["centering_text"]
    derived = derived or root / DERIVED_DIR
    probe = read_jsonl(derived / "probe.jsonl")
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    plan = score_plan(probe, read_jsonl(derived / "rewrites.jsonl"), subset, centering_text, include_fail, items)
    settings, seed = config["sampling"], int(config["seed"])
    count = int(samples or config["samples_per_text"])
    edges = tuple(float(edge) for edge in load_config(root)["lm"]["bin_edges"])
    examples = json.loads((root / config["fewshot"]["file"]).read_text(encoding="utf-8"))["examples"] if fmt == "F" else []
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    import torch
    import transformers

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    sim_format = SimFormat(tokenizer, spec, fmt, config["prompts"], examples)
    scored_pairs = [(row["text"], row["utterance"]) for part in ("human", "urewrite", "hrewrite") if part in parts
                    for row in plan[part]]
    failed = boundary_failures(sim_format, scored_pairs)
    rate = len(failed) / len(scored_pairs) if scored_pairs else 0.0
    checks = {"template_check": sim_format.template_check(), "boundary_pairs": len(scored_pairs),
              "boundary_failures": len(failed), "boundary_failure_rate": rate}
    run_dir = (results_root or root / RESULTS_DIR) / (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-sim-{key}-{fmt}" + (f"-{label}" if label else ""))
    manifest = {
        "kind": "sim-score", "quantity": "Surprisal S in nats of a simulator model. Not Jev scores.",
        "model_key": key, "format": fmt, "primary": fmt == spec.get("primary"), "model": spec["id"],
        "revision": spec["revision"], "license": spec.get("license"), "dtype": dtype, "git": git_state(root),
        "prompts": config["prompts"], "fewshot_examples": examples,
        "prefix_example": sim_format.prefix_text(centering_text), "centering_text": centering_text,
        "end_token": spec.get("end_token"), "end_text": spec.get("end_text"), "parts": parts,
        "include_fail": include_fail, "items": None if items is None else sorted(items),
        "planned": {part: len(plan[part]) for part in parts}, "samples_per_text": count, "seed": seed,
        "seed_rule": "torch.manual_seed(int(sha256(f'{seed}:{text}').hexdigest()[:8], 16)) per hypothesis text",
        "bin_edges": list(edges), "checks": checks, "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage_seconds": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    if failed:
        write_jsonl(run_dir / "boundary_failures.jsonl", [{"text": text, "utterance": utterance}
                                                          for text, utterance in failed])
    if not checks["template_check"] or rate > float(config["format_check"]["boundary_failure_limit"]):
        raise RuntimeError(f"Phase 2 self-check failed before scoring: {checks}")

    if model is None:
        model = load_model(spec["id"], spec["revision"], dtype, pick_device(device))
    model.eval()
    pad = getattr(tokenizer, "pad_token_id", None)
    end_or_eos = sim_format.end if sim_format.end is not None else tokenizer.eos_token_id
    pad = pad if pad is not None else end_or_eos
    generation = generation_config(sim_format, settings, pad)
    checks["sampling_is_exact"] = sampling_is_exact(
        model, sim_format.prefix_ids(centering_text), use_config(model, sampling_config(settings, end_or_eos, pad)))
    manifest.update({
        "device": str(next(model.parameters()).device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "sampling": {name: getattr(generation, name) for name in (
            "do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "eos_token_id",
            "pad_token_id", "stop_strings")},
    })
    write_json(run_dir / "manifest.json", manifest)
    if not checks["sampling_is_exact"]:
        raise RuntimeError(f"Phase 2 self-check failed: {checks}")

    def sequences_of(rows: list[dict]) -> list[tuple[list[int], int]]:
        return [sim_format.scored(row["text"], row["utterance"]) for row in rows]

    def finish(part: str, stage: float) -> None:
        manifest["stage_seconds"][part] = round(time.monotonic() - stage, 1)
        write_json(run_dir / "manifest.json", manifest)

    values_of: dict[str, list[float]] = {}
    if "samples" in parts:
        stage = time.monotonic()
        rows, hypotheses = [], []
        shapes = hypothesis_shape([row for row in probe if row["set"] == "clean"])
        for text in plan["samples"]:
            block = sample_block(model, sim_format, text, count, generation, text_seed(seed, text), batch_size, pad,
                                 centering_text)
            rows.extend({"text": text, **row} for row in block)
            values = [row["S"] for row in block if row["S"] is not None]
            values_of[text] = values
            mu, sigma = mean_sd(values)
            hypotheses.append({"text": text, "n_constrained": shapes[text]["n_constrained"],
                               "regime": shapes[text]["regime"], "hyp_types": sorted(shapes[text]["hyp_types"]),
                               "mu": mu, "sigma": sigma, "kept": len(values), "dropped": len(block) - len(values)})
        write_jsonl(run_dir / "lm_samples.jsonl", rows)
        write_jsonl(run_dir / "lm_hypotheses.jsonl", hypotheses)
        manifest["samples"] = tally(rows)
        finish("samples", stage)
    if "human" in parts:
        stage = time.monotonic()
        rows = plan["human"]
        sequences = sequences_of(rows)
        values = surprisals(model, sequences, batch_size, pad)
        empty = {"lm_loglik": None, "mu": None, "sigma": None, "T": None, "lm_typ_normal": None, "lm_typ_bins": None}
        out = []
        for row, value, (tokens, first) in zip(rows, values, sequences):
            samples_here = values_of.get(row["text"], [])
            found = (typicality_scores(value, samples_here, edges) if samples_here
                     else {**empty, "S": value, "lm_loglik": -value})
            out.append({"item_id": row["item_id"], "set": row["set"], "hyp_key": row["hyp_key"],
                        "tokens_scored": len(tokens) - first, **found})
        write_jsonl(run_dir / "lm_scores.jsonl", out)
        finish("human", stage)
    for part in ("urewrite", "hrewrite"):
        if part in parts:
            stage = time.monotonic()
            rows = plan[part]
            values = surprisals(model, sequences_of(rows), batch_size, pad)
            write_jsonl(run_dir / f"{part}.jsonl", [row | {"S": value} for row, value in zip(rows, values)])
            finish(part, stage)
    if "hsample" in parts:
        stage = time.monotonic()
        rows, statistics = [], []
        for unit in plan["hsample"]:
            block = sample_block(model, sim_format, unit["text"], count, generation, text_seed(seed, unit["text"]),
                                 batch_size, pad, centering_text)
            rows.extend({"unit_id": unit["unit_id"], **row} for row in block)
            kept = sum(1 for row in block if row["S"] is not None)
            statistics.append({"unit_id": unit["unit_id"], "text": unit["text"], "kept": kept,
                               "dropped": len(block) - kept})
        write_jsonl(run_dir / "hsample.jsonl", rows)
        write_jsonl(run_dir / "hsample_hypotheses.jsonl", statistics)
        manifest["hsample"] = tally(rows)
        finish("hsample", stage)
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "planned": manifest["planned"], "checks": checks,
            "stage_seconds": manifest["stage_seconds"], "samples": manifest.get("samples"),
            "hsample": manifest.get("hsample")}
