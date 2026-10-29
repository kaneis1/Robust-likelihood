"""Claims plan 2c, GPU job A: language-model surprisal for the rewrite suite.

Phase 8 itself is the existing `lm-score` run (full probe, K = 64 samples per text). This
module adds, each part writing its own jsonl into one run folder:

- `center`:   S(u | broad|impl) for every probe utterance and every Phase 8 sample, for the
              centered score S(U|H) - S(U|broad|impl) and its reference distribution.
- `urewrite`: S(U' | h) for every accepted utterance rewrite under every hypothesis of its item.
- `hrewrite`: S(U | h') for every clean probe row under every accepted rewrite of its text.
- `hsample`:  K samples under each accepted rewritten hypothesis text of the robustness subset,
              with S under h' and under broad|impl.

S, the prompt, and the sampling settings are Phase 8's (multiwoz/lm.py); the second model
family runs the scoring parts only.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

PARTS = ("center", "urewrite", "hrewrite", "hsample")
SCORING_PARTS = ("urewrite", "hrewrite")


def load_claims_config(root: Path) -> dict:
    return read_json(root / "configs" / "claims.json")


def plan_pairs(probe: list[dict], rewrites: list[dict], subset: list[str], centering_text: str,
               phase8_samples: list[dict]) -> dict[str, list[dict]]:
    """Every (hypothesis text, utterance) pair each part scores, with the fields its rows keep."""
    utterances = {row["item_id"]: row["utterance"] for row in probe}
    rows_by_item: dict[str, list[dict]] = {}
    for row in probe:
        rows_by_item.setdefault(row["item_id"], []).append(row)
    clean = [row for row in probe if row["set"] == "clean"]
    hyp_rewrites: dict[str, list[dict]] = {}
    for rewrite in rewrites:
        if rewrite["side"] == "hypothesis":
            hyp_rewrites.setdefault(rewrite["original"], []).append(rewrite)
    center = [{"source": "utterance", "item_id": item_id, "text": centering_text, "utterance": utterance}
              for item_id, utterance in sorted(utterances.items())]
    center += [{"source": "sample", "hyp_text": sample["text"], "sample": sample["sample"], "text": centering_text,
                "utterance": sample["reply"]} for sample in phase8_samples]
    urewrite = [
        {"unit_id": rewrite["unit_id"], "item_id": rewrite["item_id"], "kind": rewrite["kind"],
         "hyp_key": row["hyp_key"], "text": row["text"], "utterance": rewrite["rewrite"]}
        for rewrite in rewrites if rewrite["side"] == "utterance"
        for row in rows_by_item[rewrite["item_id"]]
    ]
    hrewrite = [
        {"unit_id": rewrite["unit_id"], "item_id": row["item_id"], "kind": rewrite["kind"], "hyp_key": row["hyp_key"],
         "text": rewrite["rewrite"], "utterance": row["utterance"]}
        for row in clean for rewrite in hyp_rewrites.get(row["text"], [])
    ]
    chosen = set(subset)
    texts = {}
    for row in clean:
        if row["item_id"] in chosen:
            for rewrite in hyp_rewrites.get(row["text"], []):
                texts.setdefault(rewrite["unit_id"], rewrite["rewrite"])
    hsample = [{"unit_id": unit_id, "text": text} for unit_id, text in sorted(texts.items())]
    return {"center": center, "urewrite": urewrite, "hrewrite": hrewrite, "hsample": hsample}


def claim1_lm(
    root: Path,
    model_key: str,
    parts: list[str],
    phase8_dir: Path,
    *,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    device: str | None = None,
    label: str | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    from robust_likelihood.lmcore import (
        load_model,
        pick_device,
        resolve_revision,
        sample_replies,
        sampling_config,
        sampling_is_exact,
        scored_sequence,
        surprisals,
        text_seed,
    )
    from robust_likelihood.multiwoz import load_config
    from robust_likelihood.multiwoz.lm import (
        SYSTEM_PROMPT,
        USER_PROMPT,
        boundary_holds,
        end_of_turn_id,
        prefix_ids,
        reply_ids,
    )

    claims = load_claims_config(root)
    settings = dict(load_config(root)["lm"])
    spec = claims["models"][model_key]
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}")
    derived = derived or root / DERIVED_DIR
    probe = read_jsonl(derived / "probe.jsonl")
    rewrites = read_jsonl(derived / "rewrites.jsonl")
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    samples = read_jsonl(phase8_dir / "lm_samples.jsonl") if "center" in parts else []
    plans = plan_pairs(probe, rewrites, subset, claims["centering_text"], samples)
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    commit = resolve_revision(spec["id"], spec["revision"])
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=commit)
    model = load_model(spec["id"], commit, dtype, pick_device(device))
    model.eval()
    end = end_of_turn_id(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
    generation = sampling_config(settings, end, pad)
    model.generation_config = generation
    checks = {"boundary_holds": all(boundary_holds(tokenizer, row["text"], row["utterance"], end)
                                    for row in (plans["urewrite"] + plans["hrewrite"])[:: 500] or probe[:25])}
    if "hsample" in parts:
        checks["sampling_is_exact"] = sampling_is_exact(model, prefix_ids(tokenizer, claims["centering_text"]), generation)
    if not all(checks.values()):
        raise RuntimeError(f"language-model self-check failed: {checks}")
    run_dir = (results_root or root / RESULTS_DIR / "claim1") / (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or 'lm-' + model_key}")
    manifest = {
        "quantity": "Language-model surprisal S in nats for the claims-plan rewrite suite.",
        "model_key": model_key, "model": spec["id"], "revision": commit, "dtype": dtype,
        "device": str(next(model.parameters()).device), "git": git_state(root),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "prompt": {"system": SYSTEM_PROMPT, "user": USER_PROMPT, "assistant": "{utterance}"},
        "centering_text": claims["centering_text"], "phase8_dir": phase8_dir.as_posix(), "parts": parts,
        "planned": {part: len(plans[part]) for part in parts}, "checks": checks,
        "sampling": {key: getattr(generation, key) for key in ("do_sample", "temperature", "top_p", "top_k",
                                                              "repetition_penalty", "max_new_tokens")},
        "samples_per_rewritten_hypothesis": int(claims["samples_per_rewritten_hypothesis"]),
        "seed": int(claims["seed"]),
        "seed_rule": "torch.manual_seed(int(sha256(f'{seed}:{text}').hexdigest()[:8], 16)) per rewritten text",
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "stage_seconds": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    prefixes: dict[str, list[int]] = {}

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        sequences = []
        for text, utterance in pairs:
            if text not in prefixes:
                prefixes[text] = prefix_ids(tokenizer, text)
            sequences.append(scored_sequence(prefixes[text], reply_ids(tokenizer, utterance), end))
        return surprisals(model, sequences, batch_size, pad)

    for part in [part for part in PARTS if part in parts and part != "hsample"]:
        stage = time.monotonic()
        rows = plans[part]
        values = score([(row["text"], row["utterance"]) for row in rows])
        write_jsonl(run_dir / f"{part}.jsonl", [
            {key: value for key, value in row.items() if part != "center" or key not in ("text",)} | {"S": found}
            for row, found in zip(rows, values)
        ])
        manifest["stage_seconds"][part] = round(time.monotonic() - stage, 1)
        write_json(run_dir / "manifest.json", manifest)
    if "hsample" in parts:
        stage = time.monotonic()
        count = int(claims["samples_per_rewritten_hypothesis"])
        seed = int(claims["seed"])
        sample_rows, statistics = [], []
        for unit in plans["hsample"]:
            prefix = prefix_ids(tokenizer, unit["text"])
            replies, dropped = sample_replies(model, prefix, count, generation, text_seed(seed, unit["text"]), end)
            decoded = [tokenizer.decode(reply) for reply in replies]
            values = surprisals(model, [scored_sequence(prefix, reply, end) for reply in replies], batch_size, pad)
            broad = score([(claims["centering_text"], reply) for reply in decoded])
            sample_rows.extend({"unit_id": unit["unit_id"], "sample": index, "reply": reply, "S": value,
                                "S_broad": centered}
                               for index, (reply, value, centered) in enumerate(zip(decoded, values, broad)))
            statistics.append({"unit_id": unit["unit_id"], "text": unit["text"], "kept": len(replies),
                               "dropped": dropped})
        write_jsonl(run_dir / "hsample.jsonl", sample_rows)
        write_jsonl(run_dir / "hsample_hypotheses.jsonl", statistics)
        manifest["stage_seconds"]["hsample"] = round(time.monotonic() - stage, 1)
        manifest["hsample_drop_rate"] = (sum(row["dropped"] for row in statistics)
                                         / max(1, count * len(statistics)))
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "planned": manifest["planned"], "revision": commit,
            "stage_seconds": manifest["stage_seconds"]}
