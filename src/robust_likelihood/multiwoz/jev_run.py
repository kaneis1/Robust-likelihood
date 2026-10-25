"""Score probe rows with Jev: one hypothesis per call, through the existing client and runner.

stated_intention is the constant find_restaurant and the hypothesis text goes in
intention_description. The internal key (G|impl, broad|expl, ...) is never sent, so
the label cannot leak breadth. Successful records are cached in
results/multiwoz/cache.jsonl and reused by later runs.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.clients import build_request
from robust_likelihood.keys import load_keys
from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR, jev_model_id
from robust_likelihood.planning import CallSpec, assert_comparison_pins, cache_key
from robust_likelihood.prompting import HypothesisSet, load_prompts, pairwise_instructions
from robust_likelihood.runner import _success_map, build_clients, execute
from robust_likelihood.storage import append_jsonl, read_json, read_jsonl, write_json, write_jsonl

DATASET = "multiwoz_size"
HYPOTHESIS_VERSION = "multiwoz_v1"
WORDINGS = {
    "likely": ("multiwoz_likely",),
    "typical": ("multiwoz_typical",),
    "both": ("multiwoz_likely", "multiwoz_typical"),
}
UNUSED_HYPOTHESES = HypothesisSet(HYPOTHESIS_VERSION, {})
QUANTITY = "Elicited yes-probabilities (Jev noul), not verified P(utterance | goal)."


class ModelIdChanged(RuntimeError):
    pass


def choose_items(items: list[dict], set_name: str, count: int | None = None, seed: int | None = None) -> list[str]:
    pool = sorted(item["item_id"] for item in items if set_name == "all" or item["set"] == set_name)
    if count is None:
        return pool
    if count > len(pool):
        raise ValueError(f"--items {count} exceeds the {len(pool)} items in set {set_name}")
    return sorted(random.Random(seed).sample(pool, count))


def call_spec(row: dict, prompt_version: str, model_id: str, intention: str, repeat_id: int | None = None) -> CallSpec:
    return CallSpec(
        example_id=row["item_id"],
        original_example_id=row["item_id"],
        transformation="none",
        task="pairwise",
        hypothesis=intention,
        hypothesis_wording=row["text"],
        hypothesis_description_version=HYPOTHESIS_VERSION,
        input_text=row["utterance"],
        prompt_version=prompt_version,
        requested_model_id=model_id,
        provider="jev",
        inference_settings={},
        repeat_id=repeat_id,
        intent=intention,
        original_intent=intention,
        label_changed=False,
        dataset=DATASET,
        pair_id=row["item_id"],
        role=row["hyp_key"],
    )


def plan_calls(
    rows: list[dict],
    item_ids: list[str],
    prompt_versions: tuple[str, ...],
    model_id: str,
    intention: str,
    repeats: int = 0,
) -> list[CallSpec]:
    """Item-major order, so a stopped run leaves whole items scored under both wordings."""
    if repeats < 0:
        raise ValueError("repeats must be >= 0")
    chosen = set(item_ids)
    repeat_ids: list[int | None] = [None, *range(1, repeats + 1)]
    return [
        call_spec(row, version, model_id, intention, repeat_id)
        for row in rows
        if row["item_id"] in chosen
        for version in prompt_versions
        for repeat_id in repeat_ids
    ]


class PromptRouter:
    """Send each spec with the prompt pack its prompt_version names."""

    def __init__(self, client, packs: dict):
        self.client = client
        self.packs = packs

    def perform(self, spec, prompts, hypotheses):
        return self.client.perform(spec, self.packs[spec.prompt_version], hypotheses)


def _done(record: dict) -> tuple:
    return record["example_id"], record["role"], record["prompt_version"], record["repeat_id"]


def current_cache(records: list[dict], specs: list[CallSpec], packs: dict) -> dict[str, dict]:
    """Cached successes for these specs whose stored prompt text is the one the pack renders now.

    The cache key names the prompt version, not its text, so an edited prompt file
    must not reuse old scores.
    """
    wanted = {cache_key(spec): pairwise_instructions(spec, packs[spec.prompt_version]) for spec in specs}
    return {
        key: record
        for key, record in _success_map(records).items()
        if key in wanted and record.get("prompt_text") == wanted[key]
    }


def score_rows(records: list[dict]) -> list[dict]:
    """item_id, hyp_key, prompt_version, repeat_id, noul for each scored (item, hypothesis, wording, repeat)."""
    found = {}
    for record in records:
        value = (record.get("parsed") or {}).get("plausibility_yes_probability")
        if record.get("status") == "success" and value is not None:
            found.setdefault(_done(record), value)
    rows = [
        {"item_id": item_id, "hyp_key": hyp_key, "prompt_version": version, "repeat_id": repeat_id, "noul": value}
        for (item_id, hyp_key, version, repeat_id), value in found.items()
    ]
    return sorted(rows, key=lambda row: (row["item_id"], row["prompt_version"], row["hyp_key"], row["repeat_id"] or 0))


def _failures(records: list[dict]) -> list[dict]:
    attempts: dict[tuple, list[dict]] = {}
    for record in records:
        attempts.setdefault(_done(record), []).append(record)
    failed = []
    for key, rows in attempts.items():
        if any(row.get("status") == "success" for row in rows):
            continue
        last = rows[-1]
        failed.append(
            {
                "item_id": key[0],
                "hyp_key": key[1],
                "prompt_version": key[2],
                "repeat_id": key[3],
                "status": last.get("status"),
                "http_status": last.get("http_status"),
                "error": last.get("error"),
                "attempts": len(rows),
            }
        )
    return failed


def git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_jev(
    root: Path,
    config: dict,
    *,
    wording: str = "both",
    set_name: str = "clean",
    items: int | None = None,
    seed: int | None = None,
    repeats: int = 0,
    max_requests: int | None = None,
    dry_run: bool = False,
    resume: Path | None = None,
    label: str = "run",
    derived: Path | None = None,
    results_root: Path | None = None,
    clients=None,
    transport=None,
    environ=None,
    sleeper=None,
    now: datetime | None = None,
) -> dict:
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be >= 0")
    models = read_json(root / "configs" / "models.json")
    model_id = jev_model_id(root)
    assert_comparison_pins({"jev": model_id})
    derived = derived or root / DERIVED_DIR
    manifest = read_json(resume / "manifest.json") if resume is not None else None
    if manifest is not None:
        selection = manifest["selection"]
        wording, set_name, repeats = selection["wording"], selection["set"], selection["repeats"]
        item_ids = selection["item_ids"]
    else:
        if seed is None and items is not None:
            seed = int(config["pilot"]["seed"])
        item_ids = choose_items(read_jsonl(derived / "items.jsonl"), set_name, items, seed)
    if wording not in WORDINGS:
        raise ValueError("wording must be likely, typical, or both")
    versions = WORDINGS[wording]
    packs = {version: load_prompts(root, version) for version in versions}
    specs = plan_calls(read_jsonl(derived / "probe.jsonl"), item_ids, versions, model_id, config["stated_intention"], repeats)
    results_parent = results_root or root / RESULTS_DIR
    cache_path = results_parent / "cache.jsonl"
    success_cache = current_cache(read_jsonl(cache_path), specs, packs)
    unique = {cache_key(spec) for spec in specs}
    if dry_run:
        summary = {
            "dry_run": True,
            "api_calls": 0,
            "planned_calls": len(specs),
            "unique_cache_keys": len(unique),
            "already_cached": len(unique & set(success_cache)),
            "http_calls_before_retries": len(unique - set(success_cache)),
            "items": len(item_ids),
            "prompt_versions": list(versions),
            "repeats": repeats,
            "requested_model_id": model_id,
            "first_requests": [build_request(spec, packs[spec.prompt_version], UNUSED_HYPOTHESES) for spec in specs[:3]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    started = now or datetime.now(timezone.utc)
    if manifest is not None:
        run_dir = resume
        existing = read_jsonl(run_dir / "calls.jsonl")
        finished = {_done(record) for record in existing if record.get("status") == "success"}
        specs = [spec for spec in specs if (spec.example_id, spec.role, spec.prompt_version, spec.repeat_id) not in finished]
        manifest.setdefault("resumed_at", []).append(started.strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        run_dir = results_parent / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label}"
        existing = []
        manifest = {
            "dataset": DATASET,
            "quantity": QUANTITY,
            "label": label,
            "git": git_state(root),
            "config": config,
            "endpoint": models["endpoints"]["jev"],
            "timeout_seconds": models["timeout_seconds"],
            "max_retries": models["max_retries"],
            "requested_model_id": model_id,
            "stated_intention": config["stated_intention"],
            "hypothesis_description_version": HYPOTHESIS_VERSION,
            "prompts": {
                version: {
                    "pairwise_question": pack.pairwise_question,
                    "classification_question": pack.classification_question,
                    "chat_pairwise": pack.chat_pairwise,
                    "chat_classification": pack.chat_classification,
                }
                for version, pack in packs.items()
            },
            "inputs": {
                "probe_sha256": _sha256(derived / "probe.jsonl"),
                "items_sha256": _sha256(derived / "items.jsonl"),
            },
            "selection": {
                "wording": wording,
                "set": set_name,
                "items": items,
                "seed": seed,
                "repeats": repeats,
                "item_ids": item_ids,
            },
            "planned_calls": len(specs),
            "unique_cache_keys": len(unique),
            "max_requests": max_requests,
            "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(run_dir / "manifest.json", manifest)
    if clients is None:
        keys = load_keys(("jev",), environ=environ, root=root)
        clients = build_clients(models, keys, ("jev",), transport)
    calls_path = run_dir / "calls.jsonl"
    returned = [record["returned_model_id"] for record in existing if record.get("returned_model_id")]
    expected = {"id": returned[0] if returned else None}

    def keep(record: dict) -> None:
        found = record.get("returned_model_id")
        append_jsonl(calls_path, [record])
        if found:
            if expected["id"] is None:
                expected["id"] = found
            elif found != expected["id"]:
                raise ModelIdChanged(f"Jev returned {found} after {expected['id']}; the run stopped")
        if record.get("status") == "success" and not record.get("cache_hit"):
            append_jsonl(cache_path, [record])
            success_cache[record["cache_key"]] = record

    clock = time.monotonic()
    aborted = None
    outcome = {"http_requests": 0, "cache_hits": 0, "stopped_reason": None}
    try:
        outcome = execute(
            specs,
            {"jev": PromptRouter(clients["jev"], packs)},
            None,
            UNUSED_HYPOTHESES,
            max_requests=max_requests,
            max_retries=int(models["max_retries"]),
            backoff_seconds=float(models["retry_backoff_seconds"]),
            sleeper=sleeper or time.sleep,
            success_cache=success_cache,
            skip_keys=set(),
            on_record=keep,
        )
    except ModelIdChanged as exc:
        aborted = str(exc)
    if not calls_path.exists():
        append_jsonl(calls_path, [])
    records = read_jsonl(calls_path)
    scores = score_rows(records)
    write_jsonl(run_dir / "scores.jsonl", scores)
    failures = _failures(records)
    latencies = [record["latency_ms"] for record in records if record.get("latency_ms") is not None]
    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(time.monotonic() - clock, 3) + manifest.get("elapsed_seconds", 0.0),
            "stopped_reason": "returned_model_id_changed" if aborted else outcome["stopped_reason"],
            "abort_message": aborted,
            "http_requests": manifest.get("http_requests", 0) + outcome["http_requests"],
            "cache_hits": manifest.get("cache_hits", 0) + outcome["cache_hits"],
            "returned_model_ids": dict(
                Counter(record.get("returned_model_id") or "none" for record in records if record.get("status") == "success")
            ),
            "scored": len(scores),
            "failures": failures,
            "status_counts": dict(Counter(record.get("status") for record in records)),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    summary = {
        "dry_run": False,
        "run_dir": str(run_dir),
        "planned_calls": manifest["planned_calls"],
        "scored": len(scores),
        "failed": len(failures),
        "http_requests": manifest["http_requests"],
        "cache_hits": manifest["cache_hits"],
        "returned_model_ids": manifest["returned_model_ids"],
        "stopped_reason": manifest["stopped_reason"],
    }
    if aborted:
        raise ModelIdChanged(f"{aborted}. Records so far are in {run_dir}")
    return summary
