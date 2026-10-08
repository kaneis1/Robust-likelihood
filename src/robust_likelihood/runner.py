from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.clients import ModelClient, UrllibTransport
from robust_likelihood.constants import CONNECTIVITY_UTTERANCES, QUANTITY
from robust_likelihood.evaluate import write_evaluation
from robust_likelihood.keys import load_keys
from robust_likelihood.pilot import load_pilot
from robust_likelihood.planning import (
    CallSpec,
    assert_comparison_pins,
    build_plan,
    cache_key,
    inference_settings_for,
    summarize_plan,
)
from robust_likelihood.synthetic import build_synthetic_plan, load_synthetic
from robust_likelihood.synthetic_families import (
    build_family_plan,
    load_synthetic_families,
    select_split,
)
from robust_likelihood.synthetic_v2 import build_synthetic_v2_plan, load_synthetic_v2
from robust_likelihood.toy_likelihood import build_toy_plan, load_toy_likelihood
from robust_likelihood.prompting import WORDING_NOTES, load_hypotheses, load_prompts
from robust_likelihood.storage import append_jsonl, read_json, read_jsonl, write_json


class UnreviewedDrafts(Exception):
    def __init__(self, example_ids: list[str]):
        self.example_ids = list(example_ids)
        super().__init__(
            f"Unreviewed draft inputs: {len(self.example_ids)} items are not approved. "
            "Pass --allow-unreviewed to mark an exploratory run."
        )


def is_approved(row: dict | None) -> bool:
    if not isinstance(row, dict):
        return False
    reviewer = row.get("reviewer")
    reviewed_at = row.get("reviewed_at")
    return (
        row.get("approved") is True
        and isinstance(reviewer, str)
        and bool(reviewer.strip())
        and isinstance(reviewed_at, str)
        and bool(reviewed_at.strip())
        and isinstance(row.get("label_changed"), bool)
    )


def unapproved_ids(items: list[dict], review: dict) -> list[str]:
    by_id = {row.get("example_id"): row for row in review.get("items", [])}
    return [item["example_id"] for item in items if not is_approved(by_id.get(item["example_id"]))]


def build_clients(config: dict, keys, providers, transport=None) -> dict:
    transport = transport or UrllibTransport()
    clients = {}
    for provider in providers:
        clients[provider] = ModelClient(
            provider=provider,
            api_key=keys.require(provider),
            endpoint=config["endpoints"][provider],
            transport=transport,
            timeout=float(config["timeout_seconds"]),
            anthropic_version=config.get("anthropic_version", "2023-06-01"),
        )
    return clients


def _record(spec: CallSpec, result, attempt_id: int, key: str, *, cache_hit: bool) -> dict:
    return {
        "call_id": f"{key}:{attempt_id}",
        "cache_key": key,
        "cache_hit": cache_hit,
        "attempt_id": attempt_id,
        "repeat_id": spec.repeat_id,
        "example_id": spec.example_id,
        "original_example_id": spec.original_example_id,
        "transformation": spec.transformation,
        "task": spec.task,
        "hypothesis": spec.hypothesis,
        "hypothesis_wording": spec.hypothesis_wording,
        "hypothesis_description_version": spec.hypothesis_description_version,
        "input_text": spec.input_text,
        "prompt_version": spec.prompt_version,
        "prompt_text": result.prompt_text,
        "provider": spec.provider,
        "requested_model_id": spec.requested_model_id,
        "returned_model_id": result.returned_model_id,
        "inference_settings": spec.inference_settings,
        "intent": spec.intent,
        "original_intent": spec.original_intent,
        "label_changed": spec.label_changed,
        "dataset": spec.dataset,
        "family": spec.family,
        "pair_id": spec.pair_id,
        "role": spec.role,
        "massive_lump_label": spec.massive_lump_label,
        "request": result.request_body,
        "response": result.response_body,
        "parsed": result.parsed,
        "latency_ms": None if cache_hit else result.latency_ms,
        "usage": result.usage,
        "error": result.error,
        "status": result.status,
        "retryable": result.retryable,
        "http_status": result.http_status,
    }


def _success_map(records: list[dict]) -> dict[str, dict]:
    found = {}
    for record in records:
        if record.get("status") == "success" and record.get("cache_key"):
            found[record["cache_key"]] = record
    return found


def execute(
    specs: list[CallSpec],
    clients: dict,
    prompts,
    hypotheses,
    *,
    max_requests: int | None,
    max_retries: int,
    backoff_seconds: float,
    sleeper,
    success_cache: dict[str, dict],
    skip_keys: set[str],
    on_record=None,
) -> dict:
    new_records = []
    http_calls = 0
    cache_hits = 0
    stopped = None

    def try_consume() -> bool:
        nonlocal http_calls
        if max_requests is not None and http_calls >= max_requests:
            return False
        http_calls += 1
        return True

    for spec in specs:
        key = cache_key(spec)
        if key in skip_keys:
            continue
        cached = success_cache.get(key)
        if cached and cached.get("status") == "success":
            copied = dict(cached)
            copied["cache_hit"] = True
            copied["latency_ms"] = None
            copied["example_id"] = spec.example_id
            copied["original_example_id"] = spec.original_example_id
            copied["transformation"] = spec.transformation
            copied["family"] = spec.family
            copied["pair_id"] = spec.pair_id
            copied["role"] = spec.role
            copied["dataset"] = spec.dataset
            copied["massive_lump_label"] = spec.massive_lump_label
            copied["intent"] = spec.intent
            copied["original_intent"] = spec.original_intent
            copied["label_changed"] = spec.label_changed
            new_records.append(copied)
            if on_record is not None:
                on_record(copied)
            cache_hits += 1
            continue
        for attempt in range(max_retries + 1):
            if not try_consume():
                stopped = "max_requests"
                break
            if attempt:
                sleeper(backoff_seconds * (2 ** (attempt - 1)))
            result = clients[spec.provider].perform(spec, prompts, hypotheses)
            record = _record(spec, result, attempt, key, cache_hit=False)
            new_records.append(record)
            if on_record is not None:
                on_record(record)
            if result.status == "success" or not result.retryable:
                break
        if stopped:
            break
    return {
        "records": new_records,
        "http_requests": http_calls,
        "stopped_reason": stopped,
        "cache_hits": cache_hits,
    }


def _load_config(root: Path) -> dict:
    return read_json(root / "configs" / "models.json")


def _items(drafts: list[dict], controls: list[dict], include_controls: bool) -> list[dict]:
    if include_controls:
        return drafts + controls
    return list(drafts)


def run_comparison(
    root: Path,
    *,
    providers: tuple[str, ...],
    dry_run: bool = False,
    max_requests: int | None = None,
    allow_unreviewed: bool = False,
    repeats: int = 0,
    hypothesis_version: str | None = None,
    include_controls: bool = False,
    resume: Path | None = None,
    results_root: Path | None = None,
    clients=None,
    environ=None,
    sleeper=None,
    now: datetime | None = None,
    model_id_overrides: dict[str, str] | None = None,
    dataset: str = "massive",
    split: str | None = None,
    prompt_version: str | None = None,
) -> dict:
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be >= 0")
    allowed = {"massive", "synthetic", "synthetic_v2", "synthetic_families", "toy_likelihood"}
    if dataset not in allowed:
        raise ValueError("dataset must be massive, synthetic, synthetic_v2, synthetic_families, or toy_likelihood")
    if split is not None and dataset != "synthetic_families":
        raise ValueError("split is only used with synthetic_families")
    config = _load_config(root)
    overrides = model_id_overrides or {}
    model_ids = {provider: overrides.get(provider, config["models"][provider]) for provider in providers}
    assert_comparison_pins(model_ids)
    chosen_prompt = prompt_version or config["prompt_version"]
    if dataset == "toy_likelihood" and chosen_prompt not in {config["prompt_version"], "toy_v1"}:
        raise ValueError(
            "toy_likelihood keeps its emission-probability question; likely and typical are alarm wording versions"
        )
    prompts = load_prompts(root, chosen_prompt)
    settings = {provider: inference_settings_for(provider, config) for provider in providers}
    synthetic_items = None
    if dataset == "synthetic":
        if include_controls:
            raise ValueError("the synthetic dataset has no MASSIVE meaning-change controls")
        if hypothesis_version not in {None, "synthetic_v1"}:
            raise ValueError("the synthetic dataset uses hypothesis version synthetic_v1")
        items, review, hypotheses, massive_hypotheses = load_synthetic(root)
        synthetic_items = items
        specs = build_synthetic_plan(
            items,
            providers,
            model_ids,
            hypotheses,
            massive_hypotheses,
            prompts.version,
            settings,
            repeats=repeats,
        )
        sampling = {}
        dataset_name = "synthetic_alarm"
    elif dataset == "synthetic_v2":
        if include_controls:
            raise ValueError("synthetic_alarm_v2 has no MASSIVE meaning-change controls")
        if hypothesis_version not in {None, "synthetic_v2"}:
            raise ValueError("synthetic_alarm_v2 uses hypothesis version synthetic_v2")
        if prompts.version not in {"v2", "likely", "typical"}:
            raise ValueError("synthetic_alarm_v2 uses prompt template v2, or the likely or typical wording")
        items, review, atomic_descriptions = load_synthetic_v2(root)
        synthetic_items = items
        specs = build_synthetic_v2_plan(
            items,
            providers,
            model_ids,
            atomic_descriptions,
            prompts.version,
            settings,
            repeats=repeats,
        )
        hypotheses = load_hypotheses(root, "synthetic_v2")
        sampling = {}
        dataset_name = "synthetic_alarm_v2"
    elif dataset == "synthetic_families":
        if include_controls:
            raise ValueError("synthetic_alarm_families has no MASSIVE meaning-change controls")
        if hypothesis_version not in {None, "synthetic_v2"}:
            raise ValueError("synthetic_alarm_families uses the frozen hypothesis version synthetic_v2")
        if prompts.version not in {"v2", "likely", "typical"}:
            raise ValueError("synthetic_alarm_families uses frozen prompt template v2, or the likely or typical wording")
        items, review, atomic_descriptions = load_synthetic_families(root)
        items, review = select_split(items, review, split)
        synthetic_items = items
        specs = build_family_plan(
            items,
            providers,
            model_ids,
            atomic_descriptions,
            prompts.version,
            settings,
            repeats=repeats,
        )
        hypotheses = load_hypotheses(root, "synthetic_v2")
        sampling = {}
        dataset_name = "synthetic_alarm_families"
    elif dataset == "toy_likelihood":
        if include_controls:
            raise ValueError("toy_likelihood has no MASSIVE meaning-change controls")
        if hypothesis_version not in {None, "toy_v1"}:
            raise ValueError("toy_likelihood uses hypothesis version toy_v1")
        items, review, hypotheses, prompts = load_toy_likelihood(root)
        synthetic_items = items
        specs = build_toy_plan(items, providers, model_ids, hypotheses, settings, repeats=repeats)
        sampling = {}
        dataset_name = "toy_likelihood"
    else:
        drafts, controls, review, sampling = load_pilot(root)
        items = _items(drafts, controls, include_controls)
        version = hypothesis_version or config["hypothesis_description_version"]
        hypotheses = load_hypotheses(root, version)
        specs = build_plan(
            items,
            providers,
            model_ids,
            hypotheses.descriptions,
            prompts.version,
            hypotheses.version,
            settings,
            repeats=repeats,
        )
        dataset_name = "massive"
    planned = summarize_plan(specs)
    bad = unapproved_ids(items, review)
    if dry_run:
        summary = {
            "dry_run": True,
            "api_calls": 0,
            "planned_requests": planned,
            "models": model_ids,
            "hypothesis_description_version": hypotheses.version,
            "prompt_version": prompts.version,
            "wording_note": WORDING_NOTES.get(prompts.version),
            "repeats": repeats,
            "include_controls": include_controls,
            "unapproved_items": len(bad),
            "live_run_allowed": allow_unreviewed or not bad,
            "exploratory_if_run": allow_unreviewed,
            "dataset": dataset_name,
            "split": split,
            "quantity": QUANTITY,
            "before_retries": True,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    if bad and not allow_unreviewed:
        raise UnreviewedDrafts(bad)
    if clients is None:
        keys = load_keys(providers, environ=environ, root=root)
        clients = build_clients(config, keys, providers)
    sleeper = sleeper or time.sleep
    results_parent = results_root or (root / "results")
    cache_path = results_parent / "cache.jsonl"
    success_cache = _success_map(read_jsonl(cache_path))
    if resume is not None:
        run_dir = resume
        existing = read_jsonl(run_dir / "responses.jsonl")
        manifest = read_json(run_dir / "run_manifest.json")
    else:
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        if dataset_name == "synthetic_alarm":
            folder = f"synthetic-{stamp}"
        elif dataset_name == "synthetic_alarm_v2":
            folder = f"synthetic-v2-{stamp}"
        elif dataset_name == "synthetic_alarm_families":
            prefix = "families" if split in {None, "all"} else f"families-{split}"
            folder = f"{prefix}-{stamp}"
        elif dataset_name == "toy_likelihood":
            folder = f"toy-{stamp}"
        else:
            folder = stamp
        if prompts.version in {"likely", "typical"}:
            folder = f"{prompts.version}-{folder}"
        if allow_unreviewed:
            folder = f"exploratory-{folder}"
        run_dir = results_parent / folder
        existing = []
        manifest = {
            "dataset": dataset_name,
            "split": split,
            "exploratory": allow_unreviewed,
            "quantity": QUANTITY,
            "prompt_version": prompts.version,
            "wording_note": WORDING_NOTES.get(prompts.version),
            "prompts": {
                "pairwise_question": prompts.pairwise_question,
                "classification_question": prompts.classification_question,
                "chat_pairwise": prompts.chat_pairwise,
                "chat_classification": prompts.chat_classification,
            },
            "hypothesis_description_version": hypotheses.version,
            "hypotheses": hypotheses.descriptions,
            "state_questions": hypotheses.state_questions,
            "items": synthetic_items,
            "dataset_checksum_sha256": sampling.get("dataset_checksum_sha256"),
            "dataset_revision": sampling.get("dataset_revision"),
            "seed": sampling.get("seed"),
            "inference_settings": settings,
            "requested_model_ids": model_ids,
            "repeats": repeats,
            "include_controls": include_controls,
            "max_requests": max_requests,
            "timeout_seconds": config["timeout_seconds"],
            "max_retries": config["max_retries"],
            "planned_requests": planned,
            "stopped_reason": None,
            "http_requests": 0,
        }
        write_json(run_dir / "run_manifest.json", manifest)
    skip_keys = set(_success_map(existing))
    responses_path = run_dir / "responses.jsonl"

    def _keep(record: dict) -> None:
        append_jsonl(responses_path, [record])
        if record.get("status") == "success" and not record.get("cache_hit"):
            append_jsonl(cache_path, [record])

    outcome = execute(
        specs,
        clients,
        prompts,
        hypotheses,
        max_requests=max_requests,
        max_retries=int(config["max_retries"]),
        backoff_seconds=float(config["retry_backoff_seconds"]),
        sleeper=sleeper,
        success_cache=success_cache,
        skip_keys=skip_keys,
        on_record=_keep,
    )
    if not responses_path.exists():
        append_jsonl(responses_path, [])
    manifest["stopped_reason"] = outcome["stopped_reason"]
    manifest["http_requests"] = manifest.get("http_requests", 0) + outcome["http_requests"]
    manifest["cache_hits"] = outcome["cache_hits"]
    write_json(run_dir / "run_manifest.json", manifest)
    metrics = write_evaluation(run_dir)
    summary = {
        "dry_run": False,
        "run_dir": str(run_dir),
        "exploratory": manifest.get("exploratory"),
        "http_requests": outcome["http_requests"],
        "cache_hits": outcome["cache_hits"],
        "stopped_reason": outcome["stopped_reason"],
        "planned_requests": planned,
        "disagreement_count": metrics.get("disagreement_count"),
        "quantity": QUANTITY,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def connect_models(
    root: Path,
    providers: tuple[str, ...],
    *,
    jev_model: str | None = None,
    environ=None,
    clients=None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """One connectivity call per selected model. This does not start the comparison."""
    config = _load_config(root)
    model_ids = {provider: config["models"][provider] for provider in providers}
    if jev_model and "jev" in model_ids:
        model_ids["jev"] = jev_model
    if clients is None:
        keys = load_keys(providers, environ=environ, root=root)
        clients = build_clients(config, keys, providers)
    calls = []
    for provider in providers:
        settings = {} if provider == "jev" else inference_settings_for(provider, config)
        result = clients[provider].probe(model_ids[provider], CONNECTIVITY_UTTERANCES, settings)
        calls.append(
            {
                "provider": provider,
                "requested_model_id": model_ids[provider],
                "returned_model_id": result.returned_model_id,
                "status": result.status,
                "http_status": result.http_status,
                "error": result.error,
                "response": result.response_body,
            }
        )
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    destination = (results_root or (root / "results")) / f"connectivity-{stamp}.json"
    payload = {
        "utterances": list(CONNECTIVITY_UTTERANCES),
        "calls": calls,
        "quantity": QUANTITY,
    }
    write_json(destination, payload)
    print(json.dumps({"connectivity": str(destination), "calls": len(calls)}, indent=2))
    return payload
