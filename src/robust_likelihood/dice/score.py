"""Scoring the dice queries: token log-probs and verbalized answers, locally or through the APIs.

Every raw answer is kept next to its parsed value. Local runs append each scorer's
rows as soon as that scorer finishes, so a run that dies keeps what it finished. API
answers are cached by provider, requested model id, prompt, and requested temperature,
so the pilot's calls are reused by the full run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.dice import DATA_DIR, RESULTS_DIR
from robust_likelihood.dice.design import S1_SYSTEM, S1_USER, S2_USER, S3_USER, pilot_pairs, s1_messages, verbal_messages
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.storage import append_jsonl, read_jsonl, write_json
from robust_likelihood.verbal import api_client, api_verbal, parse_probability

VERBAL_SCORERS = ("S2", "S3")


class ModelIdChanged(RuntimeError):
    pass


def load_queries(data_dir: Path) -> tuple[list[dict], list[dict]]:
    pairs = read_jsonl(data_dir / "pairs.jsonl")
    norm = read_jsonl(data_dir / "normalization.jsonl")
    if not pairs or not norm:
        raise FileNotFoundError(f"run `python -m robust_likelihood.dice build` first; nothing in {data_dir}")
    return pairs, norm


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _wording(config: dict, hypothesis: str) -> str:
    return config["hypotheses"][hypothesis]["wording"]


def score_local(
    root: Path,
    config: dict,
    model_key: str,
    *,
    scorers: list[str] | None = None,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    device: str | None = None,
    limit: int | None = None,
    label: str | None = None,
    data_dir: Path | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    from robust_likelihood.lmcore import (
        boundary_holds,
        end_of_turn_id,
        load_model,
        pick_device,
        prefix_ids,
        prefix_text,
        reply_ids,
        resolve_revision,
        scored_sequence,
        surprisals,
    )
    from robust_likelihood.verbal import decoding_is_exact, greedy_config, local_verbal

    spec = config["local_models"][model_key]
    scorers = list(scorers or spec["scorers"])
    pairs, norm = load_queries(data_dir or root / DATA_DIR)
    queries = pairs + norm
    if limit is not None:
        queries = queries[:limit]
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    commit = resolve_revision(spec["id"], spec.get("revision"))
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=commit)
    model = load_model(spec["id"], commit, dtype, pick_device(device))
    model.eval()
    first = queries[0]
    end = end_of_turn_id(tokenizer, s1_messages(_wording(config, first["hypothesis"]), first["k"]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
    checks = {}
    if "S1" in scorers:
        checks["boundary_holds"] = all(
            boundary_holds(tokenizer, s1_messages(_wording(config, row["hypothesis"]), row["k"]), row["text"], end)
            for row in queries[:: max(1, len(queries) // 25)]
        )
    verbal = [scorer for scorer in scorers if scorer in VERBAL_SCORERS]
    if verbal:
        probe = verbal_messages(verbal[0], _wording(config, first["hypothesis"]), first["k"], first["text"])
        checks["greedy_is_exact"] = decoding_is_exact(model, prefix_ids(tokenizer, probe), greedy_config(end, pad))
    if not all(checks.values()):
        raise RuntimeError(f"language-model self-check failed: {checks}")
    run_dir = (results_root or root / RESULTS_DIR) / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or 'local-' + model_key}"
    example = queries[0]
    manifest = {
        "kind": "local",
        "model_key": model_key,
        "model": spec["id"],
        "revision_requested": spec.get("revision"),
        "revision": commit,
        "dtype": dtype,
        "device": str(next(model.parameters()).device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "git": git_state(root),
        "config": config,
        "scorers": scorers,
        "prompts": {"S1_system": S1_SYSTEM, "S1_user": S1_USER, "S2_user": S2_USER, "S3_user": S3_USER},
        "rendered_examples": {
            "S1": prefix_text(tokenizer, s1_messages(_wording(config, example["hypothesis"]), example["k"])) + example["text"],
            **{scorer: prefix_text(tokenizer, verbal_messages(scorer, _wording(config, example["hypothesis"]),
                                                              example["k"], example["text"])) for scorer in verbal},
        },
        "decoding": {"S1": "teacher-forced surprisal of the list plus end-of-turn",
                     "verbal": "greedy, do_sample=False, num_beams=1, repetition_penalty=1.0, "
                               f"max_new_tokens={int(config['verbal_max_new_tokens'])}"},
        "end_of_turn_id": end,
        "end_of_turn_token": tokenizer.convert_ids_to_tokens(end),
        "queries": len(queries),
        "checks": checks,
        "started_at": _stamp(started),
        "stage_seconds": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    scores = run_dir / "scores.jsonl"
    if "S1" in scorers:
        stage = time.monotonic()
        keys = sorted({(row["hypothesis"], row["k"], row["text"]) for row in queries})
        sequences = [
            scored_sequence(prefix_ids(tokenizer, s1_messages(_wording(config, hypothesis), k)), reply_ids(tokenizer, text), end)
            for hypothesis, k, text in keys
        ]
        values = dict(zip(keys, surprisals(model, sequences, batch_size, pad)))
        lengths = {key: len(tokens) - start for key, (tokens, start) in zip(keys, sequences)}
        append_jsonl(scores, [
            {"query_id": row["query_id"], "scorer": "S1", "model": model_key,
             "S": values[(row["hypothesis"], row["k"], row["text"])],
             "logp": -values[(row["hypothesis"], row["k"], row["text"])],
             "tokens_scored": lengths[(row["hypothesis"], row["k"], row["text"])]}
            for row in queries
        ])
        manifest["stage_seconds"]["S1"] = round(time.monotonic() - stage, 1)
        manifest["S1_distinct_sequences"] = len(keys)
    for scorer in verbal:
        stage = time.monotonic()
        answers: dict[str, dict] = {}
        rows = []
        for row in queries:
            messages = verbal_messages(scorer, _wording(config, row["hypothesis"]), row["k"], row["text"])
            prompt = messages[-1]["content"]
            if prompt not in answers:
                answers[prompt] = local_verbal(model, tokenizer, messages, end=end, pad=pad,
                                               max_new_tokens=int(config["verbal_max_new_tokens"]))
            answer = answers[prompt]
            rows.append({"query_id": row["query_id"], "scorer": scorer, "model": model_key, "raw": answer["text"],
                         "cut_off": answer["cut_off"], "p": parse_probability(answer["text"])})
        append_jsonl(scores, rows)
        manifest["stage_seconds"][scorer] = round(time.monotonic() - stage, 1)
        manifest[f"{scorer}_distinct_prompts"] = len(answers)
        manifest[f"{scorer}_parse_failures"] = sum(1 for row in rows if row["p"] is None)
        write_json(run_dir / "manifest.json", manifest)
    manifest.update({"finished_at": _stamp(datetime.now(timezone.utc)),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "queries": len(queries), "scorers": scorers, "revision": commit,
            **{key: value for key, value in manifest.items() if key.endswith("_parse_failures")}}


def api_cache_key(provider: str, model_id: str, messages: list[dict], temperature: float | None) -> str:
    payload = json.dumps({"provider": provider, "model": model_id, "messages": messages, "temperature": temperature},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def usage_tokens(provider: str, usage: dict | None) -> tuple[int, int]:
    """Input and output tokens; Claude counts thinking as output, as it is billed."""
    usage = usage or {}
    if provider == "claude":
        cached = int(usage.get("cache_read_input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0)
        return int(usage.get("input_tokens") or 0) + cached, int(usage.get("output_tokens") or 0)
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def api_queries(pairs: list[dict], norm: list[dict], config: dict, pilot: int | None) -> list[dict]:
    if pilot:
        return pilot_pairs(pairs, pilot, int(config["pilot"]["seed"]))
    return [row for row in pairs if row["api_subset"]] + norm


def cost_estimate(config: dict, provider: str, records: list[dict], full_calls: int) -> dict:
    fresh = [record for record in records if record["status"] == "success" and not record["cache_hit"]]
    if not fresh:
        return {"calls_measured": 0}
    tokens = [usage_tokens(provider, record["usage"]) for record in fresh]
    mean_in = sum(value for value, _ in tokens) / len(tokens)
    mean_out = sum(value for _, value in tokens) / len(tokens)
    found = {
        "calls_measured": len(fresh),
        "mean_input_tokens": round(mean_in, 1),
        "mean_output_tokens": round(mean_out, 1),
        "max_output_tokens": max(value for _, value in tokens),
        "full_run_calls": full_calls,
        "full_run_input_tokens": round(mean_in * full_calls),
        "full_run_output_tokens": round(mean_out * full_calls),
        "mean_latency_s": round(sum(record["latency_ms"] or 0 for record in fresh) / len(fresh) / 1000, 2),
    }
    price = (config.get("api_prices_usd_per_mtok") or {}).get(provider)
    if price:
        found["usd_per_mtok"] = price
        found["full_run_usd"] = round(
            (mean_in * price["input"] + mean_out * price["output"]) * full_calls / 1e6, 2
        )
    return found


def score_api(
    root: Path,
    config: dict,
    providers: list[str],
    *,
    pilot: int | None = None,
    workers: int = 4,
    max_requests: int | None = None,
    label: str | None = None,
    dry_run: bool = False,
    data_dir: Path | None = None,
    results_root: Path | None = None,
    transport=None,
    environ=None,
    sleeper=time.sleep,
    now: datetime | None = None,
) -> dict:
    pairs, norm = load_queries(data_dir or root / DATA_DIR)
    queries = api_queries(pairs, norm, config, pilot)
    full_calls = len(api_queries(pairs, norm, config, None)) * len(VERBAL_SCORERS)
    requested_temperature = config.get("api_temperature")
    results_parent = results_root or root / RESULTS_DIR
    cache_path = results_parent / "api_cache.jsonl"
    cache = {record["cache_key"]: record for record in read_jsonl(cache_path) if record["status"] == "success"}
    started = now or datetime.now(timezone.utc)
    if dry_run:
        planned = len(queries) * len(VERBAL_SCORERS) * len(providers)
        first = queries[0]
        return {"dry_run": True, "api_calls": 0, "planned_calls": planned, "queries": len(queries),
                "first_prompts": [verbal_messages(scorer, _wording(config, first["hypothesis"]), first["k"],
                                                  first["text"]) for scorer in VERBAL_SCORERS]}
    run_dir = results_parent / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or ('api-pilot' if pilot else 'api')}"
    manifest = {
        "kind": "api",
        "git": git_state(root),
        "config": config,
        "pilot": pilot,
        "queries": len(queries),
        "scorers": list(VERBAL_SCORERS),
        "prompts": {"S2_user": S2_USER, "S3_user": S3_USER},
        "requested_temperature": requested_temperature,
        "started_at": _stamp(started),
        "providers": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    http_calls = 0
    stopped = None
    for provider in providers:
        client, settings = api_client(root, provider, transport=transport, environ=environ)
        temperature = requested_temperature
        returned_ids: set[str] = set()
        records = []
        jobs = []
        for scorer in VERBAL_SCORERS:
            for row in queries:
                messages = verbal_messages(scorer, _wording(config, row["hypothesis"]), row["k"], row["text"])
                jobs.append((scorer, row, messages, api_cache_key(provider, settings["model_id"], messages,
                                                                  requested_temperature)))

        def ask(messages, temperature=None):
            return api_verbal(provider, messages, client=client, settings=settings, temperature=temperature,
                              sleeper=sleeper)

        pending = []
        for job in jobs:
            if job[3] in cache:
                record = dict(cache[job[3]], cache_hit=True)
                records.append(_score_row(job, provider, record))
            else:
                pending.append(job)
        if max_requests is not None:
            room = max(0, max_requests - http_calls)
            if len(pending) > room:
                stopped = f"max_requests reached; {len(pending) - room} {provider} calls not made"
                pending = pending[:room]
        answered = []
        if pending and temperature is not None:
            # One call first: if the endpoint rejects a temperature, the rest are sent without one.
            job = pending.pop(0)
            answered.append((job, ask(job[2], temperature)))
            if answered[0][1]["temperature_rejected"]:
                temperature = None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [(job, pool.submit(ask, job[2], temperature)) for job in pending]
            answered.extend((job, future.result()) for job, future in futures)
        for job, result in answered:
            http_calls += result["attempts"]
            record = {"cache_key": job[3], "provider": provider, "requested_model_id": settings["model_id"],
                      "messages": job[2], "requested_temperature": requested_temperature, "cache_hit": False,
                      **result}
            if record["status"] == "success":
                append_jsonl(cache_path, [record])
            records.append(_score_row(job, provider, record))
            if record["returned_model_id"]:
                returned_ids.add(record["returned_model_id"])
        records.sort(key=lambda row: (row["scorer"], row["query_id"]))
        append_jsonl(run_dir / "calls.jsonl", records)
        if len(returned_ids) > 1:
            manifest["providers"][provider] = {"returned_model_ids": sorted(returned_ids)}
            write_json(run_dir / "manifest.json", manifest)
            raise ModelIdChanged(f"{provider} returned several model ids: {sorted(returned_ids)}")
        manifest["providers"][provider] = {
            "requested_model_id": settings["model_id"],
            "returned_model_ids": sorted(returned_ids),
            "endpoint": settings["endpoint"],
            "temperature_sent": temperature,
            "temperature_note": None if temperature == requested_temperature else
            "the endpoint rejected a temperature, so none was sent (the model's default sampling applies)",
            "max_tokens": settings["max_tokens"] if provider == "claude" else None,
            "calls": len(records),
            "cache_hits": sum(1 for record in records if record["cache_hit"]),
            "status": {status: sum(1 for record in records if record["status"] == status)
                       for status in sorted({record["status"] for record in records})},
            "parse_failures": sum(1 for record in records if record["status"] == "success" and record["p"] is None),
            "cost_estimate": cost_estimate(config, provider, records, full_calls),
        }
        write_json(run_dir / "manifest.json", manifest)
    manifest.update({"finished_at": _stamp(datetime.now(timezone.utc)), "http_calls": http_calls, "stopped": stopped})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "http_calls": http_calls, "stopped": stopped,
            "providers": {provider: {key: found[key] for key in ("calls", "status", "parse_failures", "cost_estimate",
                                                                 "temperature_sent", "returned_model_ids")}
                          for provider, found in manifest["providers"].items()}}


def _score_row(job, provider: str, record: dict) -> dict:
    scorer, row, _, _ = job
    text = record.get("text")
    return {
        "query_id": row["query_id"],
        "scorer": scorer,
        "model": provider,
        "status": record["status"],
        "error": record.get("error"),
        "raw": text,
        "p": parse_probability(text) if record["status"] == "success" else None,
        "returned_model_id": record.get("returned_model_id"),
        "usage": record.get("usage"),
        "temperature_sent": record.get("temperature_sent"),
        "latency_ms": record.get("latency_ms"),
        "cache_hit": record["cache_hit"],
    }
