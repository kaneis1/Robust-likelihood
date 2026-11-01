"""Claims plan 2d, job B: verbalized probabilities on the robustness subset.

One query plan over the seeded 60-item robustness subset, three runners:

- `local` (GPU): Qwen, greedy, on every original (item, hypothesis) pair and every accepted
  rewrite, in two framings. `exact` asks whether the user's first message is exactly U;
  `meaning` asks whether it asks for exactly U's annotated slots and nothing else. An
  utterance rewrite keeps U's annotation, so its meaning prompt is the original's and is
  answered once. Plus `norm`: for 20 seeded items, the exact question for 20 distinct Phase 8
  samples drawn under G|impl; the answers are summed at analysis.
- `api`: claude and gpt, exact framing only, on the original pairs, para1, hpara1, and norm.
- `jev`: the existing `run` path on para1, zh, hpara1, and hzh, through a probe whose rows carry
  the rewritten text and a hyp_key tagged with the rewrite kind (G|impl@para1), so each
  rewrite is its own call. The tag is internal and never sent.

The ground truths, exp(-S) from job A and P_sem from the samples, are read at analysis (2e).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.dice.score import api_cache_key, cost_estimate
from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.hypotheses import NAMES, base, join_names
from robust_likelihood.multiwoz.jev_run import git_state, run_jev
from robust_likelihood.multiwoz.semantic import DONTCARE, NONE, annotated_tuple, load_vocab
from robust_likelihood.storage import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from robust_likelihood.verbal import VERBAL_MAX_NEW_TOKENS, api_client, api_verbal, parse_probability

# [Frozen] prompts of the plan (2d). The goal sentence is the hypothesis text itself, which already
# ends a sentence, so no second period is added (docs/claims_decisions.md).
GOAL = "A user of a restaurant booking assistant has the following goal: {hypothesis}"
EXACT = (" What is the probability that this user's first message to the assistant is exactly: \"{utterance}\"?"
         " Answer with a single probability between 0 and 1 and nothing else.")
MEANING = (" What is the probability that this user's first message to the assistant asks for exactly the"
           " following and nothing else: {description}? Answer with a single probability between 0 and 1 and"
           " nothing else.")
SENTENCE_END = (".", "!", "?", "。", "！", "？")
FRAMINGS = ("exact", "meaning")
API_SOURCES = ("original", "para1", "hpara1", "norm")
JEV_SOURCES = ("para1", "zh", "hpara1", "hzh")
API_PROVIDERS = ("claude", "gpt")


def goal_sentence(hypothesis: str) -> str:
    text = hypothesis.strip()
    return GOAL.format(hypothesis=text if text.endswith(SENTENCE_END) else text + ".")


def exact_prompt(hypothesis: str, utterance: str) -> str:
    return goal_sentence(hypothesis) + EXACT.format(utterance=utterance)


def meaning_prompt(hypothesis: str, description: str) -> str:
    return goal_sentence(hypothesis) + MEANING.format(description=description)


def slot_description(annotated: dict[str, list[str]], goal: dict) -> str:
    """U's annotated slots in the hypothesis templates' words: "a cheap restaurant serving italian food
    in the north of town", then ", with no preference about ..." for slots the user left open on purpose.
    Of several annotated values, the goal's is used when it is one of them, else the first."""
    values = {}
    for slot in SLOTS:
        found = [value for value in annotated[slot] if value not in (NONE, DONTCARE)]
        if found:
            wanted = str(goal.get(slot, "")).lower()
            values[slot] = wanted if wanted in found else found[0]
    phrase = base(values)[len("The user wants to find "):-1]
    open_slots = [NAMES[slot] for slot in SLOTS if annotated[slot] == [DONTCARE]]
    if open_slots:
        phrase += f", with no preference about {join_names(open_slots)}"
    return phrase


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def plan_queries(
    probe: list[dict],
    rewrites: list[dict],
    items: list[dict],
    subset: list[str],
    vocab: dict,
    samples: list[dict] | None,
    *,
    norm_items: int,
    norm_samples: int,
    seed: int,
) -> list[dict]:
    """Every (framing, source, item, hypothesis, rewrite) query, item-major, with its prompt."""
    chosen = sorted(set(subset))
    by_item = {item["item_id"]: item for item in items}
    rows_by_item: dict[str, list[dict]] = {}
    for row in probe:
        if row["item_id"] in by_item and row["item_id"] in chosen:
            rows_by_item.setdefault(row["item_id"], []).append(row)
    hyp_rewrites: dict[str, list[dict]] = {}
    utt_rewrites: dict[str, list[dict]] = {}
    for rewrite in rewrites:
        if rewrite["side"] == "hypothesis":
            hyp_rewrites.setdefault(rewrite["original"], []).append(rewrite)
        else:
            utt_rewrites.setdefault(rewrite["item_id"], []).append(rewrite)
    queries: list[dict] = []

    def add(framing, source, side, row, hypothesis, utterance, description, unit_id=None, sample=None):
        query = {
            "query_id": "|".join([framing, source, row["item_id"], row["hyp_key"], unit_id or "-",
                                  "-" if sample is None else str(sample)]),
            "framing": framing, "source": source, "side": side, "item_id": row["item_id"],
            "hyp_key": row["hyp_key"], "unit_id": unit_id, "sample": sample,
            "hypothesis": hypothesis, "utterance": utterance,
        }
        if framing == "meaning":
            query["description"] = description
            query["prompt"] = meaning_prompt(hypothesis, description)
        else:
            query["prompt"] = exact_prompt(hypothesis, utterance)
        queries.append(query)

    for item_id in chosen:
        item = by_item[item_id]
        description = slot_description(annotated_tuple(item, vocab), item["goal"])
        rows = rows_by_item.get(item_id, [])
        for row in rows:
            for framing in FRAMINGS:
                add(framing, "original", "none", row, row["text"], row["utterance"], description)
            for rewrite in sorted(hyp_rewrites.get(row["text"], []), key=lambda found: found["unit_id"]):
                for framing in FRAMINGS:
                    add(framing, rewrite["kind"], "hypothesis", row, rewrite["rewrite"], row["utterance"],
                        description, rewrite["unit_id"])
        for rewrite in sorted(utt_rewrites.get(item_id, []), key=lambda found: found["unit_id"]):
            for row in rows:
                for framing in FRAMINGS:
                    add(framing, rewrite["kind"], "utterance", row, row["text"], rewrite["rewrite"], description,
                        rewrite["unit_id"])
    if samples is not None and norm_items:
        replies: dict[str, list[dict]] = {}
        for sample in samples:
            replies.setdefault(sample["text"], []).append(sample)
        for item_id in sorted(random.Random(seed).sample(chosen, min(norm_items, len(chosen)))):
            row = next(row for row in rows_by_item[item_id] if row["hyp_key"] == "G|impl")
            seen: set[str] = set()
            for sample in sorted(replies.get(row["text"], []), key=lambda found: found["sample"]):
                if sample["reply"] in seen:
                    continue
                seen.add(sample["reply"])
                add("exact", "norm", "sample", row, row["text"], sample["reply"], None, sample=sample["sample"])
                if len(seen) == norm_samples:
                    break
    return queries


def load_plan(root: Path, phase8_dir: Path | None, derived: Path | None = None) -> tuple[list[dict], dict, dict]:
    derived = derived or root / DERIVED_DIR
    claims = read_json(root / "configs" / "claims.json")
    settings = claims["verbalcheck"]
    samples = read_jsonl(phase8_dir / "lm_samples.jsonl") if phase8_dir is not None else None
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    queries = plan_queries(
        read_jsonl(derived / "probe.jsonl"), read_jsonl(derived / "rewrites.jsonl"),
        read_jsonl(derived / "items.jsonl"), subset, load_vocab(derived), samples,
        norm_items=int(settings["norm_items"]), norm_samples=int(settings["norm_samples"]), seed=int(claims["seed"]),
    )
    inputs = {name: _sha256(derived / name) for name in
              ("probe.jsonl", "rewrites.jsonl", "items.jsonl", "robustness_subset.json", "semantic_vocab.json")}
    if phase8_dir is not None:
        inputs["phase8_dir"] = phase8_dir.as_posix()
        inputs["lm_samples.jsonl"] = _sha256(phase8_dir / "lm_samples.jsonl")
    return queries, claims, inputs


def counts(queries: list[dict]) -> dict:
    found: dict[str, dict[str, int]] = {}
    for query in queries:
        found.setdefault(query["framing"], {}).setdefault(query["source"], 0)
        found[query["framing"]][query["source"]] += 1
    return found


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _messages(query: dict) -> list[dict]:
    return [{"role": "user", "content": query["prompt"]}]


def _row(query: dict, model: str, **found) -> dict:
    keep = ("query_id", "framing", "source", "side", "item_id", "hyp_key", "unit_id", "sample")
    return {key: query[key] for key in keep} | {"model": model} | found


def run_local(
    root: Path,
    model_key: str,
    phase8_dir: Path,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    limit: int | None = None,
    label: str | None = None,
    resume: Path | None = None,
    chunk: int = 250,
    derived: Path | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Greedy answers of a local model to every query, appended in chunks so a stopped run keeps them;
    with resume, a run folder's answered queries are kept and its answers reused."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    from robust_likelihood.lmcore import end_of_turn_id, load_model, pick_device, prefix_ids, prefix_text, resolve_revision
    from robust_likelihood.verbal import decoding_is_exact, greedy_config, local_verbal

    queries, claims, inputs = load_plan(root, phase8_dir, derived)
    if limit is not None:
        queries = queries[:limit]
    spec = claims["models"][model_key]
    max_new_tokens = int(claims["verbalcheck"].get("max_new_tokens", VERBAL_MAX_NEW_TOKENS))
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    commit = resolve_revision(spec["id"], spec["revision"])
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=commit)
    model = load_model(spec["id"], commit, dtype, pick_device(device))
    model.eval()
    end = end_of_turn_id(tokenizer, _messages(queries[0]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
    checks = {"greedy_is_exact": decoding_is_exact(model, prefix_ids(tokenizer, _messages(queries[0])),
                                                   greedy_config(end, pad, max_new_tokens))}
    if not all(checks.values()):
        raise RuntimeError(f"language-model self-check failed: {checks}")
    if resume is not None:
        run_dir = resume
        manifest = read_json(run_dir / "manifest.json")
        if manifest["inputs"] != inputs:
            raise RuntimeError("the inputs changed since the run started; start a new run instead of resuming")
        manifest.setdefault("resumed_at", []).append(_stamp(started))
    else:
        run_dir = (results_root or root / RESULTS_DIR / "claim1") / (
            f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or 'verbal-' + model_key}")
        examples = {}
        for query in queries:
            key = query["framing"] if query["source"] != "norm" else "norm"
            examples.setdefault(key, prefix_text(tokenizer, _messages(query)))
        manifest = {
            "quantity": "Verbalized probabilities (greedy answers, parsed) for the claims-plan robustness subset.",
            "model_key": model_key, "model": spec["id"], "revision": commit, "dtype": dtype,
            "device": str(next(model.parameters()).device), "git": git_state(root),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__},
            "prompts": {"goal": GOAL, "exact": EXACT, "meaning": MEANING}, "rendered_examples": examples,
            "decoding": f"greedy, do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens={max_new_tokens}",
            "end_of_turn_id": end, "end_of_turn_token": tokenizer.convert_ids_to_tokens(end),
            "inputs": inputs, "planned": counts(queries), "queries": len(queries),
            "distinct_prompts": len({query["prompt"] for query in queries}), "checks": checks,
            "started_at": _stamp(started),
        }
        write_jsonl(run_dir / "queries.jsonl", queries)
    write_json(run_dir / "manifest.json", manifest)
    scores_path = run_dir / "scores.jsonl"
    answers: dict[str, dict] = {}
    done: set[str] = set()
    plan = {query["query_id"]: query for query in queries}
    for row in read_jsonl(scores_path):
        done.add(row["query_id"])
        if row["query_id"] in plan:
            answers[plan[row["query_id"]]["prompt"]] = {"text": row["raw"], "cut_off": row["cut_off"],
                                                       "tokens": row["tokens"]}
    pending: list[dict] = []
    generated = 0
    for query in queries:
        if query["query_id"] in done:
            continue
        if query["prompt"] not in answers:
            answers[query["prompt"]] = local_verbal(model, tokenizer, _messages(query), end=end, pad=pad,
                                                    max_new_tokens=max_new_tokens)
            generated += 1
        answer = answers[query["prompt"]]
        pending.append(_row(query, model_key, raw=answer["text"], cut_off=answer["cut_off"], tokens=answer["tokens"],
                            p=parse_probability(answer["text"])))
        if len(pending) >= chunk:
            append_jsonl(scores_path, pending)
            pending = []
    append_jsonl(scores_path, pending)
    rows = read_jsonl(scores_path)
    manifest.update({
        "finished_at": _stamp(datetime.now(timezone.utc)),
        "elapsed_seconds": round(time.monotonic() - clock, 1) + manifest.get("elapsed_seconds", 0.0),
        "generated_this_session": generated, "scored": len(rows),
        "parse_failures": sum(1 for row in rows if row["p"] is None),
        "cut_off": sum(1 for row in rows if row["cut_off"]),
    })
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "queries": len(queries), "scored": len(rows), "generated": generated,
            "parse_failures": manifest["parse_failures"], "revision": commit}


def api_queries(queries: list[dict], pilot: int | None, seed: int) -> list[dict]:
    chosen = [query for query in queries if query["framing"] == "exact" and query["source"] in API_SOURCES]
    if pilot:
        chosen = sorted(random.Random(seed).sample(chosen, min(pilot, len(chosen))), key=lambda query: query["query_id"])
    return chosen


def run_api(
    root: Path,
    providers: list[str],
    phase8_dir: Path,
    *,
    pilot: int | None = None,
    workers: int = 4,
    max_requests: int | None = None,
    dry_run: bool = False,
    label: str | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    transport=None,
    environ=None,
    sleeper=time.sleep,
    now: datetime | None = None,
) -> dict:
    """Exact-framing answers from the API models; answers are cached by provider, model id, prompt, and
    requested temperature, so a pilot's calls are reused by the full run."""
    unknown = set(providers) - set(API_PROVIDERS)
    if unknown:
        raise ValueError(f"API providers are {API_PROVIDERS}, not {sorted(unknown)}")
    plan, claims, inputs = load_plan(root, phase8_dir, derived)
    seed = int(claims["seed"])
    queries = api_queries(plan, pilot, seed)
    full_calls = len(api_queries(plan, None, seed))
    requested_temperature = claims["verbalcheck"].get("api_temperature", 0.0)
    if dry_run:
        return {"dry_run": True, "api_calls": 0, "queries_per_provider": len(queries), "providers": providers,
                "planned_calls": len(queries) * len(providers), "planned": counts(queries),
                "first_prompts": [query["prompt"] for query in queries[:2]]}
    results_parent = results_root or root / RESULTS_DIR / "claim1"
    cache_path = results_parent / "api_cache.jsonl"
    cache = {record["cache_key"]: record for record in read_jsonl(cache_path) if record["status"] == "success"}
    started = now or datetime.now(timezone.utc)
    run_dir = results_parent / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or ('verbal-api-pilot' if pilot else 'verbal-api')}"
    prices = {"api_prices_usd_per_mtok": read_json(root / "configs" / "dice.json").get("api_prices_usd_per_mtok", {})}
    manifest = {"kind": "api", "git": git_state(root), "pilot": pilot, "queries": len(queries),
                "planned": counts(queries), "inputs": inputs,
                "prompts": {"goal": GOAL, "exact": EXACT}, "requested_temperature": requested_temperature,
                "started_at": _stamp(started), "providers": {}}
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "queries.jsonl", queries)
    http_calls = 0
    stopped = None
    for provider in providers:
        client, settings = api_client(root, provider, transport=transport, environ=environ)
        temperature = requested_temperature
        jobs = [(query, api_cache_key(provider, settings["model_id"], _messages(query), requested_temperature))
                for query in queries]
        records, pending = [], []
        for query, key in jobs:
            if key in cache:
                records.append(_api_row(query, provider, dict(cache[key], cache_hit=True)))
            else:
                pending.append((query, key))
        if max_requests is not None:
            room = max(0, max_requests - http_calls)
            if len(pending) > room:
                stopped = f"max_requests reached; {len(pending) - room} {provider} calls not made"
                pending = pending[:room]

        def ask(query, temperature=None):
            return api_verbal(provider, _messages(query), client=client, settings=settings, temperature=temperature,
                              sleeper=sleeper)

        answered = []
        if pending and temperature is not None:
            # One call first: if the endpoint rejects a temperature, the rest are sent without one.
            query, key = pending.pop(0)
            answered.append((query, key, ask(query, temperature)))
            if answered[0][2]["temperature_rejected"]:
                temperature = None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [(query, key, pool.submit(ask, query, temperature)) for query, key in pending]
            answered.extend((query, key, future.result()) for query, key, future in futures)
        returned_ids: set[str] = set()
        for query, key, result in answered:
            http_calls += result["attempts"]
            record = {"cache_key": key, "provider": provider, "requested_model_id": settings["model_id"],
                      "messages": _messages(query), "requested_temperature": requested_temperature, "cache_hit": False,
                      **result}
            if record["status"] == "success":
                append_jsonl(cache_path, [record])
            records.append(_api_row(query, provider, record))
        for record in records:
            if record["returned_model_id"]:
                returned_ids.add(record["returned_model_id"])
        records.sort(key=lambda row: row["query_id"])
        append_jsonl(run_dir / "scores.jsonl", records)
        manifest["providers"][provider] = {
            "requested_model_id": settings["model_id"], "returned_model_ids": sorted(returned_ids),
            "endpoint": settings["endpoint"], "temperature_sent": temperature,
            "max_tokens": settings["max_tokens"] if provider == "claude" else None,
            "calls": len(records), "cache_hits": sum(1 for record in records if record["cache_hit"]),
            "status": {status: sum(1 for record in records if record["status"] == status)
                       for status in sorted({record["status"] for record in records})},
            "parse_failures": sum(1 for record in records if record["status"] == "success" and record["p"] is None),
            "cost_estimate": cost_estimate(prices, provider, records, full_calls),
        }
        write_json(run_dir / "manifest.json", manifest)
        if len(returned_ids) > 1:
            raise RuntimeError(f"{provider} returned several model ids: {sorted(returned_ids)}; see {run_dir}")
    manifest.update({"finished_at": _stamp(datetime.now(timezone.utc)), "http_calls": http_calls, "stopped": stopped})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "http_calls": http_calls, "stopped": stopped,
            "providers": {provider: {key: found[key] for key in ("calls", "status", "parse_failures", "cost_estimate",
                                                                 "temperature_sent", "returned_model_ids")}
                          for provider, found in manifest["providers"].items()}}


def _api_row(query: dict, provider: str, record: dict) -> dict:
    text = record.get("text")
    return _row(query, provider, status=record["status"], error=record.get("error"), raw=text,
                p=parse_probability(text) if record["status"] == "success" else None,
                returned_model_id=record.get("returned_model_id"), usage=record.get("usage"),
                temperature_sent=record.get("temperature_sent"), latency_ms=record.get("latency_ms"),
                cache_hit=record["cache_hit"])


def jev_probe(queries: list[dict], probe: list[dict]) -> list[dict]:
    """Probe rows for the Jev rewrite run: the rewritten text in place, the kind tagged onto hyp_key."""
    rows = {(row["item_id"], row["hyp_key"]): row for row in probe}
    found = []
    for query in queries:
        if query["framing"] == "exact" and query["source"] in JEV_SOURCES:
            row = rows[(query["item_id"], query["hyp_key"])]
            found.append(dict(row, text=query["hypothesis"], utterance=query["utterance"],
                              hyp_key=f"{row['hyp_key']}@{query['source']}", rewrite_unit=query["unit_id"]))
    return found


def run_jev_rewrites(
    root: Path,
    config: dict,
    *,
    wording: str = "both",
    max_requests: int | None = None,
    dry_run: bool = False,
    resume: Path | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    transport=None,
    environ=None,
    sleeper=None,
    now: datetime | None = None,
) -> dict:
    derived = derived or root / DERIVED_DIR
    queries, _, _ = load_plan(root, None, derived)
    subset = set(json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8")))
    results_parent = results_root or root / RESULTS_DIR
    inputs = results_parent / "claim1" / "jev_input"
    write_jsonl(inputs / "probe.jsonl", jev_probe(queries, read_jsonl(derived / "probe.jsonl")))
    write_jsonl(inputs / "items.jsonl", [item for item in read_jsonl(derived / "items.jsonl") if item["item_id"] in subset])
    return run_jev(root, config, wording=wording, set_name="clean", max_requests=max_requests, dry_run=dry_run,
                   resume=resume, label="jev-rewrites", derived=inputs, results_root=results_parent,
                   transport=transport, environ=environ, sleeper=sleeper, now=now)
