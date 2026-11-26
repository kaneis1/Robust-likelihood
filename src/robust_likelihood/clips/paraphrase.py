"""Phase 4 of docs/clips_small_lm_plan.md: para1 and para2 of each instruction, through rewrites.py's pieces.

Each of the 65 instructions gets two paraphrase units. A candidate comes from the `claude` provider
(configs/models.json, the key through keys.py) with perturb.PARAPHRASE and rewrites.REPLY_ONLY, and is
accepted when it changed the text, keeps every protected word (perturb.valid), and passes the pinned NLI
model at rewrites.NLI_THRESHOLD in both directions. A rejected unit is asked again, up to
rewrites.ATTEMPTS attempts in all, as in rewrites.py. Every call and candidate is logged under
results/clips/paraphrases/<UTC time>/.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.clips.perturb import API_KINDS, PARAPHRASE, protected_set, utterance_key, valid
from robust_likelihood.multiwoz.rewrites import ATTEMPTS, NLI, NLI_MODEL, NLI_REVISION, NLI_THRESHOLD, REPLY_ONLY
from robust_likelihood.storage import append_jsonl, read_jsonl, write_json, write_jsonl

CLIPS_ROOT = Path("external") / "CLIPS.jl"


def messages(text: str) -> list[dict]:
    return [{"role": "user", "content": f"{PARAPHRASE} {REPLY_ONLY}\n\n{text}"}]


def paraphrase_units(utterances: list[dict]) -> list[dict]:
    return [{"unit_id": f"{utterance_key(row)}:{kind}", "utterance": utterance_key(row), "domain": row["domain"],
             "plan_id": row["plan_id"], "index": row["index"], "timestep": row["timestep"], "kind": kind,
             "original": row["text"]}
            for row in sorted(utterances, key=utterance_key) for kind in API_KINDS]


def plan(utterances: list[dict]) -> dict:
    units = paraphrase_units(utterances)
    return {"units": len(units), "calls_at_least": len(units) + 1, "calls_at_most": len(units) * ATTEMPTS + 1,
            "note": "one extra call first settles whether the model accepts a temperature"}


def run_paraphrases(root: Path, *, workers: int = 4, nli_device: str | None = None, transport=None, environ=None,
                    sleeper=time.sleep, nli=None, now: datetime | None = None) -> dict:
    from robust_likelihood.multiwoz.jev_run import git_state
    from robust_likelihood.verbal import api_client, api_verbal

    utterances = read_jsonl(root / "data" / "clips" / "derived" / "utterances.jsonl")
    protected = protected_set(root / CLIPS_ROOT)
    units = paraphrase_units(utterances)
    by_id = {unit["unit_id"]: unit for unit in units}
    client, settings = api_client(root, "claude", transport=transport, environ=environ)
    started = now or datetime.now(timezone.utc)
    run_dir = root / "results" / "clips" / "paraphrases" / started.strftime("%Y%m%dT%H%M%SZ")
    manifest = {"kind": "clips-paraphrases", "git": git_state(root), "units": len(units), "provider": "claude",
                "requested_model_id": settings["model_id"], "requested_temperature": 1.0,
                "prompt": f"{PARAPHRASE} {REPLY_ONLY}", "attempts": ATTEMPTS,
                "nli": {"model": NLI_MODEL, "revision": NLI_REVISION, "threshold": NLI_THRESHOLD},
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_json(run_dir / "manifest.json", manifest)
    temperature = {"value": 1.0}
    calls: list[dict] = []

    def ask(text: str) -> dict:
        found = api_verbal("claude", messages(text), client=client, settings=settings,
                           temperature=temperature["value"], sleeper=sleeper)
        if found["temperature_rejected"]:
            temperature["value"] = None
        calls.append({key: found[key] for key in ("status", "usage", "returned_model_id", "temperature_sent",
                                                  "attempts", "latency_ms")})
        return found

    nli = nli if nli is not None else NLI(nli_device)
    ask("Can you pass me the red key?")  # settles the temperature, as rewrites.py does
    accepted: dict[str, dict] = {}
    pending = list(units)
    for attempt in range(ATTEMPTS):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            answers = list(pool.map(lambda unit: ask(unit["original"]), pending))
        records = []
        for unit, answer in zip(pending, answers):
            text = answer["text"].strip() if answer["text"] else None
            records.append({"unit_id": unit["unit_id"], "attempt": attempt, "rewrite": text, "error": answer["error"],
                            "returned_model_id": answer["returned_model_id"],
                            "protected_kept": valid(unit["original"], text, protected)})
        checked = [record for record in records if record["protected_kept"]]
        scores = nli.entailment_probs([(by_id[record["unit_id"]]["original"], record["rewrite"]) for record in checked]
                                      + [(record["rewrite"], by_id[record["unit_id"]]["original"]) for record in checked])
        for index, record in enumerate(checked):
            record["nli_forward"], record["nli_backward"] = scores[index], scores[len(checked) + index]
        for record in records:
            record["accepted"] = bool(record["protected_kept"] and record.get("nli_forward") is not None
                                      and record["nli_forward"] >= NLI_THRESHOLD
                                      and record["nli_backward"] >= NLI_THRESHOLD)
            if record["accepted"]:
                accepted[record["unit_id"]] = record
        append_jsonl(run_dir / "candidates.jsonl", records)
        pending = [unit for unit in pending if unit["unit_id"] not in accepted]
    rows = [{**by_id[unit_id], "rewrite": record["rewrite"], "valid": True, "attempt": record["attempt"],
             "nli_forward": record["nli_forward"], "nli_backward": record["nli_backward"]}
            for unit_id, record in sorted(accepted.items())]
    write_jsonl(run_dir / "accepted.jsonl", rows)
    usage_in = sum(int((call["usage"] or {}).get("input_tokens") or 0) for call in calls)
    usage_out = sum(int((call["usage"] or {}).get("output_tokens") or 0) for call in calls)
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "temperature_sent": temperature["value"], "api_calls": len(calls),
                     "api_failures": sum(1 for call in calls if call["status"] != "success"),
                     "returned_model_ids": sorted({call["returned_model_id"] for call in calls
                                                   if call["returned_model_id"]}),
                     "input_tokens": usage_in, "output_tokens": usage_out, "accepted": len(rows),
                     "accepted_by_kind": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in API_KINDS}})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "units": len(units), "accepted": len(rows), "api_calls": len(calls),
            "input_tokens": usage_in, "output_tokens": usage_out}
