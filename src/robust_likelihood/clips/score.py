"""Phase 2 of docs/clips_small_lm_plan.md: the language-model score of every table key, per model [Frozen 2.3].

score(role, prompt, output) = log P(output + "\n" | prompt), summed over the continuation's tokens. The
output already carries CLIPS's leading space. The prompt and the continuation are tokenized apart, with no
special tokens, and their ids concatenated, so the prompt boundary is exact; no chat template is applied,
for any model. A continuation longer than max_tokens (64) scores -inf, written as null. The score does not
depend on the role, so each distinct (prompt, output) is scored once; scores/<model>.jsonl holds
(prompt_id, output, score), and `table` expands it with prompts.jsonl into the (role, prompt, output,
score) rows TableMixture replays (results/clips/tables/, outside git). Batching is lmcore.surprisals'
(sorted by length, batches of 32, bf16).

Determinism check: determinism_keys random keys (seed 20261126) are scored again in another layout
(reversed order, batch size 7) and the largest absolute difference is reported.
torch and transformers are imported inside the functions only.
"""

from __future__ import annotations

import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl


def load_clips_config(root: Path) -> dict:
    return read_json(root / "configs" / "clips.json")


def distinct_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """Distinct (prompt_id, output) of pairs.jsonl rows."""
    return sorted({(row["prompt_id"], row["output"]) for row in rows})


def encode(tokenizer, prompt: str, output: str, max_tokens: int) -> tuple[list[int], int] | None:
    """(prompt ids + continuation ids, index of the first continuation token), or None past max_tokens."""
    def ids(text: str) -> list[int]:
        found = tokenizer(text, add_special_tokens=False)["input_ids"]
        return list(found.tolist() if hasattr(found, "tolist") else found)

    continuation = ids(output + "\n")
    if len(continuation) > max_tokens:
        return None
    prefix = ids(prompt)
    return prefix + continuation, len(prefix)


def table_rows(rows: list[dict], prompts: dict[str, str], scores: dict[tuple[str, str], float | None]) -> list[dict]:
    """The replay table: every (role, prompt, output) of pairs.jsonl with its pair's score."""
    return [{"role": row["role"], "prompt": prompts[row["prompt_id"]], "output": row["output"],
             "score": scores[(row["prompt_id"], row["output"])]} for row in rows]


def write_table(root: Path, key: str, *, pairs_file: Path, prompts_file: Path, scores_dir: Path, out: Path) -> dict:
    rows = read_jsonl(pairs_file)
    prompts = {row["prompt_id"]: row["prompt"] for row in read_jsonl(prompts_file)}
    scores = {(row["prompt_id"], row["output"]): row["score"] for row in read_jsonl(scores_dir / f"{key}.jsonl")}
    missing = {(row["prompt_id"], row["output"]) for row in rows} - set(scores)
    if missing:
        raise ValueError(f"{key}: {len(missing)} pairs have no score")
    table = table_rows(rows, prompts, scores)
    write_jsonl(out / f"{key}.jsonl", table)
    return {"model": key, "rows": len(table), "path": (out / f"{key}.jsonl").as_posix()}


def clips_score(root: Path, key: str, *, pairs_file: Path, prompts_file: Path, out: Path, device: str | None = None,
                tokenizer=None, model=None, now: datetime | None = None) -> dict:
    import torch
    import transformers

    from robust_likelihood.lmcore import load_model, pick_device, surprisals
    from robust_likelihood.multiwoz.jev_run import git_state

    config = load_clips_config(root)
    spec, settings = config["models"][key], config["score"]
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    rows = read_jsonl(pairs_file)
    prompts = {row["prompt_id"]: row["prompt"] for row in read_jsonl(prompts_file)}
    pairs = distinct_pairs(rows)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if model is None:
        model = load_model(spec["id"], spec["revision"], settings["dtype"], pick_device(device))
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad = pad if pad is not None else 0
    encoded = [encode(tokenizer, prompts[found], output, int(settings["max_tokens"])) for found, output in pairs]
    usable = [index for index, found in enumerate(encoded) if found is not None]
    stage = time.monotonic()
    values = surprisals(model, [encoded[index] for index in usable], int(settings["batch_size"]), pad)
    seconds = time.monotonic() - stage
    scores: dict[tuple[str, str], float | None] = {pair: None for pair in pairs}
    for index, value in zip(usable, values):
        scores[pairs[index]] = -value
    rng = random.Random(f"{config['seed']}:determinism:{key}")
    chosen = rng.sample(usable, min(int(settings["determinism_keys"]), len(usable)))
    again = surprisals(model, [encoded[index] for index in reversed(chosen)], 7, pad)
    gaps = [abs(-value - scores[pairs[index]]) for index, value in zip(reversed(chosen), again)]
    write_jsonl(out / f"{key}.jsonl", [{"prompt_id": found, "output": output, "score": scores[(found, output)]}
                                       for found, output in pairs])
    prompt_tokens = [found[1] for found in encoded if found is not None]
    manifest = {
        "kind": "clips-score", "model_key": key, "model": spec["id"], "revision": spec["revision"],
        "dtype": settings["dtype"], "device": str(next(model.parameters()).device), "git": git_state(root),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "pairs_file": pairs_file.as_posix(), "prompts_file": prompts_file.as_posix(), "table_keys": len(rows),
        "distinct_pairs": len(pairs),
        "over_max_tokens": len(pairs) - len(usable), "max_tokens": settings["max_tokens"],
        "prompt_tokens": {"mean": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
                          "max": max(prompt_tokens) if prompt_tokens else None},
        "scoring_seconds": round(seconds, 1),
        "determinism": {"keys": len(chosen), "layout": "reversed order, batch size 7",
                        "max_abs_difference": max(gaps) if gaps else None},
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1),
    }
    write_json(out / f"{key}.manifest.json", manifest)
    return {"model": key, "distinct_pairs": len(pairs), "scoring_seconds": manifest["scoring_seconds"],
            "determinism_max_abs_difference": manifest["determinism"]["max_abs_difference"]}
