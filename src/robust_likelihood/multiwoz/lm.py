"""Phase 8: surprisal and normalized typicality from an open language model.

Needs the `lm` extra (torch, transformers, accelerate). Nothing heavy is imported at
module load, so the package and its tests run without them.

S(u|h) is the surprisal of utterance u as the assistant's reply to a fixed prompt that
describes the user with hypothesis text h: the reply tokens plus the end-of-turn token,
and nothing before them. For each distinct h, K replies are sampled from the same prefix
at temperature 1 with no top-k, top-p, or repetition penalty, so the samples come from
the distribution S measures. Qwen's own defaults (top_p 0.8, top_k 20, repetition
penalty 1.05) would not, so they are replaced and the replacement is checked: with these
settings the processed scores of a short generation must equal the raw logits.
"""

from __future__ import annotations

import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood import lmcore
from robust_likelihood.lmcore import (
    load_model,
    pick_device,
    reply_ids,
    resolve_revision,
    sample_replies,
    sampling_config,
    sampling_is_exact,
    scored_sequence,
    surprisals,
    text_seed,
)
from robust_likelihood.multiwoz import RESULTS_DIR
from robust_likelihood.storage import read_jsonl, write_json, write_jsonl

SYSTEM_PROMPT = "You simulate users of a restaurant booking assistant."
USER_PROMPT = (
    "Description of the user: {text}\n"
    "Write this user's first message to the assistant. Reply with the message only."
)
EXPLICIT_MARK = "The user has no preference about"


def chat(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(text=text)},
    ]


def prefix_ids(tokenizer, text: str) -> list[int]:
    return lmcore.prefix_ids(tokenizer, chat(text))


def end_of_turn_id(tokenizer) -> int:
    return lmcore.end_of_turn_id(tokenizer, chat("h"))


def boundary_holds(tokenizer, text: str, utterance: str, end: int) -> bool:
    return lmcore.boundary_holds(tokenizer, chat(text), utterance, end)


def bin_index(value: float, edges: tuple[float, ...]) -> int:
    return sum(1 for edge in edges if abs(value) >= edge)


def typicality_scores(surprisal: float, sample_surprisals: list[float], edges: tuple[float, ...]) -> dict:
    """lm_loglik = -S; T = (S - mu_h) / sigma_h; lm_typ_normal = -T^2/2; lm_typ_bins = ln mass of |T|'s bin.

    The bin mass is counted among h's own samples with add-one smoothing over the
    len(edges) + 1 bins: (count + 1) / (K + bins), K the number of kept samples.
    """
    kept = len(sample_surprisals)
    mu = sum(sample_surprisals) / kept
    sigma = math.sqrt(sum((value - mu) ** 2 for value in sample_surprisals) / (kept - 1)) if kept > 1 else 0.0
    found = {"S": surprisal, "lm_loglik": -surprisal, "mu": mu, "sigma": sigma, "T": None,
             "lm_typ_normal": None, "lm_typ_bins": None}
    if sigma > 0:
        t = (surprisal - mu) / sigma
        target = bin_index(t, edges)
        count = sum(1 for value in sample_surprisals if bin_index((value - mu) / sigma, edges) == target)
        found.update({
            "T": t,
            "lm_typ_normal": -t * t / 2,
            "lm_typ_bins": math.log((count + 1) / (kept + len(edges) + 1)),
        })
    return found


def hypothesis_shape(rows: list[dict]) -> dict[str, dict]:
    """Slots stated and wording of every distinct hypothesis text."""
    shapes = {}
    for row in rows:
        shapes.setdefault(row["text"], {
            "n_constrained": len(row["constrained"]),
            "regime": "expl" if EXPLICIT_MARK in row["text"] else "impl",
            "hyp_types": set(),
        })["hyp_types"].add(row["hyp_type"])
    return shapes


def lm_score(
    root: Path,
    config: dict,
    probe: Path,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    samples: int | None = None,
    batch_size: int = 16,
    limit_texts: int | None = None,
    label: str = "lm",
    dtype: str = "bfloat16",
    device: str | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    settings = dict(config["lm"])
    model_name = model_name or settings["model"]
    count = int(samples or settings["samples_per_hypothesis"])
    edges = tuple(float(edge) for edge in settings["bin_edges"])
    seed = int(settings["seed"])
    rows = read_jsonl(probe)
    shapes = hypothesis_shape(rows)
    texts = sorted(shapes)
    if limit_texts is not None:
        texts = texts[:limit_texts]
        rows = [row for row in rows if row["text"] in set(texts)]
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    commit = resolve_revision(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit)
    model = load_model(model_name, commit, dtype, pick_device(device))
    model.eval()
    end = end_of_turn_id(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
    generation = sampling_config(settings, end, pad)
    model.generation_config = generation
    first_prefix = prefix_ids(tokenizer, texts[0])
    checks = {
        "sampling_is_exact": sampling_is_exact(model, first_prefix, generation),
        "boundary_holds": all(boundary_holds(tokenizer, row["text"], row["utterance"], end) for row in rows[:25]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"language-model self-check failed: {checks}")
    run_dir = (results_root or root / RESULTS_DIR) / f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label}"
    manifest = {
        "quantity": "Language-model surprisal and normalized typicality, in nats. Not Jev scores.",
        "model": model_name,
        "revision_requested": revision,
        "revision": commit,
        "dtype": dtype,
        "device": str(next(model.parameters()).device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "prompt": {"system": SYSTEM_PROMPT, "user": USER_PROMPT, "assistant": "{utterance}"},
        "end_of_turn_id": end,
        "end_of_turn_token": tokenizer.convert_ids_to_tokens(end),
        "sampling": {
            key: getattr(generation, key)
            for key in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens",
                        "eos_token_id", "pad_token_id")
        },
        "samples_per_hypothesis": count,
        "seed": seed,
        "seed_rule": "torch.manual_seed(int(sha256(f'{seed}:{text}').hexdigest()[:8], 16)) per hypothesis text",
        "bin_edges": list(edges),
        "texts": len(texts),
        "rows": len(rows),
        "checks": checks,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(run_dir / "manifest.json", manifest)
    statistics = {}
    sample_rows = []
    for text in texts:
        prefix = prefix_ids(tokenizer, text)
        replies, dropped = sample_replies(model, prefix, count, generation, text_seed(seed, text), end)
        values = surprisals(model, [scored_sequence(prefix, reply, end) for reply in replies], batch_size, pad)
        statistics[text] = {"values": values, "kept": len(replies), "dropped": dropped}
        sample_rows.extend(
            {"text": text, "sample": index, "reply": tokenizer.decode(reply), "S": value}
            for index, (reply, value) in enumerate(zip(replies, values))
        )
    sequences = [
        scored_sequence(prefix_ids(tokenizer, row["text"]), reply_ids(tokenizer, row["utterance"]), end) for row in rows
    ]
    scored = surprisals(model, sequences, batch_size, pad)
    score_rows = []
    for row, value, (tokens, first) in zip(rows, scored, sequences):
        found = typicality_scores(value, statistics[row["text"]]["values"], edges)
        score_rows.append({
            "item_id": row["item_id"],
            "set": row["set"],
            "hyp_key": row["hyp_key"],
            "tokens_scored": len(tokens) - first,
            **found,
        })
    hypotheses = []
    for text in texts:
        found = statistics[text]
        values = found["values"]
        mu = sum(values) / len(values) if values else None
        sigma = (
            math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
        )
        hypotheses.append({
            "text": text,
            "n_constrained": shapes[text]["n_constrained"],
            "regime": shapes[text]["regime"],
            "hyp_types": sorted(shapes[text]["hyp_types"]),
            "mu": mu,
            "sigma": sigma,
            "kept": found["kept"],
            "dropped": found["dropped"],
        })
    write_jsonl(run_dir / "lm_scores.jsonl", score_rows)
    write_jsonl(run_dir / "lm_hypotheses.jsonl", hypotheses)
    write_jsonl(run_dir / "lm_samples.jsonl", sample_rows)
    drawn = count * len(texts)
    dropped = sum(found["dropped"] for found in statistics.values())
    manifest.update({
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1),
        "samples_drawn": drawn,
        "samples_dropped_at_max_new_tokens": dropped,
        "drop_rate": dropped / drawn if drawn else None,
        "rows_without_typicality": sum(1 for row in score_rows if row["T"] is None),
    })
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "rows": len(score_rows), "texts": len(texts), "drop_rate": manifest["drop_rate"],
            "revision": manifest["revision"]}
