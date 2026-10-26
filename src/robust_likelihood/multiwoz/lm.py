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

import copy
import hashlib
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

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


def _ids(encoded) -> list[int]:
    """apply_chat_template returns a list or a BatchEncoding, depending on the transformers version."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def prefix_ids(tokenizer, text: str) -> list[int]:
    return _ids(tokenizer.apply_chat_template(chat(text), add_generation_prompt=True, tokenize=True))


def reply_ids(tokenizer, utterance: str) -> list[int]:
    return _ids(tokenizer(utterance, add_special_tokens=False))


def end_of_turn_id(tokenizer) -> int:
    """The token that closes an assistant turn in the chat template (<|im_end|> for Qwen)."""
    prefix = tokenizer.apply_chat_template(chat("h"), add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(chat("h") + [{"role": "assistant", "content": "x"}], tokenize=False)
    if not full.startswith(prefix + "x"):
        raise ValueError("the chat template does not put the assistant reply right after the generation prompt")
    return reply_ids(tokenizer, full[len(prefix) + 1:])[0]


def scored_sequence(prefix: list[int], reply: list[int], end: int) -> tuple[list[int], int]:
    """Prefix, reply, and end-of-turn token, with the index of the first scored token."""
    return prefix + reply + [end], len(prefix)


def boundary_holds(tokenizer, text: str, utterance: str, end: int) -> bool:
    """Tokenizing prefix and reply apart gives the same ids as tokenizing the whole conversation."""
    prefix_text = tokenizer.apply_chat_template(chat(text), add_generation_prompt=True, tokenize=False)
    whole = reply_ids(tokenizer, prefix_text + utterance) + [end]
    sequence, _ = scored_sequence(prefix_ids(tokenizer, text), reply_ids(tokenizer, utterance), end)
    return whole == sequence


def surprisals(model, sequences: list[tuple[list[int], int]], batch_size: int, pad_id: int) -> list[float]:
    """-sum log p of the tokens from each sequence's first scored index to its end."""
    import torch

    device = next(model.parameters()).device
    order = sorted(range(len(sequences)), key=lambda index: len(sequences[index][0]))
    found: list[float | None] = [None] * len(sequences)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        width = max(len(sequences[index][0]) for index in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, index in enumerate(chunk):
            tokens = sequences[index][0]
            ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, :len(tokens)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for row, index in enumerate(chunk):
            tokens, first = sequences[index]
            positions = torch.log_softmax(logits[row, first - 1:len(tokens) - 1].float(), dim=-1)
            targets = torch.tensor(tokens[first:], dtype=torch.long, device=positions.device)
            found[index] = -positions.gather(1, targets[:, None]).sum().item()
    return found


def sampling_config(settings: dict, end: int, pad: int):
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=True,
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        top_k=int(settings["top_k"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        max_new_tokens=int(settings["max_new_tokens"]),
        eos_token_id=end,
        pad_token_id=pad,
    )


def with_settings(config, **changes):
    """A copy of a GenerationConfig with fields changed, so generate() gets one config object and no kwargs."""
    updated = copy.deepcopy(config)
    for key, value in changes.items():
        setattr(updated, key, value)
    return updated


def sampling_is_exact(model, prefix: list[int], config) -> bool:
    """No logits processor or warper may change the model's own next-token distribution."""
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    probe = with_settings(
        config, max_new_tokens=3, num_return_sequences=2, output_scores=True, output_logits=True,
        return_dict_in_generate=True,
    )
    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=probe)
    return len(output.scores) == len(output.logits) and all(
        torch.equal(score.float(), logit.float()) for score, logit in zip(output.scores, output.logits)
    )


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_revision(name: str, revision: str | None) -> str:
    """The commit hash a branch or tag points to, read from the snapshot folder name."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=revision, allow_patterns=["config.json"])).name


def load_model(name: str, revision: str | None, dtype: str, device: str):
    """One device, no dispatch: the model fits on one GPU. transformers 4.56 renamed torch_dtype to dtype."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    keyword = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **{keyword: getattr(torch, dtype)})
    return model.to(device)


def text_seed(seed: int, text: str) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:8], 16)


def sample_replies(model, prefix: list[int], count: int, config, seed: int, end: int) -> tuple[list[list[int]], int]:
    """Up to count replies ending at the end-of-turn token; replies cut off by max_new_tokens are dropped."""
    import torch

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        generation_config=with_settings(config, num_return_sequences=count),
    )
    replies = []
    dropped = 0
    for row in output[:, len(prefix):].tolist():
        if end in row:
            replies.append(row[: row.index(end)])
        else:
            dropped += 1
    return replies, dropped


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
