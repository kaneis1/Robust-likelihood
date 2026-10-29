"""Surprisal and exact sampling from an open chat model, shared by every experiment.

Needs the `lm` extra (torch, transformers, accelerate). Nothing heavy is imported at
module load, so the package and its tests run without them.

A prompt is a list of chat messages. S(reply | messages) is the surprisal of the reply
as the assistant's next turn: the reply tokens plus the end-of-turn token, and nothing
before them. Sampling replaces the model's own generation defaults (Qwen ships top_p
0.8, top_k 20, repetition penalty 1.05) with explicit settings, and sampling_is_exact
checks that no logits processor or warper changes the distribution being scored.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

PROBE_MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "h"}]


def _ids(encoded) -> list[int]:
    """apply_chat_template returns a list or a BatchEncoding, depending on the transformers version."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    return _ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))


def prefix_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def reply_ids(tokenizer, reply: str) -> list[int]:
    return _ids(tokenizer(reply, add_special_tokens=False))


def end_of_turn_id(tokenizer, messages: list[dict] | None = None) -> int:
    """The token that closes an assistant turn in the chat template (<|im_end|> for Qwen)."""
    messages = messages or PROBE_MESSAGES
    prefix = prefix_text(tokenizer, messages)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": "x"}], tokenize=False)
    if not full.startswith(prefix + "x"):
        raise ValueError("the chat template does not put the assistant reply right after the generation prompt")
    return reply_ids(tokenizer, full[len(prefix) + 1:])[0]


def scored_sequence(prefix: list[int], reply: list[int], end: int) -> tuple[list[int], int]:
    """Prefix, reply, and end-of-turn token, with the index of the first scored token."""
    return prefix + reply + [end], len(prefix)


def boundary_holds(tokenizer, messages: list[dict], reply: str, end: int) -> bool:
    """Tokenizing prefix and reply apart gives the same ids as tokenizing the whole conversation."""
    whole = reply_ids(tokenizer, prefix_text(tokenizer, messages) + reply) + [end]
    sequence, _ = scored_sequence(prefix_ids(tokenizer, messages), reply_ids(tokenizer, reply), end)
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
