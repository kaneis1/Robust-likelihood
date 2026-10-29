"""Verbalized probabilities: ask a model for a number, keep the raw answer, and parse it.

Local answers come from an open chat model through lmcore, decoded greedily unless
samples are asked for. Every decoding option is set explicitly and installed as the
model's own generation_config: recent transformers versions fill options left at
their global default from the model's config, which for Qwen would add a repetition
penalty of 1.05 even to greedy decoding.

API answers go through the existing transport, keys, endpoints, and model ids in
configs/models.json. The request builders in clients.py render the pilot's prompt
packs, so a free-form chat body is built here instead, without changing them.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from robust_likelihood.clients import _message_text
from robust_likelihood.keys import load_keys
from robust_likelihood.lmcore import prefix_ids, sample_replies, sampling_config, with_settings
from robust_likelihood.runner import build_clients
from robust_likelihood.storage import read_json

VERBAL_MAX_NEW_TOKENS = 48
SAMPLED = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}


def greedy_config(end: int, pad: int, max_new_tokens: int = VERBAL_MAX_NEW_TOKENS):
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=end,
        pad_token_id=pad,
    )


def decoding_is_exact(model, prefix: list[int], config) -> bool:
    """With these settings the processed scores of a short greedy generation equal the raw logits."""
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    probe = with_settings(config, max_new_tokens=3, output_scores=True, output_logits=True, return_dict_in_generate=True)
    model.generation_config = with_settings(config)
    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=probe)
    return len(output.scores) == len(output.logits) and all(
        torch.equal(score.float(), logit.float()) for score, logit in zip(output.scores, output.logits)
    )


def local_verbal(
    model,
    tok,
    messages: list[dict],
    samples: int = 0,
    *,
    end: int,
    pad: int,
    max_new_tokens: int = VERBAL_MAX_NEW_TOKENS,
    seed: int = 0,
) -> dict:
    """The greedy answer, or `samples` answers drawn at temperature 1 with no truncation.

    A greedy answer that reaches max_new_tokens without the end-of-turn token is kept
    and flagged; sampled answers that do so are dropped by sample_replies and counted.
    """
    import torch

    prefix = prefix_ids(tok, messages)
    device = next(model.parameters()).device
    if samples:
        config = sampling_config({**SAMPLED, "max_new_tokens": max_new_tokens}, end, pad)
        model.generation_config = with_settings(config)
        replies, dropped = sample_replies(model, prefix, samples, config, seed, end)
        return {"samples": [tok.decode(reply, skip_special_tokens=True) for reply in replies], "dropped": dropped}
    config = greedy_config(end, pad, max_new_tokens)
    model.generation_config = with_settings(config)
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
    row = output[0, len(prefix):].tolist()
    finished = end in row
    reply = row[: row.index(end)] if finished else row
    return {"text": tok.decode(reply, skip_special_tokens=True), "cut_off": not finished, "tokens": len(reply)}


def api_client(root: Path, provider: str, *, transport=None, environ=None):
    """A ModelClient for gpt or claude, with the pinned model id and the shared request settings."""
    if provider not in {"gpt", "claude"}:
        raise ValueError("verbalized API answers come from gpt or claude")
    config = read_json(root / "configs" / "models.json")
    keys = load_keys((provider,), environ=environ, root=root)
    client = build_clients(config, keys, (provider,), transport)[provider]
    return client, {
        "model_id": config["models"][provider],
        "endpoint": config["endpoints"][provider],
        "max_tokens": int(config.get("claude_max_tokens", 16000)),
        "max_retries": int(config["max_retries"]),
        "retry_backoff_seconds": float(config["retry_backoff_seconds"]),
    }


def chat_body(provider: str, model_id: str, messages: list[dict], *, temperature: float | None, max_tokens: int) -> dict:
    """A minimal chat request. Claude takes system text as a top-level field and needs max_tokens."""
    if provider == "claude":
        system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        body = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [message for message in messages if message["role"] != "system"],
        }
        if system:
            body["system"] = system
    elif provider == "gpt":
        body = {"model": model_id, "messages": list(messages)}
    else:
        raise ValueError(provider)
    if temperature is not None:
        body["temperature"] = temperature
    return body


def reply_text(provider: str, payload: dict) -> tuple[dict | None, str | None]:
    """The answer text with the stop reason; a refusal or an empty answer is an invalid call."""
    if provider == "claude":
        stop = payload.get("stop_reason")
        text = _message_text(
            [block for block in payload.get("content") or [] if isinstance(block, dict) and block.get("type") == "text"]
        )
    else:
        try:
            choice = payload["choices"][0]
            text = _message_text(choice["message"]["content"])
            stop = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None, "missing chat content"
    if stop == "refusal":
        return None, "refusal"
    if not text.strip():
        return None, "empty answer"
    return {"text": text, "stop_reason": stop}, None


def _temperature_rejected(result) -> bool:
    return result.http_status == 400 and "temperature" in (result.error or "").lower()


def api_verbal(
    provider: str,
    messages: list[dict],
    *,
    client,
    settings: dict,
    temperature: float | None = 0.0,
    sleeper=time.sleep,
) -> dict:
    """One answer, retried on retryable failures. Temperature 0 where supported: a 400 that
    names the temperature is resent once without it, and the record says so."""
    sent_temperature = temperature
    attempts = 0
    while True:
        body = chat_body(provider, settings["model_id"], messages, temperature=sent_temperature,
                         max_tokens=settings["max_tokens"])
        attempts += 1
        result = client._send(body, "\n\n".join(message["content"] for message in messages),
                              lambda payload: reply_text(provider, payload))
        if result.status == "failed" and sent_temperature is not None and _temperature_rejected(result):
            sent_temperature = None
            continue
        if result.status == "failed" and result.retryable and attempts <= settings["max_retries"]:
            sleeper(settings["retry_backoff_seconds"] * 2 ** (attempts - 1))
            continue
        break
    parsed = result.parsed or {}
    return {
        "status": result.status,
        "error": result.error,
        "http_status": result.http_status,
        "text": parsed.get("text"),
        "stop_reason": parsed.get("stop_reason"),
        "returned_model_id": result.returned_model_id,
        "usage": result.usage,
        "temperature_sent": sent_temperature,
        "temperature_rejected": temperature is not None and sent_temperature is None,
        "attempts": attempts,
        "latency_ms": result.latency_ms,
    }


_NUMBER = re.compile(
    r"(?<![\w.\-/])(?:"
    r"(?P<num>\d+(?:\.\d+)?)\s*/\s*(?P<den>\d+(?:\.\d+)?)"
    r"|(?P<pct>\d+(?:\.\d+)?)\s*%"
    r"|(?P<dec>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r")(?![\w/%]|\.\d)"
)
_POWER = re.compile(r"\^|\*\*|×|⁻|·\s*10|\bx\s*10\b|\btimes\s+10\b", re.IGNORECASE)
_LABEL = re.compile(r"^(?:answer|probability|p)\s*[:=]\s*", re.IGNORECASE)


def _value(match: re.Match[str]) -> float | None:
    if match.group("num") is not None:
        denominator = float(match.group("den"))
        return float(match.group("num")) / denominator if denominator else None
    if match.group("pct") is not None:
        return float(match.group("pct")) / 100.0
    return float(match.group("dec"))


def _in_range(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0


def parse_probability(text: str | None) -> float | None:
    """0.3, 30%, 3e-5, and 1/6 are read; anything else gives None.

    The whole answer should be one number, after stripping quotes, a trailing period,
    and an "Answer:"-style label. Otherwise the answer is read only if it has no power
    notation and its numbers in [0, 1] all agree within 5%, so "1/216 ≈ 0.00463" reads
    as 1/216 while "(1/6)^3" and "between 0.1 and 0.2" give None.
    """
    if not text:
        return None
    cleaned = _LABEL.sub("", text.strip().strip("`*\"' ").rstrip(".").strip())
    whole = _NUMBER.fullmatch(cleaned)
    if whole:
        value = _value(whole)
        return value if _in_range(value) else None
    if _POWER.search(text):
        return None
    found = [value for value in (_value(match) for match in _NUMBER.finditer(text)) if _in_range(value)]
    if not found:
        return None
    first = found[0]
    if all(abs(value - first) <= 0.05 * max(abs(first), abs(value)) for value in found):
        return first
    return None
