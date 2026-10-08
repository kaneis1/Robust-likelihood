from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from robust_likelihood.constants import CONNECTIVITY_QUESTION, INTENTS, RETRYABLE_STATUS
from robust_likelihood.prompting import (
    HypothesisSet,
    PromptPack,
    classification_descriptions,
    pairwise_instructions,
    render_prompt,
)
from robust_likelihood.scoring import is_probability, validate_distribution


@dataclass
class TransportResult:
    status: int | None
    payload: dict | None
    error: str | None
    retryable: bool


@dataclass
class CallResult:
    status: str
    retryable: bool
    http_status: int | None
    returned_model_id: str | None
    response_body: dict | None
    parsed: dict | None
    usage: dict | None
    error: str | None
    request_body: dict
    prompt_text: str
    latency_ms: float | None


class UrllibTransport:
    def __call__(self, url: str, headers: dict[str, str], body: dict, timeout: float) -> TransportResult:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
        except TimeoutError:
            return TransportResult(None, None, "timeout", True)
        except urllib.error.URLError as exc:
            return TransportResult(None, None, f"connection error: {exc.reason}", True)
        text = raw.decode("utf-8", errors="replace") if raw else ""
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"_raw": text[:300], "_json_error": True}
        if not isinstance(payload, dict):
            payload = {"_raw": text[:300], "_json_error": True}
        return TransportResult(status, payload, None, False)


def build_jev_request(spec, prompts: PromptPack, hypotheses: HypothesisSet) -> dict:
    if "temperature" in spec.inference_settings:
        raise ValueError("Jev is not sent a temperature")
    if spec.task == "pairwise":
        return {
            "model": spec.requested_model_id,
            "state": {
                "utterance": spec.input_text,
                "stated_intention": spec.hypothesis,
                "intention_description": spec.hypothesis_wording,
            },
            "questions": {
                "plausibility": {
                    "type": "noul",
                    "instructions": pairwise_instructions(spec, prompts),
                }
            },
        }
    return {
        "model": spec.requested_model_id,
        "state": {"utterance": spec.input_text},
        "questions": {
            "intention": {
                "type": "choice",
                "instructions": prompts.classification_question,
                "criteria": classification_descriptions(spec, hypotheses),
            }
        },
    }


def build_chat_request(spec, prompts: PromptPack, hypotheses: HypothesisSet) -> dict:
    body = {
        "model": spec.requested_model_id,
        "messages": [
            {
                "role": "user",
                "content": render_prompt(spec, prompts, hypotheses),
            }
        ],
    }
    settings = spec.inference_settings
    if "temperature" in settings:
        body["temperature"] = settings["temperature"]
    if spec.provider == "gpt" and "response_format" in settings:
        body["response_format"] = settings["response_format"]
    if spec.provider == "claude":
        body["max_tokens"] = settings.get("max_tokens", 512)
    return body


def build_request(spec, prompts: PromptPack, hypotheses: HypothesisSet) -> dict:
    if spec.provider == "jev":
        return build_jev_request(spec, prompts, hypotheses)
    if spec.provider in {"gpt", "claude"}:
        return build_chat_request(spec, prompts, hypotheses)
    raise KeyError(spec.provider)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "".join(chunks)
    return ""


def parse_json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


_NUMBER = re.compile(
    r"(?<![\d.])(?:(?P<pct>\d{1,3}(?:\.\d+)?)\s*%|(?P<unit>0(?:\.\d+)?|1(?:\.0+)?))(?!\d)"
)
_RANGE_PHRASE = re.compile(r"between\s+0(?:\.0+)?\s+and\s+1(?:\.0+)?", re.IGNORECASE)


def _probability_token(match: re.Match[str]) -> float | None:
    if match.group("pct") is not None:
        value = float(match.group("pct")) / 100.0
    else:
        value = float(match.group("unit"))
    if is_probability(value):
        return value
    return None


def _probabilities_in(text: str) -> list[float]:
    cleaned = _RANGE_PHRASE.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(cleaned):
        value = _probability_token(match)
        if value is not None:
            found.append(value)
    return found


def extract_json_object(text: str) -> dict | None:
    """Return a JSON object from the whole reply, or the first object inside it."""
    direct = parse_json_object(text)
    if direct is not None:
        return direct
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_yes_probability(text: str) -> float | None:
    embedded = extract_json_object(text)
    if embedded is not None and is_probability(embedded.get("plausibility_yes_probability")):
        return float(embedded["plausibility_yes_probability"])
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if "probab" not in sentence.lower():
            continue
        found = _probabilities_in(sentence)
        if found:
            return found[0]
    found = _probabilities_in(text)
    if len(found) == 1:
        return found[0]
    return None


def parse_intent_distribution(text: str) -> dict[str, float] | None:
    embedded = extract_json_object(text)
    parsed = validate_distribution(embedded)
    if parsed is not None:
        return parsed
    scores: dict[str, float] = {}
    for intent in INTENTS:
        label = re.compile(rf"{re.escape(intent)}|{re.escape(intent.replace('_', ' '))}", re.IGNORECASE)
        match = label.search(text)
        if match is None:
            return None
        window = text[match.end(): match.end() + 100]
        found = _probabilities_in(window)
        if not found:
            return None
        scores[intent] = found[0]
    return validate_distribution(scores)


def parse_experiment_payload(provider: str, task: str, payload: dict) -> tuple[dict | None, str | None]:
    if provider == "jev":
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            return None, "missing answers"
        if task == "pairwise":
            answer = answers.get("plausibility")
            if not isinstance(answer, dict):
                return None, "missing noul answer"
            value = answer.get("noul")
            if not is_probability(value):
                return None, "noul outside [0, 1]"
            return {"plausibility_yes_probability": float(value)}, None
        answer = answers.get("intention")
        if not isinstance(answer, dict):
            return None, "missing choice answer"
        parsed = validate_distribution(answer.get("probabilities"))
        if parsed is None:
            return None, "malformed choice distribution"
        return {"distribution": parsed}, None
    text = ""
    if provider == "gpt":
        try:
            text = _message_text(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return None, "missing chat content"
    elif provider == "claude":
        text = _message_text(payload.get("content"))
    else:
        return None, "unknown provider"
    if task == "pairwise":
        value = parse_yes_probability(text)
        if value is None:
            return None, "could not find a yes probability"
        return {"plausibility_yes_probability": value}, None
    parsed = parse_intent_distribution(text)
    if parsed is None:
        return None, "could not find a classification distribution"
    return {"distribution": parsed}, None


def _short_error(status: int, payload: dict | None) -> str:
    text = json.dumps(payload, ensure_ascii=False) if payload else ""
    return f"HTTP {status}: {text[:300]}"


class ModelClient:
    def __init__(self, provider: str, api_key: str, endpoint: str, transport, timeout: float, anthropic_version: str):
        self.provider = provider
        self.api_key = api_key
        self.endpoint = endpoint
        self.transport = transport
        self.timeout = timeout
        self.anthropic_version = anthropic_version

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider == "claude":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = self.anthropic_version
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _send(self, body: dict, prompt_text: str, interpret) -> CallResult:
        started = time.perf_counter()
        transport_result = self.transport(self.endpoint, self._headers(), body, self.timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        payload = transport_result.payload
        if transport_result.error:
            return CallResult(
                status="failed",
                retryable=transport_result.retryable,
                http_status=transport_result.status,
                returned_model_id=None,
                response_body=payload,
                parsed=None,
                usage=None,
                error=transport_result.error,
                request_body=body,
                prompt_text=prompt_text,
                latency_ms=latency_ms,
            )
        status = transport_result.status or 0
        if status >= 400 or (isinstance(payload, dict) and payload.get("_json_error") and status != 200):
            retryable = status in RETRYABLE_STATUS
            return CallResult(
                status="failed",
                retryable=retryable,
                http_status=status,
                returned_model_id=(payload or {}).get("model") if isinstance(payload, dict) else None,
                response_body=payload,
                parsed=None,
                usage=(payload or {}).get("usage") if isinstance(payload, dict) else None,
                error=_short_error(status, payload if isinstance(payload, dict) else None),
                request_body=body,
                prompt_text=prompt_text,
                latency_ms=latency_ms,
            )
        returned = payload.get("model") if isinstance(payload, dict) else None
        parsed, error = interpret(payload or {})
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if error:
            return CallResult(
                status="invalid",
                retryable=False,
                http_status=status,
                returned_model_id=returned if isinstance(returned, str) else None,
                response_body=payload,
                parsed=None,
                usage=usage if isinstance(usage, dict) else None,
                error=error,
                request_body=body,
                prompt_text=prompt_text,
                latency_ms=latency_ms,
            )
        return CallResult(
            status="success",
            retryable=False,
            http_status=status,
            returned_model_id=returned if isinstance(returned, str) else None,
            response_body=payload,
            parsed=parsed,
            usage=usage if isinstance(usage, dict) else None,
            error=None,
            request_body=body,
            prompt_text=prompt_text,
            latency_ms=latency_ms,
        )

    def perform(self, spec, prompts: PromptPack, hypotheses: HypothesisSet) -> CallResult:
        body = build_request(spec, prompts, hypotheses)
        prompt_text = render_prompt(spec, prompts, hypotheses)

        def interpret(payload: dict):
            return parse_experiment_payload(self.provider, spec.task, payload)

        return self._send(body, prompt_text, interpret)

    def probe(self, model_id: str, utterances: tuple[str, ...], inference_settings: dict) -> CallResult:
        if self.provider == "jev":
            if "temperature" in inference_settings:
                raise ValueError("Jev is not sent a temperature")
            body = {
                "model": model_id,
                "state": {"utterances": list(utterances)},
                "questions": {
                    "mentions_alarm": {
                        "type": "noul",
                        "instructions": CONNECTIVITY_QUESTION,
                    }
                },
            }
            prompt_text = CONNECTIVITY_QUESTION
        else:
            utterance_block = "\n".join(utterances)
            prompt_text = (
                f"{utterance_block}\n\n{CONNECTIVITY_QUESTION}\n"
                "Reply with JSON only, using mentions_alarm for the probability that the answer is yes."
            )
            body = {"model": model_id, "messages": [{"role": "user", "content": prompt_text}]}
            if "temperature" in inference_settings:
                body["temperature"] = inference_settings["temperature"]
            if self.provider == "gpt":
                body["response_format"] = {"type": "json_object"}
            if self.provider == "claude":
                body["max_tokens"] = inference_settings.get("max_tokens", 512)

        def interpret(payload: dict):
            model = payload.get("model")
            if not isinstance(model, str) or not model:
                return None, "response did not record a model id"
            return {"connectivity": True}, None

        return self._send(body, prompt_text, interpret)
