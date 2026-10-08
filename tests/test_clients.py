import json

import pytest

from robust_likelihood.clients import (
    CallResult,
    ModelClient,
    TransportResult,
    build_request,
    parse_experiment_payload,
)
from robust_likelihood.constants import INTENTS
from robust_likelihood.planning import build_plan
from robust_likelihood.prompting import load_hypotheses, load_prompts, render_prompt
from robust_likelihood.storage import repo_root


def _spec(provider="jev", task="pairwise"):
    descriptions = load_hypotheses(repo_root(), "v1").descriptions
    plan = build_plan(
        [
            {
                "example_id": "722__original",
                "original_example_id": "722",
                "transformation": "original",
                "utt": "make an alarm to wake me up after five hours",
                "intent": "alarm_set",
                "original_intent": "alarm_set",
                "label_changed": False,
            }
        ],
        (provider,),
        {provider: "jev-1.13.0" if provider == "jev" else "gpt-4.1-2025-04-14"},
        descriptions,
        "v1",
        "v1",
        {provider: {} if provider == "jev" else {"temperature": 0, "response_format": {"type": "json_object"}}},
    )
    return next(spec for spec in plan if spec.task == task)


class Capture:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.bodies = []
        self.headers = []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(body)
        self.headers.append(headers)
        return TransportResult(self.status, self.payload, None, False)


def _client(provider, transport):
    return ModelClient(provider, "test-key", "https://example.test", transport, 5, "2023-06-01")


def test_jev_pairwise_sends_one_noul_and_no_temperature():
    spec = _spec("jev", "pairwise")
    prompts = load_prompts(repo_root(), "v1")
    hypotheses = load_hypotheses(repo_root(), "v1")
    body = build_request(spec, prompts, hypotheses)
    assert "temperature" not in body
    assert list(body["questions"]) == ["plausibility"]
    assert body["questions"]["plausibility"]["instructions"] == prompts.pairwise_question
    assert body["state"]["stated_intention"] == "alarm_set"
    assert set(body["state"]) == {"utterance", "stated_intention", "intention_description"}
    transport = Capture(
        {
            "model": "jev-1.13.1",
            "answers": {"plausibility": {"type": "noul", "noul": 0.25}},
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )
    result = _client("jev", transport).perform(spec, prompts, hypotheses)
    assert transport.bodies[0] == body
    assert "temperature" not in transport.bodies[0]
    assert result.returned_model_id == "jev-1.13.1"
    assert result.parsed == {"plausibility_yes_probability": 0.25}
    assert transport.headers[0]["Authorization"] == "Bearer test-key"
    assert "test-key" not in json.dumps(result.response_body)


def test_jev_temperature_is_rejected_before_http():
    spec = _spec("jev", "pairwise")
    spec = type(spec)(**{**spec.__dict__, "inference_settings": {"temperature": 0}})
    transport = Capture({})
    with pytest.raises(ValueError, match="temperature"):
        _client("jev", transport).perform(spec, load_prompts(repo_root(), "v1"), load_hypotheses(repo_root(), "v1"))
    assert transport.bodies == []


def test_jev_classification_is_one_choice_over_three_intents():
    spec = _spec("jev", "classification")
    prompts = load_prompts(repo_root(), "v1")
    hypotheses = load_hypotheses(repo_root(), "v1")
    body = build_request(spec, prompts, hypotheses)
    assert list(body["questions"]) == ["intention"]
    assert list(body["questions"]["intention"]["criteria"]) == list(INTENTS)
    assert body["state"] == {"utterance": spec.input_text}
    assert "temperature" not in body


def test_chat_prompt_asks_for_the_yes_probability():
    spec = _spec("gpt", "pairwise")
    prompts = load_prompts(repo_root(), "v1")
    hypotheses = load_hypotheses(repo_root(), "v1")
    rendered = render_prompt(spec, prompts, hypotheses)
    assert prompts.pairwise_question in rendered
    body = build_request(spec, prompts, hypotheses)
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    payload = {
        "model": "gpt-4.1-2025-04-14",
        "choices": [{"message": {"content": '{"plausibility_yes_probability": 0.75}'}}],
    }
    parsed, error = parse_experiment_payload("gpt", "pairwise", payload)
    assert error is None
    assert parsed == {"plausibility_yes_probability": 0.75}


def test_malformed_distribution_is_invalid_and_not_renormalized():
    payload = {
        "model": "jev-1.13.0",
        "answers": {"intention": {"type": "choice", "probabilities": {"alarm_set": 0.2, "alarm_query": 0.2, "alarm_remove": 0.2}}},
    }
    parsed, error = parse_experiment_payload("jev", "classification", payload)
    assert parsed is None
    assert error
    missing_model = {"answers": {"plausibility": {"type": "noul", "noul": 0.5}}}
    parsed, error = parse_experiment_payload("jev", "pairwise", missing_model)
    assert error is None
    assert parsed["plausibility_yes_probability"] == 0.5


def test_natural_language_answers_are_scored():
    prompts = load_prompts(repo_root(), "v2")
    hypotheses = load_hypotheses(repo_root(), "v1")
    spec = _spec("gpt", "pairwise")
    rendered = render_prompt(spec, prompts, hypotheses)
    assert "not JSON" in rendered
    assert "Reply with JSON only" not in rendered
    prose = {
        "model": "gpt-6-astra",
        "choices": [
            {
                "message": {
                    "content": (
                        "The utterance asks for a new alarm. "
                        "The probability that the answer is yes is 0.82."
                    )
                }
            }
        ],
    }
    parsed, error = parse_experiment_payload("gpt", "pairwise", prose)
    assert error is None
    assert parsed == {"plausibility_yes_probability": 0.82}
    fenced = {
        "model": "claude-fable-5-1",
        "content": [
            {
                "type": "text",
                "text": '```json\n{"plausibility_yes_probability": 0.05}\n```\n\nThis is a request to set an alarm.',
            }
        ],
    }
    parsed, error = parse_experiment_payload("claude", "pairwise", fenced)
    assert error is None
    assert parsed == {"plausibility_yes_probability": 0.05}
    classes = {
        "model": "gpt-6-astra",
        "choices": [
            {
                "message": {
                    "content": (
                        "The probability of alarm_set is 0.7. "
                        "The probability of alarm_query is 0.2. "
                        "The probability of alarm_remove is 0.1."
                    )
                }
            }
        ],
    }
    parsed, error = parse_experiment_payload("gpt", "classification", classes)
    assert error is None
    assert parsed == {"distribution": {"alarm_set": 0.7, "alarm_query": 0.2, "alarm_remove": 0.1}}


def test_gpt_decision_sends_one_predicate_and_reads_its_probability():
    hypotheses = load_hypotheses(repo_root(), "v1")
    plan = build_plan(
        [
            {
                "example_id": "722__original",
                "original_example_id": "722",
                "transformation": "original",
                "utt": "make an alarm to wake me up after five hours",
                "intent": "alarm_set",
                "original_intent": "alarm_set",
                "label_changed": False,
            }
        ],
        ("gpt_decision",),
        {"gpt_decision": "gpt-6-luna"},
        hypotheses.descriptions,
        "likely",
        "v1",
        {"gpt_decision": {}},
    )
    spec = next(item for item in plan if item.task == "pairwise")
    prompts = load_prompts(repo_root(), "likely")
    body = build_request(spec, prompts, hypotheses)
    assert body["model"] == spec.requested_model_id
    assert "temperature" not in body
    assert body["questions"] == [
        {
            "type": "predicate",
            "name": "plausibility",
            "instructions": prompts.pairwise_question.replace("<<UTTERANCE>>", spec.input_text),
        }
    ]
    assert "Stated intention: alarm_set" in body["input"]
    assert spec.input_text in body["input"]
    parsed, error = parse_experiment_payload(
        "gpt_decision",
        "pairwise",
        {"model": "gpt-6-luna", "answers": [{"type": "predicate", "name": "plausibility", "probability": 0.25}]},
    )
    assert error is None
    assert parsed == {"plausibility_yes_probability": 0.25}
    refused, error = parse_experiment_payload(
        "gpt_decision",
        "pairwise",
        {"answers": [{"type": "refusal", "name": "plausibility"}]},
    )
    assert refused is None
    assert error == "decision refused"
    choice, error = parse_experiment_payload(
        "gpt_decision",
        "classification",
        {
            "answers": [
                {
                    "type": "choice",
                    "name": "intention",
                    "choice": "alarm_set",
                    "probabilities": [
                        {"value": "alarm_set", "probability": 0.7},
                        {"value": "alarm_query", "probability": 0.2},
                        {"value": "alarm_remove", "probability": 0.1},
                    ],
                }
            ]
        },
    )
    assert error is None
    assert choice == {"distribution": {"alarm_set": 0.7, "alarm_query": 0.2, "alarm_remove": 0.1}}


def test_http_401_is_not_retried_by_the_client_flag():
    spec = _spec("jev", "pairwise")
    transport = Capture({"error": "unauthorized"}, status=401)
    result = _client("jev", transport).perform(spec, load_prompts(repo_root(), "v1"), load_hypotheses(repo_root(), "v1"))
    assert result.status == "failed"
    assert result.retryable is False
    assert isinstance(result, CallResult)
