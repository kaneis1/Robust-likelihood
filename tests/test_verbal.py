import threading

import pytest

from robust_likelihood.clients import TransportResult
from robust_likelihood.storage import repo_root
from robust_likelihood.verbal import api_client, api_verbal, chat_body, parse_probability, reply_text

ENV = {"ANTHROPIC_API_KEY": "test-claude", "OPENAI_API_KEY": "test-gpt"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.3", 0.3),
        ("30%", 0.3),
        ("3e-5", 3e-5),
        ("1/6", 1 / 6),
        ("1/6.", 1 / 6),
        ("Answer: 0.5", 0.5),
        ("`0.25`", 0.25),
        ("**0.02**", 0.02),
        ("1/216 ≈ 0.00463", 0.00463),
        ("1/216, about 0.00463", 1 / 216),
        ("0", 0.0),
        ("I rolled 3, 5, 2 so 0.0046", 0.0046),
        ("P(6)=1/2, others 1/10. Probability = (1/10)^2 × 1/2 = 0.005\n\n0.005", 0.005),
        ("(1/6)^8 = 1/1679616 ≈ 5.954e-7\n\n0.000000595374", 5.95374e-7),
        ("The die only lands on 2, 4, or 6, so 3 is impossible.\n\n0", 0.0),
        ("(1/6)^8 ≈ 0.000000595", 5.95e-7),
    ],
)
def test_parse_probability_reads_the_four_forms(text, expected):
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text", ["", None, "abc", "1.5", "-0.3", "(1/6)^3", "6^-10", "1.65 × 10^-8", "between 0.1 and 0.2", "120%"]
)
def test_parse_probability_returns_none_otherwise(text):
    assert parse_probability(text) is None


def test_chat_body_moves_claude_system_text_and_keeps_gpt_messages():
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "p?"}]
    claude = chat_body("claude", "claude-x", messages, temperature=0.0, max_tokens=100)
    assert claude == {"model": "claude-x", "max_tokens": 100, "system": "Be brief.",
                      "messages": [{"role": "user", "content": "p?"}], "temperature": 0.0}
    gpt = chat_body("gpt", "gpt-x", messages, temperature=None, max_tokens=100)
    assert gpt == {"model": "gpt-x", "messages": messages}


def test_reply_text_skips_thinking_blocks_and_flags_refusals():
    payload = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "0.2"}]}
    assert reply_text("claude", payload) == ({"text": "0.2", "stop_reason": "end_turn"}, None)
    assert reply_text("claude", {"stop_reason": "refusal", "content": []}) == (None, "refusal")
    assert reply_text("gpt", {"choices": [{"message": {"content": "0.4"}, "finish_reason": "stop"}]})[0]["text"] == "0.4"


class RejectsTemperature:
    def __init__(self):
        self.bodies = []
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.bodies.append(body)
        if "temperature" in body:
            return TransportResult(400, {"error": {"message": "temperature is not supported for this model"}}, None, False)
        return TransportResult(200, {"model": body["model"], "stop_reason": "end_turn",
                                     "content": [{"type": "text", "text": "1/36"}],
                                     "usage": {"input_tokens": 50, "output_tokens": 7}}, None, False)


def test_api_verbal_drops_a_rejected_temperature_and_records_it():
    transport = RejectsTemperature()
    client, settings = api_client(repo_root(), "claude", transport=transport, environ=ENV)
    found = api_verbal("claude", [{"role": "user", "content": "p?"}], client=client, settings=settings,
                       temperature=0.0, sleeper=lambda seconds: None)
    assert found["status"] == "success" and found["text"] == "1/36"
    assert found["temperature_rejected"] and found["temperature_sent"] is None and found["attempts"] == 2
    assert "temperature" in transport.bodies[0] and "temperature" not in transport.bodies[1]
    assert found["returned_model_id"] == settings["model_id"]
    assert parse_probability(found["text"]) == pytest.approx(1 / 36)


def test_api_client_refuses_other_providers():
    with pytest.raises(ValueError):
        api_client(repo_root(), "jev", environ=ENV)
