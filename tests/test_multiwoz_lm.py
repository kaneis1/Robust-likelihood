import math
from types import SimpleNamespace

import pytest

from robust_likelihood.multiwoz.lm import (
    bin_index,
    boundary_holds,
    end_of_turn_id,
    hypothesis_shape,
    prefix_ids,
    reply_ids,
    scored_sequence,
    typicality_scores,
)

SPECIAL = {"<|start|>": 1, "<|end|>": 2}
VOCAB = 64


class CharTokenizer:
    """One token per character; the chat template wraps each turn in <|start|> and <|end|>."""

    def encode(self, text: str) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            for token, number in SPECIAL.items():
                if text.startswith(token, index):
                    ids.append(number)
                    index += len(token)
                    break
            else:
                ids.append(3 + ord(text[index]) % (VOCAB - 3))
                index += 1
        return ids

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        text = "".join(f"<|start|>{message['role']}\n{message['content']}<|end|>\n" for message in messages)
        if add_generation_prompt:
            text += "<|start|>assistant\n"
        return self.encode(text) if tokenize else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}


def test_end_of_turn_and_boundary_on_a_character_tokenizer():
    tokenizer = CharTokenizer()
    assert end_of_turn_id(tokenizer) == SPECIAL["<|end|>"]
    assert boundary_holds(tokenizer, "The user wants to find a restaurant.", "I need a cheap place.", 2)
    sequence, first = scored_sequence(prefix_ids(tokenizer, "h"), reply_ids(tokenizer, "hi"), 2)
    assert sequence[first:] == reply_ids(tokenizer, "hi") + [2]


def test_typicality_scores_follow_the_definitions():
    samples = [10.0, 12.0, 14.0, 16.0]
    mu, sigma = 13.0, math.sqrt(sum((value - 13.0) ** 2 for value in samples) / 3)
    found = typicality_scores(13.0 + 1.5 * sigma, samples, (1.0, 2.0))
    assert found["lm_loglik"] == pytest.approx(-(13.0 + 1.5 * sigma))
    assert found["mu"] == pytest.approx(mu) and found["sigma"] == pytest.approx(sigma)
    assert found["T"] == pytest.approx(1.5)
    assert found["lm_typ_normal"] == pytest.approx(-1.125)
    in_bin = sum(1 for value in samples if bin_index((value - mu) / sigma, (1.0, 2.0)) == 1)
    assert found["lm_typ_bins"] == pytest.approx(math.log((in_bin + 1) / (4 + 3)))
    assert bin_index(0.5, (1.0, 2.0)) == 0 and bin_index(-1.2, (1.0, 2.0)) == 1 and bin_index(2.0, (1.0, 2.0)) == 2
    flat = typicality_scores(5.0, [5.0, 5.0], (1.0, 2.0))
    assert flat["T"] is None and flat["lm_typ_normal"] is None


def test_hypothesis_shape_reads_slots_and_wording():
    rows = [
        {"text": "The user wants to find a restaurant.", "constrained": [], "hyp_type": "broad"},
        {"text": "The user wants to find a restaurant. The user has no preference about the area.",
         "constrained": ["food", "pricerange"], "hyp_type": "G"},
    ]
    shapes = hypothesis_shape(rows)
    assert shapes["The user wants to find a restaurant."] == {"n_constrained": 0, "regime": "impl", "hyp_types": {"broad"}}
    assert shapes[rows[1]["text"]]["regime"] == "expl"


def test_only_assistant_tokens_and_the_end_of_turn_are_scored():
    torch = pytest.importorskip("torch")
    from robust_likelihood.multiwoz.lm import surprisals

    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, VOCAB) + 0 * self.anchor)

    tokenizer = CharTokenizer()
    reply = reply_ids(tokenizer, "I want thai food.")
    short = scored_sequence(prefix_ids(tokenizer, "h"), reply, 2)
    long = scored_sequence(prefix_ids(tokenizer, "a much longer hypothesis description"), reply, 2)
    values = surprisals(Uniform(), [short, long], batch_size=2, pad_id=0)
    expected = (len(reply) + 1) * math.log(VOCAB)
    assert values[0] == pytest.approx(expected)
    assert values[1] == pytest.approx(expected)
