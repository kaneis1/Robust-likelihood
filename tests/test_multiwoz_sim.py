import re

import pytest

from robust_likelihood.multiwoz.lm import SYSTEM_PROMPT, USER_PROMPT
from robust_likelihood.multiwoz.sim import (
    SimFormat,
    chat_messages,
    check_goals,
    fewshot_prefix,
    flags,
    select_fewshot,
)
from robust_likelihood.storage import repo_root

PROMPTS = {
    "S_system": "You are a user talking to a restaurant booking assistant. {H}",
    "opener": "Hello, welcome to the restaurant booking service. How can I help you?",
    "B_system": "You are a real person typing to a restaurant booking assistant. {H}",
    "N_system": "{H}",
    "F_header": "Header.",
    "F_example": "\n\nCustomer: {H}\nMessage: {U}",
    "F_query": "\n\nCustomer: {H}\nMessage:",
}
QWEN = {"id": "fake/qwen", "speaker_role": "assistant", "end_token": "<|im_end|>", "separator": "", "template_kwargs": {}}
USERLM = {"id": "fake/userlm", "speaker_role": "user", "end_token": "<|eot_id|>", "separator": "\n",
          "template_kwargs": {}, "flag_tokens": ["<|endconversation|>"]}
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|endoftext|>"]


class FakeTokenizer:
    """Characters as tokens, special tokens whole; a Qwen-style or a UserLM-style chat template."""

    def __init__(self, style: str):
        self.style = style
        self.eos_token, self.eos_token_id, self.unk_token_id = "<|endoftext|>", 1005, None

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        if self.style == "qwen":
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")
        text = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>" for m in messages)
        return text + "<|start_header_id|>user<|end_header_id|>"

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in re.split("(" + "|".join(re.escape(token) for token in SPECIALS) + ")", text):
            if part in SPECIALS:
                ids.append(1000 + SPECIALS.index(part))
            else:
                ids.extend(ord(char) for char in part)
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return 1000 + SPECIALS.index(token) if token in SPECIALS else None


def test_messages_put_the_opener_in_the_role_the_model_does_not_speak():
    s = chat_messages("S", QWEN, PROMPTS, "The user wants food.")
    assert s == [{"role": "system", "content": "You are a user talking to a restaurant booking assistant. The user wants food."},
                 {"role": "user", "content": PROMPTS["opener"]}]
    assert chat_messages("S", USERLM, PROMPTS, "H.")[1] == {"role": "assistant", "content": PROMPTS["opener"]}
    assert chat_messages("N", QWEN, PROMPTS, "H.") == [{"role": "system", "content": "H."}]
    assert chat_messages("B", QWEN, PROMPTS, "H.")[0]["content"].startswith("You are a real person")
    assert chat_messages("P0", QWEN, PROMPTS, "H.") == [{"role": "system", "content": SYSTEM_PROMPT},
                                                        {"role": "user", "content": USER_PROMPT.format(text="H.")}]


def test_fewshot_prefix_and_reply():
    examples = [{"H": "H1.", "U": "u1"}, {"H": "H2.", "U": "u2"}]
    assert fewshot_prefix(PROMPTS, examples, "H.") == (
        "Header.\n\nCustomer: H1.\nMessage: u1\n\nCustomer: H2.\nMessage: u2\n\nCustomer: H.\nMessage:")
    fmt = SimFormat(FakeTokenizer("qwen"), {"id": "fake/base", "end_text": "\n"}, "F", PROMPTS, examples)
    tokens, first = fmt.scored("H.", "hi")
    assert fmt.end is None and tokens[first:] == [ord(" "), ord("h"), ord("i"), ord("\n")]
    assert fmt.boundary_holds("H.", "hi") and fmt.template_check()


def test_qwen_style_prefix_reply_and_end_token():
    fmt = SimFormat(FakeTokenizer("qwen"), QWEN, "S", PROMPTS)
    assert fmt.prefix_text("H.").endswith("<|im_start|>assistant\n")
    tokens, first = fmt.scored("H.", "ok")
    assert tokens[first:] == [ord("o"), ord("k"), 1001]
    assert fmt.boundary_holds("H.", "ok") and fmt.template_check()


def test_userlm_style_speaks_as_user_after_a_newline():
    fmt = SimFormat(FakeTokenizer("userlm"), USERLM, "S", PROMPTS)
    prefix = fmt.prefix_text("H.")
    assert "<|start_header_id|>assistant<|end_header_id|>\n" + PROMPTS["opener"] in prefix
    assert prefix.endswith("<|start_header_id|>user<|end_header_id|>\n")
    tokens, first = fmt.scored("H.", "ok")
    assert tokens[first:] == [ord("o"), ord("k"), 1004]
    assert fmt.template_check()
    native = SimFormat(FakeTokenizer("userlm"), USERLM, "N", PROMPTS).prefix_text("H.")
    assert native == "<|start_header_id|>system<|end_header_id|>\nH.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"


def test_an_unknown_end_token_is_refused():
    with pytest.raises(ValueError):
        SimFormat(FakeTokenizer("qwen"), dict(QWEN, end_token="<|nope|>"), "S", PROMPTS)


def test_flags_catch_assistant_voice_restaurant_voice_and_think_blocks():
    assert flags("Sure, I can help.", "Sure, I can help.", False)["assistant_opening"]
    assert flags("Hi. We have a table for you.", "", False)["restaurant_voice"]
    assert not flags("Do you have a cheap place?", "", False)["restaurant_voice"]
    assert flags("x", "<think>plan</think>x", False)["think_block"]
    assert not flags("x", "<think>\n\n</think>x", False)["think_block"]
    found = flags("  ", "<|endconversation|>", True, ("<|endconversation|>",))
    assert found["empty"] and found["truncated"] and found["flag_tokens"] == ["<|endconversation|>"]


def candidate(index, n, omitted=()):
    slots = ["food", "area", "pricerange"][:n]
    return {"dialogue_id": f"MUL{index:04d}", "n_constrained": n,
            "status": {slot: "omitted" if slot in omitted else "named" for slot in slots}, "H": "h", "U": "u"}


def test_fewshot_selection_is_stratified_seeded_and_keeps_an_omission():
    pool = [candidate(i, 2) for i in range(10)] + [candidate(100 + i, 3, ("food",)) for i in range(10)]
    chosen = select_fewshot(pool, 7)
    assert sorted(found["n_constrained"] for found in chosen) == [2] * 4 + [3] * 4
    assert chosen == select_fewshot(list(reversed(pool)), 7)
    with pytest.raises(ValueError):
        select_fewshot([candidate(i, 2) for i in range(10)] + [candidate(100 + i, 3) for i in range(10)], 7)


def test_sampling_stays_exact_when_the_model_ships_a_low_temperature():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.lmcore import sampling_config, sampling_is_exact
    from robust_likelihood.multiwoz.sim import use_config

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=2)).eval()
    model.generation_config = transformers.GenerationConfig(do_sample=True, temperature=0.6, top_p=0.9)
    settings = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": 3}
    config = sampling_config(settings, 63, 63)
    assert not sampling_is_exact(model, [1, 2, 3], config)
    assert sampling_is_exact(model, [1, 2, 3], use_config(model, config))


def test_check_goals_draw_two_and_three_slot_items():
    from robust_likelihood.storage import read_jsonl

    derived = repo_root() / "data" / "multiwoz" / "derived"
    goals = check_goals(read_jsonl(derived / "items.jsonl"), read_jsonl(derived / "probe.jsonl"), 20261103, 3, 3)
    assert [goal["n_constrained"] for goal in goals] == [2, 2, 2, 3, 3, 3]
    assert all(goal["H"].startswith("The user wants to find") for goal in goals)
