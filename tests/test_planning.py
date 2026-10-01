from dataclasses import replace

from robust_likelihood.planning import (
    UnpinnedModelError,
    assert_comparison_pins,
    build_plan,
    cache_key,
    cache_payload,
    inference_settings_for,
    is_pinned_jev,
    summarize_plan,
    workload,
)
from robust_likelihood.prompting import load_hypotheses, load_prompts
from robust_likelihood.storage import read_json, repo_root


DESCRIPTIONS = {
    "alarm_set": "set an alarm",
    "alarm_query": "ask about an alarm",
    "alarm_remove": "cancel an alarm",
}


def _item(index=0):
    return {
        "example_id": f"{index}__original",
        "original_example_id": str(index),
        "transformation": "original",
        "utt": f"utterance {index}",
        "intent": "alarm_set",
        "original_intent": "alarm_set",
        "label_changed": False,
    }


def _plan(n_items=1, providers=("jev", "gpt"), repeats=0, descriptions=None):
    items = [_item(index) for index in range(n_items)]
    return build_plan(
        items,
        providers,
        {"jev": "jev-1.13.0", "gpt": "gpt-4.1-2025-04-14", "claude": "claude-sonnet-4-5-20250929"},
        descriptions or DESCRIPTIONS,
        "v1",
        "v1",
        {"jev": {}, "gpt": {"temperature": 0}, "claude": {"temperature": 0, "max_tokens": 512}},
        repeats=repeats,
    )


def test_baseline_workload_is_480_requests():
    assert workload(60, 2, 3, 0) == {"classification": 120, "pairwise": 360, "total": 480}
    assert summarize_plan(_plan(60))["total"] == 480
    assert workload(60, 2, 3, 1)["total"] == 960


def test_cache_key_separates_hypotheses_and_repeats_but_not_attempts():
    spec = _plan(1, providers=("jev",))[1]
    payload = cache_payload(spec)
    assert "repeat_id" not in payload
    assert "attempt" not in payload
    same_wording = replace(spec, hypothesis="alarm_query", hypothesis_wording=spec.hypothesis_wording)
    assert cache_key(spec) != cache_key(same_wording)
    assert cache_key(spec) != cache_key(replace(spec, repeat_id=1))
    assert cache_key(spec) == cache_key(replace(spec, input_text=spec.input_text))
    assert cache_payload(replace(spec, repeat_id=1))["repeat_id"] == 1


def test_jev_alias_is_refused_for_comparison_only():
    assert is_pinned_jev("jev-1.13.0")
    assert not is_pinned_jev("jev-latest")
    assert not is_pinned_jev("jev-preview")
    assert not is_pinned_jev("jev-1.13")
    try:
        assert_comparison_pins({"jev": "jev-latest"})
    except UnpinnedModelError as exc:
        assert "jev-latest" in str(exc)
    else:
        raise AssertionError("alias was accepted")


def test_inference_settings_omit_jev_temperature():
    config = read_json(repo_root() / "configs" / "models.json")
    assert inference_settings_for("jev", config) == {}
    assert "temperature" not in inference_settings_for("jev", config)
    assert "temperature" not in inference_settings_for("gpt", config)
    assert "temperature" not in inference_settings_for("claude", config)
    assert inference_settings_for("claude", config)["max_tokens"] == 16000
    assert config["models"]["jev"] == "jev-1.13.0"
    assert config["models"]["gpt"] == "gpt-6-astra"
    assert config["models"]["claude"] == "claude-fable-5-1"
    assert config["default_comparison"] == ["jev", "gpt", "claude"]
    assert config["prompt_version"] == "v2"
    assert "response_format" not in inference_settings_for("gpt", config)


def test_prompt_files_and_hypothesis_versions():
    root = repo_root()
    prompts = load_prompts(root, "v1")
    assert prompts.pairwise_question == (
        "Assume the user has the stated intention. Is this utterance a plausible expression of that intention?"
    )
    first = _plan(1, providers=("gpt",))
    pairwise = next(spec for spec in first if spec.task == "pairwise")
    from robust_likelihood.prompting import render_prompt

    rendered = render_prompt(pairwise, prompts, load_hypotheses(root, "v1"))
    assert prompts.pairwise_question in rendered
    v1 = load_hypotheses(root, "v1")
    v2 = load_hypotheses(root, "v2")
    assert v1.descriptions != v2.descriptions
    left = _plan(1, descriptions=v1.descriptions)[1]
    right = _plan(1, descriptions=v2.descriptions)[1]
    assert cache_key(left) != cache_key(right)
