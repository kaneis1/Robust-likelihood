from pathlib import Path

import pytest

from robust_likelihood.clips import loops
from robust_likelihood.clips.data import parse_plan
from robust_likelihood.clips.record import build_pairs
from robust_likelihood.clips.score import distinct_pairs, encode, table_rows
from robust_likelihood.storage import repo_root

ROOT = repo_root()
CLIPS = ROOT / "external" / "CLIPS.jl"


def test_parse_plan_times_utterances_by_actions_before_them():
    actions, annotations, times = parse_plan("(left human)\n(wait robot)\n\n; Can you pass the key?\n(up human)\n; ok\n")
    assert actions == ["(left human)", "(wait robot)", "(up human)"]
    assert annotations == ["Can you pass the key?", "ok"] and times == [2, 3]


@pytest.mark.skipif(not CLIPS.exists(), reason="external/CLIPS.jl is not cloned")
def test_utterances_match_the_plan_counts():
    from robust_likelihood.clips.data import utterance_rows

    rows = utterance_rows(CLIPS)
    mdkg = [row for row in rows if row["domain"] == "mdkg"]
    vh = [row for row in rows if row["domain"] == "vh"]
    assert len(mdkg) == 30 and len({row["plan_id"] for row in mdkg}) == 30
    assert len(vh) == 35 and len({row["plan_id"] for row in vh}) == 25
    assert sum(1 for plan in {row["plan_id"] for row in vh} if sum(r["plan_id"] == plan for r in vh) > 1) == 10
    assert all(row["true_goal"].startswith("(has human gem") for row in mdkg)


@pytest.mark.skipif(not CLIPS.exists(), reason="external/CLIPS.jl is not cloned")
def test_copied_loops_regenerate_exactly():
    for domain, folder in loops.DOMAINS.items():
        original = (CLIPS / folder / "experiments.jl").read_text(encoding="utf-8")
        assert loops.transform(original, domain) == (ROOT / "julia" / "clips" / f"{domain}_experiments.jl").read_text(
            encoding="utf-8")


def test_transform_refuses_a_changed_original():
    with pytest.raises(ValueError):
        loops.transform("using PDDL\n", "mdkg")


RECORD = [
    {"role": "mdkg_literal", "prompt": "P1", "output": " go", "context": {"plan_id": "1.1.keys", "method": "literal"}},
    {"role": "mdkg_pragmatic", "prompt": "P2", "output": " go", "context": {"plan_id": "1.1.keys", "method": "pragmatic"}},
    {"role": "mdkg_pragmatic", "prompt": "P3", "output": " go", "context": {"plan_id": "1.1.keys", "method": "pragmatic"}},
]
PARTICLES = [
    {"plan_id": "1.1.keys", "particle": 1, "goal": 1, "cost": 1, "t": 7, "prompts": ["P2", "P3"], "probs": [0.5, 0.5],
     "output": " go"},
    {"plan_id": "1.1.keys", "particle": 2, "goal": 2, "cost": 1, "t": 7, "prompts": ["P3"], "probs": [1.0],
     "output": " go"},
]


def test_pairs_attribute_goals_and_check_both_logs():
    rows, counts = build_pairs("mdkg", RECORD, PARTICLES)
    uses = {row["prompt"]: row["uses"] for row in rows}
    assert uses["P1"] == [{"plan_id": "1.1.keys", "goal": None, "method": "literal"}]
    assert [use["goal"] for use in uses["P3"]] == [1, 2]
    assert counts["check"]["passes"] and counts["unique_pairs"] == 3
    assert counts["commands_per_goal"] == {"mean": 1.5, "max": 2}
    _, broken = build_pairs("mdkg", RECORD[:2], PARTICLES)
    assert broken["check"]["particle_prompts_not_recorded"] == 1 and not broken["check"]["passes"]
    _, empty = build_pairs("mdkg", [], [])
    assert not empty["check"]["passes"]


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(char) % 50 for char in text]}


def test_encode_splits_prompt_and_continuation_and_caps_length():
    ids, first = encode(CharTokenizer(), "Input: x\nOutput:", " go", 64)
    assert first == len("Input: x\nOutput:") and len(ids) == first + len(" go\n")
    assert encode(CharTokenizer(), "p", " " + "a" * 70, 64) is None
    rows = [{"role": "a", "prompt_id": "x1", "output": " u"}, {"role": "b", "prompt_id": "x1", "output": " u"}]
    assert distinct_pairs(rows) == [("x1", " u")]
    table = table_rows(rows, {"x1": "the prompt"}, {("x1", " u"): -3.5})
    assert [(row["role"], row["prompt"], row["score"]) for row in table] == [("a", "the prompt", -3.5),
                                                                          ("b", "the prompt", -3.5)]


def test_split_prompts_keeps_each_prompt_once():
    from robust_likelihood.clips.record import prompt_id, split_prompts

    rows = [{"role": "a", "prompt": "P", "output": " u", "uses": []}, {"role": "b", "prompt": "P", "output": " v", "uses": []}]
    pairs, prompts = split_prompts(rows)
    assert prompts == [{"prompt_id": prompt_id("P"), "prompt": "P"}]
    assert all(row["prompt_id"] == prompt_id("P") and "prompt" not in row for row in pairs)
