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


RAW = ROOT / "data" / "clips" / "raw" / "results"
# Table 1 of the paper (AAMAS 2024, arXiv v1): P(g_true), precision, recall, rel. plan length, rel. human cost,
# goal correlation, assistance correlation; None where the paper prints a dash.
PUBLISHED_TABLE1 = {
    "mdkg": {"clips": (0.74, 0.97, 0.97, 1.00, 1.00, 0.93, 0.96),
             "language_only": (0.55, 0.90, 0.83, 1.26, 1.18, 0.74, 0.83),
             "action_only": (0.31, 0.43, 0.40, 1.68, 1.46, 0.15, 0.45),
             "literal_efficient": (None, 0.65, 0.54, 1.55, 1.58, None, 0.47),
             "literal_naive": (None, 0.58, 0.47, 1.54, 1.52, None, 0.54)},
    "vh": {"clips": (0.63, 0.87, 1.00, 1.00, 1.00, None, None),
           "language_only": (0.37, 0.59, 0.96, 1.33, 1.35, None, None),
           "action_only": (0.25, 0.61, 0.84, 1.30, 1.33, None, None),
           "literal": (None, 0.64, 0.70, 1.48, 1.54, None, None)},
}


@pytest.mark.skipif(not RAW.exists(), reason="the paper's OSF result files are not downloaded")
def test_evaluate_reproduces_the_published_table_from_the_paper_files():
    from robust_likelihood.clips.evaluate import DOMAIN_DIRS, read_humans, read_rows, table1

    humans = read_humans(RAW / "doors-keys-gems" / "mean_human_responses.csv")
    keys = ("p_true", "assist_precision", "assist_recall", "rel_plan_length", "rel_human_cost", "goal_correlation",
            "assist_correlation")
    for domain, folder in DOMAIN_DIRS.items():
        rows = read_rows([RAW / folder / "literal_experiments.csv", RAW / folder / "pragmatic_experiments.csv"])
        found = table1(rows, domain, humans=humans, replicates=10)
        for method, published in PUBLISHED_TABLE1[domain].items():
            for key, value in zip(keys, published):
                block = found[method].get(key) or {}
                mine = block.get("mean", block.get("r"))
                assert (mine is None) == (value is None), (domain, method, key)
                if value is not None:
                    assert round(mine, 2) == value, (domain, method, key, mine)


def test_table1_on_two_problems():
    from robust_likelihood.clips.evaluate import table1

    def row(plan, method, modality, p, length, cost, **extra):
        return {"plan_id": plan, "infer_method": method, "modalities": modality, "estim_type": extra.pop("estim", "mode"),
                "assist_method": extra.pop("assist", "qmdp_act"), "true_goal_probs": p, "assist_precision": "1.0",
                "assist_recall": "0.5", "plan_length": length, "speaker_cost": cost, "option_count": "2", **extra}

    rows = [row("a", "pragmatic", "action, utterance", "0.8", "10", "5"), row("b", "pragmatic", "action, utterance", "0.6", "20", "10"),
            row("a", "pragmatic", "action", "0.2", "20", "10"), row("b", "pragmatic", "action", "0.4", "20", "20"),
            row("a", "literal", "", "", "30", "5", assist="literal", estim="mean"),
            row("b", "literal", "", "", "30", "5", assist="literal", estim="mean"),
            row("b", "literal", "", "", "99", "99", assist="literal", estim="mode")]
    found = table1(rows, "vh", replicates=10)
    assert found["clips"]["p_true"]["mean"] == pytest.approx(0.7)
    assert found["action_only"]["rel_plan_length"]["mean"] == pytest.approx((2.0 + 1.0) / 2)
    assert found["action_only"]["rel_human_cost"]["mean"] == pytest.approx((2.0 + 2.0) / 2)
    assert found["literal"]["rel_plan_length"]["mean"] == pytest.approx((3.0 + 1.5) / 2)
    assert found["literal"]["p_true"]["mean"] is None and "language_only" not in found


def test_protected_tokens_and_validity():
    from robust_likelihood.clips.perturb import protected_tokens, valid

    protected = frozenset({"red", "key", "keys", "two", "me", "you", "i", "not", "glasses"})
    assert protected_tokens("I'll get the red key, don't you?", protected) == {"i": 1, "red": 1, "key": 1, "not": 1,
                                                                             "you": 1}
    assert valid("Can you get two glasses?", "Could you fetch two glasses?", protected)
    assert not valid("Can you get two glasses?", "Could you fetch a couple of glasses?", protected)
    assert not valid("Can you get two glasses?", "Can you get two glasses?", protected)
    assert valid("Pass me the red key.", "pass me the red key.", protected)
    assert not valid("Get 2 keys.", "Get keys.", protected)


def test_perturbations_keep_protected_words_and_batches_put_the_original_first():
    from robust_likelihood.clips.perturb import batches, perturb_instruction, utterance_prompts

    protected = frozenset({"red", "key", "me", "you", "pass"})
    row = {"domain": "mdkg", "plan_id": "1.1.keys", "index": 0, "timestep": 7,
           "text": "Can you please pass me the red key?"}
    units = perturb_instruction(row, protected, 1, frozenset())
    kinds = {unit["kind"]: unit for unit in units}
    assert kinds["typo1"]["valid"] and "please" not in kinds["typo1"]["rewrite"]
    assert kinds["typo2"]["rewrite"] is None and not kinds["typo2"]["valid"]
    assert kinds["fill2"]["rewrite"].endswith(row["text"] + " " + kinds["fill2"]["added"][1])
    particles = [{"plan_id": "1.1.keys", "t": 7, "output": " " + row["text"], "prompts": ["P1", "P2"]},
                 {"plan_id": "1.1.keys", "t": 7, "output": " " + row["text"], "prompts": ["P2"]}]
    prompts = utterance_prompts(particles, [row])
    assert prompts == {"mdkg:1.1.keys:0": ["P1", "P2"]}
    found = batches(units, [row], prompts)
    assert len(found) == 2 and all(batch["outputs"][0] == " " + row["text"] for batch in found)
    assert len(found[0]["outputs"]) == 1 + sum(unit["valid"] for unit in units)


def test_swapping_an_instruction_moves_the_posterior_by_the_mixture_change():
    import math

    from robust_likelihood.clips.robustness import Problem, unit_effect

    particles = [{"plan_id": "p", "particle": 1, "goal": 1, "t": 3, "prompts": ["A"], "probs": [1.0]},
                 {"plan_id": "p", "particle": 2, "goal": 2, "t": 3, "prompts": ["B"], "probs": [1.0]}]
    final = {"trace_scores_1": "-1.0", "trace_scores_2": "-1.0", "utt_trace_scores_1": "-2.0",
             "utt_trace_scores_2": "-2.0"}
    problem = Problem("p", final, particles, true_goal=1)
    from robust_likelihood.clips.record import prompt_id

    logged = {prompt_id("A"): -2.0, prompt_id("B"): -2.0}.get
    original = logged
    perturbed = {prompt_id("A"): -2.0, prompt_id("B"): -2.0 + math.log(3)}.get
    found = unit_effect(problem, 3, logged, original, perturbed)
    assert found["p_true"] == 0.5 and abs(found["p_true_new"] - 0.25) < 1e-12 and found["flip"]
    assert abs(found["delta_log_bf"] + math.log(3)) < 1e-12
