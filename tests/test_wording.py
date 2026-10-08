from robust_likelihood.clients import build_request
from robust_likelihood.planning import build_plan, cache_key
from robust_likelihood.prompting import WORDING_NOTES, load_hypotheses, load_prompts, pairwise_instructions
from robust_likelihood.runner import run_comparison
from robust_likelihood.storage import repo_root


EXAMPLE = 'I think I asked you to create a new alarm for seven. When was that?'
LIKELY = (
    "Assume the user has the stated intention described in the state. "
    "Is the user likely to produce the following utterance when expressing that intention? "
    f'Utterance: "{EXAMPLE}"'
)
TYPICAL = (
    "Assume the user has the stated intention described in the state. "
    "Is the following utterance typical of what a user would say when expressing that intention? "
    f'Utterance: "{EXAMPLE}"'
)


def _spec(prompt_version, utterance=EXAMPLE):
    root = repo_root()
    descriptions = load_hypotheses(root, "v1").descriptions
    plan = build_plan(
        [
            {
                "example_id": "example",
                "original_example_id": "example",
                "transformation": "original",
                "utt": utterance,
                "intent": "alarm_query",
                "original_intent": "alarm_query",
                "label_changed": False,
            }
        ],
        ("jev",),
        {"jev": "jev-1.13.0"},
        descriptions,
        prompt_version,
        "v1",
        {"jev": {}},
    )
    return next(spec for spec in plan if spec.task == "pairwise" and spec.hypothesis == "alarm_query")


def test_likely_and_typical_keep_state_and_replace_the_question():
    root = repo_root()
    hypotheses = load_hypotheses(root, "v1")
    spec = _spec("likely")
    for version, expected in (("likely", LIKELY), ("typical", TYPICAL)):
        prompts = load_prompts(root, version)
        body = build_request(spec, prompts, hypotheses)
        assert set(body["state"]) == {"utterance", "stated_intention", "intention_description"}
        assert body["state"]["utterance"] == EXAMPLE
        assert body["state"]["stated_intention"] == "alarm_query"
        assert body["state"]["intention_description"] == hypotheses.descriptions["alarm_query"]
        assert body["questions"]["plausibility"]["instructions"] == expected
        assert pairwise_instructions(spec, prompts) == expected
    assert "surprisal" in WORDING_NOTES["typical"]
    assert "probably" in WORDING_NOTES["likely"]


def test_wording_versions_do_not_share_a_cache_key_with_v2():
    root = repo_root()
    v2 = _spec("v2")
    likely = _spec("likely")
    typical = _spec("typical")
    assert len({cache_key(v2), cache_key(likely), cache_key(typical)}) == 3
    old = load_prompts(root, "v2")
    assert old.pairwise_question == (
        "Assume the user has the stated intention. Is this utterance a plausible expression of that intention?"
    )


def test_likely_dry_run_keeps_hypothesis_definitions_and_plans_jev_only():
    root = repo_root()
    summary = run_comparison(
        root,
        providers=("jev",),
        dry_run=True,
        dataset="synthetic_families",
        split="holdout",
        prompt_version="likely",
    )
    assert summary["planned_requests"] == {"classification": 0, "pairwise": 696, "total": 696}
    assert summary["prompt_version"] == "likely"
    assert summary["hypothesis_description_version"] == "synthetic_v2"
    assert "ordinary-language typicality" not in (summary["wording_note"] or "")
    try:
        run_comparison(root, providers=("jev",), dry_run=True, dataset="toy_likelihood", prompt_version="likely")
    except ValueError as exc:
        assert "toy_likelihood" in str(exc)
    else:
        raise AssertionError("toy likelihood accepted an alarm wording")
