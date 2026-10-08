from robust_likelihood.planning import summarize_plan
from robust_likelihood.prompting import load_hypotheses
from robust_likelihood.scoring import predict
from robust_likelihood.storage import repo_root
from robust_likelihood.synthetic import build_synthetic_plan, evaluate_synthetic_records, load_synthetic
from robust_likelihood.synthetic_items import INTENTION_HYPOTHESES, authored_items


def _record(example_id, provider, hypothesis, probability):
    return {
        "cache_key": f"{provider}:{example_id}:{hypothesis}",
        "provider": provider,
        "example_id": example_id,
        "task": "pairwise",
        "hypothesis": hypothesis,
        "status": "success",
        "attempt_id": 0,
        "parsed": {"plausibility_yes_probability": probability},
    }


def _bundle(example_id, provider, scores):
    return [_record(example_id, provider, hypothesis, scores[hypothesis]) for hypothesis in INTENTION_HYPOTHESES]


def test_synthetic_dataset_is_separate_and_plans_five_intentions():
    root = repo_root()
    items, review, hypotheses, massive = load_synthetic(root)
    assert len(items) == 22
    assert {item["family"] for item in items} == {
        "meaning_preserved",
        "meaning_reversed",
        "negation_correction",
        "quoted_command",
        "multiple_commands",
        "underspecified",
    }
    assert set(hypotheses.state_questions) == {"alarm_is_enabled", "alarm_exists"}
    assert set(hypotheses.state_questions).isdisjoint(hypotheses.descriptions)
    assert set(massive.descriptions) == {"alarm_set", "alarm_query", "alarm_remove"}
    assert all(row["approved"] is True for row in review["items"])
    specs = build_synthetic_plan(
        items,
        ("jev", "gpt", "claude"),
        {"jev": "jev-1.13.0", "gpt": "gpt-6-astra", "claude": "claude-fable-5-1"},
        hypotheses,
        massive,
        "v2",
        {"jev": {}, "gpt": {}, "claude": {"max_tokens": 16000}},
    )
    summary = summarize_plan(specs)
    assert summary == {"classification": 30, "pairwise": 330, "total": 360}
    assert {spec.hypothesis for spec in specs if spec.task == "pairwise"} == set(INTENTION_HYPOTHESES)
    classified = [spec for spec in specs if spec.task == "classification"]
    assert classified
    assert all(spec.hypothesis_description_version == "v1" for spec in classified)
    assert all(spec.family != "underspecified" for spec in classified)
    context = next(spec for spec in specs if spec.example_id == "syn-under-seven-query")
    assert context.input_text.startswith("Context: Is my seven o'clock alarm enabled?")
    assert load_hypotheses(root, "v1").state_questions == {}


def test_synthetic_checks_follow_the_support_map():
    items = [
        item
        for item in authored_items()
        if item["pair_id"] in {"syn-mp-first", "syn-mr-first", "syn-neg-cancel", "syn-multi-seven-eight", "syn-under-seven"}
    ]
    enable = {hypothesis: 0.05 for hypothesis in INTENTION_HYPOTHESES}
    enable["intend_enable"] = 0.9
    disable = {hypothesis: 0.05 for hypothesis in INTENTION_HYPOTHESES}
    disable["intend_disable"] = 0.88
    query = {hypothesis: 0.05 for hypothesis in INTENTION_HYPOTHESES}
    query["intend_query"] = 0.8
    both = {hypothesis: 0.05 for hypothesis in INTENTION_HYPOTHESES}
    both["intend_delete"] = 0.7
    both["intend_create"] = 0.75
    spread = {hypothesis: 0.4 for hypothesis in INTENTION_HYPOTHESES}
    spread["intend_query"] = 0.45
    records = []
    for example_id, scores in {
        "syn-mp-first-source": enable,
        "syn-mp-first-variant": enable,
        "syn-mr-first-source": enable,
        "syn-mr-first-variant": disable,
        "syn-neg-cancel": query,
        "syn-multi-seven-eight": both,
        "syn-under-seven-bare": spread,
        "syn-under-seven-query": spread,
        "syn-under-seven-cancel": {**spread, "intend_delete": 0.7},
    }.items():
        records.extend(_bundle(example_id, "jev", scores))
    metrics, disagreements = evaluate_synthetic_records(records, items)
    assert metrics["coverage"]["by_provider"]["jev"]["succeeded"] == len(records)
    assert metrics["meaning_preserved"]["by_provider"]["jev"]["passed"] == 1
    assert metrics["meaning_reversed"]["by_provider"]["jev"]["passed"] == 1
    assert metrics["negation_correction"]["by_provider"]["jev"]["passed"] == 1
    assert metrics["multiple_commands"]["by_provider"]["jev"]["passed"] == 1
    assert metrics["underspecified"]["by_provider"]["jev"]["accuracy"] is None
    bare = metrics["underspecified"]["by_provider"]["jev"]["bare"]["scores"]
    assert predict(bare) == "tie" or bare["intend_query"] == 0.45
    cancel = metrics["underspecified"]["by_provider"]["jev"]["contexts"][-1]
    assert abs(cancel["signed_change_from_bare"]["intend_delete"] - 0.3) < 1e-9
    assert disagreements == []
