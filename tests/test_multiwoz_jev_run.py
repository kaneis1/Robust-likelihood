import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from robust_likelihood.clients import TransportResult, build_request
from robust_likelihood.multiwoz import load_config
from robust_likelihood.multiwoz.__main__ import prepare
from robust_likelihood.multiwoz.hypotheses import value_pools
from robust_likelihood.multiwoz.jev_run import (
    UNUSED_HYPOTHESES,
    ModelIdChanged,
    choose_items,
    current_cache,
    plan_calls,
    run_jev,
)
from robust_likelihood.multiwoz.probe import build_probe
from robust_likelihood.planning import cache_key
from robust_likelihood.prompting import load_prompts
from robust_likelihood.storage import read_json, read_jsonl, repo_root, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"
FIXED = datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)
ENV = {"JEV_API_KEY": "test-key"}


class FakeJev:
    def __init__(self, models=("jev-1.13.0",)):
        self.models = list(models)
        self.bodies = []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(body)
        model = self.models[min(len(self.bodies), len(self.models)) - 1]
        value = (len(body["state"]["intention_description"]) % 90 + 5) / 100
        payload = {"model": model, "answers": {"plausibility": {"type": "noul", "noul": value}}}
        return TransportResult(200, payload, None, False)


class Boom:
    def __call__(self, *args, **kwargs):
        raise AssertionError("no HTTP call is allowed")


@pytest.fixture()
def derived(tmp_path):
    folder = tmp_path / "derived"
    prepare(repo_root(), load_config(repo_root()), raw=FIXTURE, derived=folder)
    return folder


def _run(derived, results, transport, **kwargs):
    options = {"wording": "both", "set_name": "all", "derived": derived, "results_root": results,
               "transport": transport, "environ": ENV, "sleeper": lambda seconds: None, "now": FIXED}
    options.update(kwargs)
    return run_jev(repo_root(), load_config(repo_root()), **options)


def test_request_body_keeps_the_intention_constant_and_sends_the_text(derived):
    rows = read_jsonl(derived / "probe.jsonl")
    specs = plan_calls(rows, ["SNG0002"], ("multiwoz_likely", "multiwoz_typical"), "jev-1.13.0", "find_restaurant")
    assert len(specs) == 2 * sum(1 for row in rows if row["item_id"] == "SNG0002")
    texts = {row["hyp_key"]: row["text"] for row in rows if row["item_id"] == "SNG0002"}
    for spec in specs:
        pack = load_prompts(repo_root(), spec.prompt_version)
        body = build_request(spec, pack, UNUSED_HYPOTHESES)
        assert set(body) == {"model", "state", "questions"}
        assert body["model"] == "jev-1.13.0"
        assert body["state"] == {
            "utterance": "Can you find me a Chinese place?",
            "stated_intention": "find_restaurant",
            "intention_description": texts[spec.role],
        }
        question = body["questions"]["plausibility"]
        assert question["type"] == "noul"
        assert question["instructions"] == pack.pairwise_question.replace("<<UTTERANCE>>", spec.input_text)
        assert "first message to a restaurant booking assistant" in question["instructions"]
        assert "temperature" not in body
        serialized = json.dumps(body)
        assert spec.role not in serialized and "impl" not in serialized and "ctrl" not in serialized
    likely = load_prompts(repo_root(), "multiwoz_likely").pairwise_question
    typical = load_prompts(repo_root(), "multiwoz_typical").pairwise_question
    assert "Is the user likely to produce the following utterance" in likely
    assert "Is the following utterance typical of what a user would say" in typical


def test_dry_run_makes_no_calls(derived, tmp_path, capsys):
    summary = _run(derived, tmp_path / "results", Boom(), dry_run=True)
    assert summary["api_calls"] == 0
    assert summary["planned_calls"] == 2 * len(read_jsonl(derived / "probe.jsonl"))
    assert len(summary["first_requests"]) == 3
    assert summary["first_requests"][0]["state"]["stated_intention"] == "find_restaurant"
    assert "planned_calls" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_live_run_scores_every_call_and_a_rerun_hits_the_cache(derived, tmp_path):
    results = tmp_path / "results"
    first = FakeJev()
    summary = _run(derived, results, first, label="first")
    planned = summary["planned_calls"]
    assert summary["http_requests"] == len(first.bodies) == planned
    assert summary["scored"] == planned and summary["failed"] == 0
    run_dir = Path(summary["run_dir"])
    assert run_dir.name == "20261026T090000Z-first"
    scores = read_jsonl(run_dir / "scores.jsonl")
    assert set(scores[0]) == {"item_id", "hyp_key", "prompt_version", "repeat_id", "noul"}
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["returned_model_ids"] == {"jev-1.13.0": planned}
    assert manifest["prompts"]["multiwoz_likely"]["pairwise_question"].startswith("Assume the user")
    assert manifest["git"]["commit"]
    assert len(read_jsonl(results / "cache.jsonl")) == planned
    second = FakeJev()
    again = _run(derived, results, second, label="second")
    assert second.bodies == []
    assert again["cache_hits"] == planned and again["http_requests"] == 0
    assert read_jsonl(Path(again["run_dir"]) / "scores.jsonl") == scores


def test_repeats_are_new_calls_and_the_baseline_comes_from_the_cache(derived, tmp_path):
    results = tmp_path / "results"
    _run(derived, results, FakeJev(), wording="likely", label="base")
    transport = FakeJev()
    summary = _run(derived, results, transport, wording="likely", repeats=2, label="repeats")
    baseline = summary["planned_calls"] // 3
    assert len(transport.bodies) == 2 * baseline
    assert summary["cache_hits"] == baseline
    repeat_ids = {row["repeat_id"] for row in read_jsonl(Path(summary["run_dir"]) / "scores.jsonl")}
    assert repeat_ids == {None, 1, 2}


def test_shared_utterances_are_scored_once_per_text(tmp_path):
    folder = tmp_path / "derived"
    item = {"set": "clean", "utterance": "I am looking for an expensive restaurant in the centre",
            "goal": {"area": "centre", "pricerange": "expensive"}, "n_constrained": 2,
            "slot_status": {"food": "omitted", "area": "named", "pricerange": "named"}}
    items = [{**item, "item_id": "SNG0901"}, {**item, "item_id": "SNG0902"}]
    write_jsonl(folder / "items.jsonl", items)
    write_jsonl(folder / "probe.jsonl", build_probe(items, value_pools(["thai"])))
    transport = FakeJev()
    summary = _run(folder, tmp_path / "results", transport, wording="likely", label="shared")
    rows = read_jsonl(folder / "probe.jsonl")
    assert summary["planned_calls"] == len(rows)
    assert len(transport.bodies) == len({(row["utterance"], row["text"]) for row in rows}) < len(rows)
    assert summary["scored"] == len(rows)


def test_a_changed_model_id_stops_the_run(derived, tmp_path):
    with pytest.raises(ModelIdChanged, match="jev-1.13.1"):
        _run(derived, tmp_path / "results", FakeJev(["jev-1.13.0", "jev-1.13.0", "jev-1.13.1"]), label="drift")
    run_dir = next((tmp_path / "results").glob("*-drift"))
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["stopped_reason"] == "returned_model_id_changed"
    assert len(read_jsonl(run_dir / "calls.jsonl")) == 3
    assert len(read_jsonl(tmp_path / "results" / "cache.jsonl")) == 2


def test_resume_finishes_without_repeating_calls(derived, tmp_path):
    results = tmp_path / "results"
    transport = FakeJev()
    stopped = _run(derived, results, transport, label="partial", max_requests=5)
    assert stopped["stopped_reason"] == "max_requests" and stopped["scored"] == 5
    finished = _run(derived, results, transport, resume=Path(stopped["run_dir"]), max_requests=None)
    assert finished["scored"] == finished["planned_calls"]
    assert len(transport.bodies) == finished["planned_calls"]
    assert finished["http_requests"] == finished["planned_calls"]


def test_a_cached_record_with_an_older_prompt_text_is_not_reused(derived):
    rows = read_jsonl(derived / "probe.jsonl")
    specs = plan_calls(rows, ["SNG0001"], ("multiwoz_likely",), "jev-1.13.0", "find_restaurant")
    packs = {"multiwoz_likely": load_prompts(repo_root(), "multiwoz_likely")}
    good = {"status": "success", "cache_key": cache_key(specs[0]),
            "prompt_text": packs["multiwoz_likely"].pairwise_question.replace("<<UTTERANCE>>", specs[0].input_text)}
    stale = {"status": "success", "cache_key": cache_key(specs[1]), "prompt_text": "an older question"}
    assert list(current_cache([good, stale], specs, packs)) == [good["cache_key"]]


def test_item_choice_is_seeded_and_sorted():
    items = [{"item_id": f"SNG{index:04d}", "set": "clean" if index % 3 else "fail_info"} for index in range(30)]
    first = choose_items(items, "clean", 5, 20261026)
    assert first == choose_items(items, "clean", 5, 20261026) == sorted(first)
    assert len(choose_items(items, "all")) == 30
    with pytest.raises(ValueError):
        choose_items(items, "fail_info", 50, 1)
