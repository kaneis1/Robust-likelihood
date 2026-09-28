from datetime import datetime, timezone

import pytest

from robust_likelihood.clients import CallResult
from robust_likelihood.planning import UnpinnedModelError, build_plan
from robust_likelihood.runner import connect_models, execute, run_comparison
from robust_likelihood.storage import read_jsonl, repo_root


DESCRIPTIONS = {
    "alarm_set": "set an alarm",
    "alarm_query": "ask about an alarm",
    "alarm_remove": "cancel an alarm",
}


def _item():
    return {
        "example_id": "1__original",
        "original_example_id": "1",
        "transformation": "original",
        "utt": "set an alarm for seven am",
        "intent": "alarm_set",
        "original_intent": "alarm_set",
        "label_changed": False,
    }


def _specs(repeats=0):
    return build_plan(
        [_item()],
        ("jev",),
        {"jev": "jev-1.13.0"},
        DESCRIPTIONS,
        "v1",
        "v1",
        {"jev": {}},
        repeats=repeats,
    )


def _result(status="success", retryable=False, parsed=None):
    return CallResult(
        status=status,
        retryable=retryable,
        http_status=200 if status != "failed" else 500,
        returned_model_id="jev-1.13.0",
        response_body={"model": "jev-1.13.0"},
        parsed=parsed
        or {"distribution": {"alarm_set": 1.0, "alarm_query": 0.0, "alarm_remove": 0.0}},
        usage={"input_tokens": 1, "output_tokens": 1},
        error=None if status == "success" else "HTTP 500",
        request_body={"model": "jev-1.13.0"},
        prompt_text="question",
        latency_ms=4.0,
    )


class Scripted:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def perform(self, spec, prompts, hypotheses):
        self.calls += 1
        return self.results.pop(0)


def test_retries_share_a_cache_key_and_use_attempt_ids():
    client = Scripted([_result("failed", True), _result("success")])
    sleeps = []
    outcome = execute(
        _specs()[:1],
        {"jev": client},
        None,
        None,
        max_requests=None,
        max_retries=2,
        backoff_seconds=1,
        sleeper=sleeps.append,
        success_cache={},
        skip_keys=set(),
    )
    assert client.calls == 2
    assert sleeps == [1]
    assert [row["attempt_id"] for row in outcome["records"]] == [0, 1]
    assert outcome["records"][0]["cache_key"] == outcome["records"][1]["cache_key"]
    assert outcome["records"][0]["repeat_id"] is None


def test_fatal_http_is_not_retried_and_budget_stops_the_run():
    client = Scripted([_result("failed", False)])
    sleeps = []
    outcome = execute(
        _specs()[:1],
        {"jev": client},
        None,
        None,
        max_requests=None,
        max_retries=2,
        backoff_seconds=1,
        sleeper=sleeps.append,
        success_cache={},
        skip_keys=set(),
    )
    assert client.calls == 1
    assert sleeps == []
    budget_client = Scripted([_result("failed", True), _result("success")])
    limited = execute(
        _specs(),
        {"jev": budget_client},
        None,
        None,
        max_requests=1,
        max_retries=2,
        backoff_seconds=1,
        sleeper=sleeps.append,
        success_cache={},
        skip_keys=set(),
    )
    assert budget_client.calls == 1
    assert limited["stopped_reason"] == "max_requests"
    assert len(limited["records"]) == 1


def test_cache_and_resume_do_not_repeat_successful_calls():
    spec = _specs()[:1]
    client = Scripted([_result()])
    first = execute(
        spec,
        {"jev": client},
        None,
        None,
        max_requests=None,
        max_retries=0,
        backoff_seconds=0,
        sleeper=lambda seconds: None,
        success_cache={},
        skip_keys=set(),
    )
    cached = {first["records"][0]["cache_key"]: first["records"][0]}
    again = Scripted([_result()])
    second = execute(
        spec,
        {"jev": again},
        None,
        None,
        max_requests=None,
        max_retries=0,
        backoff_seconds=0,
        sleeper=lambda seconds: None,
        success_cache=cached,
        skip_keys=set(),
    )
    assert again.calls == 0
    assert second["records"][0]["cache_hit"] is True
    assert second["records"][0]["latency_ms"] is None
    resumed = execute(
        spec,
        {"jev": again},
        None,
        None,
        max_requests=None,
        max_retries=0,
        backoff_seconds=0,
        sleeper=lambda seconds: None,
        success_cache={},
        skip_keys={first["records"][0]["cache_key"]},
    )
    assert resumed["records"] == []
    assert again.calls == 0


def test_repeat_ids_are_separate_calls():
    specs = [spec for spec in _specs(repeats=1) if spec.task == "classification"]
    assert len(specs) == 2
    assert cache_keys(specs)
    client = Scripted([_result(), _result()])
    outcome = execute(
        specs,
        {"jev": client},
        None,
        None,
        max_requests=None,
        max_retries=0,
        backoff_seconds=0,
        sleeper=lambda seconds: None,
        success_cache={},
        skip_keys=set(),
    )
    assert client.calls == 2
    assert outcome["records"][0]["repeat_id"] is None
    assert outcome["records"][1]["repeat_id"] == 1
    assert outcome["records"][0]["cache_key"] != outcome["records"][1]["cache_key"]


def cache_keys(specs):
    from robust_likelihood.planning import cache_key

    assert cache_key(specs[0]) != cache_key(specs[1])
    return True


def test_alias_is_refused_before_a_comparison_call():
    class Boom:
        def perform(self, *args, **kwargs):
            raise AssertionError("http")

    with pytest.raises(UnpinnedModelError):
        run_comparison(
            repo_root(),
            providers=("jev",),
            model_id_overrides={"jev": "jev-latest"},
            clients={"jev": Boom()},
        )


def test_connectivity_allows_an_alias_and_makes_one_call(tmp_path):
    captured = {}

    class Fake:
        def probe(self, model_id, utterances, settings):
            captured["model_id"] = model_id
            captured["utterances"] = tuple(utterances)
            captured["settings"] = settings
            return _result()

    fixed = datetime(2026, 9, 27, tzinfo=timezone.utc)
    connect_models(
        repo_root(),
        ("jev",),
        jev_model="jev-latest",
        clients={"jev": Fake()},
        results_root=tmp_path,
        now=fixed,
    )
    assert captured["model_id"] == "jev-latest"
    assert len(captured["utterances"]) == 3
    assert captured["settings"] == {}
    assert list(tmp_path.glob("connectivity-*.json"))


def test_dry_run_prints_480_without_calls():
    root = repo_root()
    if not (root / "data" / "pilot" / "draft_inputs.jsonl").exists():
        pytest.skip("pilot drafts have not been prepared")
    summary = run_comparison(root, providers=("jev", "gpt"), dry_run=True)
    assert summary["api_calls"] == 0
    assert summary["planned_requests"] == {"classification": 120, "pairwise": 360, "total": 480}
    assert summary["unapproved_items"] == 0
    assert summary["live_run_allowed"] is True


def test_unreviewed_live_run_is_exploratory(tmp_path, monkeypatch):
    root = repo_root()
    drafts = root / "data" / "pilot" / "draft_inputs.jsonl"
    if not drafts.exists():
        pytest.skip("pilot drafts have not been prepared")
    monkeypatch.setattr(
        "robust_likelihood.runner.unapproved_ids",
        lambda items, review: ["722__original"],
    )

    class Boom:
        def perform(self, *args, **kwargs):
            raise AssertionError("http")

    from robust_likelihood.runner import UnreviewedDrafts

    with pytest.raises(UnreviewedDrafts):
        run_comparison(root, providers=("jev", "gpt"), clients={"jev": Boom(), "gpt": Boom()}, results_root=tmp_path)
    fixed = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)
    summary = run_comparison(
        root,
        providers=("jev", "gpt"),
        allow_unreviewed=True,
        max_requests=0,
        clients={"jev": Boom(), "gpt": Boom()},
        results_root=tmp_path,
        now=fixed,
    )
    assert summary["http_requests"] == 0
    assert summary["exploratory"] is True
    assert "exploratory" in summary["run_dir"]
    manifest = read_jsonl(tmp_path / "exploratory-20260927T080000Z" / "responses.jsonl")
    assert manifest == []
    stored = __import__("json").loads(
        (tmp_path / "exploratory-20260927T080000Z" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert stored["exploratory"] is True
    assert stored["prompts"]["pairwise_question"].startswith("Assume the user has the stated intention.")
