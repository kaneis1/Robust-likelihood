import math
import threading
from collections import Counter
from datetime import datetime, timezone

import pytest

from robust_likelihood.clients import TransportResult
from robust_likelihood.dice import load_config
from robust_likelihood.dice.__main__ import build
from robust_likelihood.dice.analyze import analyze, ols
from robust_likelihood.dice.design import (
    S1_SYSTEM,
    build_normalization,
    build_pairs,
    draw_sequences,
    face_probabilities,
    pilot_pairs,
    s1_messages,
    verbal_messages,
)
from robust_likelihood.dice.score import score_api
from robust_likelihood.storage import read_json, read_jsonl, repo_root, write_json, write_jsonl

FIXED = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
ENV = {"ANTHROPIC_API_KEY": "test-claude", "OPENAI_API_KEY": "test-gpt"}


@pytest.fixture(scope="module")
def config():
    return load_config(repo_root())


def test_dice_faces_sum_to_one_and_bad_dice_are_refused(config):
    dice = face_probabilities(config)
    assert set(dice) == {"fair", "even", "loaded"}
    broken = {"hypotheses": {"bad": {"wording": "w", "faces": {str(face): "1/5" for face in range(1, 7)}}}}
    with pytest.raises(ValueError):
        face_probabilities(broken)


def test_sequences_are_reproducible_and_follow_their_die(config):
    first, second = draw_sequences(config), draw_sequences(config)
    assert first == second and len(first) == 420
    assert set(Counter((row["k"], row["generator"]) for row in first).values()) == {20}
    assert all(face % 2 == 0 for row in first if row["generator"] == "even" for face in row["faces"])
    assert sum(1 for row in first if row["api_subset"]) == 210
    assert all(len(row["faces"]) == row["k"] for row in first)


def test_pairs_carry_exact_likelihoods(config):
    pairs = build_pairs(config, draw_sequences(config))
    assert len(pairs) == 1260
    for row in pairs:
        if row["hypothesis"] == "fair":
            assert row["true_logp"] == pytest.approx(-row["k"] * math.log(6))
        if row["hypothesis"] == "even":
            assert (row["true_logp"] is None) == any(face % 2 for face in row["faces"])
        if row["hypothesis"] == "loaded":
            sixes = row["faces"].count(6)
            assert row["true_logp"] == pytest.approx(sixes * math.log(0.5) + (row["k"] - sixes) * math.log(0.1))


def test_normalization_sets_sum_to_one(config):
    norm = build_normalization(config)
    assert len(norm) == 54
    totals = Counter()
    for row in norm:
        totals[row["norm_set"]] += row["true_p"]
    assert set(totals) == {"k1_fair", "k1_even", "k1_loaded", "k2_fair"}
    assert all(total == pytest.approx(1.0) for total in totals.values())


def test_prompts_are_the_frozen_texts(config):
    wording = config["hypotheses"]["fair"]["wording"]
    assert s1_messages(wording, 3) == [
        {"role": "system", "content": S1_SYSTEM},
        {"role": "user", "content": "The die: a fair six-sided die. Roll it 3 times and report the results in order as "
                                    "a comma-separated list of numbers. Reply with the list only."},
    ]
    assert verbal_messages("S2", wording, 2, "3, 5")[0]["content"].startswith(
        "Someone rolled a fair six-sided die 2 times and reported the results in order: 3, 5. How likely")
    assert verbal_messages("S3", wording, 2, "3, 5")[0]["content"].startswith(
        "A fair six-sided die is rolled 2 times. What is the probability that the results, in order, are exactly 3, 5?")


def test_pilot_is_a_seeded_sample_of_the_api_subset(config):
    pairs = build_pairs(config, draw_sequences(config))
    chosen = pilot_pairs(pairs, 30, 20261027)
    assert chosen == pilot_pairs(pairs, 30, 20261027)
    assert len(chosen) == 30 and all(row["api_subset"] for row in chosen)


def test_ols_recovers_a_line():
    fit = ols([1, 2, 3, 4], [1.0, -1.0, -3.0, -5.0])
    assert fit["slope"] == pytest.approx(-2.0) and fit["intercept"] == pytest.approx(3.0) and fit["r2"] == pytest.approx(1.0)


def _local_run(path, model, rows):
    write_json(path / "manifest.json", {"kind": "local", "model_key": model, "finished_at": "x"})
    write_jsonl(path / "scores.jsonl", rows)


def test_an_exact_scorer_passes_d1_and_a_flat_verbal_scorer_passes_d2(tmp_path, config):
    data = tmp_path / "data"
    build(repo_root(), config, data)
    pairs, norm = read_jsonl(data / "pairs.jsonl"), read_jsonl(data / "normalization.jsonl")
    queries = pairs + norm
    exact = [{"query_id": row["query_id"], "scorer": "S1", "model": "oracle",
              "logp": row["true_logp"] if row["true_logp"] is not None else -60.0} for row in queries]
    flat = [{"query_id": row["query_id"], "scorer": "S2", "model": "oracle", "raw": "0.5", "p": 0.5, "cut_off": False}
            for row in queries]
    _local_run(tmp_path / "run1", "oracle", exact + flat)
    small = {**config, "bootstrap": {"replicates": 50, "seed": 1}, "local_models": {"oracle": {}}}
    metrics = analyze(repo_root(), small, [tmp_path / "run1"], data)
    s1 = metrics["sources"]["oracle:S1"]
    assert s1["slopes"]["fair"]["fit"]["slope"] == pytest.approx(-math.log10(6))
    assert s1["slopes"]["fair"]["ratio_to_truth"] == pytest.approx(1.0)
    assert s1["bayes"]["fit"]["slope"] == pytest.approx(1.0)
    assert s1["normalization"]["k2_fair"]["sum"] == pytest.approx(1.0)
    assert s1["k1_accuracy"]["max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    s2 = metrics["sources"]["oracle:S2"]
    assert s2["slopes"]["fair"]["fit"]["slope"] == pytest.approx(0.0)
    assert s2["normalization"]["k2_fair"]["sum"] == pytest.approx(18.0)
    results = {row["id"]: row["pass"] for row in metrics["predictions"]}
    assert results == {"D1": True, "D2": True, "D3": False, "D4": True}
    assert metrics["acceptance"]["local_parse_failures_under_2pct"] is True


class FakeChat:
    def __init__(self):
        self.bodies = []
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.bodies.append(body)
        if "anthropic" in url:
            if "temperature" in body:
                return TransportResult(400, {"error": {"message": "`temperature` is deprecated for this model"}},
                                       None, False)
            return TransportResult(200, {"model": body["model"], "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "1/6"}],
                                         "usage": {"input_tokens": 60, "output_tokens": 300}}, None, False)
        return TransportResult(200, {"model": body["model"] + "-2026", "usage": {"prompt_tokens": 60,
                                                                                 "completion_tokens": 4},
                                     "choices": [{"message": {"content": "0.0046"}, "finish_reason": "stop"}]},
                               None, False)


def test_api_pilot_caches_answers_and_estimates_cost(tmp_path, config):
    data = tmp_path / "data"
    build(repo_root(), config, data)
    transport = FakeChat()
    first = score_api(repo_root(), config, ["claude", "gpt"], pilot=5, data_dir=data, results_root=tmp_path / "results",
                      transport=transport, environ=ENV, sleeper=lambda seconds: None, now=FIXED, workers=2)
    assert first["providers"]["claude"]["temperature_sent"] is None
    assert first["providers"]["gpt"]["temperature_sent"] == 0.0
    assert first["providers"]["claude"]["status"] == {"success": 10}
    estimate = first["providers"]["claude"]["cost_estimate"]
    assert estimate["full_run_calls"] == (630 + 54) * 2
    assert estimate["full_run_usd"] == pytest.approx((60 * 10 + 300 * 50) * 1368 / 1e6, abs=0.01)
    calls = read_jsonl(tmp_path / "results" / "20261029T090000Z-api-pilot" / "calls.jsonl")
    assert len(calls) == 20 and all(row["p"] is not None for row in calls)
    sent = len(transport.bodies)
    again = score_api(repo_root(), config, ["claude", "gpt"], pilot=5, data_dir=data, results_root=tmp_path / "results",
                      transport=transport, environ=ENV, sleeper=lambda seconds: None, now=FIXED, label="again")
    assert len(transport.bodies) == sent and again["http_calls"] == 0
    manifest = read_json(tmp_path / "results" / "20261029T090000Z-again" / "manifest.json")
    assert manifest["providers"]["gpt"]["cache_hits"] == 10


def test_greedy_config_sets_every_decoding_option():
    pytest.importorskip("transformers")
    from robust_likelihood.verbal import greedy_config

    config = greedy_config(end=2, pad=0, max_new_tokens=4)
    assert config.do_sample is False and config.num_beams == 1
    assert config.repetition_penalty == 1.0 and config.no_repeat_ngram_size == 0
    assert config.eos_token_id == 2 and config.max_new_tokens == 4
