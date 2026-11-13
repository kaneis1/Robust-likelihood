import json
from pathlib import Path

import pytest

from robust_likelihood.multiwoz.claim1 import ItemBoot
from robust_likelihood.multiwoz.sim_analyze import (
    discover,
    distinct_2,
    human_percentile,
    named,
    paired_difference,
    predictions,
    quantiles,
    style,
)


def test_human_percentile_counts_cut_off_samples_by_their_lower_bound():
    samples = [{"S": 5.0}, {"S": 20.0}, {"S": None, "S_lower": 30.0}, {"S": None, "S_lower": 1.0}]
    # 20 and the cut-off sample whose lower bound is 30 are at least 10; the one bounded by 1 is unknown.
    assert human_percentile(10.0, samples) == pytest.approx(2 / 3)
    assert human_percentile(100.0, [{"S": 5.0}, {"S": 6.0}]) == 0.0
    assert human_percentile(1.0, []) is None


def test_quantiles_distinct_and_style():
    assert quantiles([]) is None
    assert quantiles([float(value) for value in range(101)])["q50"] == 50.0
    assert distinct_2(["a b a b", "a b"]) == pytest.approx(2 / 4)
    found = style(["Hi, any cheap food?", "i want chepa food"], {"hi", "any", "cheap", "food", "i", "want"})
    assert found["greeting"] == 0.5 and found["question"] == 0.5 and found["non_dictionary"] == 0.5
    assert found["median_words"] == 4


def test_named_excludes_none_and_dontcare():
    assert named("italian") and not named("none") and not named("dontcare")


def test_paired_difference_uses_the_same_draws():
    boot = ItemBoot(["a", "b", "c", "d"], 200, 1)
    first = (["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0])
    second = (["a", "b", "c", "d"], [2.0, 3.0, 4.0, 5.0])
    found = paired_difference(boot, first, second, "mean")
    assert found["difference"] == pytest.approx(1.0)
    assert found["ci95"] == pytest.approx([1.0, 1.0])  # a constant shift survives every paired resample
    assert paired_difference(boot, first, second, "median")["difference"] == pytest.approx(1.0)


def test_discover_keeps_the_newest_finished_sim_score_run(tmp_path: Path):
    def run(name, manifest):
        path = tmp_path / name
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return path

    older = run("20261113T100000Z-sim-osim_mid-S", {"kind": "sim-score", "model_key": "osim_mid", "format": "S",
                                                     "finished_at": "x"})
    newer = run("20261113T110000Z-sim-osim_mid-S", {"kind": "sim-score", "model_key": "osim_mid", "format": "S",
                                                     "finished_at": "y"})
    run("20261113T120000Z-sim-osim_mid-S", {"kind": "sim-score", "model_key": "osim_mid", "format": "S"})
    run("20261113T130000Z-sim-check", {"kind": "sim-check"})
    assert discover([older, newer, tmp_path / "20261113T120000Z-sim-osim_mid-S", tmp_path / "20261113T130000Z-sim-check"]) \
        == {"osim_mid|S": newer}


def test_predictions_score_the_primary_runs():
    def block(pct, beyond, mu, a, b, para, extreme):
        return {"primary": True,
                "q1": {"all": {"median_percentile": {"median": pct}, "share_beyond_all": {"mean": beyond}}},
                "q2": {"naming": {"slots_mean": {"a_model": {"mean": a}, "b_model_impl": {"mean": b},
                                                  "a_corpus": 0.73, "b_corpus": 0.012}}},
                "q3": {"utterance": {"para": {"abs_delta": {"median": para}}, "zh": {"abs_delta": {"median": para}}}},
                "q4": {"glob1A": {"uniform": {"extreme_share": extreme}}},
                "q5": {"mean_sample_surprisal": mu}}

    runs = {"osim_mid|S": block(0.4, 0.1, 30.0, 0.75, 0.01, 5.0, 0.3),
            "osim_final|S": block(0.2, 0.3, 15.0, 0.85, 0.005, 5.0, 0.6),
            "userlm|S": block(0.3, 0.2, 25.0, 0.6, 0.02, 4.0, 0.4),
            "qwen25_broad|B": block(0.1, 0.5, 6.0, 0.95, 0.001, 6.0, 0.9),
            "qwen25_inst|S": block(0.0, 0.9, 5.0, 0.99, 0.0, 7.0, 1.0),
            "qwen3_inst|S": block(0.01, 0.8, 3.0, 0.97, 0.0, 7.0, 1.0)}
    found = {row["id"]: row["pass"] for row in predictions(runs)}
    assert found == {"S1": True, "S2": True, "S3": True, "S4": True, "S5": True, "S6": True}
