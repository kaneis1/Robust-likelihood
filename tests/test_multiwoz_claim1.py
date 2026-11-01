import math

import pytest

np = pytest.importorskip("numpy")

from robust_likelihood.multiwoz.claim1 import (  # noqa: E402
    ORIGINAL,
    ItemBoot,
    Scorer,
    bf_pairs,
    centered_log,
    clipped_log,
    item_meta,
    jev_scorers,
    posterior,
    rankdata,
    rewording,
    spearman,
    utterance_rows,
    verbal_scorers,
)

KEYS = ["G|impl", "G|expl", "minus:food|impl", "minus:food|expl", "broad|impl", "broad|expl", "ctrl:contra"]


def metas(items=("A", "B", "C")):
    found_items = [{"item_id": item, "goal": {"food": "italian"}, "set": "clean"} for item in items]
    probe = [{"item_id": item, "hyp_key": key, "text": f"{key} text"} for item in items for key in KEYS]
    return item_meta(found_items, probe)


def test_pairs_cover_removals_broad_and_contra():
    pairs = bf_pairs(metas()["A"])
    # Fewer than three constrained slots: the explicit pairs use G|expl (goal_key, as in metrics.py).
    assert ("G|expl", "minus:food|expl", "minus") in pairs and ("G|expl", "broad|expl", "broad") in pairs
    assert ("G|impl", "minus:food|impl", "minus") in pairs and ("G|impl", "broad|impl", "broad") in pairs
    assert ("G|impl", "ctrl:contra", "contra") in pairs and len(pairs) == 5


def test_clipped_and_centered_scores():
    log = {("A", "G|impl", ORIGINAL, ORIGINAL): -10.0, ("A", "broad|impl", ORIGINAL, ORIGINAL): -12.0,
           ("A", "ctrl:contra", ORIGINAL, ORIGINAL): -20.0, ("A", "G|impl", "hzh", ORIGINAL): -9.0,
           ("A", "G|impl", ORIGINAL, "pad"): -30.0, ("A", "broad|impl", ORIGINAL, "pad"): -29.0}
    clipped = clipped_log(log, 3.0)
    assert clipped[("A", "G|impl", ORIGINAL, ORIGINAL)] == 0.0
    assert clipped[("A", "broad|impl", ORIGINAL, ORIGINAL)] == -2.0
    assert clipped[("A", "ctrl:contra", ORIGINAL, ORIGINAL)] == -3.0
    assert clipped[("A", "G|impl", "hzh", ORIGINAL)] == 1.0  # measured against the best original text
    assert clipped[("A", "G|impl", ORIGINAL, "pad")] == -1.0  # the best under the padded utterance
    centered = centered_log(log)
    assert centered[("A", "G|impl", ORIGINAL, ORIGINAL)] == 2.0 and centered[("A", "G|impl", ORIGINAL, "pad")] == -1.0


def rewording_log(noise):
    log = {}
    for index, item in enumerate("ABC"):
        for rank, key in enumerate(KEYS):
            base = -10.0 * rank - 50.0 * index  # an item offset that centering removes
            log[(item, key, ORIGINAL, ORIGINAL)] = base
            for step, kind in enumerate(("hpara1", "hpara2", "hpara3", "hzh")):
                log[(item, key, kind, ORIGINAL)] = base + noise * (step + 1) * (-1) ** step
    return log


def test_rewording_icc_ratio_and_shift():
    boot = ItemBoot(["A", "B", "C"], 200, 1)
    exact = rewording(Scorer("s", "loglik", rewording_log(0.0), items="ABC"), metas(), boot)
    assert exact["icc1"] == pytest.approx(1.0) and exact["sd_ratio"]["median"] == 0.0
    assert exact["shift_abs"]["hzh"]["median"] == 0.0
    noisy = rewording(Scorer("s", "loglik", rewording_log(8.0), items="ABC"), metas(), boot)
    assert noisy["icc1"] < exact["icc1"] and noisy["sd_ratio"]["median"] > 0
    assert noisy["shift_abs"]["hzh"]["median"] == pytest.approx(32.0)
    assert noisy["shift_signed"]["hzh"]["median"] == pytest.approx(-32.0)


def test_utterance_rows_measure_the_bayes_factor_change_and_flips():
    log = {("A", "G|impl", ORIGINAL, ORIGINAL): -10.0, ("A", "broad|impl", ORIGINAL, ORIGINAL): -12.0,
           ("A", "G|impl", ORIGINAL, "pad"): -40.0, ("A", "broad|impl", ORIGINAL, "pad"): -39.0}
    rows = utterance_rows(Scorer("s", "loglik", log, items=["A"]), metas(["A"]))
    assert rows == [{"item_id": "A", "family": "broad", "kind": "pad", "bf0": 2.0, "bf1": -1.0, "delta": -3.0,
                     "flip": True, "raw": -30.0}]


def test_posterior_flips_only_when_a_rewording_changes_the_winner():
    log = {}
    for key in ("G|impl", "minus:food|impl", "broad|impl", "ctrl:contra"):
        log[("A", key, ORIGINAL, ORIGINAL)] = -10.0 if key == "G|impl" else -12.0
    stable = posterior(Scorer("s", "loglik", dict(log), items=["A"]), metas(["A"]), ItemBoot(["A"], 50, 1), seed=3)
    assert stable["sd_p_goal"]["mean"] == 0.0 and stable["argmax_flip_rate"]["mean"] == 0.0
    log[("A", "broad|impl", "hzh", ORIGINAL)] = -5.0
    moved = posterior(Scorer("s", "loglik", log, items=["A"]), metas(["A"]), ItemBoot(["A"], 50, 1), seed=3)
    assert 0 < moved["argmax_flip_rate"]["mean"] < 1 and moved["sd_p_goal"]["mean"] > 0


def test_ranks_share_ties_and_spearman_reads_order():
    assert rankdata([0.05, 0.01, 0.05, 0.2]).tolist() == [2.5, 1.0, 2.5, 4.0]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [5, 5, 5]) is None


def test_bootstrap_median_brackets_the_point():
    boot = ItemBoot(["A", "B", "C", "D"], 500, 7)
    found = boot.median(["A", "B", "C", "D"], [1.0, 2.0, 3.0, 4.0])
    assert found["median"] == 2.5 and found["ci95"][0] <= 2.5 <= found["ci95"][1]


def test_verbal_zeros_have_no_log_but_keep_their_probability():
    rows = [{"model": "qwen", "framing": "exact", "source": "original", "side": "none", "item_id": "A",
             "hyp_key": "G|impl", "p": 0.0},
            {"model": "claude", "framing": "exact", "source": "hpara1", "side": "hypothesis", "item_id": "A",
             "hyp_key": "G|impl", "p": 0.02, "status": "success"},
            {"model": "qwen", "framing": "exact", "source": "norm", "side": "sample", "item_id": "A",
             "hyp_key": "G|impl", "p": 0.05}]
    scorers, norm = verbal_scorers(rows)
    by_name = {scorer.name: scorer for scorer in scorers}
    qwen = by_name["qwen_verbal_exact"]
    assert qwen.like == {("A", "G|impl", ORIGINAL, ORIGINAL): 0.0} and qwen.log == {} and qwen.notes["zero"] == 1
    assert by_name["claude_verbal"].log == {("A", "G|impl", "hpara1", ORIGINAL): math.log(0.02)}
    assert len(norm) == 1 and norm[0]["scorer"] == "qwen_verbal_exact"


def test_jev_rewrites_are_read_from_their_tagged_keys():
    full = [{"item_id": "A", "hyp_key": "G|impl", "prompt_version": "multiwoz_likely", "repeat_id": None, "noul": 0.8}]
    rewrites = [{"item_id": "A", "hyp_key": "G|impl@zh", "prompt_version": "multiwoz_likely", "noul": 0.6},
                {"item_id": "A", "hyp_key": "G|impl@hzh", "prompt_version": "multiwoz_likely", "noul": 0.7}]
    (likely,) = jev_scorers(full, rewrites, {"A"}, 1e-3)
    assert set(likely.like) == {("A", "G|impl", ORIGINAL, ORIGINAL), ("A", "G|impl", ORIGINAL, "zh"),
                                ("A", "G|impl", "hzh", ORIGINAL)}
    assert likely.log[("A", "G|impl", ORIGINAL, ORIGINAL)] == pytest.approx(math.log(0.8 / 0.2))
