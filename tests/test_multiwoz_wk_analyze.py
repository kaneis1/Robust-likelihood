import pytest

from robust_likelihood.multiwoz.claim1 import ORIGINAL, ItemBoot
from robust_likelihood.multiwoz.wk_analyze import (
    orig,
    perturb_cells,
    perturb_log,
    predictions,
    raw_metrics,
    slope,
    teacher_choice,
    unit_rows,
)
from robust_likelihood.multiwoz.wk_quantile import quantile_scheme, quantile_scores, specs, weighted_quantile
from robust_likelihood.multiwoz.typv0 import load_typv0_config
from robust_likelihood.storage import repo_root


def test_weighted_quantile_and_scheme_masses():
    pairs = [(float(value), 1.0) for value in range(100)]
    assert weighted_quantile(pairs, 0.5) == 49.0 and weighted_quantile(pairs, 0.99) == 98.0
    scheme = quantile_scheme(pairs, [0.5, 0.7, 0.85, 0.95, 0.99], 0.995)
    assert scheme["masses"] == pytest.approx([0.49, 0.2, 0.15, 0.1, 0.04, 0.02])
    assert scheme["centres"][0] == pytest.approx(statistics_mean(range(49)))
    assert scheme["centres"][-1] == 99.0  # the 0.995 quantile caps the top bin
    weighted = [(0.0, 3.0), (1.0, 1.0)]
    assert weighted_quantile(weighted, 0.5) == 0.0 and weighted_quantile(weighted, 0.8) == 1.0


def statistics_mean(values):
    values = list(values)
    return sum(values) / len(values)


def test_quantile_specs_and_scores_on_given_cells():
    variants = specs(load_typv0_config(repo_root()))
    assert {"glob1Q", "cent1Q", "glob1Q:3q", "cent1Q:3q", "cent1B:6bins"} <= set(variants)
    assert variants["cent1Q:3q"]["quantiles"] == [0.5, 0.8, 0.9] and variants["cent1Q"]["top_centre_quantile"] == 0.995
    scheme = {"H": 10.0, "sigma": 2.0, "edges": [0.0, 1.0], "centres": [-1.0, 0.5, 3.0]}
    cells = [{"key": ("A", "G|impl", ORIGINAL, ORIGINAL), "S": 9.0}, {"key": ("A", "broad|impl", ORIGINAL, ORIGINAL), "S": 20.0}]
    log, bins, _ = quantile_scores(None, {"statistic": "S"}, {}, cells, scheme)
    assert bins == {cells[0]["key"]: 0, cells[1]["key"]: 2}
    assert log[cells[0]["key"]] == -(10.0 + 2.0 * -1.0) and log[cells[1]["key"]] == -(10.0 + 2.0 * 3.0)


def test_slope():
    assert slope([(1, 1.0), (2, 2.0), (4, 4.0)]) == pytest.approx(1.0)
    assert slope([(1, 1.0), (1, 2.0)]) is None


ROWS = [
    {"unit_id": "A|typo1", "item_id": "A", "kind": "typo1", "hyp_key": key, "text": key, "S_orig": s0, "S": s1,
     "tokens_orig": 10, "tokens": 11}
    for key, s0, s1 in (("G|impl", 10.0, 14.0), ("minus:food|impl", 12.0, 16.5), ("broad|impl", 15.0, 19.0),
                        ("ctrl:contra", 30.0, 33.0))
]
META = {"A": {"constrained": ["food"], "keys": ["G|impl", "minus:food|impl", "broad|impl", "ctrl:contra"],
              "texts": {}, "item": {"utterance": "x"}}}
UNITS = {"A|typo1": {"item_id": "A", "kind": "typo1", "dose": 1, "edits": [{"op": "swap", "start": 0}], "added": []}}


def test_perturb_log_cells_and_unit_rows():
    log = perturb_log(ROWS)
    assert log[("A", "G|impl", ORIGINAL, "typo1")] == -14.0 and log[("A", "G|impl", ORIGINAL, orig("typo1"))] == -10.0
    cells = perturb_cells(ROWS)
    centred = {cell["key"]: cell["cent"] for cell in cells}
    assert centred[("A", "G|impl", ORIGINAL, "typo1")] == 14.0 - 19.0
    assert centred[("A", "G|impl", ORIGINAL, orig("typo1"))] == 10.0 - 15.0
    found = unit_rows(log, META, UNITS, None)
    assert len(found) == 1
    row = found[0]
    assert row["delta_log_p"] == pytest.approx(-4.0)  # median of -4, -4.5, -4, -3
    pairs = {(pair["goal"], pair["other"]): pair for pair in row["pairs"]}
    assert pairs[("G|impl", "minus:food|impl")]["bf0"] == pytest.approx(2.0)
    assert pairs[("G|impl", "minus:food|impl")]["delta"] == pytest.approx(0.5)
    assert row["posterior"]["flip"] is False and row["op"] == "swap"
    boot = ItemBoot(["A"], 20, 1)
    metrics = raw_metrics(found, {}, boot)
    assert metrics["typo1"]["units"] == 1 and metrics["typo1"]["by_op"]["swap"]["median"] == pytest.approx(-4.0)


def test_teacher_choice_follows_the_rule():
    def models(q_para, q_typo, q_size, b_para, b_typo):
        return {"qwen3_base|F": {
            "part_c": {"cent1Q": {"agreement": {"para1": {"mean": q_para}}, "size_kept": {"impl": q_size}},
                       "cent1B:6bins": {"agreement": {"para1": {"mean": b_para}}}},
            "typicality": {"cent1Q": {"typo1": {"bin_change": {"mean": 1 - q_typo}}},
                           "cent1B:6bins": {"typo1": {"bin_change": {"mean": 1 - b_typo}}}}}}
    assert teacher_choice(models(0.80, 0.80, 1.2, 0.81, 0.81))["choice"] == "cent1Q"
    assert teacher_choice(models(0.70, 0.80, 1.2, 0.81, 0.81))["choice"] == "cent1B:6bins"
    assert teacher_choice(models(0.90, 0.90, 2.5, 0.81, 0.81))["choice"] == "cent1B:6bins"


def test_predictions_need_every_model():
    found = {item["id"]: item for item in predictions({})}
    assert not found["T1"]["pass"] and not found["T2"]["pass"] and not found["T4"]["pass"]
