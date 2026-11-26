import pytest

from robust_likelihood.multiwoz.wk_student_data import (
    build_pairs,
    held_out_foods,
    sample_split,
    scheme_for,
    target,
)
from test_multiwoz_sim_score import TinyTokenizer


class FoodExtractor:
    def extract(self, text):
        for food in ("thai", "french", "italian"):
            if food in text.lower():
                return {"food": food}
        return {"food": "none"}


SCHEME = {"H": -5.0, "sigma": 2.0, "edges": [-2.0, -1.0, 0.0, 1.0, 2.0], "centres": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
          "clip": [-2.5, 2.0]}


def test_held_out_foods_take_a_seeded_fifth():
    items = [{"goal": {"food": food}} for food in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")] + [{"goal": {}}]
    found = held_out_foods(items)
    assert len(found) == 2 and found == held_out_foods(list(reversed(items))) and found == sorted(found)


def test_target_clips_and_bins():
    found = target(-5.0 + 2.0 * 3.0, SCHEME)
    assert found["T"] == pytest.approx(3.0) and found["T_clip"] == 2.0 and found["bin"] == 5
    assert target(-5.0, SCHEME)["bin"] == 3
    fixed = scheme_for("cent1B:6bins", [(float(v), 1.0) for v in range(-3, 4)], -5.0, 2.0,
                       {"sensitivity": {"edges": {"6bins": [-2.0, -1.0, 0.0, 1.0, 2.0]}}})
    assert fixed["centres"] == [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] and fixed["clip"] == [-3.0, 3.0]


def test_build_pairs_splits_by_sample_and_holds_out_foods():
    samples = [{"text": "wants thai", "sample": 0, "S": 10.0, "S_broad": 14.0, "reply": "thai please"},
               {"text": "wants thai", "sample": 1, "S": None, "S_broad": None, "reply": "cut"},
               {"text": "wants thai", "sample": 2, "S": 9.0, "S_broad": 12.0, "reply": " "},
               {"text": "wants french", "sample": 0, "S": 8.0, "S_broad": 11.0, "reply": "french food"},
               {"text": "wants food", "sample": 0, "S": 7.0, "S_broad": 9.0, "reply": "any food"}]
    xscore = [{"text_from": "wants thai", "sample": 0, "text": "wants food", "S": 13.0},
              {"text_from": "wants food", "sample": 0, "text": "wants french", "S": 12.0}]
    pool = {"wants thai", "wants french", "wants food"}
    pairs, counts = build_pairs(samples, xscore, pool, {"french"}, FoodExtractor(), SCHEME)
    assert counts["cut_off"] == 1 and counts["empty"] == 1 and counts["usable_samples"] == 3
    by = {(pair["kind"], pair["text_from"], pair["h"]): pair for pair in pairs}
    assert by[("matched", "wants french", "wants french")]["split"] == "held_out_food"
    assert by[("cross", "wants food", "wants french")]["split"] == "held_out_food"
    matched, cross = by[("matched", "wants thai", "wants thai")], by[("cross", "wants thai", "wants food")]
    assert matched["split"] == cross["split"] == sample_split("wants thai\x1f0")
    assert matched["T"] == pytest.approx(((10.0 - 14.0) + 5.0) / 2.0)
    assert cross["T"] == pytest.approx(((13.0 - 14.0) + 5.0) / 2.0)


@pytest.mark.parametrize("run", ["R-lin", "C-mlp"])
def test_student_learns_a_toy_target_on_a_tiny_qwen3(run):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.multiwoz.wk_student import RUNS, SETTINGS, build_student, encode, predict, train_student

    torch.manual_seed(0)
    config = transformers.Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    backbone = transformers.Qwen3Model(config).eval()
    settings = dict(SETTINGS, batch_size=8, epochs=3, eval_every=5, lr_lora=1e-2, lr_head=5e-2)
    spec = RUNS[run]
    student, wrapped = build_student(backbone, 16, spec["head"], 1 if spec["loss"] == "mse" else 2, settings)
    assert wrapped == 7  # q, k, v, o, gate, up, down
    assert all(not parameter.requires_grad for name, parameter in student.named_parameters()
               if name.endswith(".weight") and ".base." in name)
    tokenizer = TinyTokenizer()
    rows = []
    for index in range(48):
        cheap = index % 2 == 0
        rows.append({"h": "cheap" if cheap else "dear", "x": "aaaa" if cheap else "zzzz z", "T_clip": 1.0 if cheap else -1.0,
                     "bin": int(cheap)})
    for row in rows:
        row["ids"] = encode(tokenizer, row["h"], row["x"], 64)
        assert row["ids"][-1] == tokenizer.eos_token_id
    result = train_student(student, rows[:40], rows[40:], spec, settings, 0, torch.device("cpu"), log=lambda _: None)
    assert result["steps"] == 15 and result["curve"] and result["state"]
    assert result["best_val_loss"] == min(row["val_loss"] for row in result["curve"])
    found, speed = predict(student, rows[40:], 0, torch.device("cpu"))
    assert len(found) == 8 and speed > 0
    if run == "R-lin":
        assert all(isinstance(value, float) for value in found)
    else:
        assert all(len(value) == 2 and sum(value) == pytest.approx(1.0) for value in found)


def test_student_values_and_verdict():
    from robust_likelihood.multiwoz.wk_student_eval import key_of, student_values, verdict

    found = student_values(0.5, SCHEME)
    assert found["bin"] == 3 and found["cont"] == -(-5.0 + 2.0 * 0.5) and found["bin_score"] == -(-5.0 + 2.0 * 0.5)
    probs = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    found = student_values(probs, SCHEME)
    assert found["T"] == pytest.approx(1.0) and found["bin"] == 3
    assert found["expected"] == pytest.approx(0.5 * -(-5.0 + 1.0) + 0.5 * -(-5.0 + 3.0))
    assert key_of({"set": "hrewrite", "item_id": "A", "hyp_key": "G|impl", "kind": "hpara1"}) == ("A", "G|impl", "hpara1",
                                                                                                    "original")
    assert key_of({"set": "synthetic_val"}) is None

    def students(rho, kept, contra):
        return {"students": {"R-lin": {"agreement": {"all": {"spearman": rho}, "synthetic_val": {"spearman": 0.9}},
                                       "forms": {"cont": {"all": {"size_kept": {"named": kept}, "contra_log_bf": contra}}}}}}
    assert verdict(students(0.8, 1.2, 4.0))["verdict"] == "The pipeline works"
    assert verdict(students(0.8, 3.0, 4.0))["verdict"] == "Learns the teacher but not the task"
    assert verdict(students(0.5, 1.0, 4.0))["verdict"] == "Does not learn the teacher"


def test_top1_credit_splits_ties():
    from robust_likelihood.multiwoz.wk_student_eval import top1_credit

    assert top1_credit([-1.0, -2.0, -3.0]) == 1.0
    assert top1_credit([-2.0, -1.0, -3.0]) == 0.0
    assert top1_credit([-1.0, -1.0, -1.0, -4.0]) == pytest.approx(1 / 3)
    assert top1_credit([-2.0, -1.0, -1.0]) == 0.0
