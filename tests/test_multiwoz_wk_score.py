import json
from pathlib import Path

import pytest

from robust_likelihood.multiwoz import rewrites
from robust_likelihood.multiwoz.semantic import Extractor, load_vocab
from robust_likelihood.multiwoz.sim_score import uperturb_plan, xscore_plan
from robust_likelihood.multiwoz.wk_units import FIXED_ITEMS, fixed_units, load_units, terse_units
from robust_likelihood.storage import read_jsonl, repo_root, write_jsonl
from test_multiwoz_sim_score import DERIVED, TinyTokenizer


def clean_items():
    return [item for item in read_jsonl(DERIVED / "items.jsonl") if item["set"] == "clean"]


def clean_probe():
    return [row for row in read_jsonl(DERIVED / "probe.jsonl") if row["set"] == "clean"]


def test_rewrite_units_are_unchanged_unless_terse_is_asked_for():
    vocab = load_vocab(DERIVED)
    items, probe = clean_items(), clean_probe()
    default = rewrites.build_units(items, probe, vocab)
    assert len(default) == 3262 and not any(unit["kind"] == "terse" for unit in default)
    terse = rewrites.build_units(items, probe, vocab, ("terse",))
    assert len(terse) == 235 and {unit["kind"] for unit in terse} == {"terse"}
    assert terse[0]["unit_id"].endswith(":terse") and terse[0]["side"] == "utterance"
    message = rewrites.llm_messages("terse", "I want cheap food.")[0]["content"]
    assert message.startswith(rewrites.TERSE) and message.endswith("I want cheap food.")
    planned = rewrites.planned_units(DERIVED, ("terse",))
    assert planned["units"] == 235 and planned["first_attempt_calls"] == 236 and planned["max_calls"] == 706


def test_write_extra_keeps_rewrites_jsonl(tmp_path: Path):
    run = tmp_path / "run"
    write_jsonl(run / "accepted.jsonl", [{"unit_id": "u:B:terse", "kind": "terse", "item_id": "B", "rewrite": "x"},
                                         {"unit_id": "u:A:terse", "kind": "terse", "item_id": "A", "rewrite": "y"}])
    found = rewrites.write_extra(run, tmp_path, "terse")
    assert found == {"file": "terse.jsonl", "rewrites": 2}
    assert [row["unit_id"] for row in read_jsonl(tmp_path / "terse.jsonl")] == ["u:A:terse", "u:B:terse"]
    assert not (tmp_path / "rewrites.jsonl").exists()
    with pytest.raises(ValueError):
        rewrites.write_extra(run, tmp_path, "para1")


def test_fixed_units_correct_restaurant_and_keep_the_slots():
    rows = fixed_units(clean_items(), Extractor(load_vocab(DERIVED)))
    assert [row["item_id"] for row in rows] == list(FIXED_ITEMS)
    assert all(row["valid"] and "restaurant" in row["rewrite"].lower() for row in rows)
    assert all("restuarant" not in row["rewrite"] and "resturant" not in row["rewrite"] for row in rows)
    assert read_jsonl(DERIVED / "fixed.jsonl") == rows


def test_load_units_takes_valid_perturbations_fixed_and_terse(tmp_path: Path):
    units = load_units(DERIVED)
    kinds = {row["kind"] for row in units}
    assert sum(1 for row in units if row["kind"] not in ("fixed", "terse", "terse_rx")) == 2058
    assert sum(1 for row in units if row["kind"] == "fixed") == 6 and "typo1" in kinds
    for name in ("perturbations.jsonl", "fixed.jsonl"):
        (tmp_path / name).write_bytes((DERIVED / name).read_bytes())
    write_jsonl(tmp_path / "terse.jsonl", [{"unit_id": "u:SNG0458:terse", "item_id": "SNG0458", "kind": "terse",
                                            "original": "o", "rewrite": "r"}])
    found = load_units(tmp_path, {"SNG0458"})
    assert {row["kind"] for row in found} >= {"fixed", "terse", "typo1"}
    assert terse_units([{"item_id": "A", "original": "o", "rewrite": "r"}])[0]["unit_id"] == "A|terse"


def test_uperturb_groups_hold_the_item_rows_and_refuse_a_foreign_original():
    probe = clean_probe()
    units = [row for row in load_units(DERIVED) if row["item_id"] == "SNG0458"]
    groups = uperturb_plan(probe, units)
    assert len(groups) == len(units)
    expected = [row["hyp_key"] for row in probe if row["item_id"] == "SNG0458"]
    assert all([row["hyp_key"] for row in group["rows"]] == expected for group in groups)
    with pytest.raises(ValueError):
        uperturb_plan(probe, [dict(units[0], original="something else")])


def test_xscore_pairs_every_other_text_of_the_item_with_usable_samples():
    probe = [row for row in clean_probe() if row["item_id"] == "SNG0458"]
    texts = sorted({row["text"] for row in probe})
    samples = []
    for text in texts:
        samples += [{"text": text, "sample": 0, "S": 1.0, "reply": "hi"},
                    {"text": text, "sample": 1, "S": None, "reply": "cut"},
                    {"text": text, "sample": 2, "S": 1.0, "reply": "  "},
                    {"text": text, "sample": 3, "S": 1.0, "reply": "ok"},
                    {"text": text, "sample": 40, "S": 1.0, "reply": "late"}]
    rows = xscore_plan(probe, samples, 32)
    assert len(rows) == len(texts) * (len(texts) - 1) * 2
    assert all(row["text_from"] != row["text"] and row["sample"] in (0, 3) for row in rows)


def test_uperturb_and_xscore_on_a_tiny_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.multiwoz.sim_score import sim_score

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=48, n_positions=1024, n_embd=16, n_layer=1, n_head=2)).eval()
    item = "SNG0458"
    texts = sorted({row["text"] for row in clean_probe() if row["item_id"] == item})
    phase2 = tmp_path / "phase2"
    write_jsonl(phase2 / "lm_samples.jsonl", [{"text": text, "sample": index, "S": 3.0, "reply": f"cheap {index}"}
                                              for text in texts for index in range(2)])
    summary = sim_score(repo_root(), "osim_mid", "S", parts=["uperturb", "xscore", "human"], items={item},
                        results_root=tmp_path, tokenizer=TinyTokenizer(), model=model, device="cpu",
                        samples_dir=phase2)
    run = Path(summary["run_dir"])
    assert run.name.split("-", 1)[1].startswith("simwk-osim_mid-S")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "sim-week" and manifest["finished_at"]
    units = load_units(DERIVED, {item})
    rows = read_jsonl(run / "uperturb.jsonl")
    assert len(rows) == len(units) * len(texts) == manifest["planned"]["uperturb"]
    assert manifest["uperturb_units"] == len(units)
    human = {row["hyp_key"]: row["S"] for row in read_jsonl(run / "lm_scores.jsonl")}
    for row in rows:  # fp32 on CPU: the co-batched original matches the human part closely
        assert row["S_orig"] == pytest.approx(human[row["hyp_key"]], abs=1e-3)
        assert row["batch"] == 2 * len(texts) and row["tokens"] > 0
    fixed = [row for row in rows if row["kind"] == "fixed"]
    assert fixed and all(row["S"] != row["S_orig"] for row in fixed)
    xs = read_jsonl(run / "xscore.jsonl")
    assert len(xs) == len(texts) * (len(texts) - 1) * 2 == manifest["planned"]["xscore"]
    with pytest.raises(ValueError):
        sim_score(repo_root(), "osim_mid", "S", parts=["xscore"])
