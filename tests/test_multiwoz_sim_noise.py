import json
from pathlib import Path

import pytest

from robust_likelihood.multiwoz.sim_noise import compare, null_rows, quantile
from robust_likelihood.storage import read_jsonl, repo_root
from test_multiwoz_sim_score import CENTER, TinyTokenizer, derived_inputs


def test_null_rows_put_the_original_utterance_back():
    rows = [{"item_id": "A", "kind": "para1", "hyp_key": "G|impl", "text": "h", "utterance": "rewritten"}]
    assert null_rows(rows, {"A": "original"}) == [rows[0] | {"utterance": "original"}]
    assert rows[0]["utterance"] == "rewritten"


def test_compare_reads_real_shifts_against_the_noise():
    noise = [{"kind": "para1", "delta": value} for value in (0.1, -0.2, 0.3, 0.1, 0.0)]
    real = [{"kind": "para2", "delta": value} for value in (2.0, -0.05, 1.0, 0.25)]
    found = compare(real, noise)["para"]
    assert found["noise_p50"] == 0.1 and found["noise_p95"] == 0.3
    assert found["real_above_noise_p95"] == 0.5  # 2.0 and 1.0 exceed 0.3
    assert compare([], [])["zh"]["real_above_noise_p95"] is None
    assert quantile([], 0.5) is None


def test_sim_noise_on_a_tiny_model_reproduces_fp32_scores(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.multiwoz.claim1 import ORIGINAL, Scorer, item_meta, token_log, utterance_rows
    from robust_likelihood.multiwoz.sim_noise import sim_noise
    from robust_likelihood.multiwoz.sim_score import score_plan

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=48, n_positions=1024, n_embd=16, n_layer=1, n_head=2)).eval()
    probe, rewrites, subset = derived_inputs()
    item = subset[0]
    summary = sim_noise(repo_root(), "osim_mid", "S", items={item}, results_root=tmp_path,
                        tokenizer=TinyTokenizer(), model=model, device="cpu")
    run = Path(summary["run_dir"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "sim-noise" and manifest["finished_at"] and manifest["batch_size"] == 32
    plan = score_plan(probe, rewrites, subset, CENTER, include_fail=False, items={item})
    nulls = read_jsonl(run / "urewrite_null.jsonl")
    original = next(row["utterance"] for row in probe if row["item_id"] == item)
    assert len(nulls) == len(plan["urewrite"]) and {row["utterance"] for row in nulls} == {original}
    human = read_jsonl(run / "human_repeat.jsonl")
    assert len(human) == len(plan["human"])
    # In fp32 on CPU the layout barely matters, so the null shifts are close to zero.
    originals = {(row["item_id"], row["hyp_key"]): -row["S"] for row in human}
    log = token_log(originals, {"urewrite": nulls}, {item})
    items = [row for row in read_jsonl(repo_root() / "data/multiwoz/derived/items.jsonl") if row["item_id"] == item]
    metas = item_meta(items, [row for row in probe if row["item_id"] == item and row["set"] == "clean"])
    rows = utterance_rows(Scorer("tiny", "loglik", log, items=[item]), metas)
    assert rows and max(abs(row["delta"]) for row in rows) < 1e-3
    assert all(key[2] == ORIGINAL for key in log)
