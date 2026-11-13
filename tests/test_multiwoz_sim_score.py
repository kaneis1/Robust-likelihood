import json
import re
from pathlib import Path

import pytest

from robust_likelihood.multiwoz.sim import load_sim_config
from robust_likelihood.multiwoz.sim_score import default_parts, mean_sd, score_plan, tally
from robust_likelihood.storage import read_jsonl, repo_root

DERIVED = repo_root() / "data" / "multiwoz" / "derived"
CENTER = "The user wants to find a restaurant."


def derived_inputs():
    probe = read_jsonl(DERIVED / "probe.jsonl")
    rewrites = read_jsonl(DERIVED / "rewrites.jsonl")
    subset = json.loads((DERIVED / "robustness_subset.json").read_text(encoding="utf-8"))
    return probe, rewrites, subset


def test_plan_samples_every_clean_text_and_scores_clean_rows_unless_fail_is_asked():
    probe, rewrites, subset = derived_inputs()
    plan = score_plan(probe, rewrites, subset, CENTER, include_fail=False)
    clean = [row for row in probe if row["set"] == "clean"]
    assert plan["samples"] == sorted({row["text"] for row in clean})
    assert len(plan["human"]) == len(clean)
    assert len(score_plan(probe, rewrites, subset, CENTER, include_fail=True)["human"]) == len(probe)
    assert {row["unit_id"] for row in plan["hsample"]} <= {row["unit_id"] for row in rewrites}


def test_plan_restricted_to_items():
    probe, rewrites, subset = derived_inputs()
    item = subset[0]
    plan = score_plan(probe, rewrites, subset, CENTER, include_fail=False, items={item})
    assert {row["item_id"] for row in plan["human"]} == {item}
    assert all(row["item_id"] == item for row in plan["urewrite"] + plan["hrewrite"])
    assert plan["hsample"]


def test_default_parts_follow_the_plan():
    config = load_sim_config(repo_root())
    assert "hsample" in default_parts("osim_mid", config)
    assert "hsample" not in default_parts("userlm", config)
    assert {spec["primary"] for spec in config["models"].values()} == {"S", "B", "F"}


def test_mean_sd_and_tally():
    assert mean_sd([]) == (None, None)
    assert mean_sd([2.0]) == (2.0, None)
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(2 ** 0.5))
    rows = [{"ended": True, "flags": []}, {"ended": False, "flags": ["truncated", "<|x|>"]}]
    assert tally(rows) == {"drawn": 2, "ended": 1, "truncated": 1, "<|x|>": 1}


SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class TinyTokenizer:
    """Specials whole, every other character folded into 45 ids: a small vocabulary a tiny model can end in."""

    eos_token, eos_token_id, pad_token_id, unk_token_id = "<|endoftext|>", 2, None, None

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in re.split("(" + "|".join(re.escape(token) for token in SPECIALS) + ")", text):
            ids.extend([SPECIALS.index(part)] if part in SPECIALS else [3 + ord(char) % 45 for char in part])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else SPECIALS[i]) if i < 3 else chr(ord("a") + (i - 3) % 26)
                       for i in ids)

    def convert_tokens_to_ids(self, token):
        return SPECIALS.index(token) if token in SPECIALS else None


def test_sim_score_end_to_end_on_a_tiny_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.multiwoz.sim_score import sim_score

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=48, n_positions=1024, n_embd=16, n_layer=1, n_head=2)).eval()
    probe, rewrites, subset = derived_inputs()
    item = subset[0]
    summary = sim_score(repo_root(), "osim_mid", "S", items={item}, samples=3, results_root=tmp_path,
                        tokenizer=TinyTokenizer(), model=model, device="cpu")
    run = Path(summary["run_dir"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checks"]["sampling_is_exact"] and manifest["checks"]["boundary_failures"] == 0
    assert manifest["primary"] and manifest["finished_at"]
    assert manifest["sampling"]["temperature"] == 1.0 and manifest["sampling"]["top_k"] == 0

    plan = score_plan(probe, rewrites, subset, CENTER, include_fail=False, items={item})
    samples = read_jsonl(run / "lm_samples.jsonl")
    assert len(samples) == 3 * len(plan["samples"])
    for row in samples:
        assert (row["S"] is not None) == row["ended"] == (row["S_broad"] is not None)
        assert (row["S_lower"] is not None) == (not row["ended"])
    hypotheses = read_jsonl(run / "lm_hypotheses.jsonl")
    assert sum(row["kept"] + row["dropped"] for row in hypotheses) == len(samples)
    scores = read_jsonl(run / "lm_scores.jsonl")
    assert len(scores) == len(plan["human"]) and all(row["S"] > 0 for row in scores)
    assert len(read_jsonl(run / "urewrite.jsonl")) == len(plan["urewrite"])
    assert len(read_jsonl(run / "hrewrite.jsonl")) == len(plan["hrewrite"])
    assert len(read_jsonl(run / "hsample.jsonl")) == 3 * len(plan["hsample"])
    assert manifest["samples"]["drawn"] == len(samples)


def test_sim_score_refuses_hsample_outside_the_plan_and_p0():
    from robust_likelihood.multiwoz.sim_score import sim_score

    with pytest.raises(ValueError):
        sim_score(repo_root(), "userlm", "S", parts=["hsample"])
    with pytest.raises(ValueError):
        sim_score(repo_root(), "qwen25_inst", "P0")
