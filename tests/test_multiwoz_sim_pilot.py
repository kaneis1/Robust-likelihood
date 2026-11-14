import json
from pathlib import Path

import pytest

from robust_likelihood.multiwoz.claim1 import ItemBoot
from robust_likelihood.multiwoz.sim import load_sim_config
from robust_likelihood.multiwoz.sim_pilot import Context, PilotFormat, alternates, pilot_plan, pilot_rows
from robust_likelihood.multiwoz.sim_pilot_report import (
    discordant,
    health,
    later_pairs,
    ranking,
    turn_coverage,
    turn_one_pairs,
)
from robust_likelihood.storage import read_jsonl, repo_root
from test_multiwoz_sim_score import TinyTokenizer

DERIVED = repo_root() / "data" / "multiwoz" / "derived"


def dialogue(*speakers):
    return [{"speaker": speaker, "utterance": f"{speaker.lower()} {index}"} for index, speaker in enumerate(speakers)]


def test_rows_draw_dialogues_that_reach_the_last_turn_and_carry_their_history():
    turns = {"A": dialogue("USER", "SYSTEM", "USER", "SYSTEM", "USER", "SYSTEM"),
             "B": dialogue("USER", "SYSTEM", "USER", "SYSTEM", "USER"),
             "C": dialogue("USER", "SYSTEM", "USER", "SYSTEM"),          # no third user turn
             "D": dialogue("USER", "USER", "SYSTEM", "USER", "SYSTEM"),  # does not alternate
             "E": dialogue("USER", "SYSTEM", "USER", "SYSTEM", "USER")}
    items = [{"item_id": key, "set": "clean"} for key in turns] + [{"item_id": "F", "set": "fail_info"}]
    assert alternates(turns["B"], 3) and not alternates(turns["C"], 3) and not alternates(turns["D"], 3)
    rows = pilot_rows(turns, items, 2, 7, [2, 3])
    assert rows == pilot_rows(turns, items, 2, 7, [2, 3])
    assert {row["item_id"] for row in rows} <= {"A", "B", "E"} and len(rows) == 4
    second = next(row for row in rows if row["turn"] == 2)
    assert [turn["speaker"] for turn in second["history"]] == ["USER", "SYSTEM"] and second["utterance"] == "user 2"
    third = next(row for row in rows if row["turn"] == 3)
    assert len(third["history"]) == 4 and third["utterance"] == "user 4"


def test_the_committed_plan_matches_the_config_and_the_probe():
    config = load_sim_config(repo_root())["phase5"]
    rows = read_jsonl(repo_root() / config["file"])
    assert len({row["item_id"] for row in rows}) == config["dialogues"]
    assert sorted({row["turn"] for row in rows}) == config["user_turns"]
    items = {row["item_id"]: row for row in read_jsonl(DERIVED / "items.jsonl")}
    assert all(items[row["item_id"]]["set"] == "clean" for row in rows)
    assert all(row["history"][0]["utterance"] == items[row["item_id"]]["utterance"] for row in rows)
    units = pilot_plan(rows, read_jsonl(DERIVED / "probe.jsonl"))
    clean = [row for row in read_jsonl(DERIVED / "probe.jsonl") if row["set"] == "clean"
             and row["item_id"] in {row["item_id"] for row in rows}]
    assert sum(len(unit["hyp_keys"]) for unit in units) == len(clean) * len(config["user_turns"])


SPEC = {"id": "tiny", "speaker_role": "assistant", "end_token": "<|im_end|>", "separator": "", "template_kwargs": {}}
PROMPTS = {"S_system": "You are a user. {H}", "B_system": "{H}", "N_system": "{H}", "opener": "How can I help?"}


def test_history_follows_the_opener_in_its_roles():
    context = Context("H1", (("USER", "cheap food"), ("SYSTEM", "what area?")))
    found = PilotFormat(TinyTokenizer(), SPEC, "S", PROMPTS)
    assert [(m["role"], m["content"]) for m in found.messages(context)] == [
        ("system", "You are a user. H1"), ("user", "How can I help?"), ("assistant", "cheap food"),
        ("user", "what area?")]
    user_spec = SPEC | {"speaker_role": "user"}
    flipped = PilotFormat(TinyTokenizer(), user_spec, "S", PROMPTS)
    assert [m["role"] for m in flipped.messages(context)] == ["system", "assistant", "user", "assistant"]
    assert found.template_check(context) and found.history_kept(context) and found.boundary_holds(context, "the east")
    assert found.prefix_text(context).endswith("what area?<|im_end|>\n<|im_start|>assistant\n")
    # The first-turn prefix is format S's own.
    assert found.prefix_text(Context("H1")) == PilotFormat(TinyTokenizer(), SPEC, "S", PROMPTS).prefix_text(Context("H1"))
    with pytest.raises(ValueError):
        PilotFormat(TinyTokenizer(), SPEC, "F", PROMPTS)


def test_sim_pilot_end_to_end_on_a_tiny_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from robust_likelihood.multiwoz.sim_pilot import sim_pilot

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=48, n_positions=2048, n_embd=16, n_layer=1, n_head=2)).eval()
    config = load_sim_config(repo_root())["phase5"]
    rows = read_jsonl(repo_root() / config["file"])
    item = rows[0]["item_id"]
    summary = sim_pilot(repo_root(), "osim_mid", dialogues={item}, samples=2, results_root=tmp_path,
                        tokenizer=TinyTokenizer(), model=model, device="cpu")
    run = Path(summary["run_dir"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    checks = manifest["checks"]
    assert checks["sampling_is_exact"] and checks["history_kept"] and checks["boundary_failures"] == 0
    assert manifest["kind"] == "sim-pilot" and manifest["finished_at"] and manifest["user_turns"] == [2, 3]
    units = pilot_plan(rows, read_jsonl(DERIVED / "probe.jsonl"), {item})
    samples = read_jsonl(run / "pilot_samples.jsonl")
    assert len(samples) == 2 * len(units) and "S_broad" not in samples[0]
    for row in samples:
        assert (row["S"] is not None) == row["ended"] and (row["S_lower"] is not None) == (not row["ended"])
    scores = read_jsonl(run / "pilot_scores.jsonl")
    assert len(scores) == sum(len(unit["hyp_keys"]) for unit in units) and all(row["S"] > 0 for row in scores)
    assert {row["turn"] for row in scores} == {2, 3}
    with pytest.raises(ValueError):
        sim_pilot(repo_root(), "qwen3_base", results_root=tmp_path)

    from robust_likelihood.multiwoz.sim_pilot_report import run_pilot_analyze

    found = run_pilot_analyze(repo_root(), runs=[run], out=tmp_path / "out", figures=False)
    assert found["runs"] == ["osim_mid|S"] and found["turns"] == ["2", "3"]
    text = (tmp_path / "out" / "pilot.md").read_text(encoding="utf-8")
    assert "## Coverage by turn, all hypotheses" in text and "## Sample health by turn" in text


def sample(S, ended=True, flags=(), reply="a b", index=0, **extra):
    return {"S": S if ended else None, "S_lower": None if ended else S, "ended": ended, "flags": list(flags),
            "reply": reply, "sample": index, **extra}


def test_turn_one_keeps_the_pilot_dialogues_and_the_first_samples():
    texts = {("A", "G|impl"): "t1", ("A", "broad|impl"): "t2", ("B", "G|impl"): "t1"}
    scores = [{"item_id": "A", "set": "clean", "hyp_key": "G|impl", "S": 5.0},
              {"item_id": "A", "set": "clean", "hyp_key": "broad|impl", "S": 9.0},
              {"item_id": "B", "set": "clean", "hyp_key": "G|impl", "S": 1.0}]
    samples = [sample(value, index=index, text="t1") for index, value in enumerate([4.0, 6.0, 100.0])]
    samples += [sample(1.0, index=0, text="t2")]
    pairs, used = turn_one_pairs(scores, samples, texts, {("A", 1): "hi there"}, {"A"}, 2)
    assert [(pair["item"], pair["key"], len(pair["samples"])) for pair in pairs] == [("A", "G|impl", 2),
                                                                                    ("A", "broad|impl", 1)]
    assert len(used) == 3  # sample 2 of t1 is beyond the first two


def test_coverage_health_and_ranking():
    boot = ItemBoot(["A", "B"], 200, 1)
    rows = [sample(4.0, item_id="A", turn=2, text="t", index=0), sample(6.0, item_id="A", turn=2, text="t", index=1),
            sample(30.0, ended=False, flags=["truncated"], item_id="B", turn=2, text="t", index=0),
            sample(2.0, flags=["empty"], reply="", item_id="B", turn=2, text="t", index=1)]
    scores = [{"item_id": "A", "turn": 2, "hyp_key": "G|impl", "text": "t", "S": 5.0},
              {"item_id": "B", "turn": 2, "hyp_key": "G|impl", "text": "t", "S": 10.0}]
    pairs, used = later_pairs(scores, rows, {("A", 2): "one two", ("B", 2): "x"}, 2)
    found, columns = turn_coverage(pairs, used, {"A": "G|impl", "B": "G|impl"}, boot)
    # A: one of two samples is at least 5; B: the cut-off sample's bound 30 counts, the 2.0 sample does not.
    assert sorted(columns["all"][1]) == [0.5, 0.5]
    assert found["all"]["share_beyond_all"]["mean"] == 0.0
    assert found["health"]["cut_off"] == 0.25 and found["health"]["empty"] == 0.25
    assert health([sample(1.0, flags=["<|endconversation|>"])])["other_flag"] == 1.0

    def block(median, beyond):
        return {"turns": {"1": {"all": {"median_percentile": {"median": median},
                                        "share_beyond_all": {"mean": beyond}}}}}

    runs = {"x": block(0.3, 0.0), "y": block(0.0, 0.9), "z": block(0.0, 0.5)}
    assert ranking(runs, "1") == ["x", "z", "y"]
    assert discordant(["x", "z", "y"], ["z", "x", "y"]) == [["x", "z"]]
    assert discordant(["x", "y"], ["x", "y"]) == []
