import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, load_config
from robust_likelihood.multiwoz.__main__ import prepare
from robust_likelihood.multiwoz.hypotheses import (
    base,
    control_seed,
    explicit,
    food_pool,
    item_hypotheses,
    join_names,
    value_pools,
)
from robust_likelihood.multiwoz.probe import APPENDIX_A4, build_probe, probe_summary
from robust_likelihood.storage import read_jsonl, repo_root

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"
EXPLICIT_SENTENCE = re.compile(r"^ The user has no preference about (the type of food|the area|the price range)"
                               r"((, | or |, or )(the type of food|the area|the price range))*\.$")


def _item(goal, status, item_id="SNG9999"):
    return {"item_id": item_id, "goal": goal, "slot_status": status}


def _probe_rows():
    return read_jsonl(repo_root() / DERIVED_DIR / "probe.jsonl")


def test_plan_example_texts():
    goal = {"food": "italian", "pricerange": "expensive"}
    assert base(goal) == "The user wants to find an expensive restaurant serving italian food."
    assert explicit(goal) == (
        "The user wants to find an expensive restaurant serving italian food. "
        "The user has no preference about the area."
    )
    assert explicit({"pricerange": "expensive"}) == (
        "The user wants to find an expensive restaurant. The user has no preference about the type of food or the area."
    )
    assert explicit({"food": "italian"}) == (
        "The user wants to find a restaurant serving italian food. "
        "The user has no preference about the area or the price range."
    )
    assert explicit({}) == (
        "The user wants to find a restaurant. "
        "The user has no preference about the type of food, the area, or the price range."
    )
    full = {"food": "modern european", "area": "centre", "pricerange": "moderate"}
    assert explicit(full) == base(full) == (
        "The user wants to find a moderately priced restaurant serving modern european food in the centre of town."
    )


def test_articles_and_lists():
    assert base({"pricerange": "expensive"}).startswith("The user wants to find an expensive restaurant")
    assert base({"pricerange": "cheap"}).startswith("The user wants to find a cheap restaurant")
    assert base({"pricerange": "moderate"}).startswith("The user wants to find a moderately priced restaurant")
    assert base({"area": "east"}) == "The user wants to find a restaurant in the east of town."
    assert join_names(["A"]) == "A"
    assert join_names(["A", "B"]) == "A or B"
    assert join_names(["A", "B", "C"]) == "A, B, or C"


def test_regime_texts_differ_only_by_the_explicit_sentence():
    rows = _probe_rows()
    by_item = defaultdict(dict)
    for row in rows:
        by_item[row["item_id"]][row["hyp_key"]] = row
    pairs = 0
    for hypotheses in by_item.values():
        for key, row in hypotheses.items():
            if row["regime"] != "expl":
                continue
            implicit = hypotheses[key.replace("|expl", "|impl")]
            assert row["values"] == implicit["values"]
            assert row["text"].startswith(implicit["text"])
            assert EXPLICIT_SENTENCE.match(row["text"][len(implicit["text"]):])
            pairs += 1
    assert pairs == sum(1 for row in rows if row["regime"] == "expl")


def test_no_key_or_type_reaches_a_text():
    rows = _probe_rows()
    labels = {row["hyp_key"] for row in rows} | ({row["hyp_type"] for row in rows} - {"G"})
    labels |= {"impl", "expl", "ctrl", "minus", "contra", "tie2", "broad", "|"}
    for row in rows:
        for label in labels:
            assert label not in row["text"], (label, row["text"])
        assert not re.search(r"\bG\b", row["text"])


def test_three_slot_goals_have_no_separate_explicit_goal_row():
    rows = _probe_rows()
    keys = defaultdict(set)
    sizes = {}
    for row in rows:
        keys[row["item_id"]].add(row["hyp_key"])
        if row["hyp_key"] == "G|impl":
            sizes[row["item_id"]] = len(row["constrained"])
    assert set(sizes.values()) == {2, 3}
    for item_id, size in sizes.items():
        assert ("G|expl" in keys[item_id]) == (size == 2)


def test_each_control_changes_exactly_one_slot_to_a_new_value():
    rows = _probe_rows()
    goals = {row["item_id"]: row["values"] for row in rows if row["hyp_key"] == "G|impl"}
    controls = [row for row in rows if row["hyp_type"].startswith("ctrl:")]
    assert Counter(row["hyp_key"] for row in controls) == {"ctrl:contra": 359, "ctrl:tie": 214, "ctrl:tie2": 11}
    for row in controls:
        goal = goals[row["item_id"]]
        assert set(row["values"]) == set(goal)
        changed = [slot for slot in goal if row["values"][slot] != goal[slot]]
        assert changed == [row["changed"]["slot"]]
        assert row["changed"]["from"] == goal[changed[0]]
        assert row["changed"]["to"].lower() != goal[changed[0]].lower()
    texts = defaultdict(dict)
    for row in controls:
        texts[row["item_id"]][row["hyp_key"]] = row["text"]
    assert all(found["ctrl:tie2"] != found["ctrl:tie"] for found in texts.values() if "ctrl:tie2" in found)


def test_control_slots_follow_turn0_status():
    named = {"food": "named", "area": "omitted", "pricerange": "named"}
    item = _item({"food": "italian", "area": "east", "pricerange": "cheap"}, named)
    pools = value_pools(["italian", "thai", "welsh"])
    rows = {row["hyp_key"]: row for row in item_hypotheses(item, pools)}
    assert rows["ctrl:contra"]["changed"]["slot"] in {"food", "pricerange"}
    assert rows["ctrl:tie"]["changed"]["slot"] == "area"
    silent = _item({"food": "italian", "area": "east"}, {"food": "omitted", "area": "omitted", "pricerange": "omitted"})
    keys = [row["hyp_key"] for row in item_hypotheses(silent, pools)]
    assert "ctrl:contra" not in keys and keys[-2:] == ["ctrl:tie2", "ctrl:tie"]


def test_seeding_is_deterministic():
    assert control_seed("SNG0001", "contra") == int(hashlib.sha256(b"SNG0001:contra").hexdigest()[:8], 16)
    item = _item({"food": "italian", "area": "east", "pricerange": "cheap"},
                 {"food": "named", "area": "omitted", "pricerange": "named"})
    pools = value_pools(["british", "chinese", "italian", "thai", "welsh"])
    assert item_hypotheses(item, pools) == item_hypotheses(dict(item), pools)
    texts = {item_hypotheses({**item, "item_id": f"SNG{index:04d}"}, pools)[-2]["text"] for index in range(40)}
    assert len(texts) > 1


def test_fixture_rows_and_food_pool(tmp_path):
    summary = prepare(repo_root(), load_config(repo_root()), raw=FIXTURE, derived=tmp_path)
    rows = read_jsonl(tmp_path / "probe.jsonl")
    per_item = Counter(row["item_id"] for row in rows)
    assert per_item == {
        "SNG0001": 9, "SNG0002": 11, "SNG0003": 9, "SNG0005": 10, "SNG0011": 9, "SNG0013": 10, "SNG0014": 10,
    }
    assert summary["probe"]["by_set"] == {"clean": 59, "fail_info": 9}
    assert summary["probe"]["food_pool_size"] == 8
    assert summary["appendix_differences"]["A4"]
    assert set(rows[0]) >= {"item_id", "set", "utterance", "hyp_key", "hyp_type", "regime", "constrained", "removed", "text"}
    goals = {"x": {"goal": {"restaurant": {"info": {"food": "Thai"}, "fail_info": {"food": "dontcare"}}}}}
    assert food_pool(goals) == ["thai"]


def test_committed_probe_matches_appendix_a4():
    rows = _probe_rows()
    summary = probe_summary(rows)
    assert summary["rows"] == APPENDIX_A4["rows"]
    assert summary["by_set"] == APPENDIX_A4["by_set"]
    assert len({row["item_id"] for row in rows}) == 370
    assert len({row["text"] for row in rows if row["hyp_type"] == "broad"}) == 2
