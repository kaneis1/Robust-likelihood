from pathlib import Path

from robust_likelihood.multiwoz.corpus import (
    APPENDIX_A1,
    differences,
    first_goal,
    first_restaurant_turn,
    item_population,
    load_goals,
    load_turns,
    pooled_population,
    select_items,
    turn_status,
    value_status,
)

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"
OTHER = ("hotel", "train", "taxi", "attraction", "police", "hospital")


def _corpus():
    return load_goals(FIXTURE), load_turns(FIXTURE)


def test_selection_keeps_sng_restaurant_only_slot_goals():
    goals, turns = _corpus()
    items, summary = select_items(goals, turns, OTHER)
    by_id = {item["item_id"]: item for item in items}
    assert list(by_id) == ["SNG0001", "SNG0002", "SNG0003", "SNG0005", "SNG0011", "SNG0013", "SNG0014"]
    assert summary["dialogues"] == {"v21": 14, "v22": 13}
    assert summary["missing_in_v22"] == ["SNG0012"]
    assert summary["restaurant_goal"] == 12
    assert summary["restaurant_only"] == 9
    assert summary["sng_restaurant_only"] == 8
    assert summary["name_based"] == {"clean": 1, "fail_info": 0}
    assert summary["slot_based"] == {"clean": 6, "fail_info": 1}
    assert summary["clean_by_n_constrained"] == {"2": 5, "3": 1}
    assert summary["no_named_constrained"] == 1
    assert summary["utterance_mismatches"] == ["SNG0013"]
    first = by_id["SNG0001"]
    assert first["set"] == "clean" and first["has_fail_info"] is False
    assert first["goal"] == {"food": "italian", "pricerange": "expensive"}
    assert first["n_constrained"] == 2
    assert first["slot_status"] == {"food": "named", "area": "omitted", "pricerange": "named"}
    assert first["slot_match"] == {"food": True, "area": None, "pricerange": True}
    assert first["utterance"] == "I am looking for an expensive Italian restaurant."
    assert first["utterance_matches_v21"] is True


def test_fail_info_wins_and_dontcare_goals_are_not_constrained():
    goals, turns = _corpus()
    items, _ = select_items(goals, turns, OTHER)
    by_id = {item["item_id"]: item for item in items}
    assert first_goal(goals["SNG0003"]["goal"]) == {"food": "korean", "area": "north"}
    assert by_id["SNG0003"]["set"] == "fail_info"
    assert by_id["SNG0003"]["goal"] == {"food": "korean", "area": "north"}
    assert by_id["SNG0003"]["slot_match"]["food"] is True
    assert by_id["SNG0005"]["goal"] == {"area": "west", "pricerange": "moderate"}
    assert by_id["SNG0005"]["slot_status"]["food"] == "dontcare"
    assert by_id["SNG0011"]["slot_match"] == {"food": True, "area": False, "pricerange": None}


def test_clean_turn0_table_counts_dontcare_apart_from_named():
    goals, turns = _corpus()
    _, summary = select_items(goals, turns, OTHER)
    table = summary["clean_turn0"]
    assert table["food"] == {
        "constrained": 4, "named": 4, "named_mismatch": 0, "omitted": 0, "dontcare": 0,
        "no_pref": 2, "no_pref_named": 0, "no_pref_dontcare": 1,
    }
    assert table["area"]["named"] == 2 and table["area"]["named_mismatch"] == 1
    assert table["area"]["no_pref_named"] == 1
    assert table["pricerange"]["omitted"] == 4


def test_status_and_delta_against_the_previous_user_turn():
    assert value_status([]) == "omitted"
    assert value_status(["dontcare"]) == "dontcare"
    assert value_status(["centre", "center"]) == "named"
    _, turns = _corpus()
    dialogue = turns["MUL0007"]
    earlier = dialogue[2]["restaurant"]
    later = dialogue[4]["restaurant"]
    assert turn_status(later) == {"food": "named", "area": "named", "pricerange": "omitted"}
    assert turn_status(later, earlier) == {"food": "omitted", "area": "named", "pricerange": "omitted"}


def test_first_restaurant_turn_and_pooled_population():
    goals, turns = _corpus()
    index, turn, previous = first_restaurant_turn(turns["MUL0007"])
    assert index == 2
    assert turn["utterance"] == "Thanks. I also want Indian food."
    assert previous == {"active_intent": "NONE", "requested_slots": [], "slot_values": {}}
    assert first_restaurant_turn(turns["PMUL0010"]) is None
    rows, counts = pooled_population(goals, turns)
    assert counts == {"slot_based_goal": 11, "no_restaurant_turn": 1, "population": 10}
    by_id = {row["dialogue_id"]: row for row in rows}
    assert "SNG0004" not in by_id and "PMUL0010" not in by_id
    assert by_id["MUL0007"]["turn_index"] == 2
    assert by_id["MUL0007"]["constrained"] == ["food", "area"]
    assert by_id["MUL0007"]["status"] == {"food": "named", "area": "omitted", "pricerange": "omitted"}
    assert by_id["WOZ0008"]["constrained"] == ["food"]
    assert by_id["WOZ0008"]["status"]["pricerange"] == "dontcare"


def test_item_population_has_the_pooled_row_shape():
    goals, turns = _corpus()
    items, _ = select_items(goals, turns, OTHER)
    rows = item_population(items)
    pooled, _ = pooled_population(goals, turns)
    assert set(rows[0]) == set(pooled[0])
    assert rows[0] == {
        "dialogue_id": "SNG0001",
        "turn_index": 0,
        "constrained": ["food", "pricerange"],
        "status": {"food": "named", "area": "omitted", "pricerange": "named"},
        "match": {"food": True, "area": None, "pricerange": True},
    }


def test_differences_name_each_leaf():
    expected = {"a": 1, "b": {"c": 2, "d": 3}}
    assert differences(expected, {"a": 1, "b": {"c": 2, "d": 3}, "extra": 9}) == []
    assert differences(expected, {"a": 1, "b": {"c": 5}}) == [
        "b.c: expected 2, got 5",
        "b.d: expected 3, got nothing",
    ]
    assert APPENDIX_A1["slot_based"] == {"clean": 235, "fail_info": 135}
