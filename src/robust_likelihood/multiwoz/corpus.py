"""MultiWOZ goals, the analysis population, and slot status by turn.

Goals come from 2.1 data.json. Turns and their state annotations come from the
2.2 dialogue files. Every rule here is a frozen Phase 2 definition in
docs/multiwoz_size_principle_plan.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SLOTS = ("food", "area", "pricerange")

APPENDIX_A1 = {
    "dialogues": {"v21": 10438, "v22": 10437},
    "restaurant_goal": 4692,
    "restaurant_only": 1311,
    "sng_restaurant_only": 442,
    "name_based": {"clean": 44, "fail_info": 28},
    "slot_based": {"clean": 235, "fail_info": 135},
    "clean_by_n_constrained": {"2": 124, "3": 111},
    "no_named_constrained": 11,
    "clean_turn0": {
        "food": {"constrained": 195, "named": 142, "named_mismatch": 0, "omitted": 53, "no_pref": 40, "no_pref_named": 2},
        "area": {"constrained": 186, "named": 138, "named_mismatch": 3, "omitted": 48, "no_pref": 49, "no_pref_named": 0},
        "pricerange": {
            "constrained": 200, "named": 150, "named_mismatch": 1, "omitted": 50, "no_pref": 35, "no_pref_named": 1,
        },
    },
}


def dialogue_key(dialogue_id: str) -> str:
    return dialogue_id[: -len(".json")] if dialogue_id.endswith(".json") else dialogue_id


def load_goals(raw: Path) -> dict[str, dict]:
    """Goal and first logged text of every 2.1 dialogue."""
    data = json.loads((raw / "MultiWOZ_2.1" / "data.json").read_text(encoding="utf-8"))
    return {
        dialogue_key(dialogue_id): {"goal": dialogue["goal"], "first_text": dialogue["log"][0]["text"]}
        for dialogue_id, dialogue in data.items()
    }


def restaurant_state(turn: dict) -> dict | None:
    for frame in turn.get("frames", []):
        if frame.get("service") == "restaurant":
            return frame.get("state")
    return None


def load_turns(raw: Path) -> dict[str, list[dict]]:
    """Speaker, utterance, and restaurant state of every 2.2 turn."""
    found = {}
    for path in sorted((raw / "MultiWOZ_2.2").glob("*/dialogues_*.json")):
        for dialogue in json.loads(path.read_text(encoding="utf-8")):
            found[dialogue_key(dialogue["dialogue_id"])] = [
                {"speaker": turn["speaker"], "utterance": turn["utterance"], "restaurant": restaurant_state(turn)}
                for turn in dialogue["turns"]
            ]
    return found


def restaurant_goal(goal: dict) -> dict:
    return goal.get("restaurant") or {}


def is_restaurant_only(goal: dict, other_domains) -> bool:
    return bool(restaurant_goal(goal)) and not any(goal.get(domain) for domain in other_domains)


def first_goal(goal: dict) -> dict:
    """info updated by fail_info, so fail_info wins: the constraints the user starts with."""
    restaurant = restaurant_goal(goal)
    merged = dict(restaurant.get("info") or {})
    merged.update(restaurant.get("fail_info") or {})
    return merged


def constrained_slots(g_first: dict) -> list[str]:
    return [slot for slot in SLOTS if g_first.get(slot) not in (None, "dontcare")]


def goal_set(goal: dict) -> str:
    return "fail_info" if restaurant_goal(goal).get("fail_info") else "clean"


def slot_values(state: dict | None, slot: str) -> list[str]:
    if not state:
        return []
    return list((state.get("slot_values") or {}).get(f"restaurant-{slot}", []))


def value_status(values: list[str]) -> str:
    if not values:
        return "omitted"
    if "dontcare" in values:
        return "dontcare"
    return "named"


def turn_status(state: dict | None, previous: dict | None = None) -> dict[str, str]:
    """Turn 0 reads its state. A later turn reads the change since the previous USER turn."""
    statuses = {}
    for slot in SLOTS:
        current = slot_values(state, slot)
        if previous is not None and set(current) == set(slot_values(previous, slot)):
            statuses[slot] = "omitted"
        else:
            statuses[slot] = value_status(current)
    return statuses


def matches(goal_value: str, values: list[str]) -> bool:
    return goal_value.lower() in {value.lower() for value in values}


def slot_match(g_first: dict, constrained: list[str], status: dict, values: dict) -> dict:
    """Whether a named constrained slot carries the goal value; None where no comparison applies."""
    return {
        slot: matches(g_first[slot], values[slot]) if slot in constrained and status[slot] == "named" else None
        for slot in SLOTS
    }


def is_restaurant_turn(state: dict | None) -> bool:
    if not state:
        return False
    return state.get("active_intent", "NONE") != "NONE" or bool(state.get("slot_values"))


def first_restaurant_turn(turns: list[dict]) -> tuple[int, dict, dict | None] | None:
    """Index, turn, and previous USER restaurant state of the first USER turn about restaurants."""
    previous = None
    for index, turn in enumerate(turns):
        if turn["speaker"] != "USER":
            continue
        if is_restaurant_turn(turn["restaurant"]):
            return index, turn, previous
        previous = turn["restaurant"] or {}
    return None


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def select_items(goals: dict[str, dict], turns: dict[str, list[dict]], other_domains) -> tuple[list[dict], dict]:
    """The slot-based analysis items and the Appendix A1 counts."""
    counts = Counter()
    name_based = Counter()
    items = []
    for item_id in sorted(goals):
        goal = goals[item_id]["goal"]
        if not restaurant_goal(goal):
            continue
        counts["restaurant_goal"] += 1
        if not is_restaurant_only(goal, other_domains):
            continue
        counts["restaurant_only"] += 1
        if not item_id.startswith("SNG"):
            continue
        counts["sng_restaurant_only"] += 1
        g_first = first_goal(goal)
        chosen = goal_set(goal)
        if "name" in g_first:
            name_based[chosen] += 1
            continue
        if item_id not in turns:
            raise ValueError(f"{item_id} has no MultiWOZ 2.2 turns")
        first = turns[item_id][0]
        if first["speaker"] != "USER":
            raise ValueError(f"{item_id} turn 0 is spoken by {first['speaker']}, not USER")
        constrained = constrained_slots(g_first)
        status = turn_status(first["restaurant"])
        values = {slot: slot_values(first["restaurant"], slot) for slot in SLOTS}
        items.append(
            {
                "item_id": item_id,
                "set": chosen,
                "utterance": first["utterance"],
                "goal": {slot: g_first[slot] for slot in constrained},
                "n_constrained": len(constrained),
                "slot_status": status,
                "slot_match": slot_match(g_first, constrained, status, values),
                "has_fail_info": chosen == "fail_info",
                "utterance_matches_v21": normalize(first["utterance"]) == normalize(goals[item_id]["first_text"]),
                "turn0_values": values,
            }
        )
    summary = {
        "dialogues": {"v21": len(goals), "v22": len(turns)},
        "missing_in_v22": sorted(set(goals) - set(turns)),
        "restaurant_goal": counts["restaurant_goal"],
        "restaurant_only": counts["restaurant_only"],
        "sng_restaurant_only": counts["sng_restaurant_only"],
        "name_based": {name: name_based[name] for name in ("clean", "fail_info")},
        "slot_based": {name: sum(1 for item in items if item["set"] == name) for name in ("clean", "fail_info")},
        "clean_by_n_constrained": _by_size([item for item in items if item["set"] == "clean"]),
        "no_named_constrained": sum(1 for item in items if not _named_constrained(item)),
        "clean_turn0": turn0_table([item for item in items if item["set"] == "clean"]),
        "utterance_mismatches": [item["item_id"] for item in items if not item["utterance_matches_v21"]],
    }
    return items, summary


def _named_constrained(item: dict) -> list[str]:
    return [slot for slot in item["goal"] if item["slot_status"][slot] == "named"]


def _by_size(items: list[dict]) -> dict[str, int]:
    sizes = Counter(item["n_constrained"] for item in items)
    return {str(size): sizes[size] for size in sorted(sizes)}


def turn0_table(items: list[dict]) -> dict[str, dict[str, int]]:
    table = {}
    for slot in SLOTS:
        constrained = [item for item in items if slot in item["goal"]]
        no_pref = [item for item in items if slot not in item["goal"]]
        table[slot] = {
            "constrained": len(constrained),
            "named": sum(1 for item in constrained if item["slot_status"][slot] == "named"),
            "named_mismatch": sum(1 for item in constrained if item["slot_match"][slot] is False),
            "omitted": sum(1 for item in constrained if item["slot_status"][slot] == "omitted"),
            "dontcare": sum(1 for item in constrained if item["slot_status"][slot] == "dontcare"),
            "no_pref": len(no_pref),
            "no_pref_named": sum(1 for item in no_pref if item["slot_status"][slot] == "named"),
            "no_pref_dontcare": sum(1 for item in no_pref if item["slot_status"][slot] == "dontcare"),
        }
    return table


def pooled_population(goals: dict[str, dict], turns: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """First restaurant turn of every dialogue, any prefix, with a slot-based restaurant goal."""
    rows = []
    counts = Counter()
    for dialogue_id in sorted(goals):
        goal = goals[dialogue_id]["goal"]
        if not restaurant_goal(goal):
            continue
        g_first = first_goal(goal)
        if "name" in g_first:
            continue
        counts["slot_based_goal"] += 1
        if dialogue_id not in turns:
            counts["missing_in_v22"] += 1
            continue
        first = first_restaurant_turn(turns[dialogue_id])
        if first is None:
            counts["no_restaurant_turn"] += 1
            continue
        index, turn, previous = first
        constrained = constrained_slots(g_first)
        status = turn_status(turn["restaurant"], previous if index else None)
        values = {slot: slot_values(turn["restaurant"], slot) for slot in SLOTS}
        rows.append(
            {
                "dialogue_id": dialogue_id,
                "turn_index": index,
                "constrained": constrained,
                "status": status,
                "match": slot_match(g_first, constrained, status, values),
            }
        )
    counts["population"] = len(rows)
    return rows, dict(counts)


def item_population(items: list[dict]) -> list[dict]:
    """Turn-0 rows of analysis items, in the shape pooled_population returns."""
    return [
        {
            "dialogue_id": item["item_id"],
            "turn_index": 0,
            "constrained": [slot for slot in SLOTS if slot in item["goal"]],
            "status": item["slot_status"],
            "match": item["slot_match"],
        }
        for item in items
    ]


def differences(expected, actual, path: str = "") -> list[str]:
    """Every leaf where actual differs from expected, as 'path: expected X, got Y'."""
    if isinstance(expected, dict):
        found = []
        for key, value in expected.items():
            here = f"{path}.{key}" if path else str(key)
            if not isinstance(actual, dict) or key not in actual:
                found.append(f"{here}: expected {value}, got nothing")
                continue
            found.extend(differences(value, actual[key], here))
        return found
    return [] if expected == actual else [f"{path}: expected {expected}, got {actual}"]
