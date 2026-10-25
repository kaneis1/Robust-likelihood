"""Hypothesis texts for the size-principle probe. The templates are frozen.

A hypothesis is a set of slot values. base() states them in one sentence. explicit()
adds one sentence naming the slots it leaves open, so the two wordings share the
first sentence word for word and differ only in that statement of no preference.

Keys such as G|impl or minus:food|expl are internal. They never enter a text.
"""

from __future__ import annotations

import hashlib
import random

from robust_likelihood.multiwoz.corpus import SLOTS, restaurant_goal

PRICE = {"cheap": "cheap", "moderate": "moderately priced", "expensive": "expensive"}
NAMES = {"food": "the type of food", "area": "the area", "pricerange": "the price range"}
AREAS = ("centre", "north", "south", "east", "west")
PRICES = ("cheap", "moderate", "expensive")


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def base(goal: dict) -> str:
    """The user wants to find {a|an} [PRICE ]restaurant[ serving {food} food][ in the {area} of town]."""
    noun = "restaurant"
    if "pricerange" in goal:
        noun = f"{PRICE[goal['pricerange']]} {noun}"
    text = f"The user wants to find {article(noun)} {noun}"
    if "food" in goal:
        text += f" serving {goal['food'].lower()} food"
    if "area" in goal:
        text += f" in the {goal['area']} of town"
    return text + "."


def join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def explicit(goal: dict) -> str:
    """base(goal), then the slots it leaves open in the order food, area, pricerange."""
    open_slots = [NAMES[slot] for slot in SLOTS if slot not in goal]
    if not open_slots:
        return base(goal)
    return f"{base(goal)} The user has no preference about {join_names(open_slots)}."


def food_pool(goals: dict[str, dict]) -> list[str]:
    """Every food value in any restaurant goal's info or fail_info, except dontcare."""
    found = set()
    for record in goals.values():
        restaurant = restaurant_goal(record["goal"])
        for part in ("info", "fail_info"):
            value = (restaurant.get(part) or {}).get("food")
            if value is not None and value != "dontcare":
                found.add(value.lower())
    return sorted(found)


def value_pools(foods: list[str]) -> dict[str, list[str]]:
    return {"food": sorted(foods), "area": sorted(AREAS), "pricerange": sorted(PRICES)}


def control_seed(item_id: str, kind: str) -> int:
    return int(hashlib.sha256(f"{item_id}:{kind}".encode("utf-8")).hexdigest()[:8], 16)


def changed_goal(
    goal: dict,
    candidates: list[str],
    item_id: str,
    kind: str,
    pools: dict[str, list[str]],
    avoid: tuple[str, str] | None = None,
) -> tuple[dict, dict]:
    """The goal with one candidate slot set to another pool value, chosen by the seeded generator.

    The new value is never the goal value. avoid is a (slot, value) already used by
    another control of the same item, so a second tie control is not a copy of the first.
    """
    rng = random.Random(control_seed(item_id, kind))
    slot = rng.choice(sorted(candidates, key=SLOTS.index))
    options = [value for value in pools[slot] if value.lower() != goal[slot].lower()]
    if avoid is not None and avoid[0] == slot:
        options = [value for value in options if value != avoid[1]]
    value = rng.choice(options)
    changed = dict(goal)
    changed[slot] = value
    return changed, {"slot": slot, "from": goal[slot], "to": value}


def _row(hyp_key: str, hyp_type: str, regime: str, values: dict, removed: list[str], text: str, changed=None) -> dict:
    row = {
        "hyp_key": hyp_key,
        "hyp_type": hyp_type,
        "regime": regime,
        "constrained": [slot for slot in SLOTS if slot in values],
        "removed": removed,
        "values": {slot: values[slot] for slot in SLOTS if slot in values},
        "text": text,
    }
    if changed is not None:
        row["changed"] = changed
    return row


def item_hypotheses(item: dict, pools: dict[str, list[str]]) -> list[dict]:
    """Every hypothesis row for one item, in a fixed order."""
    goal = {slot: item["goal"][slot] for slot in SLOTS if slot in item["goal"]}
    constrained = list(goal)
    status = item["slot_status"]
    rows = [_row("G|impl", "G", "impl", goal, [], base(goal))]
    if explicit(goal) != base(goal):
        rows.append(_row("G|expl", "G", "expl", goal, [], explicit(goal)))
    for slot in constrained:
        narrower = {key: value for key, value in goal.items() if key != slot}
        rows.append(_row(f"minus:{slot}|impl", f"minus:{slot}", "impl", narrower, [slot], base(narrower)))
        rows.append(_row(f"minus:{slot}|expl", f"minus:{slot}", "expl", narrower, [slot], explicit(narrower)))
    rows.append(_row("broad|impl", "broad", "impl", {}, list(constrained), base({})))
    rows.append(_row("broad|expl", "broad", "expl", {}, list(constrained), explicit({})))
    named = [slot for slot in constrained if status[slot] == "named"]
    omitted = [slot for slot in constrained if status[slot] == "omitted"]
    tie = None
    if omitted:
        values, tie = changed_goal(goal, omitted, item["item_id"], "tie", pools)
        tie_row = _row("ctrl:tie", "ctrl:tie", "impl", values, [], base(values), tie)
    if named:
        values, change = changed_goal(goal, named, item["item_id"], "contra", pools)
        rows.append(_row("ctrl:contra", "ctrl:contra", "impl", values, [], base(values), change))
    else:
        avoid = None if tie is None else (tie["slot"], tie["to"])
        values, change = changed_goal(goal, omitted, item["item_id"], "tie2", pools, avoid)
        rows.append(_row("ctrl:tie2", "ctrl:tie2", "impl", values, [], base(values), change))
    if omitted:
        rows.append(tie_row)
    return rows
