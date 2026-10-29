"""A deterministic slot extractor: text to (food, area, pricerange), each a value, dontcare, or none.

The vocabulary is every food value in a restaurant goal or the 2.1 ontology (compound
ontology values such as "kosher|british" are split) and the area and price values of the
2.2 schema. The synonym list is [Frozen]. Matching is case-insensitive on word
boundaries. "No preference" phrasings are read first, then foods (longest first), then
price and area in what is left, so "north american" is a food and not an area, and
"inexpensive" is cheap and not expensive.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from robust_likelihood.multiwoz.corpus import SLOTS, load_goals
from robust_likelihood.multiwoz.hypotheses import food_pool

VOCAB_FILE = "semantic_vocab.json"
NONE = "none"
DONTCARE = "dontcare"

# [Frozen] synonyms, extended after the rewrite pilot with the user's agreement (docs/claims_decisions.md):
# adjectival areas, downtown, and -priced/-cost price phrasings.
AREA_SYNONYMS = {"city centre": "centre", "city center": "centre", "centre": "centre", "center": "centre",
                 "central": "centre", "downtown": "centre", "north": "north", "south": "south", "east": "east",
                 "west": "west", "northern": "north", "southern": "south", "eastern": "east", "western": "west"}
PRICE_SYNONYMS = {"inexpensive": "cheap", "cheap": "cheap", "budget": "cheap", "low-priced": "cheap",
                  "low priced": "cheap", "low-cost": "cheap", "low cost": "cheap",
                  "moderately priced": "moderate", "mid-range": "moderate", "mid range": "moderate",
                  "reasonably priced": "moderate", "moderate": "moderate", "moderately": "moderate",
                  "moderately-priced": "moderate", "mid-priced": "moderate", "mid priced": "moderate",
                  "expensive": "expensive", "upscale": "expensive", "high-end": "expensive",
                  "high end": "expensive", "pricey": "expensive", "pricy": "expensive", "high-priced": "expensive",
                  "high priced": "expensive", "costly": "expensive"}
SLOT_WORDS = {
    "food": r"food|cuisine|type of (?:food|cuisine|restaurant)|kind of (?:food|cuisine)",
    "area": r"area|location|part of (?:the )?(?:town|city)|where(?: it is)?|side of town|neighbou?rhood",
    "pricerange": r"price(?: range)?|pricerange|cost|how (?:much|expensive)",
}
DONTCARE_PHRASES = {
    "food": [r"any (?:type|kind|sort) of (?:food|cuisine)", r"any (?:food|cuisine)", r"any type(?! of (?:area|price))"],
    "area": [r"any (?:area|location|part of (?:the )?(?:town|city)|side of town|neighbou?rhood)"],
    "pricerange": [r"any price(?: range)?", r"any (?:budget|cost)", r"price (?:range )?(?:is not|isn't) important"],
}
CARE = (
    r"(?:do(?:es)?n'?t|do(?:es)? not) (?:really )?(?:care|matter|mind)|no preference|not (?:important|picky|fussy)"
    r"|(?:do(?:es)?n'?t|do(?:es)? not) have (?:a|any) (?:particular |specific )?preference"
    r"|has no (?:particular |specific )?preference|(?:is|are|am) (?:not particular|flexible|open)"
)


def build_vocab(raw: Path) -> dict:
    goals = load_goals(raw)
    foods = set(food_pool(goals))
    ontology = json.loads((raw / "MultiWOZ_2.1" / "ontology.json").read_text(encoding="utf-8"))
    for value in ontology["restaurant-semi-food"]:
        foods.update(part.strip().lower() for part in re.split(r"[|>]", value))
    foods.discard(DONTCARE)
    schema = json.loads((raw / "MultiWOZ_2.2" / "schema.json").read_text(encoding="utf-8"))
    values = {}
    for service in schema:
        if service["service_name"] == "restaurant":
            for slot in service["slots"]:
                if slot["name"] in ("restaurant-area", "restaurant-pricerange"):
                    values[slot["name"].split("-", 1)[1]] = sorted(slot["possible_values"])
    return {"food": sorted(foods), "area": values["area"], "pricerange": values["pricerange"],
            "area_synonyms": AREA_SYNONYMS, "price_synonyms": PRICE_SYNONYMS}


def load_vocab(derived: Path) -> dict:
    return json.loads((derived / VOCAB_FILE).read_text(encoding="utf-8"))


def _alternation(phrases) -> str:
    return "|".join(re.escape(phrase) for phrase in sorted(phrases, key=lambda phrase: (-len(phrase), phrase)))


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text)


class Extractor:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.food = re.compile(rf"\b(?:{_alternation(vocab['food'])})\b")
        self.area = re.compile(rf"\b(?:{_alternation(vocab['area_synonyms'])})\b")
        self.price = re.compile(rf"\b(?:{_alternation(vocab['price_synonyms'])})\b")
        # A "no preference" match consumes only the nearest slot phrase, so one phrase can cover a
        # list: "no preference about the type of food, the area, or the price range".
        self.dontcare = {
            slot: [re.compile(rf"\b{phrase}\b") for phrase in phrases] + [
                re.compile(rf"\b(?P<slot>{SLOT_WORDS[slot]})\b[^.!?,;]{{0,30}}?\b(?:{CARE})\b"),
                re.compile(rf"\b(?:{CARE})\b[^.!?;]{{0,30}}?\b(?P<slot>{SLOT_WORDS[slot]})\b"),
            ]
            for slot, phrases in DONTCARE_PHRASES.items()
        }

    def extract(self, text: str) -> dict[str, str]:
        remaining = normalize_text(text)
        found = {slot: NONE for slot in SLOTS}

        def take(pattern: re.Pattern) -> str | None:
            nonlocal remaining
            match = pattern.search(remaining)
            if match is None:
                return None
            group = "slot" if "slot" in pattern.groupindex else 0
            remaining = remaining[: match.start(group)] + "|" + remaining[match.end(group):]
            return match.group(0)

        for slot in SLOTS:
            for pattern in self.dontcare[slot]:
                if take(pattern) is not None:
                    found[slot] = DONTCARE
                    break
        if found["food"] == NONE:
            value = take(self.food)
            if value is not None:
                found["food"] = value
        if found["pricerange"] == NONE:
            value = take(self.price)
            if value is not None:
                found["pricerange"] = self.vocab["price_synonyms"][value]
        if found["area"] == NONE:
            value = take(self.area)
            if value is not None:
                found["area"] = self.vocab["area_synonyms"][value]
        return found


def canonical(slot: str, value: str, vocab: dict) -> str:
    value = normalize_text(value).strip()
    if value == DONTCARE:
        return DONTCARE
    if slot == "area":
        return vocab["area_synonyms"].get(value, value)
    if slot == "pricerange":
        return vocab["price_synonyms"].get(value, value)
    return value


def annotated_tuple(item: dict, vocab: dict) -> dict[str, list[str]]:
    """The 2.2 turn-0 annotation per slot: its canonical values, [dontcare], or [none]."""
    found = {}
    for slot in SLOTS:
        status = item["slot_status"][slot]
        if status == "omitted":
            found[slot] = [NONE]
        elif status == "dontcare":
            found[slot] = [DONTCARE]
        else:
            found[slot] = sorted({canonical(slot, value, vocab) for value in item["turn0_values"][slot]})
    return found


def hypothesis_tuple(row: dict) -> dict[str, str]:
    """The slots a probe hypothesis states: its values, dontcare where an explicit text says no preference, else none."""
    values = row.get("values") or {}
    unstated = DONTCARE if row["regime"] == "expl" else NONE
    return {slot: values[slot] if slot in row["constrained"] else unstated for slot in SLOTS}


def hypothesis_check(rows: list[dict], vocab: dict) -> dict:
    """Every distinct probe hypothesis text must extract to the slots it was built from."""
    extractor = Extractor(vocab)
    texts = {}
    for row in rows:
        texts.setdefault(row["text"], hypothesis_tuple(row))
    mismatches = []
    for text, expected in sorted(texts.items()):
        found = extractor.extract(text)
        if found != expected:
            mismatches.append({"text": text, "expected": expected, "extracted": found})
    return {"texts": len(texts), "agreement": 1 - len(mismatches) / len(texts) if texts else None,
            "mismatches": mismatches}


def _kind(value: str) -> str:
    return value if value in (NONE, DONTCARE) else "value"


def semantic_check(items: list[dict], vocab: dict) -> dict:
    """Per-slot agreement of the extractor with the 2.2 turn-0 annotations, and the confusion table."""
    extractor = Extractor(vocab)
    agree = Counter()
    confusion = {slot: Counter() for slot in SLOTS}
    disagreements = []
    for item in items:
        extracted = extractor.extract(item["utterance"])
        truth = annotated_tuple(item, vocab)
        for slot in SLOTS:
            ok = extracted[slot] in truth[slot]
            agree[slot] += ok
            expected = _kind(truth[slot][0])
            got = _kind(extracted[slot])
            cell = f"{expected}->{got}" if expected != "value" or got != "value" else ("value->same" if ok else "value->other")
            confusion[slot][cell] += 1
            if not ok:
                disagreements.append({"item_id": item["item_id"], "slot": slot, "annotated": truth[slot],
                                      "extracted": extracted[slot], "utterance": item["utterance"]})
    total = len(items)
    rates = {slot: agree[slot] / total for slot in SLOTS}
    return {
        "items": total,
        "agreement": rates,
        "passes_95": all(rate >= 0.95 for rate in rates.values()),
        "confusion": {slot: dict(sorted(confusion[slot].items())) for slot in SLOTS},
        "disagreements": disagreements,
    }


def check_markdown(result: dict) -> str:
    lines = ["# Semantic extractor check", "",
             f"Regex slot extractor against the MultiWOZ 2.2 turn-0 annotations of the {result['items']} items.",
             "Acceptance: at least 95% agreement on every slot.", "",
             "| Slot | Agreement |", "|---|---|"]
    lines += [f"| {slot} | {100 * rate:.1f}% |" for slot, rate in result["agreement"].items()]
    lines += ["", f"Passes: {'yes' if result['passes_95'] else 'no'}.", "", "## Confusion (annotated → extracted)", "",
              "`value->same` and `value->other` split named slots by whether the value agrees.", ""]
    cells = sorted({cell for found in result["confusion"].values() for cell in found})
    lines += ["| Cell | " + " | ".join(SLOTS) + " |", "|---|" + "---|" * len(SLOTS)]
    lines += [f"| {cell} | " + " | ".join(str(result["confusion"][slot].get(cell, 0)) for slot in SLOTS) + " |"
              for cell in cells]
    lines += ["", "## Disagreements", "", "| Item | Slot | Annotated | Extracted | Utterance |", "|---|---|---|---|---|"]
    lines += [f"| {row['item_id']} | {row['slot']} | {', '.join(row['annotated'])} | {row['extracted']} | "
              f"{row['utterance'].replace('|', '/')} |" for row in result["disagreements"]]
    hypotheses = result.get("hypotheses")
    if hypotheses:
        lines += ["", "## Hypothesis texts", "",
                  f"Every distinct probe hypothesis text ({hypotheses['texts']}) against the slots it was built from:",
                  f"{100 * hypotheses['agreement']:.1f}% extract exactly.", ""]
        lines += [f"- {row['text']} expected {row['expected']}, extracted {row['extracted']}"
                  for row in hypotheses["mismatches"]]
    return "\n".join(lines) + "\n"
