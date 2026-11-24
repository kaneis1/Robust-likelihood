"""Controlled typo, casual-style and length perturbations of human first turns.

Xuan's question (17 Nov): a likelihood should ignore what people ignore when they read, such as typos
and padding, and respond only to meaning. The Claude-written `typo` rewrites of the claims run mix
several typos with dropped words, so they cannot say how much one typo costs. This module makes the
perturbations deterministic and dosed:

- typo1, typo2, typo4: k character typos in k different words, nested (typo2 contains typo1's edit).
  A typo swaps two adjacent interior letters, deletes an interior letter, replaces one with a
  neighbouring QWERTY key, or doubles one. First and last letters stay. Words carrying a slot value,
  a slot word, a "no preference" phrase or a negation are protected, so the meaning is unchanged.
- slottypo1: one typo inside a slot-value word ("itallian"). A reader still understands it; the
  regex extractor does not, so these units are expected to fail the extractor check.
- lower, nopunct, noapos: lower-case everything; drop sentence punctuation; drop apostrophes.
- fill1, fill2, fill4: k content-free sentences added around the turn, nested.

Every unit records its edits, the words it added, and whether the slot extractor reads the same
(food, area, pricerange) before and after. A unit whose meaning-preserving check fails is kept with
valid = false and left out of the analysis.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from robust_likelihood.multiwoz.semantic import CARE, DONTCARE_PHRASES, SLOT_WORDS, Extractor

TYPO_DOSES = (1, 2, 4)
FILL_DOSES = (1, 2, 4)
OPS = ("swap", "delete", "neighbour", "double")
MIN_WORD = 4
PROTECTED_WORDS = {"not", "don't", "dont", "doesn't", "doesnt", "never", "without", "nothing", "any",
                   "anything", "anywhere", "either", "neither", "nor", "except", "rather", "instead", "only"}
QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
# Content-free sentences, in the order a person would write them:
# fill1 = turn + thanks; fill2 = greeting + turn + thanks; fill4 = greeting + lead-in + turn + thanks + closing.
GREETINGS = ("Hi there.", "Hello, I hope you're having a good day.", "Good afternoon.")
LEADINS = ("Sorry to bother you.", "I have a quick question.", "I hope this is the right place to ask.")
THANKS = ("Thank you.", "Thanks so much for your help.", "Thanks in advance.")
CLOSINGS = ("Have a nice day.", "Looking forward to your answer.", "I really appreciate it.")
LEXICON_FILE = "typo_lexicon.txt"
WORD = re.compile(r"[A-Za-z][A-Za-z']*[A-Za-z]|[A-Za-z]")


def unit_seed(seed: int, *parts: str) -> int:
    key = ":".join([str(seed), *parts])
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def neighbours() -> dict[str, str]:
    found: dict[str, set[str]] = {}
    for row_index, row in enumerate(QWERTY_ROWS):
        for col, letter in enumerate(row):
            near = set()
            for other_row in range(max(0, row_index - 1), min(len(QWERTY_ROWS), row_index + 2)):
                for other_col in (col - 1, col, col + 1):
                    if 0 <= other_col < len(QWERTY_ROWS[other_row]):
                        near.add(QWERTY_ROWS[other_row][other_col])
            near.discard(letter)
            found[letter] = near
    return {letter: "".join(sorted(near)) for letter, near in found.items()}


NEIGHBOURS = neighbours()


def protected_spans(text: str, extractor: Extractor) -> list[tuple[int, int]]:
    """Character spans of slot values, slot words, "no preference" phrases. Same length as text."""
    lowered = text.lower().replace("’", "'").replace("‘", "'")
    patterns = [extractor.food, extractor.area, extractor.price]
    patterns += [re.compile(rf"\b(?:{words})\b") for words in SLOT_WORDS.values()]
    patterns += [re.compile(rf"\b(?:{CARE})\b")]
    patterns += [re.compile(rf"\b{phrase}\b") for phrases in DONTCARE_PHRASES.values() for phrase in phrases]
    spans = []
    for pattern in patterns:
        spans += [(match.start(), match.end()) for match in pattern.finditer(lowered)]
    return spans


def slot_value_spans(text: str, extractor: Extractor) -> list[tuple[int, int]]:
    lowered = text.lower().replace("’", "'").replace("‘", "'")
    spans = []
    for pattern in (extractor.food, extractor.area, extractor.price):
        spans += [(match.start(), match.end()) for match in pattern.finditer(lowered)]
    return spans


def words(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in WORD.finditer(text)]


def overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def typo_candidates(text: str, extractor: Extractor) -> list[tuple[int, int, str]]:
    spans = protected_spans(text, extractor)
    return [(start, end, word) for start, end, word in words(text)
            if len(word) >= MIN_WORD and word.isalpha() and word.lower() not in PROTECTED_WORDS
            and not overlaps(start, end, spans)]


def slot_typo_candidates(text: str, extractor: Extractor) -> list[tuple[int, int, str]]:
    spans = slot_value_spans(text, extractor)
    return [(start, end, word) for start, end, word in words(text)
            if len(word) >= MIN_WORD and word.isalpha() and overlaps(start, end, spans)]


def match_case(letter: str, like: str) -> str:
    return letter.upper() if like.isupper() else letter


def apply_op(word: str, op: str, rng: random.Random) -> str | None:
    """One typo in an interior position; None if this op cannot change the word."""
    interior = range(1, len(word) - 1)
    if op == "swap":
        options = [i for i in range(1, len(word) - 2) if word[i].lower() != word[i + 1].lower()]
        if not options:
            return None
        i = rng.choice(options)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        options = [i for i in interior if word[i - 1].lower() != word[i].lower()]
        if not options:
            return None
        i = rng.choice(options)
        return word[:i] + word[i + 1:]
    if op == "neighbour":
        options = [i for i in interior if word[i].lower() in NEIGHBOURS]
        if not options:
            return None
        i = rng.choice(options)
        letter = rng.choice(NEIGHBOURS[word[i].lower()])
        return word[:i] + match_case(letter, word[i]) + word[i + 1:]
    if op == "double":
        options = list(interior)
        i = rng.choice(options)
        return word[: i + 1] + word[i] + word[i + 1:]
    raise ValueError(op)


def edit_words(text: str, chosen: list[tuple[int, int, str]], rng: random.Random,
               lexicon: frozenset[str] = frozenset()) -> tuple[str, list[dict]]:
    """Apply one typo to each chosen word, ops drawn at random; returns text and the edit log.

    A typo that spells a word of the lexicon ("dinner" to "diner") is redrawn, up to three times per op."""
    edits = []
    replacements = {}
    for start, end, word in chosen:
        ops = list(OPS)
        rng.shuffle(ops)
        done = False
        for op in ops:
            for _ in range(3):
                new = apply_op(word, op, rng)
                if new is not None and new != word and new.lower() not in lexicon:
                    replacements[(start, end)] = new
                    edits.append({"start": start, "word": word, "op": op, "typo": new})
                    done = True
                    break
            if done:
                break
    pieces, cursor = [], 0
    for (start, end), new in sorted(replacements.items()):
        pieces += [text[cursor:start], new]
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), edits


def nopunct(text: str) -> str:
    text = re.sub(r"[.,!?;:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def noapos(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", text)


def perturb_utterance(utterance: str, item_id: str, extractor: Extractor, seed: int,
                      lexicon: frozenset[str] = frozenset()) -> list[dict]:
    before = extractor.extract(utterance)
    units: list[dict] = []

    def add(kind: str, dose: int | None, text: str, edits: list[dict] | None = None,
            added: list[str] | None = None, available: int | None = None) -> None:
        after = extractor.extract(text)
        units.append({
            "unit_id": f"{item_id}|{kind}", "item_id": item_id, "kind": kind, "dose": dose,
            "original": utterance, "rewrite": text, "edits": edits or [], "added": added or [],
            "available": available, "changed": text != utterance,
            "extract_before": before, "extract_after": after,
            "valid": text != utterance and (kind == "slottypo1" or after == before),
        })

    candidates = typo_candidates(utterance, extractor)
    rng = random.Random(unit_seed(seed, item_id, "typo"))
    order = candidates[:]
    rng.shuffle(order)
    for dose in TYPO_DOSES:
        if len(order) >= dose:
            # Nested doses: the same words in the same order, and the same op draws for shared words.
            text, edits = edit_words(utterance, order[:dose], random.Random(unit_seed(seed, item_id, "ops")), lexicon)
            add(f"typo{dose}", dose, text, edits=edits, available=len(candidates))

    slot_words = slot_typo_candidates(utterance, extractor)
    if slot_words:
        pick = random.Random(unit_seed(seed, item_id, "slottypo")).choice(slot_words)
        text, edits = edit_words(utterance, [pick], random.Random(unit_seed(seed, item_id, "slotops")), lexicon)
        add("slottypo1", 1, text, edits=edits, available=len(slot_words))

    add("lower", None, utterance.lower())
    add("nopunct", None, nopunct(utterance))
    add("noapos", None, noapos(utterance))

    fill_rng = random.Random(unit_seed(seed, item_id, "fill"))
    greeting, leadin, thanks, closing = (fill_rng.choice(options) for options in (GREETINGS, LEADINS, THANKS, CLOSINGS))
    layouts = {1: ([], [thanks]), 2: ([greeting], [thanks]), 4: ([greeting, leadin], [thanks, closing])}
    for dose in FILL_DOSES:
        before_parts, after_parts = layouts[dose]
        text = " ".join([*before_parts, utterance.strip(), *after_parts])
        add(f"fill{dose}", dose, text, added=[*before_parts, *after_parts])
    return units


def build_lexicon(raw: Path, common_words: list[str], min_dialogues: int = 2) -> list[str]:
    """Real words a typo must not spell: the given common English words (the committed list was
    wordfreq's top_n_list("en", 50000)) plus every word in at least min_dialogues MultiWOZ 2.2
    dialogues. Written once to LEXICON_FILE and read from there, so the module needs no dependency."""
    counts: Counter = Counter()
    for path in sorted((raw / "MultiWOZ_2.2").glob("*/dialogues_*.json")):
        for dialogue in json.loads(path.read_text(encoding="utf-8")):
            seen = set()
            for turn in dialogue["turns"]:
                seen.update(word.lower() for _, _, word in words(turn["utterance"]))
            counts.update(seen)
    found = {word for word, count in counts.items() if count >= min_dialogues}
    found |= {word.lower() for word in common_words if WORD.fullmatch(word)}
    return sorted(found)


def load_lexicon(derived: Path) -> frozenset[str]:
    return frozenset((derived / LEXICON_FILE).read_text(encoding="utf-8").split())


def build(items: list[dict], extractor: Extractor, seed: int, lexicon: frozenset[str] = frozenset()) -> list[dict]:
    """Perturbations of each item's first turn, in item order."""
    rows = []
    for item in sorted(items, key=lambda found: found["item_id"]):
        rows += perturb_utterance(item["utterance"], item["item_id"], extractor, seed, lexicon)
    return rows


def summary(rows: list[dict]) -> dict:
    kinds = Counter(row["kind"] for row in rows)
    valid = Counter(row["kind"] for row in rows if row["valid"])
    ops = Counter(edit["op"] for row in rows if row["kind"] == "typo1" for edit in row["edits"])
    extractor_changed = Counter(row["kind"] for row in rows if row["extract_after"] != row["extract_before"])
    return {"units": dict(sorted(kinds.items())), "valid": dict(sorted(valid.items())),
            "typo1_ops": dict(sorted(ops.items())), "extractor_changed": dict(sorted(extractor_changed.items()))}
