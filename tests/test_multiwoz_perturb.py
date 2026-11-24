import random

import pytest

from robust_likelihood.multiwoz import perturb
from robust_likelihood.multiwoz.semantic import AREA_SYNONYMS, PRICE_SYNONYMS, Extractor

VOCAB = {
    "food": ["british", "european", "indian", "italian", "modern european"],
    "area": ["centre", "east", "north", "south", "west"],
    "pricerange": ["cheap", "expensive", "moderate"],
    "area_synonyms": AREA_SYNONYMS,
    "price_synonyms": PRICE_SYNONYMS,
}
EXTRACTOR = Extractor(VOCAB)
TURN = "Hello, I would like to find a cheap Italian restaurant somewhere in the north, please."
SEED = 20261124


def kinds(rows):
    return {row["kind"]: row for row in rows}


def test_typo_candidates_skip_slot_values_slot_words_and_negations():
    text = "I don't care about the price range but I want italian food somewhere in the north"
    found = {word for _, _, word in perturb.typo_candidates(text, EXTRACTOR)}
    assert "italian" not in found and "north" not in found and "price" not in found
    assert "range" not in found and "care" not in found and "food" not in found
    assert "somewhere" in found and "about" in found and "want" in found


@pytest.mark.parametrize("op", perturb.OPS)
def test_each_op_keeps_first_and_last_letter_and_changes_the_word(op):
    rng = random.Random(1)
    for word in ("restaurant", "somewhere", "would", "please"):
        new = perturb.apply_op(word, op, rng)
        assert new is not None and new != word
        assert new[0] == word[0] and new[-1] == word[-1]
        assert abs(len(new) - len(word)) <= 1


def test_typo_doses_are_nested_and_keep_the_meaning():
    rows = kinds(perturb.perturb_utterance(TURN, "X1", EXTRACTOR, SEED))
    one, two, four = rows["typo1"], rows["typo2"], rows["typo4"]
    assert [len(row["edits"]) for row in (one, two, four)] == [1, 2, 4]
    first = one["edits"][0]
    assert first in two["edits"] and first in four["edits"]
    assert all(row["valid"] and row["extract_after"] == row["extract_before"] for row in (one, two, four))


def test_typos_never_spell_a_lexicon_word():
    lexicon = frozenset({"woud", "wuold", "wolud", "woukd", "wouldd"})
    for seed in range(40):
        rows = kinds(perturb.perturb_utterance(TURN, "X1", EXTRACTOR, seed, lexicon))
        for edit in rows["typo4"]["edits"]:
            assert edit["typo"].lower() not in lexicon


def test_slot_typo_hits_a_slot_value_and_is_kept():
    rows = kinds(perturb.perturb_utterance(TURN, "X1", EXTRACTOR, SEED))
    slot = rows["slottypo1"]
    assert slot["edits"][0]["word"].lower() in {"cheap", "italian", "north"}
    assert slot["valid"]


def test_style_edits_and_fill_layouts():
    rows = kinds(perturb.perturb_utterance("I'm after a cheap place in the north, thanks!", "X2", EXTRACTOR, SEED))
    assert rows["lower"]["rewrite"] == "i'm after a cheap place in the north, thanks!"
    assert rows["nopunct"]["rewrite"] == "I'm after a cheap place in the north thanks"
    assert rows["noapos"]["rewrite"] == "Im after a cheap place in the north, thanks!"
    fill1, fill2, fill4 = rows["fill1"], rows["fill2"], rows["fill4"]
    assert len(fill1["added"]) == 1 and fill1["added"][0] in perturb.THANKS
    assert fill2["added"][0] in perturb.GREETINGS and fill2["added"][1] == fill1["added"][0]
    assert fill4["added"][0] == fill2["added"][0] and fill4["added"][1] in perturb.LEADINS
    assert fill4["added"][3] in perturb.CLOSINGS
    assert all(row["valid"] for row in (fill1, fill2, fill4))


def test_unchanged_style_edit_is_not_valid():
    rows = kinds(perturb.perturb_utterance("cheap food in the north", "X3", EXTRACTOR, SEED))
    assert not rows["lower"]["valid"] and not rows["noapos"]["valid"]


def test_build_is_deterministic():
    items = [{"item_id": "B", "utterance": TURN}, {"item_id": "A", "utterance": "Any cheap place in the west?"}]
    first = perturb.build(items, EXTRACTOR, SEED)
    assert first == perturb.build(list(reversed(items)), EXTRACTOR, SEED)
    assert first[0]["item_id"] == "A"
