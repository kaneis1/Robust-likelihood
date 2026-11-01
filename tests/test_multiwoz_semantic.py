import pytest

from robust_likelihood.multiwoz.semantic import AREA_SYNONYMS, PRICE_SYNONYMS, Extractor, annotated_tuple, semantic_check

VOCAB = {
    "food": ["british", "european", "indian", "modern european", "north american", "north indian"],
    "area": ["centre", "east", "north", "south", "west"],
    "pricerange": ["cheap", "expensive", "moderate"],
    "area_synonyms": AREA_SYNONYMS,
    "price_synonyms": PRICE_SYNONYMS,
}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I want an expensive Indian place in the centre.", ("indian", "centre", "expensive")),
        ("Somewhere inexpensive serving North American food.", ("north american", "none", "cheap")),
        ("A moderately priced restaurant in the city center, please.", ("none", "centre", "moderate")),
        ("Any type of food is fine, but it should be in the north and cheap.", ("dontcare", "north", "cheap")),
        ("I need a modern European restaurant; the area doesn't matter.", ("modern european", "dontcare", "none")),
        ("Find me a north indian place, any price range.", ("north indian", "none", "dontcare")),
        ("Hello, I need somewhere to eat.", ("none", "none", "none")),
        ("A high-end British restaurant in the west.", ("british", "west", "expensive")),
        ("The user wants to find a restaurant. The user has no preference about the type of food, the area, or the "
         "price range.", ("dontcare", "dontcare", "dontcare")),
        ("The user wants to find a cheap restaurant. The user has no preference about the type of food or the area.",
         ("dontcare", "dontcare", "cheap")),
        ("The user is looking for an inexpensive restaurant in the northern part of town. The user does not have a "
         "preference regarding the type of cuisine.", ("dontcare", "north", "cheap")),
        ("The user is looking for a restaurant. The user does not have any preference regarding the type of food, "
         "the area, or the price range.", ("dontcare", "dontcare", "dontcare")),
        ("The user does not have a preference regarding the cuisine type, the location, or the price range.",
         ("dontcare", "dontcare", "dontcare")),
        ("I don't care about the food, but it must be in the west and the price range should be cheap.",
         ("dontcare", "west", "cheap")),
        ("A high-priced Indian place downtown.", ("indian", "centre", "expensive")),
        ("Know anywhere that has european food?", ("european", "none", "none")),
    ],
)
def test_extractor_reads_values_synonyms_and_no_preference(text, expected):
    found = Extractor(VOCAB).extract(text)
    assert (found["food"], found["area"], found["pricerange"]) == expected


def test_semantic_check_scores_agreement_against_the_annotation():
    items = [
        {"item_id": "A", "utterance": "Cheap indian food in the centre.",
         "slot_status": {"food": "named", "area": "named", "pricerange": "named"},
         "turn0_values": {"food": ["indian"], "area": ["center"], "pricerange": ["cheap"]}},
        {"item_id": "B", "utterance": "Something british, please.",
         "slot_status": {"food": "named", "area": "omitted", "pricerange": "dontcare"},
         "turn0_values": {"food": ["british"], "area": [], "pricerange": ["dontcare"]}},
    ]
    assert annotated_tuple(items[0], VOCAB)["area"] == ["centre"]
    result = semantic_check(items, VOCAB)
    assert result["agreement"] == {"food": 1.0, "area": 1.0, "pricerange": 0.5}
    assert result["confusion"]["pricerange"] == {"dontcare->none": 1, "value->same": 1}
    assert [row["item_id"] for row in result["disagreements"]] == ["B"]
