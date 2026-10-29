import random
import threading
from datetime import datetime, timezone

from robust_likelihood.clients import TransportResult
from robust_likelihood.multiwoz.rewrites import (
    CLOSERS,
    OPENERS,
    audit_rows,
    build_units,
    llm_messages,
    pad,
    protected_spans,
    run_rewrites,
    typo,
)
from robust_likelihood.multiwoz.semantic import AREA_SYNONYMS, PRICE_SYNONYMS, Extractor
from robust_likelihood.storage import read_jsonl, repo_root, write_json, write_jsonl

VOCAB = {
    "food": ["british", "indian", "italian"],
    "area": ["centre", "east", "north", "south", "west"],
    "pricerange": ["cheap", "expensive", "moderate"],
    "area_synonyms": AREA_SYNONYMS,
    "price_synonyms": PRICE_SYNONYMS,
}
ENV = {"ANTHROPIC_API_KEY": "test-claude"}
FIXED = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)


def test_typo_never_touches_slot_words_and_is_seeded():
    extractor = Extractor(VOCAB)
    text = "I am looking for a cheap italian restaurant in the centre of town for tonight."
    protected = protected_spans(text, extractor)
    first = typo(text, random.Random(5), protected)
    assert first == typo(text, random.Random(5), protected) and first != text
    assert extractor.extract(first) == extractor.extract(text)
    for word in ("cheap", "italian", "centre"):
        assert word in first
    assert abs(len(first) - len(text)) <= round(len(text) / 12)


def test_pad_wraps_the_message_in_a_fixed_opener_and_closer():
    padded = pad("Cheap food in the north.", random.Random(1))
    assert any(padded.startswith(opener) for opener in OPENERS)
    assert any(padded.endswith(closer) for closer in CLOSERS)
    assert Extractor(VOCAB).extract(padded) == Extractor(VOCAB).extract("Cheap food in the north.")


def test_hypothesis_prompt_keeps_no_preference_only_for_explicit_texts():
    assert "no preference" in llm_messages("hpara1", "x", explicit=True)[0]["content"]
    assert "no preference" not in llm_messages("hpara1", "x", explicit=False)[0]["content"]
    assert llm_messages("zh", "hello")[0]["content"].startswith("Translate this text into Chinese.")


ITEMS = [
    {"item_id": "SNG1", "set": "clean", "utterance": "I want a cheap italian restaurant in the north.",
     "slot_status": {"food": "named", "area": "named", "pricerange": "named"},
     "turn0_values": {"food": ["italian"], "area": ["north"], "pricerange": ["cheap"]}},
]
PROBE = [
    {"item_id": "SNG1", "set": "clean", "regime": "impl", "constrained": ["food", "area", "pricerange"],
     "values": {"food": "italian", "area": "north", "pricerange": "cheap"},
     "text": "The user wants to find a cheap restaurant serving italian food in the north of town."},
    {"item_id": "SNG1", "set": "clean", "regime": "expl", "constrained": [], "values": {},
     "text": "The user wants to find a restaurant. The user has no preference about the type of food, the area, "
             "or the price range."},
]


def test_units_cover_every_kind_with_the_slots_to_keep():
    units = build_units(ITEMS, PROBE, VOCAB)
    assert len(units) == 6 + 2 * 4
    hypothesis = [unit for unit in units if unit["side"] == "hypothesis" and unit["explicit"]]
    assert hypothesis[0]["expected"] == {"food": ["dontcare"], "area": ["dontcare"], "pricerange": ["dontcare"]}


class FakeClaude:
    """Paraphrases keep the text; Chinese is a marker; back-translation drops the price on the first try."""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompts = []

    def __call__(self, url, headers, body, timeout):
        prompt = body["messages"][0]["content"]
        with self.lock:
            self.prompts.append(prompt)
            backs = sum(1 for seen in self.prompts if seen.startswith("Translate this text into English"))
        text = prompt.split("\n\n", 1)[1]
        if prompt.startswith("Translate this text into Chinese"):
            reply = "ZH:" + text
        elif prompt.startswith("Translate this text into English"):
            reply = text[3:] if backs > 2 else text[3:].replace("cheap ", "")
        else:
            reply = "Well, " + text
        return TransportResult(200, {"model": body["model"], "stop_reason": "end_turn",
                                     "content": [{"type": "text", "text": reply}],
                                     "usage": {"input_tokens": 40, "output_tokens": 20}}, None, False)


class FakeNLI:
    def entailment_probs(self, pairs):
        return [0.95 for _ in pairs]


def test_rewrites_regenerate_failures_and_record_acceptance(tmp_path):
    derived = tmp_path / "derived"
    write_jsonl(derived / "items.jsonl", ITEMS)
    write_jsonl(derived / "probe.jsonl", PROBE)
    write_json(derived / "semantic_vocab.json", VOCAB)
    summary = run_rewrites(repo_root(), derived=derived, results_root=tmp_path / "results", transport=FakeClaude(),
                           environ=ENV, sleeper=lambda seconds: None, nli=FakeNLI(), now=FIXED, workers=1)
    assert summary["accepted"] == summary["units"] == 14
    candidates = read_jsonl(tmp_path / "results" / "20261029T090000Z-full" / "candidates.jsonl")
    retried = [row for row in candidates if row["attempt"] == 1]
    assert retried and all(row["unit_id"].endswith(("zh", "hzh")) for row in retried)
    assert all(row["accepted"] for row in retried)
    assert summary["acceptance"]["typo"]["rate"] == 1.0
    accepted = read_jsonl(tmp_path / "results" / "20261029T090000Z-full" / "accepted.jsonl")
    rows = audit_rows(accepted, per_side=2)
    assert [row["side"] for row in rows] == ["utterance", "utterance", "hypothesis", "hypothesis"]
