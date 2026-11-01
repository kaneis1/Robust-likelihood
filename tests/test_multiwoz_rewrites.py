import hashlib
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.clients import TransportResult
from robust_likelihood.multiwoz.rewrites import (
    AUDIT_REJECTED,
    CLOSERS,
    OPENERS,
    audit_rows,
    build_units,
    llm_messages,
    pad,
    protected_spans,
    recheck_run,
    run_rewrites,
    typo,
    write_derived,
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


def test_recheck_accepts_only_regex_rejections_the_current_extractor_reads(tmp_path):
    derived = tmp_path / "derived"
    write_jsonl(derived / "items.jsonl", ITEMS)
    write_jsonl(derived / "probe.jsonl", PROBE)
    # The run's extractor does not know "cheap", so every text with a price fails the regex check;
    # the no-preference text passes it and is then rejected by NLI.
    narrow = dict(VOCAB, price_synonyms={key: value for key, value in PRICE_SYNONYMS.items() if key != "cheap"})
    write_json(derived / "semantic_vocab.json", narrow)
    transport = FakeClaude()
    summary = run_rewrites(repo_root(), derived=derived, results_root=tmp_path / "results", transport=transport,
                           environ=ENV, sleeper=lambda seconds: None, nli=RejectingNLI(), now=FIXED, workers=1)
    assert summary["accepted"] == 0
    calls = len(transport.prompts)
    run_dir = Path(summary["run_dir"])
    write_json(derived / "semantic_vocab.json", VOCAB)
    found = recheck_run(repo_root(), run_dir, nli=FakeNLI(), derived=derived, results_root=tmp_path / "recheck",
                        now=FIXED)
    assert len(transport.prompts) == calls
    accepted = set(found["accepted_units"])
    assert {"u:SNG1:pad", "u:SNG1:para1"} <= accepted
    explicit = hashlib.sha256(PROBE[1]["text"].encode("utf-8")).hexdigest()[:12]
    assert not any(unit.startswith(f"h:{explicit}:") for unit in accepted)  # NLI rejected these in the run
    stored = {(row["unit_id"], row["attempt"]): row for row in read_jsonl(run_dir / "candidates.jsonl")}
    rechecked = read_jsonl(Path(found["run_dir"]) / "accepted.jsonl")
    assert all(not stored[(row["unit_id"], row["attempt"])]["slots_kept"] for row in rechecked)
    items = [{"item_id": f"SNG{index:04d}", "set": "clean"} for index in range(60)]
    merged = write_derived(run_dir, tmp_path / "out", items, extra_runs=(Path(found["run_dir"]),))
    assert merged["rewrites"] == len(accepted)


def test_write_derived_leaves_out_audit_rejected_rewrites(tmp_path):
    run_dir = tmp_path / "run"
    rejected = next(iter(AUDIT_REJECTED))
    write_jsonl(run_dir / "accepted.jsonl", [{"unit_id": rejected}, {"unit_id": "u:SNG0001:pad"}])
    items = [{"item_id": f"SNG{index:04d}", "set": "clean"} for index in range(60)]
    found = write_derived(run_dir, tmp_path / "derived", items)
    assert found["rewrites"] == 1 and found["audit_rejected"] == 1
    assert [row["unit_id"] for row in read_jsonl(tmp_path / "derived" / "rewrites.jsonl")] == ["u:SNG0001:pad"]


class RejectingNLI:
    def entailment_probs(self, pairs):
        return [0.1 for _ in pairs]


def test_pad_is_checked_by_regex_only(tmp_path):
    derived = tmp_path / "derived"
    write_jsonl(derived / "items.jsonl", ITEMS)
    write_jsonl(derived / "probe.jsonl", PROBE)
    write_json(derived / "semantic_vocab.json", VOCAB)
    summary = run_rewrites(repo_root(), derived=derived, results_root=tmp_path / "results", transport=FakeClaude(),
                           environ=ENV, sleeper=lambda seconds: None, nli=RejectingNLI(), now=FIXED, workers=1)
    assert {kind for kind, found in summary["acceptance"].items() if found["accepted"]} == {"pad"}
