import json
import threading
from datetime import datetime, timezone

from robust_likelihood.clients import TransportResult
from robust_likelihood.multiwoz import load_config
from robust_likelihood.multiwoz.semantic import AREA_SYNONYMS, PRICE_SYNONYMS
from robust_likelihood.multiwoz.verbalcheck import (
    api_queries,
    exact_prompt,
    jev_probe,
    meaning_prompt,
    plan_queries,
    run_api,
    run_jev_rewrites,
    slot_description,
)
from robust_likelihood.storage import read_jsonl, repo_root, write_json, write_jsonl

VOCAB = {
    "food": ["indian", "italian", "north american"],
    "area": ["centre", "east", "north", "south", "west"],
    "pricerange": ["cheap", "expensive", "moderate"],
    "area_synonyms": AREA_SYNONYMS,
    "price_synonyms": PRICE_SYNONYMS,
}
ENV = {"ANTHROPIC_API_KEY": "test-claude", "OPENAI_API_KEY": "test-gpt"}
FIXED = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


def item(item_id, utterance, status, values, goal):
    return {"item_id": item_id, "set": "clean", "utterance": utterance, "slot_status": status, "turn0_values": values,
            "goal": goal}


ITEMS = [
    item("A", "A cheap italian place in the north.", {"food": "named", "area": "named", "pricerange": "named"},
         {"food": ["italian"], "area": ["north"], "pricerange": ["cheap"]},
         {"food": "italian", "area": "north", "pricerange": "cheap"}),
    item("B", "Any food is fine, somewhere central.", {"food": "dontcare", "area": "named", "pricerange": "omitted"},
         {"food": ["dontcare"], "area": ["centre"], "pricerange": []},
         {"food": "indian", "area": "centre", "pricerange": "expensive"}),
]
PROBE = [
    {"item_id": "A", "set": "clean", "hyp_key": "G|impl", "regime": "impl", "utterance": ITEMS[0]["utterance"],
     "text": "The user wants to find a cheap restaurant serving italian food in the north of town."},
    {"item_id": "A", "set": "clean", "hyp_key": "broad|impl", "regime": "impl", "utterance": ITEMS[0]["utterance"],
     "text": "The user wants to find a restaurant."},
    {"item_id": "B", "set": "clean", "hyp_key": "G|impl", "regime": "impl", "utterance": ITEMS[1]["utterance"],
     "text": "The user wants to find an expensive restaurant serving indian food in the centre of town."},
    {"item_id": "B", "set": "clean", "hyp_key": "broad|impl", "regime": "impl", "utterance": ITEMS[1]["utterance"],
     "text": "The user wants to find a restaurant."},
]
REWRITES = [
    {"unit_id": "u:A:para1", "side": "utterance", "kind": "para1", "item_id": "A", "original": ITEMS[0]["utterance"],
     "rewrite": "Something cheap and italian up north."},
    {"unit_id": "u:A:zh", "side": "utterance", "kind": "zh", "item_id": "A", "original": ITEMS[0]["utterance"],
     "rewrite": "北边便宜的意大利餐厅。"},
    {"unit_id": "h:b:hpara1", "side": "hypothesis", "kind": "hpara1", "item_id": None,
     "original": "The user wants to find a restaurant.", "rewrite": "The user is looking for a restaurant."},
    {"unit_id": "h:b:hzh", "side": "hypothesis", "kind": "hzh", "item_id": None,
     "original": "The user wants to find a restaurant.", "rewrite": "用户想找一家餐厅。"},
]
SAMPLES = [{"text": PROBE[0]["text"], "sample": index, "reply": reply}
           for index, reply in enumerate(["one", "two", "one", "three"])] + [
          {"text": PROBE[2]["text"], "sample": 0, "reply": "four"}]


def plan(norm_items=2, norm_samples=2):
    return plan_queries(PROBE, REWRITES, ITEMS, ["A", "B"], VOCAB, SAMPLES, norm_items=norm_items,
                        norm_samples=norm_samples, seed=7)


def test_prompts_follow_the_plan_without_a_doubled_period():
    exact = exact_prompt("The user wants to find a restaurant.", "Hi there")
    assert exact == ("A user of a restaurant booking assistant has the following goal: The user wants to find a "
                     "restaurant. What is the probability that this user's first message to the assistant is "
                     "exactly: \"Hi there\"? Answer with a single probability between 0 and 1 and nothing else.")
    assert "餐厅。 What is" in exact_prompt("用户想找一家餐厅。", "x")
    assert "goal: The user wants food. What" in exact_prompt("The user wants food", "x")
    assert meaning_prompt("H.", "a restaurant").endswith(
        "asks for exactly the following and nothing else: a restaurant? Answer with a single probability between "
        "0 and 1 and nothing else.")


def test_slot_description_uses_the_template_words_and_open_slots():
    from robust_likelihood.multiwoz.semantic import annotated_tuple

    assert slot_description(annotated_tuple(ITEMS[0], VOCAB), ITEMS[0]["goal"]) == \
        "a cheap restaurant serving italian food in the north of town"
    assert slot_description(annotated_tuple(ITEMS[1], VOCAB), ITEMS[1]["goal"]) == \
        "a restaurant in the centre of town, with no preference about the type of food"
    several = {"food": ["american", "north american"], "area": ["none"], "pricerange": ["none"]}
    assert slot_description(several, {"food": "north american"}) == "a restaurant serving north american food"
    assert slot_description({"food": ["none"], "area": ["none"], "pricerange": ["none"]}, {}) == "a restaurant"


def test_plan_covers_originals_rewrites_and_distinct_norm_samples():
    queries = plan()
    assert len({query["query_id"] for query in queries}) == len(queries)
    exact = [query for query in queries if query["framing"] == "exact"]
    meaning = [query for query in queries if query["framing"] == "meaning"]
    # 4 originals, 2 hypothesis rewrites x 2 broad rows, 2 utterance rewrites x 2 rows of item A.
    assert len(meaning) == 4 + 4 + 4
    assert len(exact) == len(meaning) + 3  # plus norm: A has 3 distinct samples (2 taken), B has 1
    norm = [query for query in exact if query["source"] == "norm"]
    assert [(query["item_id"], query["utterance"]) for query in norm] == [("A", "one"), ("A", "two"), ("B", "four")]
    # An utterance rewrite keeps U's annotation, so its meaning prompt is the original's.
    by_source = {(query["source"], query["hyp_key"]): query for query in meaning if query["item_id"] == "A"}
    assert by_source[("para1", "G|impl")]["prompt"] == by_source[("original", "G|impl")]["prompt"]
    assert by_source[("hpara1", "broad|impl")]["prompt"] != by_source[("original", "broad|impl")]["prompt"]


def test_api_and_jev_take_their_sources():
    queries = plan()
    api = api_queries(queries, None, 7)
    assert {query["source"] for query in api} == {"original", "para1", "hpara1", "norm"}
    assert all(query["framing"] == "exact" for query in api)
    assert len(api_queries(queries, 3, 7)) == 3
    rows = jev_probe(queries, PROBE)
    assert {row["hyp_key"] for row in rows} == {"G|impl@para1", "broad|impl@para1", "G|impl@zh", "broad|impl@zh",
                                                 "broad|impl@hpara1", "broad|impl@hzh"}
    hzh = next(row for row in rows if row["hyp_key"] == "broad|impl@hzh" and row["item_id"] == "B")
    assert hzh["text"] == "用户想找一家餐厅。" and hzh["utterance"] == ITEMS[1]["utterance"]


def write_inputs(tmp_path):
    derived = tmp_path / "derived"
    write_jsonl(derived / "items.jsonl", ITEMS)
    write_jsonl(derived / "probe.jsonl", PROBE)
    write_jsonl(derived / "rewrites.jsonl", REWRITES)
    write_json(derived / "semantic_vocab.json", VOCAB)
    (derived / "robustness_subset.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")
    phase8 = tmp_path / "phase8"
    write_jsonl(phase8 / "lm_samples.jsonl", SAMPLES)
    return derived, phase8


class FakeChat:
    def __init__(self):
        self.bodies = []
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.bodies.append(body)
        if "anthropic" in url:
            if "temperature" in body:
                return TransportResult(400, {"error": {"message": "`temperature` is not supported"}}, None, False)
            return TransportResult(200, {"model": body["model"], "stop_reason": "end_turn",
                                         "content": [{"type": "text", "text": "0.001"}],
                                         "usage": {"input_tokens": 90, "output_tokens": 40}}, None, False)
        return TransportResult(200, {"model": body["model"], "usage": {"prompt_tokens": 80, "completion_tokens": 3},
                                     "choices": [{"message": {"content": "1e-6"}, "finish_reason": "stop"}]},
                               None, False)


def test_api_run_parses_caches_and_drops_a_rejected_temperature(tmp_path):
    derived, phase8 = write_inputs(tmp_path)
    transport = FakeChat()
    first = run_api(repo_root(), ["claude", "gpt"], phase8, derived=derived, results_root=tmp_path / "results",
                    transport=transport, environ=ENV, sleeper=lambda seconds: None, now=FIXED, workers=2)
    planned = len(api_queries(plan(20, 20), None, 0))
    assert first["providers"]["claude"]["calls"] == planned and first["providers"]["gpt"]["calls"] == planned
    assert first["providers"]["claude"]["temperature_sent"] is None
    assert first["providers"]["gpt"]["temperature_sent"] == 0.0
    rows = read_jsonl(tmp_path / "results" / f"{FIXED.strftime('%Y%m%dT%H%M%SZ')}-verbal-api" / "scores.jsonl")
    assert {row["p"] for row in rows if row["model"] == "claude"} == {0.001}
    assert {row["p"] for row in rows if row["model"] == "gpt"} == {1e-6}
    calls = len(transport.bodies)
    again = run_api(repo_root(), ["claude", "gpt"], phase8, derived=derived, results_root=tmp_path / "results",
                    transport=transport, environ=ENV, sleeper=lambda seconds: None, now=FIXED, label="again")
    assert len(transport.bodies) == calls and again["http_calls"] == 0


def test_jev_dry_run_plans_both_wordings_for_each_rewrite(tmp_path):
    derived, _ = write_inputs(tmp_path)
    summary = run_jev_rewrites(repo_root(), load_config(repo_root()), wording="both", dry_run=True, derived=derived,
                               results_root=tmp_path / "results")
    assert summary["api_calls"] == 0
    assert summary["planned_calls"] == 2 * len(read_jsonl(tmp_path / "results" / "claim1" / "jev_input" / "probe.jsonl"))
    assert summary["items"] == 2
