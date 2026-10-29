from robust_likelihood.multiwoz.claim1_lm import plan_pairs

PROBE = [
    {"item_id": "A", "set": "clean", "hyp_key": "G|impl", "text": "G text", "utterance": "u A"},
    {"item_id": "A", "set": "clean", "hyp_key": "broad|impl", "text": "broad text", "utterance": "u A"},
    {"item_id": "B", "set": "clean", "hyp_key": "G|impl", "text": "G text B", "utterance": "u B"},
    {"item_id": "C", "set": "fail_info", "hyp_key": "G|impl", "text": "G text", "utterance": "u C"},
]
REWRITES = [
    {"unit_id": "u:A:para1", "side": "utterance", "kind": "para1", "item_id": "A", "original": "u A",
     "rewrite": "u A again"},
    {"unit_id": "h:1:hpara1", "side": "hypothesis", "kind": "hpara1", "original": "G text", "rewrite": "G text again"},
    {"unit_id": "h:2:hzh", "side": "hypothesis", "kind": "hzh", "original": "G text B", "rewrite": "G 文本"},
]


def test_plan_pairs_covers_each_part():
    samples = [{"text": "G text", "sample": 0, "reply": "a sample"}]
    plans = plan_pairs(PROBE, REWRITES, ["A"], "broad text", samples)
    assert [(row["source"], row["utterance"]) for row in plans["center"]] == [
        ("utterance", "u A"), ("utterance", "u B"), ("utterance", "u C"), ("sample", "a sample")]
    assert all(row["text"] == "broad text" for row in plans["center"])
    assert [(row["hyp_key"], row["text"], row["utterance"]) for row in plans["urewrite"]] == [
        ("G|impl", "G text", "u A again"), ("broad|impl", "broad text", "u A again")]
    # Hypothesis rewrites score clean rows only: item C is fail_info.
    assert [(row["item_id"], row["text"], row["utterance"]) for row in plans["hrewrite"]] == [
        ("A", "G text again", "u A"), ("B", "G 文本", "u B")]
    # Sampling under rewritten texts is limited to the robustness subset (item A).
    assert plans["hsample"] == [{"unit_id": "h:1:hpara1", "text": "G text again"}]
