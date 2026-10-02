"""Toy emission rules with known likelihoods. These are not claims about real users."""

from __future__ import annotations

DATASET_VERSION = "toy_likelihood"
HYPOTHESIS_VERSION = "toy_v1"
PROMPT_VERSION = "toy_v1"
EVALUATION_RULE_VERSION = "toy_likelihood_v1"
REVIEWER = "assistant wording check"
REVIEWED_AT = "2026-10-02"

HYPOTHESES = ("toy_create", "toy_disable", "toy_query")

RULES = {
    "by_intention": (
        "A toy system has exactly one hidden intention, which is Create, Disable, or Query. "
        "It then emits exactly one of three sentences. These probabilities are defined rules of the toy system, "
        "not estimates from real users. "
        "If the intention is Create, it emits 'Please change my alarm' with probability 0.60, "
        "'Please stop my alarm' with probability 0.10, and 'Please check my alarm' with probability 0.30. "
        "If the intention is Disable, it emits 'Please change my alarm' with probability 0.30, "
        "'Please stop my alarm' with probability 0.60, and 'Please check my alarm' with probability 0.10. "
        "If the intention is Query, it emits 'Please change my alarm' with probability 0.10, "
        "'Please stop my alarm' with probability 0.30, and 'Please check my alarm' with probability 0.60."
    ),
    "by_sentence": (
        "A toy system emits exactly one sentence, and the hidden intention is exactly one of Create, Disable, or Query. "
        "These are defined rules, not frequencies from real users. "
        "The sentence 'Please change my alarm' is emitted with probability 0.60 when the intention is Create, "
        "0.30 when the intention is Disable, and 0.10 when the intention is Query. "
        "The sentence 'Please stop my alarm' is emitted with probability 0.10 when the intention is Create, "
        "0.60 when the intention is Disable, and 0.30 when the intention is Query. "
        "The sentence 'Please check my alarm' is emitted with probability 0.30 when the intention is Create, "
        "0.10 when the intention is Disable, and 0.60 when the intention is Query."
    ),
    "intention_reversed": (
        "These rules define a toy system. They are not claims about real users. "
        "The system has one hidden intention and emits one sentence. "
        "When the intention is Query, the emission probabilities are 0.10 for 'Please change my alarm', "
        "0.30 for 'Please stop my alarm', and 0.60 for 'Please check my alarm'. "
        "When the intention is Disable, the emission probabilities are 0.30 for 'Please change my alarm', "
        "0.60 for 'Please stop my alarm', and 0.10 for 'Please check my alarm'. "
        "When the intention is Create, the emission probabilities are 0.60 for 'Please change my alarm', "
        "0.10 for 'Please stop my alarm', and 0.30 for 'Please check my alarm'."
    ),
}

SENTENCES = {
    "change": "Please change my alarm",
    "stop": "Please stop my alarm",
    "check": "Please check my alarm",
}

TARGETS = {
    "change": {"toy_create": 0.60, "toy_disable": 0.30, "toy_query": 0.10},
    "stop": {"toy_create": 0.10, "toy_disable": 0.60, "toy_query": 0.30},
    "check": {"toy_create": 0.30, "toy_disable": 0.10, "toy_query": 0.60},
}


def authored_items() -> list[dict]:
    rows = []
    for rules_id, rules in RULES.items():
        for sentence_id, sentence in SENTENCES.items():
            rows.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "hypothesis_version": HYPOTHESIS_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "evaluation_rule_version": EVALUATION_RULE_VERSION,
                    "example_id": f"toy-{rules_id}-{sentence_id}",
                    "pair_id": rules_id,
                    "family_id": rules_id,
                    "family": "toy_likelihood",
                    "category": "toy_likelihood",
                    "split": "all",
                    "role": sentence_id,
                    "side": "emission",
                    "variant": rules_id,
                    "utt": f"Rules:\n{rules}\n\nEmitted sentence:\n{sentence}",
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "rules_id": rules_id,
                    "evaluation": "known_likelihood",
                    "support": None,
                    "plan_hypotheses": None,
                    "targets": dict(TARGETS[sentence_id]),
                    "notes": "The targets are the toy system's emission probabilities, not estimated user frequencies.",
                }
            )
    _validate(rows)
    return rows


def _validate(rows: list[dict]) -> None:
    if len(rows) != 9:
        raise ValueError("expected 9 toy items")
    if len({row["utt"] for row in rows}) != 9:
        raise ValueError("toy inputs are not unique")
    for sentence_id, targets in TARGETS.items():
        if set(targets) != set(HYPOTHESES):
            raise ValueError(sentence_id)
        if abs(sum(targets.values()) - 1.0) > 1e-9:
            raise ValueError(sentence_id)
    for intention in HYPOTHESES:
        column = [TARGETS[sentence_id][intention] for sentence_id in SENTENCES]
        if abs(sum(column) - 1.0) > 1e-9:
            raise ValueError(intention)


def review_manifest() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "instructions": (
            "The probabilities are rules of a toy generator. "
            "They are not frequencies from real users. "
            "Each row is approved when the sentence and the target table match."
        ),
        "items": [
            {
                "example_id": row["example_id"],
                "family": row["family"],
                "role": row["role"],
                "utt": row["utt"],
                "evaluation": row["evaluation"],
                "label_changed": False,
                "approved": True,
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            }
            for row in authored_items()
        ],
    }
