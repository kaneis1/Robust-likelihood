"""Reviewed synthetic_alarm_v2 items.

Prompt template v2 is unchanged. These rows use hypothesis definitions synthetic_v2.
They are not the 22-item synthetic_v1 set, and --dataset synthetic does not load them.
"""

from __future__ import annotations

ATOMIC = (
    "intend_create",
    "intend_enable",
    "intend_disable",
    "intend_delete",
    "intend_query",
)

PLAN_FIELDS = ("action", "object", "arguments", "order")

DATASET_VERSION = "synthetic_alarm_v2"
HYPOTHESIS_VERSION = "synthetic_v2"
PROMPT_VERSION = "v2"
EVALUATION_RULE_VERSION = "synthetic_eval_v2"
REVIEWER = "assistant wording check"
REVIEWED_AT = "2026-10-02"

DELETE_THEN_CREATE = "Currently requests deleting the morning alarm, then creating a new alarm for 9:00."
CREATE_THEN_DELETE = "Currently requests creating a new alarm for 9:00, then deleting the morning alarm."
ONLY_DELETE = "Currently requests only deleting the morning alarm."
ONLY_CREATE = "Currently requests only creating a new alarm for 9:00."


def _atomic(high: str) -> dict[str, str]:
    if high not in ATOMIC:
        raise ValueError(high)
    return {hypothesis: "supported" if hypothesis == high else "rejected" for hypothesis in ATOMIC}


def _plan(supported_id: str) -> list[dict]:
    texts = {
        "plan_delete_then_create": DELETE_THEN_CREATE,
        "plan_create_then_delete": CREATE_THEN_DELETE,
        "plan_only_delete": ONLY_DELETE,
        "plan_only_create": ONLY_CREATE,
    }
    contents = {
        "plan_delete_then_create": {
            "action": ["delete", "create"],
            "object": ["the morning alarm", "a new alarm"],
            "arguments": ["entirely", "for 9:00"],
            "order": ["delete the morning alarm", "create a new alarm for 9:00"],
        },
        "plan_create_then_delete": {
            "action": ["create", "delete"],
            "object": ["a new alarm", "the morning alarm"],
            "arguments": ["for 9:00", "entirely"],
            "order": ["create a new alarm for 9:00", "delete the morning alarm"],
        },
        "plan_only_delete": {
            "action": ["delete"],
            "object": ["the morning alarm"],
            "arguments": ["entirely"],
            "order": ["delete the morning alarm"],
        },
        "plan_only_create": {
            "action": ["create"],
            "object": ["a new alarm"],
            "arguments": ["for 9:00"],
            "order": ["create a new alarm for 9:00"],
        },
    }
    rows = []
    for hypothesis_id, text in texts.items():
        content = contents[hypothesis_id]
        if set(content) != set(PLAN_FIELDS):
            raise ValueError(hypothesis_id)
        rows.append(
            {
                "id": hypothesis_id,
                "text": text,
                "expected": "supported" if hypothesis_id == supported_id else "rejected",
                **content,
            }
        )
    return rows


def _row(
    example_id: str,
    pair_id: str,
    family: str,
    role: str,
    utt: str,
    *,
    evaluation: str,
    support: dict[str, str] | None = None,
    plan_hypotheses: list[dict] | None = None,
    notes: str,
) -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "example_id": example_id,
        "pair_id": pair_id,
        "family": family,
        "role": role,
        "utt": utt,
        "evaluation": evaluation,
        "support": support,
        "plan_hypotheses": plan_hypotheses,
        "notes": notes,
    }


def authored_items() -> list[dict]:
    create = _atomic("intend_create")
    enable = _atomic("intend_enable")
    disable = _atomic("intend_disable")
    delete = _atomic("intend_delete")
    query = _atomic("intend_query")
    rows = [
        _row(
            "v2-op-create",
            "v2-create",
            "explicit_operation",
            "source",
            "Create a new alarm.",
            evaluation="scored",
            support=create,
            notes="Explicit create. The other four atomic hypotheses are rejected.",
        ),
        _row(
            "v2-op-create-polite",
            "v2-create",
            "explicit_operation",
            "polite",
            "Could you create a new alarm?",
            evaluation="scored",
            support=create,
            notes="Grammatically a question. It currently requests create, so intend_query is rejected.",
        ),
        _row(
            "v2-op-enable",
            "v2-enable",
            "explicit_operation",
            "item",
            "Enable my existing alarm.",
            evaluation="scored",
            support=enable,
            notes="Explicit enable of an existing alarm.",
        ),
        _row(
            "v2-op-disable",
            "v2-disable-mention",
            "explicit_operation",
            "source",
            "Disable my alarm, but keep it saved.",
            evaluation="scored",
            support=disable,
            notes="Explicit disable. The saved alarm is kept, so delete is rejected.",
        ),
        _row(
            "v2-op-delete",
            "v2-delete-mention",
            "explicit_operation",
            "source",
            "Delete my alarm entirely.",
            evaluation="scored",
            support=delete,
            notes="Explicit delete. Disable is rejected because the alarm is not kept.",
        ),
        _row(
            "v2-op-turn-off",
            "v2-turn-off",
            "explicit_operation",
            "item",
            "Turn off my alarm.",
            evaluation="scored",
            support=disable,
            notes="Task convention maps 'turn off' to disable. This is a convention, not a claim that ordinary language has no other reading.",
        ),
        _row(
            "v2-op-delete-polite",
            "v2-delete-polite",
            "explicit_operation",
            "item",
            "Could you delete my alarm entirely?",
            evaluation="scored",
            support=delete,
            notes="Polite question that currently requests deletion. intend_query is rejected.",
        ),
        _row(
            "v2-op-cancel-explicit-delete",
            "v2-cancel-explicit-delete",
            "explicit_operation",
            "item",
            "Cancel it by deleting the saved alarm.",
            evaluation="scored",
            support=delete,
            notes="Cancel is specified as deletion, so this row is a scored delete. Bare cancel is a different row.",
        ),
        _row(
            "v2-amb-cancel",
            "v2-amb-cancel",
            "ambiguity_probe",
            "item",
            "Cancel my alarm.",
            evaluation="not_scored",
            support=None,
            notes="Bare cancel stays ambiguous. Scores may be stored. They are not a pass or a failure.",
        ),
        _row(
            "v2-quote-delete",
            "v2-delete-mention",
            "quoted_command",
            "mention",
            "Did I say 'delete my alarm entirely'?",
            evaluation="scored",
            support=query,
            notes="The quoted sentence matches the explicit delete request. The user is asking whether it was said, so only intend_query is supported.",
        ),
        _row(
            "v2-quote-disable",
            "v2-disable-mention",
            "quoted_command",
            "mention",
            "Did I say 'disable my alarm, but keep it saved'?",
            evaluation="scored",
            support=query,
            notes="The quoted sentence matches the explicit disable request. The quoted operation is not being issued.",
        ),
        _row(
            "v2-plan-delete-then-create",
            "v2-plan-morning-nine",
            "ordered_plan",
            "forward",
            "Delete the morning alarm, and after that create one for nine.",
            evaluation="scored_plan",
            plan_hypotheses=_plan("plan_delete_then_create"),
            notes="Primary score is the complete plan. Atomic action scores are a secondary diagnostic and do not decide this row.",
        ),
        _row(
            "v2-plan-create-then-delete",
            "v2-plan-morning-nine",
            "ordered_plan",
            "reversed",
            "Create one for nine, and after that delete the morning alarm.",
            evaluation="scored_plan",
            plan_hypotheses=_plan("plan_create_then_delete"),
            notes="Same two operations in the other order. A preference for one sequence fails this pair.",
        ),
    ]
    _validate(rows)
    return rows


def _validate(rows: list[dict]) -> None:
    if len(rows) != 13:
        raise ValueError(f"expected 13 synthetic_v2 items, found {len(rows)}")
    forward = next(row for row in rows if row["example_id"] == "v2-plan-delete-then-create")
    reverse = next(row for row in rows if row["example_id"] == "v2-plan-create-then-delete")
    for row in rows:
        if row["evaluation"] == "not_scored":
            if row["support"] is not None or row["plan_hypotheses"] is not None:
                raise ValueError(f"{row['example_id']} is an unscored probe with a support map")
            continue
        if row["evaluation"] == "scored":
            if set(row["support"]) != set(ATOMIC):
                raise ValueError(row["example_id"])
            if row["plan_hypotheses"] is not None:
                raise ValueError(row["example_id"])
            continue
        if row["evaluation"] != "scored_plan":
            raise ValueError(row["evaluation"])
        supported = [item for item in row["plan_hypotheses"] if item["expected"] == "supported"]
        if len(supported) != 1:
            raise ValueError(row["example_id"])
        for item in row["plan_hypotheses"]:
            if item["id"].startswith("plan_only_") and item["expected"] != "rejected":
                raise ValueError(item["id"])
            if "only" not in item["text"] and item["id"].startswith("plan_only_"):
                raise ValueError(item["text"])
    forward_supported = next(item["id"] for item in forward["plan_hypotheses"] if item["expected"] == "supported")
    reverse_supported = next(item["id"] for item in reverse["plan_hypotheses"] if item["expected"] == "supported")
    if forward_supported == reverse_supported:
        raise ValueError("reversed plan supports the same complete hypothesis")
    if {forward_supported, reverse_supported} != {"plan_delete_then_create", "plan_create_then_delete"}:
        raise ValueError("complete plans did not swap")


def review_manifest() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "instructions": (
            "Approved is JSON true only after the utterance is checked against the support map "
            "or the plan hypotheses. Bare cancel is not scored. "
            "These rows are not the synthetic_v1 set."
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
