"""Forty alarm families. Prompt v2 and the synthetic_v2 definitions stay frozen.

Each family has a canonical sentence, a paraphrase, and a minor typo.
Ambiguity families also have a clarified trio. Analysis is by family.
Two families in each category are the development split. The rest are held out.
"""

from __future__ import annotations

from robust_likelihood.synthetic_v2_items import (
    ATOMIC,
    HYPOTHESIS_VERSION,
    PLAN_FIELDS,
    PROMPT_VERSION,
    REVIEWED_AT,
    REVIEWER,
    _atomic,
)

DATASET_VERSION = "synthetic_alarm_families"
EVALUATION_RULE_VERSION = "synthetic_eval_v3"
CATEGORIES = (
    "requests_versus_mentions",
    "negation_and_correction",
    "action_order",
    "ambiguity_and_clarification",
)


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            cost = 0 if char == other else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _phrase(kind: str, obj: str, argument: str) -> tuple[str, str, dict]:
    if kind == "delete":
        text = f"deleting {obj} entirely"
        short = f"delete {obj} entirely"
        content = {"action": ["delete"], "object": [obj], "arguments": ["entirely"], "order": [short]}
    elif kind == "create":
        text = f"creating a new alarm for {argument}"
        short = f"create a new alarm for {argument}"
        content = {"action": ["create"], "object": ["a new alarm"], "arguments": [f"for {argument}"], "order": [short]}
    elif kind == "enable":
        text = f"enabling {obj}"
        short = f"enable {obj}"
        content = {"action": ["enable"], "object": [obj], "arguments": ["turn on"], "order": [short]}
    elif kind == "disable":
        text = f"disabling {obj} while keeping it saved"
        short = f"disable {obj} while keeping it saved"
        content = {"action": ["disable"], "object": [obj], "arguments": ["keep saved"], "order": [short]}
    elif kind == "query_enabled":
        text = f"asking whether {obj} is enabled"
        short = f"ask whether {obj} is enabled"
        content = {"action": ["query"], "object": [obj], "arguments": ["is enabled"], "order": [short]}
    elif kind == "query_exists":
        text = f"asking whether {obj} exists"
        short = f"ask whether {obj} exists"
        content = {"action": ["query"], "object": [obj], "arguments": ["exists"], "order": [short]}
    else:
        raise ValueError(kind)
    return text, short, content


def _plans(first: dict, second: dict) -> list[dict]:
    first_text, _, first_content = _phrase(first["kind"], first["object"], first.get("argument", ""))
    second_text, _, second_content = _phrase(second["kind"], second["object"], second.get("argument", ""))
    bundles = {
        "plan_temporal": (
            f"Currently requests {first_text}, then {second_text}.",
            "supported",
            [first_content, second_content],
        ),
        "plan_reversed": (
            f"Currently requests {second_text}, then {first_text}.",
            "rejected",
            [second_content, first_content],
        ),
        "plan_only_first": (
            f"Currently requests only {first_text}.",
            "rejected",
            [first_content],
        ),
        "plan_only_second": (
            f"Currently requests only {second_text}.",
            "rejected",
            [second_content],
        ),
    }
    rows = []
    for hypothesis_id, (text, expected, parts) in bundles.items():
        content = {
            "action": [value for part in parts for value in part["action"]],
            "object": [value for part in parts for value in part["object"]],
            "arguments": [value for part in parts for value in part["arguments"]],
            "order": [value for part in parts for value in part["order"]],
        }
        if set(content) != set(PLAN_FIELDS):
            raise ValueError(hypothesis_id)
        rows.append({"id": hypothesis_id, "text": text, "expected": expected, **content})
    return rows


def _item(
    example_id: str,
    family_id: str,
    category: str,
    split: str,
    role: str,
    utt: str,
    *,
    evaluation: str,
    side: str,
    support: dict[str, str] | None,
    plan_hypotheses: list[dict] | None,
    focus_hypothesis: str | None,
    contrast_hypothesis: str | None,
    surface_order_matches_temporal: bool | None,
    notes: str,
) -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "example_id": example_id,
        "pair_id": family_id,
        "family_id": family_id,
        "family": category,
        "category": category,
        "split": split,
        "role": role,
        "side": side,
        "variant": role.split("_")[-1],
        "utt": utt,
        "evaluation": evaluation,
        "support": support,
        "plan_hypotheses": plan_hypotheses,
        "focus_hypothesis": focus_hypothesis,
        "contrast_hypothesis": contrast_hypothesis,
        "surface_order_matches_temporal": surface_order_matches_temporal,
        "notes": notes,
    }


def _atomic_family(spec: dict) -> list[dict]:
    support = _atomic(spec["support"])
    rows = []
    for variant, utt in spec["versions"].items():
        rows.append(
            _item(
                f"{spec['id']}__{variant}",
                spec["id"],
                spec["category"],
                spec["split"],
                variant,
                utt,
                evaluation="scored",
                side="request",
                support=support,
                plan_hypotheses=None,
                focus_hypothesis=spec["support"],
                contrast_hypothesis=spec["contrast"],
                surface_order_matches_temporal=None,
                notes=spec["notes"],
            )
        )
    return rows


def _order_family(spec: dict) -> list[dict]:
    plans = _plans(spec["first"], spec["second"])
    rows = []
    for variant, utt in spec["versions"].items():
        rows.append(
            _item(
                f"{spec['id']}__{variant}",
                spec["id"],
                "action_order",
                spec["split"],
                variant,
                utt,
                evaluation="scored_plan",
                side="request",
                support=None,
                plan_hypotheses=plans,
                focus_hypothesis="plan_temporal",
                contrast_hypothesis="plan_reversed",
                surface_order_matches_temporal=spec["surface_order_matches_temporal"],
                notes=spec["notes"],
            )
        )
    return rows


def _ambiguity_family(spec: dict) -> list[dict]:
    support = _atomic(spec["clarified_support"])
    rows = []
    for variant, utt in spec["ambiguous"].items():
        rows.append(
            _item(
                f"{spec['id']}__ambiguous__{variant}",
                spec["id"],
                "ambiguity_and_clarification",
                spec["split"],
                f"ambiguous_{variant}",
                utt,
                evaluation="not_scored",
                side="ambiguous",
                support=None,
                plan_hypotheses=None,
                focus_hypothesis=spec["clarified_support"],
                contrast_hypothesis=None,
                surface_order_matches_temporal=None,
                notes=spec["notes"],
            )
        )
    for variant, utt in spec["clarified"].items():
        rows.append(
            _item(
                f"{spec['id']}__clarified__{variant}",
                spec["id"],
                "ambiguity_and_clarification",
                spec["split"],
                f"clarified_{variant}",
                utt,
                evaluation="scored",
                side="clarified",
                support=support,
                plan_hypotheses=None,
                focus_hypothesis=spec["clarified_support"],
                contrast_hypothesis=None,
                surface_order_matches_temporal=None,
                notes=spec["notes"],
            )
        )
    return rows


def _mention_specs() -> list[dict]:
    category = "requests_versus_mentions"
    notes = "The current request asks about an earlier instruction. The named action is mentioned, not requested."
    return [
        {
            "id": "mention-01",
            "category": category,
            "split": "dev",
            "support": "intend_query",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "I remember saying 'delete my alarm.' Can you tell me when I said that?",
                "paraphrase": "When did I say 'delete my alarm'?",
                "typo": "I remember saying 'delete my alram.' Can you tell me when I said that?",
            },
        },
        {
            "id": "mention-02",
            "category": category,
            "split": "dev",
            "support": "intend_query",
            "contrast": "intend_disable",
            "notes": notes,
            "versions": {
                "canonical": "Did I tell you 'disable my alarm, but keep it saved'?",
                "paraphrase": "Was it me who said to disable the alarm and keep it saved?",
                "typo": "Did I tell you 'disabel my alarm, but keep it saved'?",
            },
        },
        {
            "id": "mention-03",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_create",
            "notes": notes,
            "versions": {
                "canonical": "I think I asked you to create a new alarm for seven. When was that?",
                "paraphrase": "When did I ask for a new alarm for seven?",
                "typo": "I think I asked you to create a new alram for seven. When was that?",
            },
        },
        {
            "id": "mention-04",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_enable",
            "notes": notes,
            "versions": {
                "canonical": "Please check whether I already told you to enable the evening alarm.",
                "paraphrase": "Did I previously ask you to turn the evening alarm on?",
                "typo": "Please check whether I already told you to enable the evning alarm.",
            },
        },
        {
            "id": "mention-05",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_disable",
            "notes": notes,
            "versions": {
                "canonical": "In my last message I wrote 'turn off my alarm.' What time did I send that?",
                "paraphrase": "When did I write the message that said to turn off my alarm?",
                "typo": "In my last message I wrote 'turn off my alram.' What time did I send that?",
            },
        },
        {
            "id": "mention-06",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "Can you find the earlier instruction where I said to delete the noon alarm entirely?",
                "paraphrase": "Look up when I said 'delete the noon alarm entirely.'",
                "typo": "Can you find the earlier instrution where I said to delete the noon alarm entirely?",
            },
        },
        {
            "id": "mention-07",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_create",
            "notes": notes,
            "versions": {
                "canonical": "Yesterday I said 'create a weekday alarm.' Can you tell me the time I said it?",
                "paraphrase": "What time yesterday did I ask you to create a weekday alarm?",
                "typo": "Yesterday I said 'create a weekdy alarm.' Can you tell me the time I said it?",
            },
        },
        {
            "id": "mention-08",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_enable",
            "notes": notes,
            "versions": {
                "canonical": "Did I say 'enable my first scheduled alarm' this morning?",
                "paraphrase": "Was the line 'enable my first scheduled alarm' something I said this morning?",
                "typo": "Did I say 'enable my first schedled alarm' this morning?",
            },
        },
        {
            "id": "mention-09",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "When did I say 'cancel it by deleting the saved alarm'?",
                "paraphrase": "I want the time I asked you to cancel it by deleting the saved alarm.",
                "typo": "When did I say 'cancel it by deleting the saved alram'?",
            },
        },
        {
            "id": "mention-10",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_disable",
            "notes": notes,
            "versions": {
                "canonical": "Remind me whether I already said 'keep my backup alarm saved but stop it from ringing.'",
                "paraphrase": "Did I previously tell you to disable the backup alarm and keep it saved?",
                "typo": "Remind me whether I already said 'keep my bakup alarm saved but stop it from ringing.'",
            },
        },
    ]


def _negation_specs() -> list[dict]:
    category = "negation_and_correction"
    notes = "The affirmative request is supported. The operation the user cancels or corrects is rejected."
    return [
        {
            "id": "negation-01",
            "category": category,
            "split": "dev",
            "support": "intend_disable",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "Don't delete my alarm—just disable it and keep it saved.",
                "paraphrase": "Leave the alarm saved. I do not want it deleted; turn it off.",
                "typo": "Don't delte my alarm—just disable it and keep it saved.",
            },
        },
        {
            "id": "negation-02",
            "category": category,
            "split": "dev",
            "support": "intend_delete",
            "contrast": "intend_disable",
            "notes": notes,
            "versions": {
                "canonical": "Don't disable it. Delete the morning alarm entirely.",
                "paraphrase": "I do not want the morning alarm merely turned off. Remove it completely.",
                "typo": "Don't disabel it. Delete the morning alarm entirely.",
            },
        },
        {
            "id": "negation-03",
            "category": category,
            "split": "holdout",
            "support": "intend_enable",
            "contrast": "intend_create",
            "notes": notes,
            "versions": {
                "canonical": "Don't create a new one. Enable the evening alarm that is already there.",
                "paraphrase": "The evening alarm already exists, so turn it on instead of making a new alarm.",
                "typo": "Don't create a new one. Enable the evning alarm that is already there.",
            },
        },
        {
            "id": "negation-04",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_enable",
            "notes": notes,
            "versions": {
                "canonical": "Don't enable anything. Tell me whether the seven o'clock alarm is on.",
                "paraphrase": "I am not asking you to turn the seven o'clock alarm on. Is it enabled?",
                "typo": "Don't enable anything. Tell me whether the seven o'clock alram is on.",
            },
        },
        {
            "id": "negation-05",
            "category": category,
            "split": "holdout",
            "support": "intend_delete",
            "contrast": "intend_disable",
            "notes": notes,
            "versions": {
                "canonical": "Don't just turn it off. Delete the weekday alarm completely.",
                "paraphrase": "Turning the weekday alarm off is not enough. Delete it entirely.",
                "typo": "Don't just turn it off. Delete the weekdy alarm completely.",
            },
        },
        {
            "id": "negation-06",
            "category": category,
            "split": "holdout",
            "support": "intend_disable",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "Don't delete the noon alarm. Disable it and keep it saved instead.",
                "paraphrase": "Keep the noon alarm saved and turn it off. Do not delete it.",
                "typo": "Don't delte the noon alarm. Disable it and keep it saved instead.",
            },
        },
        {
            "id": "negation-07",
            "category": category,
            "split": "holdout",
            "support": "intend_create",
            "contrast": "intend_query",
            "notes": notes,
            "versions": {
                "canonical": "Don't tell me the status. Create a new alarm for 8:15.",
                "paraphrase": "Skip any question about what I already have. Make a new alarm for 8:15.",
                "typo": "Don't tell me the status. Create a new alram for 8:15.",
            },
        },
        {
            "id": "negation-08",
            "category": category,
            "split": "holdout",
            "support": "intend_disable",
            "contrast": "intend_enable",
            "notes": notes,
            "versions": {
                "canonical": "I said enable, but that was a mistake. Disable the first scheduled alarm and keep it saved.",
                "paraphrase": "Do not turn the first scheduled alarm on. Turn it off and leave it saved.",
                "typo": "I said enable, but that was a mistake. Disabel the first scheduled alarm and keep it saved.",
            },
        },
        {
            "id": "negation-09",
            "category": category,
            "split": "holdout",
            "support": "intend_disable",
            "contrast": "intend_delete",
            "notes": notes,
            "versions": {
                "canonical": "Don't cancel it by deleting the saved alarm. Keep it and turn it off.",
                "paraphrase": "Do not remove the saved weekend alarm. Turn it off and leave it stored.",
                "typo": "Don't cancel it by deleting the saved alram. Keep it and turn it off.",
            },
        },
        {
            "id": "negation-10",
            "category": category,
            "split": "holdout",
            "support": "intend_query",
            "contrast": "intend_create",
            "notes": notes,
            "versions": {
                "canonical": "Don't create a six o'clock alarm. I want to know whether one already exists.",
                "paraphrase": "I am not asking for a new six o'clock alarm. Does one already exist?",
                "typo": "Don't create a six o'clock alram. I want to know whether one already exists.",
            },
        },
    ]


def _order_specs() -> list[dict]:
    notes = "The supported plan follows the temporal order. The reversed plan and both single-action plans are rejected."
    return [
        {
            "id": "order-01",
            "split": "dev",
            "surface_order_matches_temporal": False,
            "first": {"kind": "delete", "object": "the morning alarm"},
            "second": {"kind": "create", "object": "a new alarm", "argument": "9:00"},
            "notes": notes,
            "versions": {
                "canonical": "Before creating the nine o'clock alarm, delete the morning alarm.",
                "paraphrase": "Delete the morning alarm first, and create the nine o'clock alarm after that.",
                "typo": "Before creating the nine o'clock alram, delete the morning alarm.",
            },
        },
        {
            "id": "order-02",
            "split": "dev",
            "surface_order_matches_temporal": True,
            "first": {"kind": "delete", "object": "the evening alarm"},
            "second": {"kind": "create", "object": "a new alarm", "argument": "10:00"},
            "notes": notes,
            "versions": {
                "canonical": "After you delete the evening alarm, create one for ten.",
                "paraphrase": "Delete the evening alarm, and afterwards create a new alarm for 10:00.",
                "typo": "After you delete the evening alram, create one for ten.",
            },
        },
        {
            "id": "order-03",
            "split": "holdout",
            "surface_order_matches_temporal": True,
            "first": {"kind": "enable", "object": "the weekday alarm"},
            "second": {"kind": "query_enabled", "object": "the weekday alarm"},
            "notes": notes,
            "versions": {
                "canonical": "First enable the weekday alarm, then tell me whether it is on.",
                "paraphrase": "Turn the weekday alarm on, and after that say whether it is enabled.",
                "typo": "First enable the weekdy alarm, then tell me whether it is on.",
            },
        },
        {
            "id": "order-04",
            "split": "holdout",
            "surface_order_matches_temporal": False,
            "first": {"kind": "disable", "object": "the seven o'clock alarm"},
            "second": {"kind": "create", "object": "a new alarm", "argument": "noon"},
            "notes": notes,
            "versions": {
                "canonical": "Create the noon alarm after the seven o'clock alarm has been disabled and kept.",
                "paraphrase": "Disable the seven o'clock alarm and keep it, and after that create the noon alarm.",
                "typo": "Create the noon alram after the seven o'clock alarm has been disabled and kept.",
            },
        },
        {
            "id": "order-05",
            "split": "holdout",
            "surface_order_matches_temporal": True,
            "first": {"kind": "delete", "object": "the backup alarm"},
            "second": {"kind": "enable", "object": "the first scheduled alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Delete the backup alarm entirely, then enable the first scheduled alarm.",
                "paraphrase": "Remove the backup alarm completely, and afterwards switch on the first scheduled alarm.",
                "typo": "Delete the backup alram entirely, then enable the first scheduled alarm.",
            },
        },
        {
            "id": "order-06",
            "split": "holdout",
            "surface_order_matches_temporal": False,
            "first": {"kind": "disable", "object": "the weekend alarm"},
            "second": {"kind": "query_exists", "object": "the weekend alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Before you ask whether the weekend alarm exists, disable it and keep it saved.",
                "paraphrase": "Disable the weekend alarm and keep it saved, and after that tell me whether it exists.",
                "typo": "Before you ask whether the weeknd alarm exists, disable it and keep it saved.",
            },
        },
        {
            "id": "order-07",
            "split": "holdout",
            "surface_order_matches_temporal": True,
            "first": {"kind": "query_enabled", "object": "the 6 a.m. alarm"},
            "second": {"kind": "delete", "object": "the 6 a.m. alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Tell me if the 6 a.m. alarm is enabled, and after that delete it entirely.",
                "paraphrase": "After you say whether the 6 a.m. alarm is enabled, delete it completely.",
                "typo": "Tell me if the 6 a.m. alram is enabled, and after that delete it entirely.",
            },
        },
        {
            "id": "order-08",
            "split": "holdout",
            "surface_order_matches_temporal": True,
            "first": {"kind": "enable", "object": "the ten o'clock alarm"},
            "second": {"kind": "delete", "object": "the afternoon alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Enable the ten o'clock alarm before deleting the afternoon alarm.",
                "paraphrase": "Turn the ten o'clock alarm on, and afterwards delete the afternoon alarm entirely.",
                "typo": "Enable the ten o'clock alram before deleting the afternoon alarm.",
            },
        },
        {
            "id": "order-09",
            "split": "holdout",
            "surface_order_matches_temporal": False,
            "first": {"kind": "create", "object": "a new alarm", "argument": "5:30"},
            "second": {"kind": "disable", "object": "the midnight alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Before turning off the midnight alarm and keeping it, create a new alarm for 5:30.",
                "paraphrase": "Create a new alarm for 5:30, and after that turn off the midnight alarm but keep it saved.",
                "typo": "Before turning off the midnight alram and keeping it, create a new alarm for 5:30.",
            },
        },
        {
            "id": "order-10",
            "split": "holdout",
            "surface_order_matches_temporal": False,
            "first": {"kind": "create", "object": "a new alarm", "argument": "11:00"},
            "second": {"kind": "delete", "object": "the early alarm"},
            "notes": notes,
            "versions": {
                "canonical": "Delete the early alarm after the late alarm for 11:00 has been created.",
                "paraphrase": "Once a new alarm for 11:00 has been created, delete the early alarm entirely.",
                "typo": "Delete the early alram after the late alarm for 11:00 has been created.",
            },
        },
    ]


def _ambiguity_specs() -> list[dict]:
    notes = "The ambiguous wording has no gold winner. The clarified wording names one operation."
    return [
        {
            "id": "ambig-01",
            "split": "dev",
            "clarified_support": "intend_disable",
            "notes": notes,
            "ambiguous": {
                "canonical": "Do something about my morning alarm.",
                "paraphrase": "My morning alarm needs something done.",
                "typo": "Do somthing about my morning alarm.",
            },
            "clarified": {
                "canonical": "Keep it saved, but stop it ringing.",
                "paraphrase": "Leave the morning alarm stored and stop it from going off.",
                "typo": "Keep it saved, but stop it ringng.",
            },
        },
        {
            "id": "ambig-02",
            "split": "dev",
            "clarified_support": "intend_query",
            "notes": notes,
            "ambiguous": {
                "canonical": "What about my evening alarm?",
                "paraphrase": "About that evening alarm.",
                "typo": "What about my evning alarm?",
            },
            "clarified": {
                "canonical": "Is the evening alarm enabled?",
                "paraphrase": "Tell me whether the evening alarm is turned on.",
                "typo": "Is the evening alram enabled?",
            },
        },
        {
            "id": "ambig-03",
            "split": "holdout",
            "clarified_support": "intend_delete",
            "notes": notes,
            "ambiguous": {
                "canonical": "Handle the seven o'clock alarm.",
                "paraphrase": "Take care of the seven o'clock alarm.",
                "typo": "Handle the seven o'clock alram.",
            },
            "clarified": {
                "canonical": "Delete the seven o'clock alarm entirely.",
                "paraphrase": "Remove the seven o'clock alarm completely.",
                "typo": "Delte the seven o'clock alarm entirely.",
            },
        },
        {
            "id": "ambig-04",
            "split": "holdout",
            "clarified_support": "intend_create",
            "notes": notes,
            "ambiguous": {
                "canonical": "I need something done about a weekday alarm.",
                "paraphrase": "A weekday alarm needs some kind of change.",
                "typo": "I need somthing done about a weekday alarm.",
            },
            "clarified": {
                "canonical": "Create a new weekday alarm for 7:30.",
                "paraphrase": "Make a new alarm for 7:30 on weekdays.",
                "typo": "Create a new weekdy alarm for 7:30.",
            },
        },
        {
            "id": "ambig-05",
            "split": "holdout",
            "clarified_support": "intend_enable",
            "notes": notes,
            "ambiguous": {
                "canonical": "The noon alarm is a problem.",
                "paraphrase": "There is an issue with the noon alarm.",
                "typo": "The noon alram is a problem.",
            },
            "clarified": {
                "canonical": "Turn the noon alarm on.",
                "paraphrase": "Enable the existing noon alarm.",
                "typo": "Turn the noon alram on.",
            },
        },
        {
            "id": "ambig-06",
            "split": "holdout",
            "clarified_support": "intend_disable",
            "notes": notes,
            "ambiguous": {
                "canonical": "Do something with my first scheduled alarm.",
                "paraphrase": "My first scheduled alarm needs attention.",
                "typo": "Do somthing with my first scheduled alarm.",
            },
            "clarified": {
                "canonical": "Disable my first scheduled alarm and keep it saved.",
                "paraphrase": "Turn off my first scheduled alarm but leave it stored.",
                "typo": "Disabel my first scheduled alarm and keep it saved.",
            },
        },
        {
            "id": "ambig-07",
            "split": "holdout",
            "clarified_support": "intend_delete",
            "notes": notes,
            "ambiguous": {
                "canonical": "Sort out the backup alarm.",
                "paraphrase": "The backup alarm needs sorting out.",
                "typo": "Sort out the bakup alarm.",
            },
            "clarified": {
                "canonical": "Delete the backup alarm entirely.",
                "paraphrase": "Remove the backup alarm completely.",
                "typo": "Delte the backup alarm entirely.",
            },
        },
        {
            "id": "ambig-08",
            "split": "holdout",
            "clarified_support": "intend_enable",
            "notes": notes,
            "ambiguous": {
                "canonical": "My 6 a.m. alarm.",
                "paraphrase": "It is about my 6 a.m. alarm.",
                "typo": "My 6 a.m. alram.",
            },
            "clarified": {
                "canonical": "Enable my existing 6 a.m. alarm.",
                "paraphrase": "Switch on the 6 a.m. alarm I already have.",
                "typo": "Enabel my existing 6 a.m. alarm.",
            },
        },
        {
            "id": "ambig-09",
            "split": "holdout",
            "clarified_support": "intend_query",
            "notes": notes,
            "ambiguous": {
                "canonical": "Can you deal with the weekend alarm?",
                "paraphrase": "The weekend alarm still needs to be dealt with.",
                "typo": "Can you deal with the weeknd alarm?",
            },
            "clarified": {
                "canonical": "Tell me whether the weekend alarm exists.",
                "paraphrase": "Does the weekend alarm currently exist?",
                "typo": "Tell me whether the weeknd alarm exists.",
            },
        },
        {
            "id": "ambig-10",
            "split": "holdout",
            "clarified_support": "intend_create",
            "notes": notes,
            "ambiguous": {
                "canonical": "Something is wrong with the ten o'clock alarm.",
                "paraphrase": "The ten o'clock alarm is not right.",
                "typo": "Something is worng with the ten o'clock alarm.",
            },
            "clarified": {
                "canonical": "Create a new alarm for 10:10.",
                "paraphrase": "I need a fresh alarm set for 10:10.",
                "typo": "Create a new alram for 10:10.",
            },
        },
    ]


def authored_items() -> list[dict]:
    rows: list[dict] = []
    for spec in _mention_specs() + _negation_specs():
        rows.extend(_atomic_family(spec))
    for spec in _order_specs():
        rows.extend(_order_family(spec))
    for spec in _ambiguity_specs():
        rows.extend(_ambiguity_family(spec))
    _validate(rows)
    return rows


def _versions(rows: list[dict]) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        grouped.setdefault((row["family_id"], row["side"]), {})[row["variant"]] = row["utt"]
    return grouped


def _validate(rows: list[dict]) -> None:
    if len(rows) != 150:
        raise ValueError(f"expected 150 family utterances, found {len(rows)}")
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example ids")
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        if row["hypothesis_version"] != HYPOTHESIS_VERSION or row["prompt_version"] != PROMPT_VERSION:
            raise ValueError(row["example_id"])
        if row["category"] not in CATEGORIES or row["split"] not in {"dev", "holdout"}:
            raise ValueError(row["example_id"])
        if row["focus_hypothesis"] is None:
            raise ValueError(row["example_id"])
        by_family.setdefault(row["family_id"], []).append(row)
    if len(by_family) != 40:
        raise ValueError("expected 40 families")
    for category in CATEGORIES:
        members = {family_id for family_id, group in by_family.items() if group[0]["category"] == category}
        if len(members) != 10:
            raise ValueError(category)
        dev = {family_id for family_id in members if by_family[family_id][0]["split"] == "dev"}
        if len(dev) != 2:
            raise ValueError(f"{category} dev split")
    for family_id, group in by_family.items():
        if len({row["split"] for row in group}) != 1 or len({row["category"] for row in group}) != 1:
            raise ValueError(family_id)
        versions = _versions(group)
        for (grouped_id, side), texts in versions.items():
            if set(texts) != {"canonical", "paraphrase", "typo"}:
                raise ValueError(grouped_id)
            if _edit_distance(texts["canonical"], texts["typo"]) == 0 or _edit_distance(texts["canonical"], texts["typo"]) > 3:
                raise ValueError(f"{grouped_id} {side} typo distance {_edit_distance(texts['canonical'], texts['typo'])}")
            if _edit_distance(texts["canonical"], texts["paraphrase"]) < 10:
                raise ValueError(f"{grouped_id} {side} paraphrase is too close")
        if group[0]["category"] == "action_order":
            plans = group[0]["plan_hypotheses"]
            expected = {row["id"]: row["expected"] for row in plans}
            if expected != {
                "plan_temporal": "supported",
                "plan_reversed": "rejected",
                "plan_only_first": "rejected",
                "plan_only_second": "rejected",
            }:
                raise ValueError(family_id)
            if not plans[2]["text"].startswith("Currently requests only"):
                raise ValueError(family_id)
            if plans[0]["order"] != list(reversed(plans[1]["order"])):
                raise ValueError(family_id)
        elif group[0]["category"] == "ambiguity_and_clarification":
            ambiguous = [row for row in group if row["side"] == "ambiguous"]
            clarified = [row for row in group if row["side"] == "clarified"]
            if len(ambiguous) != 3 or len(clarified) != 3:
                raise ValueError(family_id)
            if any(row["evaluation"] != "not_scored" or row["support"] is not None for row in ambiguous):
                raise ValueError(family_id)
            if any(row["evaluation"] != "scored" for row in clarified):
                raise ValueError(family_id)
        else:
            for row in group:
                supported = [key for key, level in row["support"].items() if level == "supported"]
                if supported != [row["focus_hypothesis"]] or row["contrast_hypothesis"] not in ATOMIC:
                    raise ValueError(row["example_id"])
                if row["support"][row["contrast_hypothesis"]] != "rejected":
                    raise ValueError(row["example_id"])


def review_manifest() -> dict:
    return {
        "dataset_version": DATASET_VERSION,
        "hypothesis_version": HYPOTHESIS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_rule_version": EVALUATION_RULE_VERSION,
        "instructions": (
            "Approved is JSON true after the utterance is checked against its family label. "
            "Ambiguous rows have no gold winner. Development and holdout are split by family. "
            "Prompt template v2 and the synthetic_v2 definitions are unchanged."
        ),
        "items": [
            {
                "example_id": row["example_id"],
                "family_id": row["family_id"],
                "family": row["family"],
                "role": row["role"],
                "split": row["split"],
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
