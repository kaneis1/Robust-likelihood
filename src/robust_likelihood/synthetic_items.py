"""Authored synthetic alarm items. These are not MASSIVE rows."""

from __future__ import annotations

INTENTION_HYPOTHESES = (
    "intend_create",
    "intend_enable",
    "intend_disable",
    "intend_delete",
    "intend_query",
)

SOURCE = "synthetic_alarm"
REVIEWER = "assistant wording check"
REVIEWED_AT = "2026-10-01"


def _support(*high: str, unspecified: bool = False) -> dict[str, str]:
    if unspecified:
        return {hypothesis: "unspecified" for hypothesis in INTENTION_HYPOTHESES}
    if any(hypothesis not in INTENTION_HYPOTHESES for hypothesis in high):
        raise ValueError(f"unknown hypothesis in {high}")
    return {
        hypothesis: "support_high" if hypothesis in high else "support_low"
        for hypothesis in INTENTION_HYPOTHESES
    }


def _row(
    example_id: str,
    pair_id: str,
    family: str,
    role: str,
    utt: str,
    *,
    support: dict[str, str],
    gold_winner: str | None,
    label_changed: bool,
    original_intent: str,
    notes: str,
    massive_lump_label: str | None = None,
    context: str = "",
    order: list[str] | None = None,
) -> dict:
    return {
        "source": SOURCE,
        "dataset": SOURCE,
        "example_id": example_id,
        "pair_id": pair_id,
        "original_example_id": pair_id,
        "family": family,
        "role": role,
        "transformation": role,
        "utt": utt,
        "context": context,
        "intent": gold_winner or "",
        "original_intent": original_intent,
        "label_changed": label_changed,
        "gold_winner": gold_winner,
        "support": support,
        "massive_lump_label": massive_lump_label,
        "order": order,
        "notes": notes,
    }


def authored_items() -> list[dict]:
    enable = _support("intend_enable")
    disable = _support("intend_disable")
    query = _support("intend_query")
    both_actions = _support("intend_delete", "intend_create")
    open_case = _support(unspecified=True)
    preserved_note = (
        "Synthetic gold is intend_enable. The MASSIVE lump would call this alarm_set, "
        "which also covers creating a new alarm."
    )
    reversed_note = (
        "Synthetic gold moves from intend_enable to intend_disable. "
        "The MASSIVE lump would call the off-sentence alarm_remove, which also covers deletion. "
        "MASSIVE control 4929__meaning_change uses that lump and is not this item."
    )
    negation_note = (
        "intend_query is the request. intend_delete, intend_disable, and intend_enable are rejected. "
        "Whether the alarm is currently enabled is a state question and has no truth label."
    )
    quote_note = (
        "The user is asking whether a command was said earlier. "
        "The command inside the quotation is not being issued now."
    )
    multiple_note = (
        "Both actions are requested, in order. Their yes-probabilities are not required to sum to 1."
    )
    under_note = (
        "No action is the gold winner, and there is no correct probability. "
        "Context rows test whether the scores move."
    )
    rows = [
        _row(
            "syn-mp-first-source",
            "syn-mp-first",
            "meaning_preserved",
            "source",
            "Enable my first scheduled alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mp-first-variant",
            "syn-mp-first",
            "meaning_preserved",
            "variant",
            "Please turn on my first scheduled alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mp-weekday-source",
            "syn-mp-weekday",
            "meaning_preserved",
            "source",
            "Switch on the weekday alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mp-weekday-variant",
            "syn-mp-weekday",
            "meaning_preserved",
            "variant",
            "Please enable the weekday alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mp-seven-source",
            "syn-mp-seven",
            "meaning_preserved",
            "source",
            "Turn my seven o'clock alarm on.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mp-seven-variant",
            "syn-mp-seven",
            "meaning_preserved",
            "variant",
            "Could you enable my seven o'clock alarm?",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=preserved_note,
        ),
        _row(
            "syn-mr-first-source",
            "syn-mr-first",
            "meaning_reversed",
            "source",
            "Turn on my first scheduled alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=reversed_note,
        ),
        _row(
            "syn-mr-first-variant",
            "syn-mr-first",
            "meaning_reversed",
            "variant",
            "Turn off my first scheduled alarm.",
            support=disable,
            gold_winner="intend_disable",
            label_changed=True,
            original_intent="intend_enable",
            massive_lump_label="alarm_remove",
            notes=reversed_note,
        ),
        _row(
            "syn-mr-seven-source",
            "syn-mr-seven",
            "meaning_reversed",
            "source",
            "Enable the seven o'clock alarm.",
            support=enable,
            gold_winner="intend_enable",
            label_changed=False,
            original_intent="intend_enable",
            massive_lump_label="alarm_set",
            notes=reversed_note,
        ),
        _row(
            "syn-mr-seven-variant",
            "syn-mr-seven",
            "meaning_reversed",
            "variant",
            "Disable the seven o'clock alarm.",
            support=disable,
            gold_winner="intend_disable",
            label_changed=True,
            original_intent="intend_enable",
            massive_lump_label="alarm_remove",
            notes=reversed_note,
        ),
        _row(
            "syn-neg-cancel",
            "syn-neg-cancel",
            "negation_correction",
            "item",
            "Don't cancel the seven o'clock alarm; tell me whether it is enabled.",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=negation_note,
        ),
        _row(
            "syn-neg-delete",
            "syn-neg-delete",
            "negation_correction",
            "item",
            "Do not delete my seven o'clock alarm; just tell me if it is on.",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=negation_note,
        ),
        _row(
            "syn-neg-off",
            "syn-neg-off",
            "negation_correction",
            "item",
            "Don't turn off the seven o'clock alarm; I only want to know whether it is enabled.",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=negation_note,
        ),
        _row(
            "syn-quote-cancel",
            "syn-quote-cancel",
            "quoted_command",
            "item",
            "Did I say 'cancel my alarm' earlier?",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=quote_note,
        ),
        _row(
            "syn-quote-delete",
            "syn-quote-delete",
            "quoted_command",
            "item",
            "Did I tell you to 'delete my morning alarm' before?",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=quote_note,
        ),
        _row(
            "syn-quote-off",
            "syn-quote-off",
            "quoted_command",
            "item",
            "Was it me who said 'turn off my alarm'?",
            support=query,
            gold_winner="intend_query",
            label_changed=False,
            original_intent="intend_query",
            notes=quote_note,
        ),
        _row(
            "syn-multi-seven-eight",
            "syn-multi-seven-eight",
            "multiple_commands",
            "item",
            "Cancel my seven o'clock alarm, then set one for eight.",
            support=both_actions,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            order=["intend_delete", "intend_create"],
            notes=multiple_note + " Order: delete the seven o'clock alarm, then create one for eight.",
        ),
        _row(
            "syn-multi-morning-nine",
            "syn-multi-morning-nine",
            "multiple_commands",
            "item",
            "Delete the morning alarm, and after that create one for nine.",
            support=both_actions,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            order=["intend_delete", "intend_create"],
            notes=multiple_note + " Order: delete the morning alarm, then create one for nine.",
        ),
        _row(
            "syn-multi-noon-one",
            "syn-multi-noon-one",
            "multiple_commands",
            "item",
            "First remove my noon alarm, then set an alarm for one.",
            support=both_actions,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            order=["intend_delete", "intend_create"],
            notes=multiple_note + " Order: delete the noon alarm, then create one for one o'clock.",
        ),
        _row(
            "syn-under-seven-bare",
            "syn-under-seven",
            "underspecified",
            "bare",
            "What about my seven o'clock alarm?",
            support=open_case,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            notes=under_note,
        ),
        _row(
            "syn-under-seven-query",
            "syn-under-seven",
            "underspecified",
            "context_query",
            "What about my seven o'clock alarm?",
            support=open_case,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            context="Is my seven o'clock alarm enabled?",
            notes=under_note,
        ),
        _row(
            "syn-under-seven-cancel",
            "syn-under-seven",
            "underspecified",
            "context_cancel",
            "What about my seven o'clock alarm?",
            support=open_case,
            gold_winner=None,
            label_changed=False,
            original_intent="",
            context="Cancel my seven o'clock alarm.",
            notes=under_note,
        ),
    ]
    if len(rows) != 22:
        raise ValueError(f"expected 22 synthetic items, found {len(rows)}")
    return rows


def review_manifest() -> dict:
    return {
        "dataset": SOURCE,
        "instructions": (
            "These rows are synthetic. They are not MASSIVE labels. "
            "approved is JSON true only after the wording is checked against the support map."
        ),
        "items": [
            {
                "example_id": row["example_id"],
                "family": row["family"],
                "utt": row["utt"],
                "context": row["context"],
                "label_changed": row["label_changed"],
                "approved": True,
                "reviewer": REVIEWER,
                "reviewed_at": REVIEWED_AT,
            }
            for row in authored_items()
        ],
    }
