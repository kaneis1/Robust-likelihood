import json
import platform
from pathlib import Path

from robust_likelihood.constants import INTENTS, MEANING_PRESERVING, SCHEMA_TRANSFORMATIONS
from robust_likelihood.drafts import AUTHORED_IDS, DRAFTS
from robust_likelihood.massive import dataset_sha256, read_canonical_lines
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl


class PilotMismatch(RuntimeError):
    pass


def rows_for_intent(rows: list[dict], intent: str) -> list[dict]:
    pool = [
        row
        for row in rows
        if row.get("intent") == intent and row.get("partition") == "train"
    ]
    pool.sort(key=lambda row: int(row["id"]))
    return pool


def select_originals(rows: list[dict], intents: tuple[str, ...] | list[str], per_intent: int, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    selected: list[dict] = []
    for intent in intents:
        pool = rows_for_intent(rows, intent)
        if len(pool) < per_intent:
            raise PilotMismatch(f"{intent} has {len(pool)} train rows, need {per_intent}")
        chosen = rng.sample(pool, per_intent)
        chosen.sort(key=lambda row: int(row["id"]))
        selected.extend(chosen)
    return selected


def _meaning_preserving_rows(massive_id: str, draft: dict, source: dict) -> list[dict]:
    texts = {
        "original": draft["utt"],
        "paraphrase": draft["paraphrase"],
        "minor_typo": draft["minor_typo"],
        "repetition": f"{draft['utt']} {draft['utt']}",
    }
    rows = []
    for transformation in MEANING_PRESERVING:
        rows.append(
            {
                "example_id": f"{massive_id}__{transformation}",
                "original_example_id": massive_id,
                "massive_id": massive_id,
                "split": "train",
                "locale": source.get("locale", "en-US"),
                "intent": draft["intent"],
                "original_intent": draft["intent"],
                "transformation": transformation,
                "utt": texts[transformation],
                "source_utt": draft["utt"],
                "label_changed": False,
            }
        )
    return rows


def _control_row(massive_id: str, draft: dict, source: dict) -> dict:
    change = draft["meaning_change"]
    return {
        "example_id": f"{massive_id}__meaning_change",
        "original_example_id": massive_id,
        "massive_id": massive_id,
        "split": "train",
        "locale": source.get("locale", "en-US"),
        "intent": change["intent"],
        "original_intent": draft["intent"],
        "transformation": "meaning_change",
        "utt": change["utt"],
        "source_utt": draft["utt"],
        "label_changed": True,
    }


def build_pilot_rows(selected: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    ids = [str(row["id"]) for row in selected]
    if ids != AUTHORED_IDS:
        raise PilotMismatch(
            "Sampled train ids do not match the authored drafts. "
            f"Got {ids}."
        )
    by_id = {str(row["id"]): row for row in selected}
    drafts: list[dict] = []
    controls: list[dict] = []
    originals: list[dict] = []
    for massive_id in AUTHORED_IDS:
        source = by_id[massive_id]
        draft = DRAFTS[massive_id]
        if source["utt"] != draft["utt"] or source["intent"] != draft["intent"]:
            raise PilotMismatch(f"Train row {massive_id} does not match the authored utterance")
        if draft["meaning_change"]["intent"] == draft["intent"]:
            raise PilotMismatch(f"Meaning-change control for {massive_id} did not change the label")
        originals.append(
            {
                "example_id": f"{massive_id}__original",
                "original_example_id": massive_id,
                **source,
            }
        )
        drafts.extend(_meaning_preserving_rows(massive_id, draft, source))
        controls.append(_control_row(massive_id, draft, source))
    return originals, drafts, controls


def review_items(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        items.append(
            {
                "example_id": row["example_id"],
                "original_example_id": row["original_example_id"],
                "transformation": row["transformation"],
                "intent": row["intent"],
                "utt": row["utt"],
                "label_changed": row["label_changed"],
                "reviewer": None,
                "reviewed_at": None,
                "approved": False,
            }
        )
    return items


def load_train_rows(path: Path) -> list[dict]:
    rows = []
    for line in read_canonical_lines(path):
        row = json.loads(line)
        if row.get("partition") != "train":
            raise PilotMismatch("Refusing a non-train row while preparing the pilot")
        rows.append(row)
    return rows


def prepare(root: Path, output_dir: Path | None = None, train_path: Path | None = None) -> dict:
    config = read_json(root / "configs" / "pilot.json")
    if list(config["intents"]) != list(INTENTS):
        raise PilotMismatch("Pilot config intents must be alarm_set, alarm_query, alarm_remove")
    train = train_path or (root / "data" / "massive" / config["locale"] / "train.jsonl")
    selected = select_originals(
        load_train_rows(train),
        tuple(config["intents"]),
        int(config["per_intent"]),
        int(config["seed"]),
    )
    originals, drafts, controls = build_pilot_rows(selected)
    if any(row["transformation"] == "polite_restatement" for row in drafts):
        raise PilotMismatch("polite_restatement is schema-only")
    unknown = {row["transformation"] for row in drafts + controls} - set(SCHEMA_TRANSFORMATIONS)
    if unknown:
        raise PilotMismatch(f"Unknown transformations: {sorted(unknown)}")
    destination = output_dir or (root / "data" / "pilot")
    checksum_path = root / "data" / "massive" / config["locale"] / "checksum.json"
    checksum = read_json(checksum_path)
    verified = False
    try:
        verified = dataset_sha256(checksum_path.parent) == checksum["sha256"]
    except OSError:
        verified = False
    sampling = {
        "dataset": config["dataset"],
        "dataset_revision": config["dataset_revision"],
        "config": config["config"],
        "locale": config["locale"],
        "split": config["split"],
        "seed": config["seed"],
        "per_intent": config["per_intent"],
        "intents": list(config["intents"]),
        "example_ids": [row["id"] for row in originals],
        "sampler": (
            "For each intent, keep partition train, sort by numeric id, "
            "draw Random(seed).sample, then sort the chosen ids."
        ),
        "dataset_checksum_sha256": checksum["sha256"],
        "dataset_checksum_verified": verified,
        "official_test_split": "untouched",
        "python_version": platform.python_version(),
        "transformations_emitted": list(MEANING_PRESERVING),
        "schema_transformations": list(SCHEMA_TRANSFORMATIONS),
    }
    review = {
        "instructions": (
            "Set reviewer, reviewed_at, and approved to true only after a person checks the utterance. "
            "The comparison run refuses these drafts until then."
        ),
        "items": review_items(drafts + controls),
    }
    write_jsonl(destination / "originals.jsonl", originals)
    write_jsonl(destination / "draft_inputs.jsonl", drafts)
    write_jsonl(destination / "meaning_change_controls.jsonl", controls)
    write_json(destination / "sampling_manifest.json", sampling)
    write_json(destination / "review_manifest.json", review)
    return {
        "originals": len(originals),
        "draft_inputs": len(drafts),
        "meaning_change_controls": len(controls),
        "review_items": len(review["items"]),
        "dataset_checksum_verified": verified,
        "output_dir": str(destination),
    }


def load_pilot(root: Path) -> tuple[list[dict], list[dict], dict, dict]:
    directory = root / "data" / "pilot"
    return (
        read_jsonl(directory / "draft_inputs.jsonl"),
        read_jsonl(directory / "meaning_change_controls.jsonl"),
        read_json(directory / "review_manifest.json"),
        read_json(directory / "sampling_manifest.json"),
    )
