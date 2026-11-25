"""Week of 24 Nov, Part A (docs/week_nov24_plan.md): every utterance unit the `uperturb` part scores.

Three sources, in the schema of derived/perturbations.jsonl (unit_id "<item>|<kind>"):

- perturbations.jsonl [Frozen]: the dosed typo, style and filler units (perturb.py); only valid units.
- fixed (A3): the six human first turns that misspell "restaurant", with the word corrected. Here the
  human turn is the perturbed text's opposite: the original carries the natural typo and the unit
  removes it. Descriptive only (n = 6).
- terse (A2): the accepted `terse` rewrites of rewrites.py (derived/terse.jsonl), once that API run exists.
- terse_rx (sensitivity, not in the plan's predictions): per unit, the first candidate of the same run that
  passes the regex slot check, whatever its NLI (derived/terse_rx.jsonl). NLI in the backward direction
  rejects most telegraphic rewrites ("thai restaurant west" does not entail the full sentence), so the
  accepted `terse` set keeps the less terse ones; terse_rx shows what that selection hides.
"""

from __future__ import annotations

import re
from pathlib import Path

from robust_likelihood.multiwoz.semantic import Extractor
from robust_likelihood.storage import read_jsonl

FIXED_ITEMS = ("SNG0458", "SNG0472", "SNG0494", "SNG0524", "SNG0575", "SNG0580")
MISSPELLED = re.compile(r"\b(?:restuarant|resturant)\b", re.IGNORECASE)
PERTURBATIONS = "perturbations.jsonl"
FIXED = "fixed.jsonl"
TERSE = "terse.jsonl"
TERSE_RX = "terse_rx.jsonl"


def fix_restaurant(text: str) -> tuple[str, list[dict]]:
    edits = [{"start": match.start(), "word": match.group(0), "op": "fix",
              "typo": "Restaurant" if match.group(0)[0].isupper() else "restaurant"}
             for match in MISSPELLED.finditer(text)]
    return MISSPELLED.sub(lambda match: "Restaurant" if match.group(0)[0].isupper() else "restaurant", text), edits


def fixed_units(items: list[dict], extractor: Extractor) -> list[dict]:
    """One unit per FIXED_ITEMS item: its first turn with "restaurant" spelled correctly."""
    by_id = {item["item_id"]: item for item in items}
    rows = []
    for item_id in FIXED_ITEMS:
        utterance = by_id[item_id]["utterance"]
        text, edits = fix_restaurant(utterance)
        before, after = extractor.extract(utterance), extractor.extract(text)
        rows.append({"unit_id": f"{item_id}|fixed", "item_id": item_id, "kind": "fixed", "dose": len(edits),
                     "original": utterance, "rewrite": text, "edits": edits, "added": [], "available": None,
                     "changed": text != utterance, "extract_before": before, "extract_after": after,
                     "valid": text != utterance and before == after})
    return rows


def terse_units(rows: list[dict]) -> list[dict]:
    """Accepted terse rewrites (rewrites.py accepted.jsonl rows) in the perturbation schema."""
    found = []
    for row in rows:
        item_id = row["item_id"]
        found.append({"unit_id": f"{item_id}|terse", "item_id": item_id, "kind": "terse", "dose": None,
                      "original": row["original"], "rewrite": row["rewrite"], "edits": [], "added": [],
                      "available": None, "changed": row["rewrite"] != row["original"], "valid": True,
                      "nli_forward": row.get("nli_forward"), "nli_backward": row.get("nli_backward")})
    return found


def terse_rx_rows(candidates: list[dict], units: dict[str, dict]) -> list[dict]:
    """The first regex-passing candidate per terse unit (candidates.jsonl of the terse run)."""
    chosen: dict[str, dict] = {}
    for record in sorted(candidates, key=lambda found: (found["unit_id"], found["attempt"])):
        if record["unit_id"] not in chosen and record.get("slots_kept") and record.get("rewrite"):
            chosen[record["unit_id"]] = record
    return [{"unit_id": unit_id, "item_id": units[unit_id]["item_id"], "kind": "terse_rx",
             "original": units[unit_id]["original"], "rewrite": record["rewrite"], "attempt": record["attempt"],
             "nli_forward": record.get("nli_forward"), "nli_backward": record.get("nli_backward"),
             "accepted": record.get("accepted")}
            for unit_id, record in sorted(chosen.items())]


def load_units(derived: Path, items: set[str] | None = None) -> list[dict]:
    """The valid units uperturb scores: perturbations, then fixed and terse where their files exist."""
    rows = [row for row in read_jsonl(derived / PERTURBATIONS) if row["valid"]]
    if (derived / FIXED).exists():
        rows += [row for row in read_jsonl(derived / FIXED) if row["valid"]]
    if (derived / TERSE).exists():
        rows += terse_units(read_jsonl(derived / TERSE))
    if (derived / TERSE_RX).exists():
        rows += [dict(row, unit_id=row["unit_id"].replace("|terse", "|terse_rx"), kind="terse_rx")
                 for row in terse_units(read_jsonl(derived / TERSE_RX))]
    return [row for row in rows if items is None or row["item_id"] in items]
