"""python -m robust_likelihood.multiwoz download | prepare"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RAW_DIR, load_config
from robust_likelihood.multiwoz.corpus import (
    APPENDIX_A1,
    SLOTS,
    differences,
    item_population,
    load_goals,
    load_turns,
    pooled_population,
    select_items,
)
from robust_likelihood.multiwoz.download import EXPECTED_DIALOGUES, ChecksumMismatch, download
from robust_likelihood.multiwoz.reference import appendix_checks, build_reference, reference_markdown
from robust_likelihood.storage import repo_root, write_json, write_jsonl


def _write_compact(path: Path, payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(root: Path, config: dict, raw: Path | None = None, derived: Path | None = None) -> dict:
    """Phases 2 and 3: select the analysis items, then estimate the reference rates."""
    if list(config["slots"]) != list(SLOTS):
        raise ValueError(f"configs/multiwoz_size_principle.json slots must be {list(SLOTS)}")
    raw = raw or root / RAW_DIR
    derived = derived or root / DERIVED_DIR
    goals = load_goals(raw)
    turns = load_turns(raw)
    items, selection = select_items(goals, turns, config["other_domains"])
    write_jsonl(derived / "items.jsonl", items)
    pooled, pooled_counts = pooled_population(goals, turns)
    populations = {
        "match_clean_turn0": item_population([item for item in items if item["set"] == "clean"]),
        "match_all_turn0": item_population(items),
        "pooled_first_restaurant_turn": pooled,
        "pooled_no_woz": [row for row in pooled if not row["dialogue_id"].startswith("WOZ")],
    }
    table, replicates = build_reference(populations, config)
    table["replicates_file"] = "reference_replicates.json"
    table["replicates_sha256"] = _write_compact(derived / "reference_replicates.json", replicates)
    table["selection"] = selection
    table["pooled_counts"] = pooled_counts
    checks = {"A1": differences(APPENDIX_A1, selection), **appendix_checks(table)}
    table["appendix_checks"] = checks
    write_json(derived / "reference_table.json", table)
    (derived / "reference_table.md").write_text(
        reference_markdown(table, selection, pooled_counts), encoding="utf-8", newline="\n"
    )
    return {
        "items": len(items),
        "selection": selection,
        "pooled_counts": pooled_counts,
        "primary_reference": table["variants"]["primary"],
        "appendix_differences": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.multiwoz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("prepare")
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ChecksumMismatch, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    root = repo_root()
    config = load_config(root)
    if args.command == "download":
        summary = download(root, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["dialogues"] != EXPECTED_DIALOGUES:
            print(f"Dialogue counts differ from Appendix A: expected {EXPECTED_DIALOGUES}", file=sys.stderr)
            return 1
        return 0
    if args.command == "prepare":
        summary = prepare(root, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        failed = False
        for table, problems in summary["appendix_differences"].items():
            if problems:
                failed = True
                print(f"Appendix {table} differs; stop and ask:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
            else:
                print(f"Appendix {table} matches.")
        return 1 if failed else 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
