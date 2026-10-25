"""python -m robust_likelihood.multiwoz download | prepare"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RAW_DIR, load_config
from robust_likelihood.multiwoz.corpus import APPENDIX_A1, SLOTS, differences, load_goals, load_turns, select_items
from robust_likelihood.multiwoz.download import EXPECTED_DIALOGUES, ChecksumMismatch, download
from robust_likelihood.storage import repo_root, write_jsonl


def prepare(root: Path, config: dict, raw: Path | None = None, derived: Path | None = None) -> dict:
    """Phase 2: select the analysis items and check them against Appendix A1."""
    if list(config["slots"]) != list(SLOTS):
        raise ValueError(f"configs/multiwoz_size_principle.json slots must be {list(SLOTS)}")
    raw = raw or root / RAW_DIR
    derived = derived or root / DERIVED_DIR
    goals = load_goals(raw)
    turns = load_turns(raw)
    items, selection = select_items(goals, turns, config["other_domains"])
    write_jsonl(derived / "items.jsonl", items)
    return {
        "items": len(items),
        "selection": selection,
        "appendix_a1_differences": differences(APPENDIX_A1, selection),
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
        problems = summary["appendix_a1_differences"]
        if problems:
            print("Appendix A1 differs; stop and ask before Phase 3:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print("Appendix A1 matches.")
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
