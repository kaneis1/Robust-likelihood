"""python -m robust_likelihood.multiwoz download | prepare | run

`run --label pilot` also writes pilot_report.md into the run folder.
"""

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
from robust_likelihood.keys import MissingAPIKey, UnrecognizedKeyLabels
from robust_likelihood.multiwoz.download import EXPECTED_DIALOGUES, ChecksumMismatch, download
from robust_likelihood.multiwoz.hypotheses import food_pool, value_pools
from robust_likelihood.multiwoz.jev_run import ModelIdChanged, run_jev
from robust_likelihood.multiwoz.probe import APPENDIX_A4, build_probe, probe_summary
from robust_likelihood.multiwoz.reference import appendix_checks, build_reference, reference_markdown
from robust_likelihood.multiwoz.report import pilot_report
from robust_likelihood.planning import UnpinnedModelError
from robust_likelihood.storage import read_json, repo_root, write_json, write_jsonl


def _write_compact(path: Path, payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(root: Path, config: dict, raw: Path | None = None, derived: Path | None = None) -> dict:
    """Phases 2 to 4: select the analysis items, estimate the reference rates, and write the probe set."""
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
    pools = value_pools(food_pool(goals))
    probe = build_probe(items, pools)
    write_jsonl(derived / "probe.jsonl", probe)
    probe_counts = probe_summary(probe)
    probe_counts["food_pool_size"] = len(pools["food"])
    checks = {
        "A1": differences(APPENDIX_A1, selection),
        **appendix_checks(table),
        "A4": differences(APPENDIX_A4, probe_counts),
    }
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
        "probe": probe_counts,
        "appendix_differences": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.multiwoz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--wording", choices=["likely", "typical", "both"], default="both")
    run_parser.add_argument("--set", dest="set_name", choices=["clean", "fail_info", "all"], default="clean")
    run_parser.add_argument("--items", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--repeats", type=int, default=0)
    run_parser.add_argument("--max-requests", type=int, default=None)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", default=None)
    run_parser.add_argument("--label", default="run")
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (
        ChecksumMismatch,
        ModelIdChanged,
        MissingAPIKey,
        UnrecognizedKeyLabels,
        UnpinnedModelError,
        FileNotFoundError,
        ValueError,
    ) as exc:
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
    if args.command == "run":
        summary = run_jev(
            root,
            config,
            wording=args.wording,
            set_name=args.set_name,
            items=args.items,
            seed=args.seed,
            repeats=args.repeats,
            max_requests=args.max_requests,
            dry_run=args.dry_run,
            resume=None if args.resume is None else Path(args.resume),
            label=args.label,
        )
        if summary["dry_run"]:
            return 0
        run_dir = Path(summary["run_dir"])
        if read_json(run_dir / "manifest.json")["label"] == "pilot":
            report = run_dir / "pilot_report.md"
            report.write_text(pilot_report(run_dir, root / DERIVED_DIR), encoding="utf-8", newline="\n")
            summary["pilot_report"] = str(report)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
