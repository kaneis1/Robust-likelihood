"""python -m robust_likelihood.multiwoz download | prepare | run | analyze | lm-score | report

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
from robust_likelihood.multiwoz.metrics import analyze
from robust_likelihood.multiwoz.probe import APPENDIX_A4, build_probe, probe_summary
from robust_likelihood.multiwoz.reference import appendix_checks, build_reference, reference_markdown
from robust_likelihood.multiwoz.report import metrics_markdown, pilot_report, summary_markdown, write_figures
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
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("run_dir")
    analyze_parser.add_argument("--repeats", default=None, help="folder of the repeat-check run")
    analyze_parser.add_argument("--lm", default=None, help="folder of the lm-score run")
    lm_parser = sub.add_parser("lm-score", help="Phase 8; needs the lm extra and a GPU")
    lm_parser.add_argument("--probe", default=str(DERIVED_DIR / "probe.jsonl"))
    lm_parser.add_argument("--model", default=None, help="defaults to the configured model")
    lm_parser.add_argument("--revision", default=None, help="Hugging Face revision; the hash used is recorded")
    lm_parser.add_argument("--samples", type=int, default=None, help="samples per hypothesis text")
    lm_parser.add_argument("--batch-size", type=int, default=16)
    lm_parser.add_argument("--limit-texts", type=int, default=None, help="smoke test on the first N texts")
    lm_parser.add_argument("--dtype", default="bfloat16")
    lm_parser.add_argument("--device", default=None, help="cuda, mps, or cpu; detected when omitted")
    lm_parser.add_argument("--label", default="lm")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("run_dir")
    report_parser.add_argument("--lm", default=None, help="folder of the lm-score run")
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
    if args.command == "analyze":
        run_dir = Path(args.run_dir)
        metrics = analyze(
            run_dir,
            root / DERIVED_DIR,
            config,
            repeats_dir=None if args.repeats is None else Path(args.repeats),
            lm_dir=None if args.lm is None else Path(args.lm),
        )
        write_json(run_dir / "metrics.json", metrics)
        (run_dir / "metrics.md").write_text(metrics_markdown(metrics), encoding="utf-8", newline="\n")
        for row in metrics["predictions"]:
            result = row.get("status", "pending") if row["pass"] is None else ("pass" if row["pass"] else "fail")
            print(f"{row['id']}: {result}  value={row.get('value')}")
        print(f"Wrote {run_dir / 'metrics.json'} and {run_dir / 'metrics.md'}")
        return 0
    if args.command == "lm-score":
        from robust_likelihood.multiwoz.lm import lm_score

        summary = lm_score(
            root,
            config,
            Path(args.probe),
            model_name=args.model,
            revision=args.revision,
            samples=args.samples,
            batch_size=args.batch_size,
            limit_texts=args.limit_texts,
            label=args.label,
            dtype=args.dtype,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        run_dir = Path(args.run_dir)
        lm_dir = None if args.lm is None else Path(args.lm)
        metrics = read_json(run_dir / "metrics.json")
        figures = write_figures(run_dir, root / DERIVED_DIR, metrics, lm_dir)
        summary = summary_markdown(
            metrics,
            read_json(root / DERIVED_DIR / "reference_table.json"),
            read_json(run_dir / "manifest.json"),
            config["multiwoz"]["commit"],
            None if lm_dir is None else read_json(lm_dir / "manifest.json"),
        )
        (run_dir / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
        print(f"Wrote {run_dir / 'summary.md'} and {len(figures)} figure files")
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
