"""Gaussian typicality test.

    python -m robust_likelihood.gaussian run --preset quick
    python -m robust_likelihood.gaussian run --preset full --experiments e0,e1,e2,e3,e4,e5
    python -m robust_likelihood.gaussian run --preset full --dry-run
    python -m robust_likelihood.gaussian report results/gaussian/<run-folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from robust_likelihood.gaussian.experiments import EXPERIMENT_NAMES, PRESETS, parse_experiments, plan_summary, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robust_likelihood.gaussian",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run experiments and write the report")
    run_parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    run_parser.add_argument("--experiments", default=",".join(EXPERIMENT_NAMES))
    run_parser.add_argument("--seed", type=int, default=None, help="override the preset's base seed")
    run_parser.add_argument("--results-root", type=Path, default=None, help="default: results/gaussian")
    run_parser.add_argument("--dry-run", action="store_true", help="print the plan and draw nothing")
    report_parser = sub.add_parser("report", help="regenerate figures and summary.md from a run folder")
    report_parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "report":
        from robust_likelihood.gaussian.report import write_report

        if not (args.run_dir / "metrics.jsonl").is_file():
            print(f"{args.run_dir} has no metrics.jsonl", file=sys.stderr)
            return 2
        summary = write_report(args.run_dir)
        print(f"Wrote {summary}")
        return 0

    try:
        experiments = parse_experiments(args.experiments)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    preset = PRESETS[args.preset]
    if args.seed is not None:
        preset = replace(preset, base_seed=args.seed)
    if args.dry_run:
        print(json.dumps(plan_summary(preset, experiments), indent=2, sort_keys=True))
        return 0
    run_dir = run(preset, experiments, results_root=args.results_root)
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
