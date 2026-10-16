"""Gaussian typicality test.

    python -m robust_likelihood.gaussian run --preset quick --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

from robust_likelihood.gaussian.experiments import EXPERIMENT_NAMES, PRESETS, parse_experiments, plan_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.gaussian", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    run_parser.add_argument("--experiments", default=",".join(EXPERIMENT_NAMES))
    run_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        experiments = parse_experiments(args.experiments)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(plan_summary(PRESETS[args.preset], experiments), indent=2, sort_keys=True))
    if not args.dry_run:
        print("Experiments are not implemented yet; this prints the plan only.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
