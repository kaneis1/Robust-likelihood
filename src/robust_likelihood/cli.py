from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robust_likelihood.evaluate import write_evaluation
from robust_likelihood.keys import MissingAPIKey, UnrecognizedKeyLabels
from robust_likelihood.massive import DatasetDrift, download
from robust_likelihood.pilot import PilotMismatch, prepare
from robust_likelihood.planning import UnpinnedModelError
from robust_likelihood.runner import UnreviewedDrafts, connect_models, run_comparison
from robust_likelihood.storage import read_json, repo_root


def parse_models(text: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in text.split(",") if part.strip())
    allowed = {"jev", "gpt", "claude"}
    unknown = [model for model in models if model not in allowed]
    if not models or unknown or len(set(models)) != len(models):
        raise ValueError(f"Models must be a unique subset of jev,gpt,claude; got {text!r}")
    return models


def _default_models(root, command_default: list[str] | None = None) -> str:
    config = read_json(root / "configs" / "models.json")
    chosen = command_default if command_default is not None else config["default_comparison"]
    return ",".join(chosen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robust-likelihood")
    sub = parser.add_subparsers(dest="command", required=True)

    download_parser = sub.add_parser("download")
    download_parser.add_argument("--locale", default="en-US")

    sub.add_parser("prepare")

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--max-requests", type=int, default=None)
    run_parser.add_argument("--allow-unreviewed", action="store_true")
    run_parser.add_argument("--repeats", type=int, default=0)
    run_parser.add_argument("--models", default=None)
    run_parser.add_argument("--hypothesis-version", default=None)
    run_parser.add_argument("--include-controls", action="store_true")
    run_parser.add_argument("--resume", default=None)

    connect_parser = sub.add_parser("connect")
    connect_parser.add_argument("--models", default="jev")
    connect_parser.add_argument("--jev-model", default=None)

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("run_dir")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (
        UnpinnedModelError,
        UnreviewedDrafts,
        UnrecognizedKeyLabels,
        MissingAPIKey,
        DatasetDrift,
        PilotMismatch,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    root = repo_root()
    if args.command == "download":
        summary = download(root, locale=args.locale)
        print(
            f"Verified {summary['locale']} MASSIVE sha256 {summary['sha256']} "
            f"rows {summary['rows']}"
        )
        return 0
    if args.command == "prepare":
        summary = prepare(root)
        print(
            f"Wrote {summary['draft_inputs']} draft inputs, "
            f"{summary['meaning_change_controls']} meaning-change controls, "
            f"and {summary['review_items']} unreviewed manifest rows to {summary['output_dir']}"
        )
        return 0
    if args.command == "run":
        providers = parse_models(args.models or _default_models(root))
        run_comparison(
            root,
            providers=providers,
            dry_run=args.dry_run,
            max_requests=args.max_requests,
            allow_unreviewed=args.allow_unreviewed,
            repeats=args.repeats,
            hypothesis_version=args.hypothesis_version,
            include_controls=args.include_controls,
            resume=None if args.resume is None else Path(args.resume),
        )
        return 0
    if args.command == "connect":
        connect_models(root, parse_models(args.models), jev_model=args.jev_model)
        return 0
    if args.command == "evaluate":
        metrics = write_evaluation(Path(args.run_dir))
        print(f"Wrote metrics with {metrics.get('disagreement_count', 0)} disagreements")
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
