"""python -m robust_likelihood.dice build | score | analyze | report

score --local needs the lm extra and a GPU (or a large Mac with MPS); score --api spends
API credit, so run --pilot first and stop for the cost estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from robust_likelihood.dice import DATA_DIR, DOCS_DIR, RESULTS_DIR, load_config
from robust_likelihood.dice.design import build_normalization, build_pairs, draw_sequences
from robust_likelihood.keys import MissingAPIKey, UnrecognizedKeyLabels
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.storage import read_json, repo_root, write_json, write_jsonl


def build(root: Path, config: dict, data_dir: Path | None = None) -> dict:
    data_dir = data_dir or root / DATA_DIR
    sequences = draw_sequences(config)
    pairs = build_pairs(config, sequences)
    norm = build_normalization(config)
    digests = {}
    for name, rows in (("sequences", sequences), ("pairs", pairs), ("normalization", norm)):
        path = data_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "sequences": len(sequences),
        "pairs": len(pairs),
        "api_subset_pairs": sum(1 for row in pairs if row["api_subset"]),
        "normalization_queries": len(norm),
        "impossible_pairs": sum(1 for row in pairs if row["true_logp"] is None),
        "sha256": digests,
    }


def latest_runs(results: Path) -> list[Path]:
    """The newest finished local run per model and the newest finished full API run."""
    found: dict[str, Path] = {}
    for run_dir in sorted(path for path in results.iterdir() if (path / "manifest.json").is_file()):
        manifest = read_json(run_dir / "manifest.json")
        if "finished_at" not in manifest:
            continue
        if manifest["kind"] == "local":
            found[f"local:{manifest['model_key']}:{manifest.get('variant')}"] = run_dir
        elif not manifest.get("pilot") and not manifest.get("stopped"):
            found["api"] = run_dir
    return [found[key] for key in sorted(found)]


def run_summary(run_dir: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    if manifest["kind"] == "local":
        text = (f"{manifest['model']} at revision {manifest['revision'][:12]}, {manifest['dtype']} on "
                f"{manifest['device']}, {manifest['queries']} queries, scorers {', '.join(manifest['scorers'])}"
                f"{' (no-space lists)' if manifest.get('variant') == 'nospace' else ''}, "
                f"{manifest.get('elapsed_seconds')} s")
        models = {manifest["model_key"]: {"id": manifest["model"], "revision": manifest["revision"]}}
    else:
        parts = []
        models = {}
        for provider, found in manifest["providers"].items():
            temperature = "temperature 0" if found["temperature_sent"] == 0 else "no temperature (rejected)"
            parts.append(f"{provider} {', '.join(found['returned_model_ids']) or found['requested_model_id']} "
                         f"({temperature}, {found['calls']} calls)")
            models[provider] = {"id": found["requested_model_id"], "returned": found["returned_model_ids"]}
        text = "; ".join(parts)
    return {"run_dir": run_dir.as_posix(), "kind": manifest["kind"], "summary": text, "models": models,
            "git": manifest.get("git"), "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"), "elapsed_seconds": manifest.get("elapsed_seconds"),
            "checks": manifest.get("checks"), "versions": manifest.get("versions")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.dice")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    score = sub.add_parser("score")
    where = score.add_mutually_exclusive_group(required=True)
    where.add_argument("--local", action="store_true", help="open models; needs the lm extra and a GPU")
    where.add_argument("--api", action="store_true", help="gpt and claude through the existing transport")
    score.add_argument("--model", default="all", help="local model key from configs/dice.json, or all")
    score.add_argument("--scorers", default=None, help="comma-separated subset of the model's scorers")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--dtype", default="bfloat16")
    score.add_argument("--device", default=None)
    score.add_argument("--limit", type=int, default=None, help="local smoke test on the first N queries")
    score.add_argument("--nospace", action="store_true", help="S1 only, lists written as 3,5,2 (sensitivity)")
    score.add_argument("--providers", default=None, help="comma-separated; defaults to configs/dice.json")
    score.add_argument("--pilot", type=int, default=None, help="score N seeded pairs of the API subset and stop")
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--max-requests", type=int, default=None)
    score.add_argument("--dry-run", action="store_true")
    score.add_argument("--label", default=None)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--runs", nargs="*", default=None, help="run folders; defaults to the newest of each kind")
    analyze.add_argument("--out", default=str(DOCS_DIR))
    report = sub.add_parser("report")
    report.add_argument("--out", default=str(DOCS_DIR))
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (MissingAPIKey, UnrecognizedKeyLabels, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    root = repo_root()
    config = load_config(root)
    if args.command == "build":
        print(json.dumps(build(root, config), indent=2, sort_keys=True))
        return 0
    if args.command == "score" and args.local:
        from robust_likelihood.dice.score import score_local

        keys = list(config["local_models"]) if args.model == "all" else [args.model]
        for key in keys:
            summary = score_local(
                root, config, key,
                scorers=None if args.scorers is None else args.scorers.split(","),
                batch_size=args.batch_size, dtype=args.dtype, device=args.device, limit=args.limit,
                nospace=args.nospace,
                label=None if args.label is None else f"{args.label}-{key}",
            )
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "score":
        from robust_likelihood.dice.score import score_api

        providers = config["api_providers"] if args.providers is None else args.providers.split(",")
        summary = score_api(root, config, providers, pilot=args.pilot, workers=args.workers,
                            max_requests=args.max_requests, label=args.label, dry_run=args.dry_run)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.command == "analyze":
        from robust_likelihood.dice.analyze import analyze

        runs = [Path(path) for path in args.runs] if args.runs else latest_runs(root / RESULTS_DIR)
        # Committed results name run folders relative to the repository, not by a machine's absolute path.
        runs = [run.resolve().relative_to(root.resolve()) if run.resolve().is_relative_to(root.resolve()) else run
                for run in runs]
        if not runs:
            raise FileNotFoundError(f"no finished runs in {root / RESULTS_DIR}")
        metrics = analyze(root, config, runs, root / DATA_DIR)
        out = root / args.out
        write_json(out / "metrics.json", metrics)
        write_json(out / "manifest.json", {
            "git": git_state(root),
            "config": config,
            "data_sha256": {name: hashlib.sha256((root / DATA_DIR / f"{name}.jsonl").read_bytes()).hexdigest()
                            for name in ("sequences", "pairs", "normalization")},
            "counts": metrics["counts"],
            "runs": [run_summary(run) for run in runs],
        })
        for row in metrics["predictions"]:
            result = row.get("status", "pending") if row["pass"] is None else ("pass" if row["pass"] else "fail")
            print(f"{row['id']} {row['model']}: {result}  value={row['value']}")
        print(f"Wrote {out / 'metrics.json'} and {out / 'manifest.json'}")
        return 0
    if args.command == "report":
        from robust_likelihood.dice.report import summary_markdown, write_figures

        out = root / args.out
        metrics = read_json(out / "metrics.json")
        figures = write_figures(out, metrics)
        (out / "summary.md").write_text(summary_markdown(metrics, read_json(out / "manifest.json")),
                                        encoding="utf-8", newline="\n")
        print(f"Wrote {out / 'summary.md'} and {len(figures)} figure files")
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
