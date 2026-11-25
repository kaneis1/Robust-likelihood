"""python -m robust_likelihood.clips <command>; docs/clips_small_lm_plan.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_likelihood.storage import repo_root, write_jsonl

CLIPS_ROOT = Path("external") / "CLIPS.jl"
DERIVED = Path("data") / "clips" / "derived"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.clips")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("copy-loops", help="write julia/clips/<domain>_experiments.jl from the pinned CLIPS.jl")
    sub.add_parser("utterances", help="write data/clips/derived/utterances.jsonl")
    pairs_parser = sub.add_parser("pairs", help="Phase 1: pairs.jsonl and the counts, from the record logs")
    pairs_parser.add_argument("--smoke", action="store_true", help="read the smoke-run record logs instead")
    score_parser = sub.add_parser("score", help="Phase 2: the table of one model (needs a GPU)")
    score_parser.add_argument("--model", required=True, help="a key of configs/clips.json")
    score_parser.add_argument("--pairs", default=str(DERIVED / "pairs.jsonl"))
    score_parser.add_argument("--prompts", default=str(DERIVED / "prompts.jsonl"))
    score_parser.add_argument("--out", default=str(DERIVED / "scores"))
    score_parser.add_argument("--device", default=None)
    table_parser = sub.add_parser("table", help="expand a model's scores into the TableMixture replay table")
    table_parser.add_argument("--model", required=True)
    table_parser.add_argument("--pairs", default=str(DERIVED / "pairs.jsonl"))
    table_parser.add_argument("--prompts", default=str(DERIVED / "prompts.jsonl"))
    table_parser.add_argument("--out", default="results/clips/tables")
    download_parser = sub.add_parser("download", help="fetch the models of configs/clips.json at their revisions")
    download_parser.add_argument("--models", default=None, help="comma-separated keys (default: all)")
    args = parser.parse_args(argv)
    root = repo_root()
    if args.command == "copy-loops":
        from robust_likelihood.clips.loops import copy_loops

        for path in copy_loops(root / CLIPS_ROOT, root / "julia" / "clips"):
            print(f"Wrote {path}")
        return 0
    if args.command == "utterances":
        from robust_likelihood.clips.data import utterance_rows

        rows = utterance_rows(root / CLIPS_ROOT)
        write_jsonl(root / DERIVED / "utterances.jsonl", rows)
        print(json.dumps({domain: sum(1 for row in rows if row["domain"] == domain) for domain in ("mdkg", "vh")}))
        return 0
    if args.command == "pairs":
        from robust_likelihood.clips.record import run_pairs, split_prompts
        from robust_likelihood.storage import write_json

        rows, counts = run_pairs(root, suffix="-smoke" if args.smoke else "")
        name = "pairs_smoke" if args.smoke else "pairs"
        pairs, prompts = split_prompts(rows)
        write_jsonl(root / DERIVED / f"{name}.jsonl", pairs)
        write_jsonl(root / DERIVED / ("prompts_smoke.jsonl" if args.smoke else "prompts.jsonl"), prompts)
        write_json(root / DERIVED / f"{name}_counts.json", counts)
        print(json.dumps(counts, indent=2))
        return 0
    if args.command == "score":
        from robust_likelihood.clips.score import clips_score

        print(json.dumps(clips_score(root, args.model, pairs_file=root / args.pairs, prompts_file=root / args.prompts,
                                     out=root / args.out, device=args.device), indent=2))
        return 0
    if args.command == "table":
        from robust_likelihood.clips.score import write_table

        print(json.dumps(write_table(root, args.model, pairs_file=root / args.pairs, prompts_file=root / args.prompts,
                                     scores_dir=root / DERIVED / "scores", out=root / args.out)))
        return 0
    if args.command == "download":
        from huggingface_hub import snapshot_download

        from robust_likelihood.clips.score import load_clips_config

        models = load_clips_config(root)["models"]
        for key in (args.models.split(",") if args.models else models):
            spec = models[key]
            path = snapshot_download(spec["id"], revision=spec["revision"],
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
            print(key, path)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
