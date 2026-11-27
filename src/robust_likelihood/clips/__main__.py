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
    evaluate_parser = sub.add_parser("evaluate", help="Phase 3: Table 1 per model from the replays' result files")
    evaluate_parser.add_argument("--models", default=None, help="comma-separated keys (default: all with a replay)")
    evaluate_parser.add_argument("--reference", default="qwen3_8b", help="the model whose CLIPS the relative metrics use")
    evaluate_parser.add_argument("--out", default="docs/clips_results")
    perturb_parser = sub.add_parser("perturb", help="Phase 4: perturbations.jsonl and the co-batched batches.jsonl")
    perturb_parser.add_argument("--paraphrases", default=None, help="a paraphrase run whose accepted units to add")
    para_parser = sub.add_parser("paraphrase", help="Phase 4: para1 and para2 through claude (API; STOP first)")
    para_parser.add_argument("--dry-run", action="store_true", help="count the units and calls; no API call")
    para_parser.add_argument("--workers", type=int, default=4)
    para_parser.add_argument("--nli-device", default=None)
    perturbed_parser = sub.add_parser("score-perturbed", help="Phase 4: score batches.jsonl with one model (GPU)")
    perturbed_parser.add_argument("--model", required=True)
    perturbed_parser.add_argument("--device", default=None)
    robust_parser = sub.add_parser("robustness", help="Phase 4: the posterior under each perturbed instruction")
    robust_parser.add_argument("--models", default=None)
    robust_parser.add_argument("--out", default="docs/clips_results")
    sub.add_parser("noplanner-build", help="Phase 6: the mDKG states, map prompts and pairs (no planner)")
    noplanner_score_parser = sub.add_parser("score-noplanner", help="Phase 6: score the map prompts with one model (GPU)")
    noplanner_score_parser.add_argument("--model", required=True)
    noplanner_score_parser.add_argument("--device", default=None)
    noplanner_parser = sub.add_parser("noplanner", help="Phase 6: P(g_true) without the planner, against the replays")
    noplanner_parser.add_argument("--models", default=None)
    noplanner_parser.add_argument("--out", default="docs/clips_results")
    report_parser = sub.add_parser("report", help="Phase 5: summary.md, figures and the scorecard")
    report_parser.add_argument("--out", default="docs/clips_results")
    report_parser.add_argument("--no-figures", action="store_true")
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
    if args.command == "evaluate":
        from robust_likelihood.clips.evaluate import run_evaluate
        from robust_likelihood.clips.score import load_clips_config

        models = args.models.split(",") if args.models else list(load_clips_config(root)["models"])
        result = run_evaluate(root, models, root / args.out, args.reference)
        print(json.dumps({"runs": result["runs"], "action_only_check": result["action_only_check"]}, indent=2))
        return 0
    if args.command == "perturb":
        from robust_likelihood.clips import perturb
        from robust_likelihood.multiwoz.perturb import load_lexicon
        from robust_likelihood.storage import read_json, read_jsonl, write_json

        seed = read_json(root / "configs" / "clips.json")["seed"]
        utterances = read_jsonl(root / DERIVED / "utterances.jsonl")
        units = perturb.build_units(utterances, perturb.protected_set(root / CLIPS_ROOT), seed,
                                    load_lexicon(root / "data" / "multiwoz" / "derived"))
        if args.paraphrases:
            units += read_jsonl(Path(args.paraphrases) / "accepted.jsonl")
        particles = [row for domain in ("mdkg", "vh")
                     for row in read_jsonl(root / "results" / "clips" / "record" / f"{domain}.particles.jsonl")]
        batches = perturb.batches(units, utterances, perturb.utterance_prompts(particles, utterances))
        write_jsonl(root / DERIVED / "perturbations.jsonl", units)
        write_jsonl(root / DERIVED / "batches.jsonl", batches)
        found = {"units": perturb.counts(units), "batches": len(batches),
                 "sequences": sum(len(row["outputs"]) for row in batches),
                 "paraphrase_run": args.paraphrases}
        write_json(root / DERIVED / "perturbations_counts.json", found)
        print(json.dumps(found, indent=2))
        return 0
    if args.command == "paraphrase":
        from robust_likelihood.clips import paraphrase
        from robust_likelihood.storage import read_jsonl

        if args.dry_run:
            print(json.dumps(paraphrase.plan(read_jsonl(root / DERIVED / "utterances.jsonl")), indent=2))
            return 0
        print(json.dumps(paraphrase.run_paraphrases(root, workers=args.workers, nli_device=args.nli_device), indent=2))
        return 0
    if args.command == "score-perturbed":
        from robust_likelihood.clips.robustness import score_batches

        print(json.dumps(score_batches(root, args.model, batches_file=root / DERIVED / "batches.jsonl",
                                       prompts_file=root / DERIVED / "prompts.jsonl",
                                       out=root / DERIVED / "scores_perturbed", device=args.device), indent=2))
        return 0
    if args.command == "robustness":
        from robust_likelihood.clips.robustness import run_robustness
        from robust_likelihood.clips.score import load_clips_config

        models = args.models.split(",") if args.models else list(load_clips_config(root)["models"])
        result = run_robustness(root, models, root / args.out)
        print(json.dumps({model: {scope: {kind: {"units": block["units"],
                                                  "abs_delta_p_true": block["abs_delta_p_true"]["median"],
                                                  "argmax_flip": block["argmax_flip"]["rate"]}
                                           for kind, block in found.items()}
                                  for scope, found in blocks.items()}
                          for model, blocks in result["models"].items()}, indent=1))
        return 0
    if args.command == "noplanner-build":
        from robust_likelihood.clips.noplanner import build

        print(json.dumps(build(root)))
        return 0
    if args.command == "score-noplanner":
        from robust_likelihood.clips.noplanner import score_noplanner

        print(json.dumps(score_noplanner(root, args.model, device=args.device), indent=2))
        return 0
    if args.command == "noplanner":
        from robust_likelihood.clips.noplanner import run_noplanner
        from robust_likelihood.clips.score import load_clips_config

        models = args.models.split(",") if args.models else list(load_clips_config(root)["models"])
        result = run_noplanner(root, models, root / args.out)
        print(json.dumps({model: {"p_true": block["p_true"]["mean"], "top1": block["top1"]["mean"],
                                  "goal_r": (block.get("goal_correlation") or {}).get("r"),
                                  "language_only_p_true": block.get("language_only_p_true"),
                                  "minus_language_only": (block.get("minus_language_only") or {}).get("mean")}
                          for model, block in result["models"].items()}, indent=1))
        return 0
    if args.command == "report":
        from robust_likelihood.clips.report import run_report

        print(json.dumps(run_report(root, root / args.out, figures=not args.no_figures), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
