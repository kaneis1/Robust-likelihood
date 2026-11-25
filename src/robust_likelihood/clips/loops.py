"""Copies of CLIPS.jl's two experiment loops, with every difference listed here (docs/clips_small_lm_plan.md 2.2).

CLIPS's experiments.jl writes its results into its own folder (`@__DIR__/results`), inside external/,
which this project never edits, so the loops are copied into julia/clips/ instead of run in place.
`copy_loops` writes the copies from the pinned originals by exact replacements, each asserted to apply
once, so the copies can be regenerated and audited:

1. Before CLIPS's src files: include julia/clips/bridge.jl and name the domain, so the src files build
   TableMixtures (bridge.jl explains how), then configure record or replay mode from the environment.
2. CLIPS's src files and dataset are read from external/CLIPS.jl/<domain>/.
3. Results go to $CLIPS_RESULTS (default results/clips/unnamed/<domain>/) instead of external/.
4. The plans are those named in $CLIPS_PLANS, if set (smoke runs); otherwise all, in CLIPS's order.
5. The record log notes the plan and the method (literal or pragmatic) of every language-model call.
6. In record mode every score is 0, so the assistance planning that follows inference (it calls no
   language model; belief updates during assistance constrain `speak` to false) and the result files
   are skipped. Replay runs the loops in full.
7. In record mode, after pragmatic inference, each particle's goal, cost profile and utterance prompts at
   every utterance time go to $CLIPS_PARTICLE_LOG (bridge.jl's record_particles), so the per-goal commands
   can be checked; the language model sees only prompts, so the record log cannot say which goal asked.
8. VirtualHome only: `global goals` at the top of the plan loop. virtual-home/src/inference.jl's verbose
   printer reads a global `goals` (`1:length(goals)`), which the loop assigns. In the REPL, where CLIPS's
   loops were presumably run, that assignment makes a global; run as a script it makes a loop-local
   variable and the printer fails with UndefVarError. Declaring it global restores the REPL behavior.
"""

from __future__ import annotations

from pathlib import Path

DOMAINS = {"mdkg": "doors-keys-gems", "vh": "virtual-home"}
SRC_FILES = ("utils", "plan_io", "heuristics", "utterances", "inference", "assistance")
HEADER = "# Copied from external/CLIPS.jl/{folder}/experiments.jl at 6536a2a by `python -m robust_likelihood.clips " \
         "copy-loops`;\n# every difference is listed in src/robust_likelihood/clips/loops.py.\n"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {old!r}, found {count}")
    return text.replace(old, new)


def transform(original: str, domain: str) -> str:
    folder = DOMAINS[domain]
    text = original.replace("\r\n", "\n")
    includes = "".join(f'include("src/{name}.jl")\n' for name in SRC_FILES)
    bridged = (f'include(joinpath(@__DIR__, "bridge.jl"))  # (1) TableMixtures, record or replay\n'
               f'CLIPS_DOMAIN[] = "{domain}"\n'
               f'CLIPS_DIR = joinpath(CLIPS_ROOT, "{folder}")  # (2)\n'
               + "".join(f'include(joinpath(CLIPS_DIR, "src", "{name}.jl"))\n' for name in SRC_FILES)
               + "CLIPS_MODE = configure_from_env!()\nRECORD = CLIPS_MODE == :record\n")
    text = replace_once(text, includes, bridged)
    if text.count('joinpath(@__DIR__, "dataset"') < 3:
        raise ValueError("dataset paths not found")
    text = text.replace('joinpath(@__DIR__, "dataset"', 'joinpath(CLIPS_DIR, "dataset"')  # (2)
    text = replace_once(text, 'RESULTS_DIR = joinpath(@__DIR__, "results")',
                        f'RESULTS_DIR = results_dir("{domain}")  # (3)')
    text = replace_once(text, "for plan_id in PLAN_IDS\n    println(\"=== Plan $plan_id ===\")\n",
                        "for plan_id in chosen_plans(PLAN_IDS)  # (4)\n    println(\"=== Plan $plan_id ===\")\n"
                        "    CONTEXT[\"plan_id\"] = plan_id  # (5)\n"
                        + ("    global goals  # (8)\n" if domain == "vh" else ""))
    text = replace_once(text, "        # Infer distribution over commands\n",
                        "        # Infer distribution over commands\n        CONTEXT[\"method\"] = \"literal\"  # (5)\n")
    text = replace_once(text, "        # Set up planners\n",
                        "        if !RECORD  # (6) assistance and results only when replaying\n        # Set up planners\n")
    text = replace_once(text, "        GC.gc()\n        CSV.write(df_path, df)\n    end\n",
                        "        GC.gc()\n        CSV.write(df_path, df)\n        end  # (6)\n    end\n")
    text = replace_once(text, "            pragmatic_inference_results = pragmatic_goal_inference(\n",
                        "            CONTEXT[\"method\"] = \"pragmatic\"  # (5)\n"
                        "            pragmatic_inference_results = pragmatic_goal_inference(\n")
    text = replace_once(text, "            sips_config = pragmatic_inference_results.sips_config\n",
                        "            sips_config = pragmatic_inference_results.sips_config\n"
                        "            RECORD && record_particles(pragmatic_inference_results.pf, utterance_times)  # (7)\n")
    text = replace_once(text, "            CSV.write(inference_df_path, inference_df)\n",
                        "            RECORD || CSV.write(inference_df_path, inference_df)  # (6)\n")
    text = replace_once(text, "            for modalities in MODALITIES\n",
                        "            for modalities in (RECORD ? () : MODALITIES)  # (6)\n")
    return HEADER.format(folder=folder) + text


def copy_loops(clips_root: Path, out: Path) -> list[Path]:
    written = []
    for domain, folder in DOMAINS.items():
        original = (clips_root / folder / "experiments.jl").read_text(encoding="utf-8")
        path = out / f"{domain}_experiments.jl"
        path.write_text(transform(original, domain), encoding="utf-8", newline="\n")
        written.append(path)
    return written
