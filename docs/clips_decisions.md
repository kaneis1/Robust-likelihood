# CLIPS with small local language models: decisions

The plan is `docs/clips_small_lm_plan.md`. Changes to **[Frozen]** items are recorded here with the
reason, and only after the user agrees. Interpretations of points the plan leaves open are recorded too,
so the analysis can be audited.

## Phase 0: setup (2026-11-26)

- **Branch** `clips-small-lm`, started in a separate worktree while the week-of-24-Nov GPU jobs ran, and
  based on `origin/week-nov24` once that branch was pushed with its results.
- **CLIPS.jl** is cloned at `6536a2a86d6b4837481a00e630a2e302c90a5278` into `external/CLIPS.jl`, which
  `.gitignore` excludes and nothing edits; every job checks that its `git status` is clean.
- **The Julia environment** is `julia/clips/Project.toml`, not CLIPS.jl's own: instantiating in
  `external/` would write a Manifest there. Its `[deps]` and `[compat]` are CLIPS.jl's pins, less the
  three plotting packages the experiment scripts never load (CairoMakie, GLMakie, PDDLViz; GLMakie needs
  a display). `julia/clips/Manifest.toml` records the resolution: Gen 0.4.6, PDDL 0.2.17,
  SymbolicPlanners 0.1.23, InversePlanning 0.0.1 from `cosilab/InversePlanning.jl#release-0.0.1`,
  GenGPT3 0.2.1, GenParticleFilters 0.2.3, on Julia 1.10.7 (CLIPS needs ≥ 1.6).
- **Where it runs.** The SoC cluster has no Julia and no module system. `scripts/clips_setup.sh` (a CPU
  job, 10 minutes) installs Julia and the environment on a compute node's local disk and packs them into
  `~/envs/clips-julia.tar`, which `scripts/clips_job.sh` unpacks on its node, as the Python environment
  is handled: the home directory is NFS, slow for many small files, and the login node caps memory at
  1 GB, too little to precompile.
- **Human data are available**, so the Phase 0 STOP does not apply. The OSF project (osf.io/v8ru7) is
  public; the setup job downloads its `datasets` and `results` folders into `data/clips/raw/`
  (ignored by git), among them `results/doors-keys-gems/mean_human_responses.csv` and
  `all_human_responses.csv`, and the paper's own result files (`pragmatic_experiments.csv`,
  `literal_experiments.csv` for both domains).
- **`utterances.jsonl`**: 30 mDKG instructions (one per problem) and 35 VirtualHome instructions over 25
  problems, 10 of which have more than one, as the plan says. A time step is the number of observed
  actions before the instruction, as CLIPS's `parse_plan` counts it. mDKG rows also carry the simplified
  utterance the literal listener reads (`dataset/utterances/literal`).

## The table mixture (plan 2.2)

- `julia/clips/table_mixture.jl` defines `TableMixture(role; temperature)`, a Gen generative function
  whose trace has GenGPT3's `GPT3MixtureTrace` fields and addresses (`prompts`, `probs`, `scores`,
  `score`, `:post_probs`, …), because CLIPS reads `utt_trace.prompts` and `utt_trace.scores` directly.
  Record mode logs each (role, prompt, output) once and scores 0; replay mode raises an error naming
  any missing key.
- **How the mixtures are swapped.** `julia/clips/bridge.jl` defines `GPT3Mixture` in `Main` before any
  CLIPS source file is included. When `src/utterances.jl` runs `using GenGPT3` and then
  `GPT3Mixture(model=…)`, Julia keeps `Main`'s binding (it prints "using GenGPT3.GPT3Mixture in module
  Main conflicts with an existing identifier"), so each mixture CLIPS defines is a `TableMixture`:
  curie → `mdkg_pragmatic`, text-davinci-002 → `mdkg_literal`, davinci-002 at τ 1.0 → `vh_pragmatic`,
  davinci-002 at τ 0.5 → `vh_literal`. The effect is the plan's rebinding of the two globals, with
  CLIPS's source files unchanged.
- `simulate` raises an error: there is no language model to sample from, and CLIPS only scores
  observed utterances. Belief updates during assistance constrain `speak` to false, so it is never asked
  to sample.
- The observed output carries CLIPS's leading space: both `literal_command_inference` and
  `pragmatic_goal_inference` prepend " " to an utterance that lacks one before constraining `:output`.

## The experiment loops are copied (plan 2.2's fallback)

CLIPS's `experiments.jl` writes into `@__DIR__/results`, inside `external/`, so it cannot run in place.
`python -m robust_likelihood.clips copy-loops` writes `julia/clips/{mdkg,vh}_experiments.jl` from the
pinned originals by exact replacements, each asserted to apply once; `tests/test_clips.py` checks that
the committed copies regenerate exactly. The differences, as listed in `src/robust_likelihood/clips/loops.py`:

1. `bridge.jl` is included before CLIPS's source files; the mode comes from the environment.
2. Source and dataset are read from `external/CLIPS.jl/<domain>/`.
3. Results go to `$CLIPS_RESULTS` (default `results/clips/unnamed/<domain>/`).
4. `$CLIPS_PLANS` can name a subset of plans (smoke runs).
5. The record log notes the plan and method of each language-model call.
6. In record mode every score is 0, so the assistance planning after inference (no language model) and
   the result files are skipped. Pragmatic inference uses one particle per (goal, cost profile) with no
   resampling or rejuvenation, so the prompts it asks for do not depend on the scores: a record pass
   asks for exactly the keys a replay needs.
7. In record mode, each particle's goal, cost profile and utterance prompts at every utterance time are
   logged (`record_particles`), because the language model sees only prompts.
8. VirtualHome only: `global goals` at the top of the plan loop. `virtual-home/src/inference.jl`'s verbose
   printer reads a global `goals`, which the loop assigns; in the REPL that makes a global, but in a
   script it makes a loop-local variable and the printer fails (found in the VH smoke run).

## Smoke test (plan Phase 0, step 2)

- `testbed.jl` hardcodes its plan (`1.1.keys` for mDKG, `2.5` for VirtualHome), so the smoke test runs
  the copied loops in record mode for the plan's two problems instead: mDKG `1.1.keys` and VirtualHome
  `1.1` (the plan's "1.1.stew_chicken": VirtualHome plan ids are the first two fields of the file name,
  the third is the true goal).
- SoC jobs 1201728 and 1201729, 1.3 and 1.5 minutes each, most of it Julia's compilation: 73 keys for
  `1.1.keys` (62 literal, 11 pragmatic; 8 particles, 4 goals × 2 cost profiles) and 80 for VH `1.1`
  (54 literal, 26 pragmatic; 6 particles). Both pass the record-log check (`pairs --smoke`).

## Models (plan 2.4), checked on the Hugging Face Hub on 2026-11-26

Revisions are pinned in `configs/clips.json`. The four 8B-class models are the simulator study's own
revisions, already on the cluster; Qwen3-1.7B-Base, Qwen3-4B-Base and SmolLM2-1.7B are downloaded on the
login node (`python -m robust_likelihood.clips download`, Xet disabled because its Rust client exceeds
the login node's memory cap).
All seven models' snapshots are complete in the cluster's Hugging Face cache at the pinned revisions
(checked 2026-11-26); `osim_mid` is the simulator study's checkpoint. The CPU download job (1201741)
failed because the Python environment exists only on the login node and GPU nodes, so the downloads
ran on the login node instead.

## Phase 1: the record pass (2026-11-26, STOP 2)

SoC CPU jobs 1201735 (mDKG, 35 min 47 s) and 1201736 (VirtualHome, 4 min 4 s), commit `760aecc`
(rebased as `cf8f4f0`). Most of mDKG's time is the literal listener, which asks every lifted command
of every step.

| | mDKG | VirtualHome |
|---|---|---|
| Problems | 30 | 25 |
| Goals per problem | 4 | 6–12 (mean 10.1) |
| Commands per goal | mean 4.1, max 10 | mean 12.1, max 52 |
| Unique prompts | 410 | 332 |
| Unique (role, prompt, output) keys | 4,247 (3,974 literal, 273 pragmatic) | 3,899 (2,146 literal, 1,753 pragmatic) |
| Check (particle prompts = pragmatic keys; prompts distinct; weights sum to 1) | passes | passes |

- The check compares the prompts each particle's utterance model built from its own plan
  (`enumerate_commands`, logged by `record_particles`) with the pragmatic keys the record log holds,
  in both directions; there are no mismatches, no duplicate prompts within a particle, and every
  particle's command weights sum to 1.
- Scores do not depend on the role, so Phase 2 scores **7,295 distinct (prompt, output) pairs** per
  model. Prompt plus output: mean 1,326 characters, max 2,492, 9.7 million in all (about 2.5 million
  tokens per model).

## Phase 2 estimate (STOP 3)

- One GPU job (`gpu` partition, one H100) scores the models in turn, bf16, batch 32, plus the
  200-key determinism check per model.
- Measured reference: the week-of-24-Nov `uperturb` job scored 24,011 pairs of similar total length
  with each 8B model in 5–8 minutes. Estimate here: 3–8 minutes of scoring per 8B model, 1–3 for the
  smaller ones, plus 1–3 minutes to load each model from NFS; **under one H100-hour for all eight**,
  well inside the plan's "under one H100-hour per model".
- The plan's shared-prefix cache is not used: the prompts are short and batching is already fast.
