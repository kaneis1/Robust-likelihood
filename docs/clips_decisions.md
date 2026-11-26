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
- Estimate from the token count: about 2.5 million tokens per model. A bf16 8B forward pass on one
  H100 handles at least 10,000 tokens per second with padding, so about 4 minutes of scoring per 8B
  model and 1–2 for the smaller ones, plus 1–3 minutes to load each model from NFS. For scale, the
  week-of-24-Nov `uperturb` job scored 24,011 shorter pairs per 8B model in 5–8 minutes. **Under one
  H100-hour for all eight**, well inside the plan's "under one H100-hour per model"; the job's
  manifests record the measured time.
- The plan's shared-prefix cache is not used: the prompts are short and batching is already fast.

## Decided at STOPs 2 and 3 (user, 2026-11-26)

The user approved the record counts and one GPU job scoring all eight models. Before submitting, the
cluster checkout moved from `week-nov24` (with CLIPS files copied in untracked) to `clips-small-lm`;
the 15 untracked copies were older versions of tracked files (11 identical, 4 from before the
prompt-id split) and were removed first. SoC job 1205678 (`sim_job.sh clips`, gpu partition).

## Phase 2 finding: batch layout moves a score by up to about 0.5 nats

The determinism check (200 keys rescored in reversed order at batch size 7) finds a largest gap of
0.47 nats for `qwen3_0p6b`, 0.59 for `qwen3_1p7b` and 0.51 for `smollm2_1p7b`. Scores are bf16 logits
put through a float32 log-softmax; a different batch shape runs different kernels, and the rounding
differences add up over a continuation of up to 64 tokens. In Phase 3 each key is scored once, so this
is fixed noise on every score. In Phase 4 it would leak into every Δ, so each perturbed instruction's
pairs are scored in the same batches as the original's, as the week-of-24-Nov `uperturb` part did.
Later scoring runs also report the median and p90 gap.

## Phase 3: the Table 1 definitions (plan 3; not stated in the plan, settled against the paper)

The plan names the metrics, not their exact form. Each definition below was chosen because it
reproduces **every cell** of the paper's Table 1 (both domains, all methods, the humans row) from the
paper's own result files on OSF; `tests/test_clips.py` checks this whenever the files are present.

- **Methods.** CLIPS = pragmatic inference with modalities "action, utterance"; language-only IP =
  "utterance"; action-only IP = "action". All three use offline QMDP assistance (`qmdp_act`, estim
  mode): section 3.1 of the paper, "we ran Algorithm 2 in offline mode, fixing the inferred posterior".
  The literal listeners use estim mean (literal naive and efficient in mDKG, literal in VH).
- **P(g_true)**, **assistance precision and recall**: means over problems of `true_goal_probs`,
  `assist_precision`, `assist_recall`; the paper's brackets are standard errors over problems.
- **Relative plan length and human cost**: per problem, `plan_length` and `speaker_cost` divided by
  the reference CLIPS's, then the mean over problems. The reference is CLIPS with `qwen3_8b`, as the
  plan says. The human's cost is the `speaker_cost` column; `plan_cost` gives other numbers.
- **Goal correlation** (mDKG): Pearson r over the 30 × 4 (problem, goal) cells of `goal_probs_k`
  against the mean human ratings. **Assistance correlation**: Pearson r over each problem's real
  assistance options (the first `option_count`) against the mean human ratings; including the padded
  zeros gives larger r for the baselines (action-only 0.50 against the published 0.45).
- Intervals: bootstrap over problems, 2,000 replicates, seed 20261126.
- **The setup check** (plan: STOP if action-only IP misses by more than 0.02): action-only P(g_true)
  against 0.31 (mDKG) and 0.25 (VH). The paper's files give 0.314 and 0.250. The check also reports the
  largest per-problem gap against the paper's file.

## Phase 3 runs (2026-11-26)

- **Setup check passes.** `qwen3_8b` replays, SoC CPU jobs 1205800 (mDKG) and 1205801 (VH, 16 minutes).
  Action-only IP gives P(g_true) 0.2499 over the 25 VH problems against the published 0.25, and
  matches the paper's own result file problem by problem (largest gap 0.0; mDKG's first two problems
  4e-10). The planner and the action model are therefore the paper's, and only the language model
  differs.
- A replay runs every assistance policy for every modality: about 7 minutes per mDKG problem (about
  3.5 hours per model) and 16 minutes for all of VH. The other 14 replays (seven models × two domains)
  are SoC jobs 1205867–1205880 on the `long` partition.

## Phase 4 design (2026-11-26, before any perturbed score exists)

- **No new Julia replays.** Pragmatic inference keeps one particle per (goal, cost profile) and never
  resamples, and an instruction enters a particle's trace score only through its TableMixture term. So
  the posterior under a swapped instruction is the replay's final trace score per particle, minus the
  original term, plus the new one (`robustness.py`). Checked on the `qwen3_8b` replays: with the
  original instruction it reproduces the logged P(g_true) to 3e-15 (VH, 35 instructions; mDKG), each
  modality's goal posterior is the particle softmax summed by goal (to 3e-15), and the experiments
  file's posteriors equal the inference file's final step. The true goal's index, which the loops do
  not log, is the k whose `goal_probs_k` equals `true_goal_probs` in all three modalities' rows
  (unique for every problem).
- **Co-batched scores.** Each (instruction, prompt) is one batch holding the original output and every
  valid perturbed output of that instruction; both sides of every Δ use these scores (2,068 batches,
  13,142 sequences before paraphrases). The Phase 2 table enters only as the term subtracted from the
  replay's trace.
- **Protected list.** As the plan lists it, plus the heads of VirtualHome's compound type names (fork,
  knife, glass, board): the instructions say "forks" and "glasses", while the PDDL says `cutleryfork`
  and `waterglass`. Contractions are expanded before comparing ("I'll" counts as "i", "don't" as
  "not"). Fill sentences contain no protected word ("Thank you." would add "you"): openers "Hey.",
  "Hi.", "Quick favour.", "Real quick."; thanks "Thanks!", "Thanks.", "Cheers.".
- **Valid units before paraphrases:** typo1 29/30 mDKG and 19/35 VH, typo2 15 and 11 (short
  instructions have few words of four or more letters that are not protected); fill1, fill2, lower,
  nopunct all 65.
- **Paraphrase prompt** (`perturb.PARAPHRASE`, in rewrites.py's form): "Rewrite this instruction so it
  means exactly the same thing. Keep every colour, object, number and person (me, you) and add
  nothing. Reply with the rewritten text only." Accepted when the protected words are kept and the
  NLI model gives at least 0.8 both ways; up to 3 attempts per unit. 130 units: 131 to 391 calls.
- **How C5 and C6 are read**, fixed now: on the two domains pooled (65 instructions), with the per-domain
  values reported next to them. "Base models of 1.7B or more" are `qwen3_1p7b`, `qwen3_4b`, `qwen3_8b`
  and `smollm2_1p7b`; C6 compares `qwen3_8b_inst` with `qwen3_8b` and `qwen25_7b_inst` with nothing
  (it has no base counterpart in the list), so C6 reads the Qwen3 pair only.

## Phase 4 runs (2026-11-26)

- **Decided at STOP 4 (user):** the paraphrase run and one GPU job scoring the perturbed instructions
  with all eight models.
- **Paraphrases:** run `results/clips/paraphrases/20261126T114927Z` (manifest copied to
  `data/clips/derived/paraphrases_manifest.json`), `claude-fable-5-1` at temperature 1, 153 calls, no
  failure, 11,504 input and 2,242 output tokens. 124 of 130 units accepted (para1 61, para2 63). The 6
  rejected units all broke the protected-word rule on every attempt, e.g. "Grab another plate." →
  "Get one more plate." (adds "one"), "Give me a red key for this door?" → "Could you give me a red key
  for this door?" (adds "you"). 33 accepted para2 rewrites are identical to their para1: short
  instructions leave little room. They are kept as units (the effect is then the same), and C5 reads
  para1 as the plan says.

## Phase 3 replays: a loop fix and plan chunks (2026-11-26)

- Three VH replays (`qwen3_1p7b`, `qwen3_8b_inst`, `qwen25_7b_inst`; jobs 1205870, 1205876, 1205878)
  failed with BoundsError in the literal listener's top-5 printout. Fixed as loop difference (9)
  (`loops.py`); their partial output was moved to `results/clips/<model>/vh-failed-<job>/` and the three
  replays were resubmitted (jobs 1206077–1206079). The VH replays of the other five models were
  unaffected (more than five commands survived for every problem).
- mDKG replays take 15–20 minutes per problem (four literal and eighteen pragmatic assistance runs), so
  a full sequential replay is 8–10 hours. The eight sequential mDKG jobs were cancelled after about an
  hour, and each model's unfinished problems were resubmitted as chunks of four (`clips_job.sh replay
  mdkg <model> <plans>`, results in `plans-<first plan>/` subfolders). A problem counts as finished when
  its last row (modalities "action, utterance", `qmdp_plan`, estim mean) is written; problems finished
  before the cancellation keep their rows, and the evaluation keeps, per problem, the rows of the
  newest file. The cluster's limit of 32 submitted jobs per user refused 28 of the 56 chunks; a feeder
  on the login node (`~/feed_chunks.sh`, under nohup) submits them as slots free up.
- **Perturbed scores:** SoC GPU job 1206071 (one H100 NVL MIG 3g.47gb slice, as every GPU job here),
  57 minutes for the eight models (2 to 11 minutes each; small batches use the slice less well than
  Phase 2), 15,344 sequences per model, none over 64 tokens. The manifests name commit 0812419 for
  `qwen3_0p6b` and be05574 for the rest: the cluster checkout was fast-forwarded while the job ran,
  and nothing between the two touches the scoring code.
- **GPU naming.** `gpu:h100-47` on the SoC cluster is a MIG slice (3g.47gb: 47 GB, about 3/7 of an H100
  NVL's compute), not a whole H100; "one H100" earlier in this file means that slice.
