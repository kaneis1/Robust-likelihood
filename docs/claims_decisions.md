# Claims plan decisions

The plan is `docs/claims_plan.md`. Changes to **[Frozen]** items are recorded here with
the reason, and only after the user agrees. Interpretations of points the plan leaves
open are recorded too, so the analysis can be audited.

## Environment

- Branch `claims-123` from `origin/multiwoz-size-principle` at `1908d74` (Phase 8 is not on
  `main` yet).
- Local machine: Windows, Python 3.14, no torch. GPU work runs on the NUS SoC compute
  cluster, which schedules with Slurm, so the job scripts carry `#SBATCH` directives next
  to the LSF and PBS placeholders the plan asks for.
- **SoC cluster environment.** The home directory is NFS: large files write fast, but the
  many small files of a torch install take hours. Login-node `/tmp` has a 50 MB per-user
  quota and login processes are capped at 1 GB of virtual memory (torch cannot be
  imported there). So the environment is built in `/var/tmp/cuizq-rl` on the login node
  (torch 2.14.1+cu126, transformers 4.57.6, accelerate 1.15.0, huggingface_hub 0.36.2;
  `~/envs/cuizq-rl.freeze.txt`), packed into `~/envs/cuizq-rl.tar`, and unpacked onto the
  compute node's local disk by each job. No job runs `pip install`. transformers is held
  below 5. The code reaches the cluster as a git bundle of `claims-123`, so run manifests
  record real commit hashes. Models are downloaded on the login node into
  `~/.cache/huggingface`, and jobs run with `HF_HUB_OFFLINE=1`.
- The `lm` extra now needs `transformers>=4.47`: `Olmo2ForCausalLM` first ships in 4.47.0.
- Model ids were checked on the Hugging Face Hub on 2026-10-29. All three are public and
  ungated. Revisions at that date, pinned in `configs/dice.json` for step 1:
  - `Qwen/Qwen2.5-7B-Instruct` `a09a35458c702b33eeacc393d103063234e8bc28`
  - `allenai/OLMo-2-1124-7B-Instruct` `470b1fba1ae01581f270116362ee4aa1b97f4c84`
  - `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`

## Shared refactor

- `multiwoz/lm.py` keeps its prompt and thin wrappers (`prefix_ids`, `end_of_turn_id`,
  `boundary_holds` take a hypothesis text there); everything else moved to `lmcore.py`
  unchanged, and its tests pass unchanged.
- The plan says "as in `lm_probs.py`"; no such file exists. Decoding follows
  `multiwoz/lm.py`: every option is set explicitly **and** installed as the model's own
  `generation_config`. Recent transformers versions fill options left at their global
  default from the model's config, which for Qwen would add a repetition penalty of 1.05
  even to greedy decoding. Greedy runs are checked the same way sampling is: the
  processed scores of a short generation must equal the raw logits.
- `parse_probability` reads a whole answer that is one of `0.3`, `30%`, `3e-5`, `1/6`.
  Otherwise it reads the answer only if it has no power notation and all its numbers in
  [0, 1] agree within 5% (so "1/216 ≈ 0.00463" reads, "(1/6)^3" does not). Raw text is
  always kept.

## Step 1: dice

- The frozen S1 prompt reads "Roll it 1 times" at k = 1, and S2 reads "…each equally often
  1 times" for the even die. The prompts are kept verbatim.
- **API temperature.** `claude-fable-5-1` rejects `temperature` (thinking is always on),
  so Claude is called with no temperature, which the plan allows ("where supported"). The
  code sends temperature 0 first and drops it only after a 400 that names it; the
  manifest records what was sent. Claude's server-side refusal fallback is **not**
  enabled: it would answer some queries with a different model. A refusal is recorded as
  a failed call.
- **Pilot (2026-10-29, `results/dice/20261029T075735Z-api-pilot`).** 30 seeded pairs × S2, S3 ×
  {claude, gpt} = 120 calls, all successful. `gpt-6-astra` also rejected `temperature`, so
  neither provider was sent one. Claude shows its working before the number on most
  answers, so `parse_probability` now also reads a last line that is a bare number, or a
  final "= x" / "≈ x"; that is the model's own stated answer. With it, 0 of 120 pilot
  answers fail to parse (40 of Claude's 60 did before). `analyze` re-parses every raw
  answer, so all runs are read by the same parser.
- **API normalization queries.** The API runs also ask the 54 normalization queries (18 at
  k = 1, 36 at k = 2) under both framings, so D3 and D4 can be read for the API models.
  That is 216 calls beyond the 630-pair subset.
- **Scores of exactly 0** and unparsed answers have no log score. They are counted and
  left out of slope and Bayes-factor fits, never imputed; normalization sums add the
  parsed answers and report the number missing.
- **Operational definitions of the predictions.**
  - D1: OLS slope of log10 score on k over `fair` sequences scored under `fair`; pass if
    slope ÷ true slope ∈ [0.5, 2] and R² > 0.9.
  - D2: the same slope for S2; pass if |slope| < 0.15.
  - D3: mean |verbalized − true| over the 18 k = 1 normalization queries ≤ 0.10. The max
    error and the share within ±0.10 are reported next to it.
  - D4: sum of S2 over the 36 k = 2 `fair` sequences > 2.
- **S1 format sensitivity (agreed with the user, 2026-10-29).** Qwen writes dice lists as
  `3,4`; the frozen observation `3, 5, 2` costs it 13–16 nats once at every k ≥ 2. A
  secondary scorer `S1-nospace` scores the same sequences as `3,5,2`. It is reported next
  to S1 and is not part of the D1 scorecard; the frozen S1 results are unchanged.
- **Bootstrap.** Slope CIs resample sequences within each k (2,000 replicates, seed
  20261027), so every replicate keeps the design in k.

## Step 2: MultiWOZ rewrites

- **2a check.** The extractor reads "no preference" lists ("no preference about the type of
  food, the area, or the price range") by consuming only the nearest slot phrase, and is
  checked on every distinct probe hypothesis text as well as on the 370 utterances.
- **Rewrite pilot (2026-10-29, `results/multiwoz/rewrites/20261029T085727Z-pilot`).** 20 seeded
  clean items and their 128 hypothesis texts, 1,093 Claude calls. Claude accepted
  temperature 1.0 here (it rejected 0 in the dice runs), so the plan's temperature 1 was
  sent. Two systematic failures, both changed with the user's agreement:
  - **[Frozen] synonym list extended.** The regex check rejected most genuine hypothesis
    paraphrases (accepted 66–73% after three attempts, biased toward template wording)
    because Claude writes "northern part of town", "downtown", "high-priced", and "does not
    have a preference regarding…". Added: northern/southern/eastern/western, downtown;
    low-priced/low-cost → cheap; mid-priced/moderately-priced → moderate; high-priced,
    pricy, costly → expensive; "does not have a (particular) preference", "has no
    particular preference", "is flexible/open/not particular" as no-preference phrasings.
    Removed "anywhere" as an any-area phrase (not in the frozen list; it read "Know anywhere
    that has european food?" as area dontcare). On the pilot candidates the regex pass rate
    goes from about 40% to about 97% for hypothesis paraphrases and to 100% for utterance
    rewrites; 2.2 agreement is food 98.9%, area 99.2% (from 98.6%), price 97.0%, and every
    probe hypothesis text still extracts exactly.
  - **[Frozen] NLI check not applied to `pad`.** The original never entails the added
    greeting, so original → padded entailment was 0.02–0.3 and only 60% of pads survived,
    biased toward the openers NLI tolerates. `pad` is meaning-preserving by construction
    (fixed neutral opener and closer, message untouched) and is checked by regex only.
    Every other kind keeps both checks.
- Full-run estimate from the pilot: about 5,200 calls, about $13; approved.
- **Full rewrite run (2026-10-29, `results/multiwoz/rewrites/20261029T103256Z-full`).** 3,262
  units, 3,142 accepted (96%), 3,933 calls, 314,493 input and 127,399 output tokens (about
  $9.50). verbose and zh are the lowest at 86%. Five items (SNG01720, SNG0465, SNG0552,
  SNG0561, SNG0748) lose most utterance rewrites because the regex reading of the original
  disagrees with its 2.2 annotation, the reference the frozen check uses. One accepted
  para2 (SNG02148) repeats the original word for word; it is kept, as the frozen checks do
  not reject it. Accepted rewrites: `data/multiwoz/derived/rewrites.jsonl`; the seeded
  60-item robustness subset: `data/multiwoz/derived/robustness_subset.json`.
- **Rewrite audit (2026-11-01, `docs/claim1_results/rewrite_audit.csv`).** The user judged
  99 of the 100 sampled rewrites meaning-preserving (50 of 50 hypothesis-side, 49 of 50
  utterance-side), above the plan's 95%, so step 2 goes on. The one failure,
  `u:SNG0606:typo`, misspells "price" as "rice" ("in the moderate rice rnage"). The typo
  generator protects the slot values the extractor reads, as the plan says, but not cue
  words such as "price", "range", "food", and "town": 104 of the 229 accepted typo rewrites
  misspell one, almost always into a non-word, and SNG0606 is the only one found where the
  misspelling is another word that changes the reading. With the user's agreement that one
  rewrite is left out of `data/multiwoz/derived/rewrites.jsonl` (`AUDIT_REJECTED` in
  `multiwoz/rewrites.py`, so a rerun of `write_derived` keeps it out): 3,141 rewrites, 228
  of them typo. It stays in the run's `accepted.jsonl`, and the typo rule is unchanged.
- **GPU job A (2026-11-01, SoC jobs 1054843 `phase8` and 1054844 `rewrites`, approved by the
  user).** Both ran on `xgpi18` (H100 NVL, MIG 3g.47gb) at commit `a7a1a52` and completed in
  28 and 31 minutes, about 1 GPU-hour against the plan's 4–8. Outputs, copied to `results/`:
  - Phase 8, `results/multiwoz/20261101T091346Z-phase8`: all 3,727 probe rows; 983 texts × 64
    samples, 3 dropped at the token cap; `sampling_is_exact` and `boundary_holds` both true.
  - Centering, `results/multiwoz/claim1/20261101T093512Z-center-qwen`: 63,279 scores (the 370
    utterances and the 62,909 samples under `broad|impl`).
  - Qwen, `…/claim1/20261101T094135Z-rewrites-qwen`: 13,197 utterance-rewrite and 8,678
    hypothesis-rewrite scores; 891 rewritten texts × 64 = 57,024 samples.
  - OLMo, `…/claim1/20261101T100831Z-rewrites-olmo`: 13,197 and 8,678 scores.

  Every planned row is present, with no missing or non-finite score. Both logs carry PyTorch
  allocator "allocation failed with OOM" warnings; the code catches no out-of-memory error, so
  these are allocations the allocator retried after freeing its cache, not lost work.
