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
- **Bootstrap.** Slope CIs resample sequences within each k (2,000 replicates, seed
  20261027), so every replicate keeps the design in k.
