# MultiWOZ size-principle decisions

The plan is `docs/multiwoz_size_principle_plan.md`. Changes to **[Frozen]** items are
recorded here with the reason, and only after the user agrees. Interpretations of
points the plan leaves open are recorded too, so the analysis can be audited.

## Environment

- The macOS machine had only the system Python 3.9. The repo needs 3.11 or later, so a
  Python 3.12 virtual environment was made with `uv` at `~/.venvs/robust-likelihood`,
  outside OneDrive, and the package was installed editable into it.
- Existing CRLF files keep CRLF when edited (`pyproject.toml`, `.gitignore`). New files
  are LF, as in the formal typicality commit. The copied prompt packs keep the CRLF
  bytes of their sources.

## Phase 2: one Appendix A1 cell differs

`prepare` reproduces every A1 count except one. Clean set, food, "No-pref named":
Appendix A1 says 2; the frozen status rule gives 1.

- The second clean no-preference item is SNG0512, "Hi! I'm looking for a cheap restaurant
  in the north, any type!" Its turn-0 food value is `["dontcare"]`. The frozen rule makes
  that `dontcare`, not `named`, so it is reported as `no_pref_dontcare: 1`.
- The same rule reproduces every pooled A2 count exactly: 3,927 dialogues; food 1,809/3,108
  and 5/819; area 1,730/3,104 and 12/823; price 1,758/3,167 and 10/760; excluding WOZ,
  3/567, 8/559, and 8/498. Counting `dontcare` as named would instead give no-preference
  counts of 11/819, 36/823, and 26/760, which contradict A2 and A3.
- The primary reference does not use this cell: `a` comes from constrained clean items and
  `b` from the pooled population. Only the V2 sensitivity variant (both rates from
  `match_all_turn0`) uses it: food `b` is 1.5/63 under the frozen rule, 2.5/63 otherwise.

Decision, 26 October 2026: the user chose to keep the frozen rule. The A1 check in
`corpus.APPENDIX_A1` now expects food `no_pref_named: 1` and `no_pref_dontcare: 1`. The
plan document keeps its original table.

## Phase 3: where the bootstrap replicates live

The 2,000 replicates of every population are stored as integer counts
(`named, constrained, no_pref_named, no_pref` per slot) in
`data/multiwoz/derived/reference_replicates.json`, written compactly (about 400 KB) so
`reference_table.json` stays readable. `reference_table.json` records that file's sha256.
Each population's resample is drawn from
`random.Random(int(sha256(f"{seed}:{population}").hexdigest()[:16], 16))` over rows in
dialogue-id order, so Phase 6 can redraw the clean-item resample of replicate b and pair
Jev's scores with the a-rates of the same resampled items.

## Phase 4: points the plan leaves open

- Keys use the slot names: `minus:pricerange|impl`, not the plan example's `minus:price`.
  Figures and tables label that slot "price".
- `hyp_type` is the key without its regime (`G`, `minus:food`, `broad`, `ctrl:contra`, ...).
  Controls are rendered with `base()`, so their regime is `impl`.
- Probe rows carry two fields beyond the plan's list: `values`, the slot values the text
  states, and, on controls, `changed` (slot, old value, new value). Both make the
  one-slot-difference check auditable without the raw data.
- For the 11 items with no named constrained slot, `ctrl:tie2` uses its own seed
  (`{item_id}:tie2`) but skips the slot and value that `ctrl:tie` already chose, so the
  second tie control is never a copy of the first. All 11 `tie2` texts differ from `tie`.
- No goal in the corpus has food `dontcare`, so the food pool is 91 values.
- The probe has 3,727 rows (clean 2,355; fail_info 1,372) and 707 distinct non-control texts.

## Phase 5: running Jev

- The 25 October pilot left no files in the repository, so `pilot_report.md` reconstructs its
  layout from the plan: per removal and wording, Jev's logit gap beside the corpus log-ratio.
  `run --label pilot` writes the report into the run folder.
- Calls are planned item by item, both wordings together, so a stopped run leaves whole items.
  `ModelClient` and `runner.execute` are used unchanged; a small router hands each call the
  prompt pack its `prompt_version` names.
- Three items share the utterance "I am looking for an expensive restaurant in the centre",
  so 24 of the 7,454 planned calls repeat an earlier (utterance, text) pair. Each success is
  added to the in-memory cache as it arrives, so those 24 are served from it.
- The cache key names the prompt version but not its text. A cached record is reused only if
  its stored prompt text equals what the pack renders now.
- `--repeats R` keeps the repo's meaning: the baseline (repeat id none) plus repeats 1 to R.
  The repeat check's baselines come from the full run's cache, so it makes about 900 new calls.
- `--seed` defaults to the configured pilot seed when `--items` is given. Resuming re-reads
  the selection from the run's manifest and skips (item, hypothesis, wording, repeat) tuples
  that already succeeded. The model-id check counts successful records only.

## Phase 6: metric details the plan leaves open

- For a full removal, L_E differs between items (which slots, named or omitted), so a group's
  OG is mean(L_J) − mean(L_E) over its items. Single-slot groups have one L_E.
- "Mean over slots" (P1, P2) is the average of the three slot-level OGs, recomputed in every
  bootstrap replicate.
- The plan names the wording only for P1. P2 to P5 are scored on `multiwoz_likely` as well;
  `typical` is reported beside them.
- P3, "full removal with all slots named": every constrained slot named. The clean set has no
  three-slot item with all three named, so this is the 106 two-slot items naming both.
- P4 pools every same-breadth pair (`G` for two-slot goals, each `minus:X`, `broad`).
- P5 uses `ctrl:tie`. The 11 `ctrl:tie2` controls are reported separately and pooled.
- P6 averages the named single-slot mean L over slots and both wordings. "|L| ≤ 0.5 or
  negative" is read as L ≤ 0.5.
- Monotonicity compares scores on the logit scale; clipping can only add ties, and the tie
  count is reported.
- Per-hypothesis bias centres each item's scores on the mean of all its scored rows,
  controls included.
- Bootstrap: when every clean item is analysed, replicate b redraws exactly the clean-item
  resample the reference used for a (`pairing_verified` checks the first 25 replicates
  against the stored counts), and b comes from pooled replicate b. Other item sets are
  resampled with their own generator.
- Figure bins: within each |C|, groups are taken in increasing k_named and pooled until a
  bin has 15 items; a short last bin joins the previous one. Clean bins: 2 slots, 0–1
  named (18 items); 2 slots, 2 named (106); 3 slots, 0–1 named (15); 3 slots, 2 named (96).

## Phase 8: the language model

- One explicit device (`cuda`, then `mps`, then `cpu`, or `--device`) instead of
  `device_map="auto"`: the 7B model fits on one GPU, and accelerate's dispatch crashed with
  a segmentation fault on the macOS test machine.
- The revision is resolved to a commit hash before loading (the snapshot folder name), the
  model and tokenizer are loaded at that hash, and the hash goes in the manifest.
  transformers 5 no longer sets `config._commit_hash`.
- Sampling settings go to `generate` only through one `GenerationConfig` copy, and
  `model.generation_config` is replaced too, so Qwen's defaults cannot be merged back in.
  Two self-checks run before any scoring: the processed scores of a short generation must
  equal the raw logits, and tokenizing prefix and reply apart must give the same ids as
  tokenizing the whole conversation.
- σ_h is the sample standard deviation (n − 1). The bin smoothing uses the kept samples:
  (count + 1) / (K_kept + 3). Every sample's text and surprisal is saved in
  `lm_samples.jsonl`.
- A local smoke test (Qwen2.5-0.5B-Instruct on CPU and MPS, 3 texts, 8 samples each)
  passed both self-checks, dropped no samples, and recorded the revision hash. It is not
  the Phase 8 result.

## Phase 9: figures

- Colours are slots 1 to 3 of the reference categorical palette, the set documented to pass
  every pairwise colour-vision check. Node was not available to rerun the validator script.
  Wording is carried by marker shape (filled circle explicit, open square implicit), and the
  CSV beside each figure is its table view.
- The y = x diagonal is dashed as the plan asks. The zero lines are solid hairlines, like the
  axes. Figures are drawn for print on a light background.
