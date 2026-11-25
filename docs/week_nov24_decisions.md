# Week of 24 Nov: decisions

The plan is `docs/week_nov24_plan.md`. Changes to **[Frozen]** items are recorded here with the
reason, and only after the user agrees. Interpretations of points the plan leaves open are recorded
too, so the analysis can be audited.

## Environment and checks

- Branch `week-nov24`, as pushed: `origin/simulator-study` (`82d42d6`) plus the typo-suite commit
  (`33b3280`) and the plan (`02ff33a`). Locally torch is installed, so the 9 tests the plan counts as
  skipped run: 260 passed at the start.
- **Appendix A counts all match**: 235 / 2,355 / 463; 2,233 rows, 2,058 valid; every per-kind count;
  20,637 scoring pairs; eligible typo words 3 / 44 / 60 / 41 / 30 / 57; 6 natural "restaurant" typos;
  9,084 (h, h′) pairs, 290,688 cross pairs, 29,632 matched pairs. `perturbations.jsonl` regenerates
  byte for byte from `perturb.build` with the committed lexicon and seed 20261124.
- **One difference from the plan's text (§2.1, not Appendix A):** the typo1 op counts in the frozen
  file are delete 57, double 60, neighbour 54, swap 64; the plan says delete 64, double 58,
  neighbor 54, swap 59. The frozen file is kept.

## Interpretations

### Part A

- **A2 `terse` prompt.** "Rewrite this message as the shortest message a real user might type with the
  same preferences. Keep every detail (cuisine, area, price, people, day, time) and every statement of
  no preference, and add nothing. Reply with the rewritten text only." The detail list is the frozen
  paraphrase prompt's: the frozen NLI check runs both ways, so a rewrite that drops a booking detail
  fails it. `terse` is an opt-in kind of `rewrites.py`; its accepted rewrites go to
  `data/multiwoz/derived/terse.jsonl`, and `rewrites.jsonl` and the claims run are unchanged.
- **A3 `fixed`.** The six turns (SNG0458, SNG0472, SNG0494, SNG0524, SNG0575, SNG0580) misspell
  "restaurant" as "restuarant" or "resturant"; the unit corrects it. Here the human turn is the
  original, so a unit's delta is corrected minus misspelled. `data/multiwoz/derived/fixed.jsonl`.
- **Co-batching.** Each unit is one batch: the item's original turn and the unit's text under every
  clean hypothesis of the item (9–11 texts, so 18–22 sequences, not 32). Every delta is taken
  against the original scored in that batch (`S_orig`), never against Phase 2's human score.
- **Delta log p of a unit** is the median over the item's hypothesis texts of `S_orig − S`. Per typo:
  divided by the dose. For fillers and `terse`, also per added token (median over texts of the
  delta divided by the change in scored tokens).
- **Inference effect.** log BF over `claim1.bf_pairs` (size-principle pairs in both regimes, broad,
  contradiction), l = −S. A sign change is bf0 · bf1 < 0. Posteriors are claim1's softmax over
  `posterior_classes` (G|impl, minus:X|impl, broad|impl, ctrl:contra).
- **Dose response.** One point per unit: its dose and its mean |Delta log BF| over its pairs (or its
  Delta log p). Least squares over all units of the family (typo 1/2/4, fill 1/2/4); item bootstrap.
- **Slot typos.** The slot hit is the one whose value span holds the edited word. The ratio is
  log BF after ÷ before for (G|impl, minus:X|impl), median over units with log BF before > 1.
- **Typicality under Part A.** Each model's own Phase 2 samples are the reference (pool statistics
  as in the simulator study). The original and the unit are both the co-batched scores; `cent`
  subtracts the same batch's broad|impl score.
- **Predictions.** T2's "larger" means each instruct model's median is above the largest human-like
  median. T5's "each added sentence" means the median Delta log p ÷ number of added sentences is at
  most −5 at every fill dose, for every model.

### Part C

- The pooled sample distribution is the mixture with one weight per text (uniform), each text's
  samples sharing it equally, which is the mixture whose moments give glob's H and sigma. A weighted
  quantile is the smallest value whose cumulative weight reaches q; a value on an edge goes up, as
  `typv0.bin_index` does. The top bin's c_b is the pooled 0.995 quantile in the sensitivity (0.50,
  0.80, 0.90) as well.
- "Size principle kept" is mean L_J ÷ L_E for named single-slot pairs, implicit regime, as the
  simulator summary reports it; the explicit regime is reported next to it.
- **D4's reference.** The 86.4% for `cent1B` with `qwen3_base` is bin agreement over both paraphrase
  kinds (para1 and para2); under para1 alone it is 87.3%. Both are reported.

### Runs

- Runs for this plan have the manifest kind `sim-week` and folders `<UTC>-simwk-<model>-<format>`, so
  `sim_analyze.discover` still finds each model's Phase 2 run as its newest `sim-score` run.
- `xscore` excludes samples that were cut off or are empty: 289,514 of the 290,688 cross pairs.
- A tokenizer-only check on the login node found no boundary failures for any of the six models on
  any `uperturb` pair, or for `qwen3_base` on any `xscore` pair.

## Decided at STOPs 1 and 2 (user, 2026-11-26)

- **`terse` API call approved** (stated as 236 first-attempt calls, at most 706, about $0.70).
- **All three GPU jobs approved**: `uperturb` for the six models, `hsample` and `xscore` for
  `qwen3_base`·F (stated as about 5.3 GPU-hours).

## Smoke runs (2026-11-26, about 7 GPU-minutes, before STOP 2)

- SoC jobs 1200907 and 1200908, H100 NVL MIG 3g.47gb, commit `620164b`, `qwen3_base`·F.
- `uperturb` on the first 3 clean items: 286 pairs (572 sequences) in 19 s, about 30 sequences/s.
  97% of the co-batched originals equal Phase 2's human score exactly (largest gap 0.48 nats).
- `xscore` on the same items: 10,412 pairs in 286 s, 36 pairs/s.
- `hsample`, 2 reworded hypotheses × 64 samples: 16.7 s, 8.4 s per hypothesis.
- The `xscore` job logged CUDA allocator retries ("allocation failed with OOM", recovered); it
  finished normally.

## The `terse` run (2026-11-26)

- The first attempt (`results/multiwoz/rewrites/20261125T191957Z-terse`) crashed at its first NLI
  batch: the local GPU's architecture is not supported by the JIT of the local torch (2.7.1+cu118),
  which DeBERTa's relative-position code triggers. About 236 calls were spent; candidates are
  written only after the NLI, so nothing was kept.
- The rerun (`…/20261125T192143Z-terse`) ran the NLI on the CPU: 556 calls, 59,262 input and
  7,079 output tokens, `claude-fable-5-1`, temperature 1.0 sent. **In total 792 calls, about $1.30,
  above the 706 calls and $0.70 stated at STOP 1.**
- The NLI ran locally with transformers 4.56.2 (the cluster has 4.57.6), at the pinned revision; its
  fast tokenizer loads from `tokenizer.json`, without sentencepiece.
- **Only 101 of 235 units are accepted (43%).** Of 555 candidates: 342 fail NLI only backward
  (rewrite ⇒ original), 75 both ways, 20 forward only, 17 the regex check. A telegraphic rewrite
  ("thai restaurant west") does not entail the full sentence for the NLI model, so the frozen
  check keeps the less terse rewrites: accepted rewrites have a median 0.44 of the original's words.
- **Sensitivity set `terse_rx`** (not in the plan; outside the predictions): per unit, the first
  candidate of the same run that passes the regex check, whatever its NLI, as the frozen `pad`
  rule does. 230 units, median 0.35 of the original's words. Reported apart from `terse`.

## GPU jobs (2026-11-26)

| Job | What | Partition |
|---|---|---|
| 1201097 | `uperturb`, six models in turn, 2,395 units (24,011 pairs per model) | gpu |
| 1201013 | `hsample`, `qwen3_base`·F, 894 reworded hypotheses × 64 | gpu-long |
| 1201014 | `xscore`, `qwen3_base`·F, 289,514 pairs | gpu-long |
