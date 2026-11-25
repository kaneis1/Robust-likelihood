# Plan: is the simulator ranking robust? Few-shot controls and a replication on 675 new dialogues

Execution plan for Claude Code, repository `kaneis1/Robust-likelihood`. Save as `docs/robustness_plan.md`. Run this before `docs/goal_inference_plan.md`.

## 0. Purpose

The simulator study ranked Qwen3-8B-Base with format F first on coverage, fidelity and robustness. Two things make that ranking weaker than it looks:

1. **The base model is the only model that sees examples.** Format F shows it eight real MultiWOZ first turns. OSim-8B-Mid, UserLM-8B and the instruct models get none. The lead could belong to the examples, not to the model.
2. **Everything rests on 235 dialogues, one draw of eight examples, and one sampling seed.** Several choices (the recommended typicality variant, the recommended reference) were made after looking at those 235.

More dialogues fix the second problem and do nothing for the first, so this plan does both:

- **Part A, few-shot controls** on the original 235 dialogues: vary the number of examples, vary which examples, give the same examples to the chat models, and try examples from another domain.
- **Part B, replication** on 675 restaurant dialogues the study has never touched, with every definition frozen.
- **Part C, pooled estimates** over all 910 dialogues, with intervals and rank stability.

| Q | Question |
|---|---|
| **F1** | How much of the base model's lead comes from the eight examples? |
| **F2** | Does the result depend on which eight examples were drawn? |
| **F3** | Do the examples have to come from the restaurant domain? |
| **F4** | Do the simulator-study findings hold on new dialogues? |
| **F5** | With 910 dialogues, how certain is the ranking of the models? |

The study ends with two verdicts: whether the base model's lead is its own (rule in Part A), and which findings replicate (rule in Part B).

## 1. Rules

- Create branch `simulator-robustness` from `origin/simulator-study`. Reuse `corpus.py`, `hypotheses.py`, `probe.py`, `reference.py`, `rewrites.py`, `sim.py`, `sim_score.py`, `sim_analyze.py`, `sim_report.py` and `typv0.py`, and don't change their behavior. Add parameters whose defaults keep the old behavior.
- Same rules as the earlier plans:
  - preserve CRLF; stdlib-first; heavy dependencies as lazy optional extras
  - keys only via `keys.py`; never print, log, or commit a key
  - `python -m pytest` and a commit per phase
  - **[Frozen]** items change only via `docs/robustness_decisions.md` plus asking the user
  - if a count in Appendix A does not match yours, report the difference at the next STOP; don't adjust a frozen rule to make it match
- **Every analysis definition is the simulator study's, unchanged:** the probe builder, the coverage percentile, a_model and b_model from the regex extractor, the Occam gap, the rewrite metrics, the typicality variants and their bins. If you find yourself redefining a metric for the new sets, stop and ask.
- **STOP and ask the user** at four points:
  1. After Phase 1 (new sets, example pool, example draws, hotel examples, the S8 prompt).
  2. Before any API call (Phase 2 rewrites), with a dry run that prints the number of calls.
  3. Before submitting GPU jobs (Phase 3), with unit counts and a GPU-hour estimate.
  4. After the Part A analysis, before deciding whether format S8 joins Part B.
- Settings go in `configs/robustness.json`. Seed 20261116 for everything new. Bootstrap 2,000 replicates over dialogues.

## 2. Frozen definitions

### 2.1 Dialogue sets

All sets hold clean (no `fail_info`), slot-based (no restaurant name in the goal) restaurant goals with 2 or 3 constrained slots, and score the first user turn.

| Set | What | Dialogues |
|---|---|---|
| `disc` | the 235 clean SNG items of the simulator study | 235 |
| `rep_ssng` | `SSNG…` restaurant-only dialogues: the same kind of dialogue as `disc` | 105 |
| `rep_multi` | `MUL…`/`PMUL…` dialogues whose first user turn opens the restaurant domain and nothing else, minus the example pool | 570 |

- "Opens the restaurant domain and nothing else" is the test in `sim.fewshot_candidates`: the restaurant frame has an active intent, and no other frame has an active intent or slot values.
- Unlike `fewshot_candidates`, `rep_multi` applies no filter on the first turn's content. Turns that name a different value, say `dontcare`, or name no slot stay in, as they do in `disc`.
- **`WOZ…` dialogues are excluded.** Their logged goals do not match what users say: 194 value mismatches across 574 two-slot dialogues, against 4 in `disc`.
- Each set gets its own derived folder (`data/multiwoz/derived_rep_ssng/`, `…_rep_multi/`) with `items.jsonl`, `probe.jsonl`, the reference table, and rewrites. `sim_score` already takes a `derived` argument; expose it on the command line as `--derived`, and add the same option to every other command that reads a derived folder.
- **Corpus reference per set:** run `reference.py` on each set for a, b and the corpus log ratios. Naming rates differ between sets (Appendix A), so every Occam gap is compared with its own set's corpus value.

### 2.2 Example pool

- The pool `E` is 64 of the 462 candidates of `sim.fewshot_candidates`, drawn with the seed, half 2-slot and half 3-slot. It must contain the eight existing examples.
- `E` is excluded from `rep_multi`. Every example set in this plan comes from `E`.

### 2.3 Few-shot conditions (Part A, on `disc`)

| Label | Model | Examples |
|---|---|---|
| `F0`, `F1`, `F2`, `F4`, `F16` | `qwen3_base` | the first k of the existing eight; `F16` appends eight more from `E`; `F0` is the header and the query only |
| `F8` | `qwen3_base` | the existing eight. Reuse the simulator-study run. |
| `F8·d2` … `F8·d5` | `qwen3_base` | four further draws of eight from `E` with `select_fewshot`, disjoint from each other and from the existing eight |
| `FX8` | `qwen3_base` | eight hotel first turns (§2.4) |
| `S8` | `osim_mid`, `userlm`, `osim_final`, `qwen25_inst` | format S with the existing eight in the system prompt (§2.5) |

That is 14 new runs. For each example set, record its own naming rate: named constrained slots over constrained slots. The existing eight name 11 of 20.

### 2.4 Hotel examples for `FX8` [Frozen after Phase 1]

- Candidates: `SNG…`/`SSNG…` hotel-only dialogues with a clean, slot-based goal; first user turn.
- Goal text: write `hotel_text(goal)` in the style of `hypotheses.base`, covering `pricerange`, `type`, `area`, `stars`, `parking` and `internet`. Example: `The user wants to find a cheap guesthouse in the north of town with 4 stars and free parking.`
- Draw eight with the seed, half with 3 and half with 4 constrained slots.
- Header for `FX8` only: `The following are first messages that customers typed to a booking assistant, each after a short description of the customer.` The query line stays a restaurant customer.

### 2.5 Format S8 [Frozen after Phase 1]

System prompt:

```
You are a user talking to a restaurant booking assistant. {H}

For reference, here are first messages that other customers typed, each after a short description of that customer.

Customer: {H_1}
Message: {U_1}

…

Customer: {H_8}
Message: {U_8}
```

Then the fixed assistant opener, then the human turn scored as the reply, exactly as in format S.

### 2.6 Parts and metrics

- Parts per run: `samples` (64 per distinct hypothesis text), `human`, `urewrite`. Not `hrewrite` or `hsample`.
- Headline measures, all from `sim-analyze`:
  - **Coverage:** median human percentile; share of human turns beyond all samples; nats per word, human and samples.
  - **Fidelity:** a_model, b_model; Occam gap for a named and an omitted slot (implicit wording), next to the set's corpus value.
  - **Robustness:** median |Δ log BF| under `para1`, `typo` and `zh`.
  - **Typicality:** share of human pairs in the extreme bin (`glob1A`); size principle kept (`glob1B`, `cent1B`); bin agreement under `para1` (`cent1B`).
  - **Health:** flagged samples (empty, cut off, assistant voice).

## Phase 1: Sets, examples, prompts (CPU, raw corpus)

Command: `python -m robust_likelihood.multiwoz rb-prepare`.

1. Build `rep_ssng` and `rep_multi` with the existing item, probe and reference code. Give every item a `set` field.
2. Draw `E`, the four further example draws, and the eight more examples for `F16`. Write them to `data/multiwoz/derived/fewshot_draws.json`.
3. Write `hotel_text`, draw the hotel examples.
4. Add formats `F` with an `examples` argument (count and draw), `FX8`, and `S8` to `sim.py`.
5. Check every count against Appendix A; write the comparison to `docs/robustness_decisions.md`.
6. Tests on the mini fixture: set membership, pool exclusion, disjoint draws, the S8 prompt, the hotel text.

**STOP.** Show: the count comparison; the reference table of each set; all five example draws with their naming rates; the eight hotel examples; one rendered S8 prompt per model; five samples per new format from a small run if a GPU or MPS device is at hand.

## Phase 2: Rewrites for the new first turns (API)

- Kinds `para1`, `typo`, `zh`, through the existing `rewrites.py` pipeline and checks, unchanged.
- 675 turns × 3 kinds = 2,025 units before retries. Dry run first. **STOP with the count.**
- Export 60 accepted rewrites to a CSV for the user to audit.

## Phase 3: Scoring (GPU)

| Runs | Set | Count |
|---|---|---|
| Part A: the 14 conditions of §2.3 | `disc` | 14 |
| Part B: `qwen3_base`·F8, `osim_mid`·S, `osim_final`·S, `userlm`·S, `qwen25_inst`·S | `rep_ssng` and `rep_multi` | 10 |
| Seed repeat: `samples` only, seed + 1, for `qwen3_base`·F8 and `osim_mid`·S | `disc` | 2 |

- Sizes: `disc` has 2,355 pairs and 463 hypothesis texts. `rep_ssng` has about 1,050 pairs and 306 texts; `rep_multi` about 5,800 pairs and 700 texts.
- A full simulator-study run took about 0.5 H100-hour per model. My estimate is 5–7 H100-hours for Part A and 8–12 for Part B. Measure on one run and report your own estimate.

**STOP before submitting,** with the estimate and the job list. Parts A and B can be submitted together.

## Phase 4: Analysis and report

Commands: `rb-analyze`, then `rb-report` → `docs/robustness_results/summary.md`, `metrics.json`, `manifest.json`, figures.

### Part A: few-shot controls

Report one table of the headline measures with a row per condition, then:

- **The k-curve:** each headline measure against k = 0, 1, 2, 4, 8, 16.
- **The draw spread:** the range and SD of each headline measure over the five draws of eight, next to the dialogue-bootstrap interval of a single draw. Plot a_model against each draw's own naming rate.
- **Equal information:** `S8` against `S` for each chat model, and `S8` against `F8`, with paired dialogue-bootstrap intervals.
- **Domain of the examples:** `FX8` against `F8`.

Pre-registered predictions (failing them is a result):

| ID | Prediction |
|---|---|
| A1 | Without examples the base model fails: `F0` has more than 20% of human turns beyond all samples, or more than 10% flagged samples. |
| A2 | Saturation: `F4` is within 0.05 of `F8` on median percentile and within 1 nat on the Occam gap for a named slot. |
| A3 | Draw spread is small: over the five draws, the median percentile ranges by at most 0.10, the Occam gap by at most 1.5 nats, and a_model by at most 0.15. |
| A4 | The examples carry the lead: with `S8`, OSim-8B-Mid's median percentile comes within 0.10 of `F8`'s and its a_model rises above 0.5. |
| A5 | Examples do not rescue the instruct model: `qwen25_inst`·S8 still has more than 80% of human turns beyond all samples. |
| A6 | Coverage transfers across domains and naming rates do not: `FX8` keeps the share beyond all samples under 2%, and its a_model differs from `F8`'s by more than 0.10. |

**Verdict for F1 [Frozen].** Compare `qwen3_base`·F8 with each chat model's `S8` on three measures: median percentile (higher is better), |Occam gap for a named slot − corpus value| (lower), and median |Δ log BF| under `para1` (lower).

- "The lead is the base model's own" if F8 is better than every chat model's S8 on at least two of the three, with paired-bootstrap intervals that exclude 0.
- "The lead comes from the examples" if some chat model's S8 matches or beats F8 on at least two of the three.
- Otherwise "mixed": report which measures go which way.

**STOP after Part A.** Report the table and the verdict. If the verdict is "the lead comes from the examples" or "mixed", ask whether to add the best `S8` model to Part B.

### Part B: replication

For each of `rep_ssng` and `rep_multi`, the headline table for the five models next to the `disc` values, with intervals.

Replication criteria [Frozen], judged per set:

| ID | Finding from the simulator study | Holds on the new set if |
|---|---|---|
| R1 | Human turns fall inside base and simulator samples | beyond all samples is under 5% for `qwen3_base`, `osim_mid` and `userlm`, and over 90% for `qwen25_inst` |
| R2 | The coverage ranking | `qwen3_base`, `userlm` and `osim_mid` are the top three by median percentile, and `osim_final` is above `qwen25_inst` |
| R3 | The Occam overshoot shrinks | the Occam gap for a named slot is within 50% of the set's corpus value for `qwen3_base`, and at least twice it for `qwen25_inst` |
| R4 | Paraphrase shifts drop | median \|Δ log BF\| under `para1` is at most 2 nats for the four human-like models and at least 4 for `qwen25_inst` |
| R5 | Typicality regains signal | under 50% of human pairs are in the extreme bin with `osim_mid` as reference, and over 95% with `qwen25_inst` |
| R6 | RL narrows the distribution | `osim_final`'s mean sample surprisal is at least 30% below `osim_mid`'s, and its share beyond all samples is higher |

- A finding **replicates** if it holds on `rep_ssng`, which is the same kind of dialogue as `disc`.
- `rep_multi` is a transfer test: users who also have other errands say less in their first turn (Appendix A). Report it separately. A difference there is a finding about the population, not a failed replication.
- Format F's examples come from the `rep_multi` population. Say so next to every `rep_multi` number for `qwen3_base`.

### Part C: pooled estimates

Over all 910 dialogues, stratified by set:

- every headline measure per model with a 95% dialogue-bootstrap interval
- paired differences `qwen3_base` − `osim_mid` and `qwen3_base` − `userlm` with intervals
- **rank stability:** for each headline measure, the share of bootstrap replicates in which each model ranks first
- the seed repeat: the change in each sample-based measure between the two seeds, next to the bootstrap interval

### Report

`summary.md` leads with three things: the F1 verdict, the replication table (R1–R6 × two sets), and the updated recommendation for the reference model. Then Parts A, B and C with one table and one figure each, the prediction scorecard, and notes for Xuan with every deviation from this plan.

**Figures** (`.png` + `.pdf` + `.csv`):

1. The k-curve: median percentile, Occam gap and paraphrase shift against k, with the five draws shown at k = 8.
2. `S` against `S8` per chat model, with `F8` as a reference line.
3. The replication: each headline measure per model on `disc`, `rep_ssng` and `rep_multi`, with intervals.
4. Rank stability: share of replicates ranked first, per measure and model.

Commit `docs/robustness_results/` and the derived data files, without raw outputs.

## Not in this plan

- A second domain. MultiWOZ has 342 hotel-only dialogues with clean, slot-based goals (3–4 constrained slots out of six). That is the next step if the paper needs a cross-domain claim; it needs hotel hypothesis templates, a hotel slot extractor and a hotel corpus reference. Here hotel dialogues serve only as examples for `FX8`.
- Later turns. That is `docs/goal_inference_plan.md`.
- Hypothesis rewording on the new sets, and the `fail_info` dialogues.

## Appendix A: counts to verify

Computed on the pinned corpus with the repository's own functions.

| Quantity | Expected |
|---|---|
| `SSNG` restaurant-only, clean, slot-based (2-slot / 3-slot) | 105 (56 / 49) |
| `MUL`/`PMUL` clean, slot-based, 2–3 slots, first turn opens only the restaurant domain | 634 (MUL 254, PMUL 380; 2-slot 320, 3-slot 314) |
| Few-shot candidates (`sim.fewshot_candidates`) | 462 (2-slot 234, 3-slot 228), including the existing 8 |
| Example pool `E` / `rep_multi` | 64 / 570 |
| All sets together | 910 dialogues, 675 of them new |
| Probe rows / distinct hypothesis texts, `disc` | 2,355 / 463 |
| Probe rows / distinct hypothesis texts, `rep_ssng` | 1,050 / 306 |
| Probe rows / distinct hypothesis texts, all 634 multi-domain items | 6,442 / 728 |
| First-turn a, food / area / price, `disc` | 0.73 / 0.74 / 0.75 |
| First-turn a, `rep_ssng` | 0.70 / 0.77 / 0.76 |
| First-turn a, `MUL` items (254) | 0.77 / 0.67 / 0.63 |
| First-turn a, `PMUL` items (380) | 0.67 / 0.54 / 0.58 |
| Multi-domain items whose first turn names no constrained slot | 64 of 634 |
| Hotel-only, clean, slot-based (`SNG` / `SSNG`) | 233 / 109 |

## Appendix B: commands

```
git checkout -b simulator-robustness origin/simulator-study
python -m pip install -e ".[dev,figures,lm]"                                          # login node
python -m robust_likelihood.multiwoz rb-prepare                                       # Phase 1 → STOP
python -m robust_likelihood.multiwoz rewrites --derived data/multiwoz/derived_rep_ssng --dry-run   # Phase 2 → STOP
python -m robust_likelihood.multiwoz sim-score --model qwen3_base --format F --examples 4          # Part A, one per condition
python -m robust_likelihood.multiwoz sim-score --model osim_mid --format S8
python -m robust_likelihood.multiwoz sim-score --model osim_mid --format S --derived data/multiwoz/derived_rep_multi   # Part B
python -m robust_likelihood.multiwoz rb-analyze --runs results/multiwoz/*-sim-*
python -m robust_likelihood.multiwoz rb-report
```
