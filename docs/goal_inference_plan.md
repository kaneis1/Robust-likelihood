# Plan: does typicality add anything to raw log-likelihood? Multi-turn goal inference on MultiWOZ

Execution plan for Claude Code, repository `kaneis1/Robust-likelihood`. Save as `docs/goal_inference_plan.md`.

## Amendments (16 Nov)

Where this block and the text below disagree, this block wins.

1. **Run order.** Run `docs/robustness_plan.md` first. Branch `goal-inference` from `origin/simulator-robustness`, not from `origin/simulator-study`.
2. **More dialogues.** Add the `SSNG…` restaurant-only dialogues (`rep_ssng` of the robustness plan: 105 clean, slot-based, the same kind of dialogue as the 235) to the dialogue set. With all 105 that is 340 dialogues, 1,524 scored turns, 2,540 candidates and 11,430 units; item 3 removes eight of them. Appendix A covers the 235 only: produce the same table for the SSNG dialogues at the Phase 1 STOP. Report every result for the two sets separately and pooled.
3. **Examples for format FM.** §2.5 draws the example dialogues from restaurant-only dialogues outside the probe. Those are now only `WOZ…` dialogues, whose logged goals do not match what users say (194 value mismatches across 574 two-slot dialogues). Don't use them. Reserve eight SSNG dialogues as the FM example pool (seeded; four 2-slot and four 3-slot; 3–5 user turns; at least two in which a constrained slot is first named in user turn 2), remove them from the dialogue set, and draw the four examples from them. The other four are a second draw.
4. **What the robustness run decides.**
   - If its draw-spread prediction (A3) fails, run `qwen3_base`·FM with both example draws and report both.
   - If its verdict is "the lead comes from the examples" or "mixed", ask the user at the Phase 2 STOP whether to add `osim_mid`·S with the four example dialogues in the system prompt as a third sampled reference.
5. **Costs** grow by a factor of about 1.45: roughly 11,100 Jev calls, 3,450 rewrite units, and 7–13 H100-hours per sampled model. The STOP points still decide.

## 0. Purpose

The simulator study (`docs/simulator_results/summary.md`) showed that a human-like reference model removes most of what the claims run blamed on token log-probs:

- Human first turns fall inside the samples of Qwen3-8B-Base (few-shot), OSim-8B-Mid and UserLM-8B: 0–0.7% lie beyond all 64 samples, against 96–100% for instruct models.
- The raw log-likelihood Occam gap for a named slot is 5.2 nats (base) and 7.2 (OSim-8B-Mid), against 4.19 in the corpus and 14.1 for Qwen2.5-7B-Instruct.
- A paraphrase moves a log Bayes factor by 1.0–1.9 nats, against 6–8 for instruct models.
- Typicality regains signal with these references, but its bins agree under paraphrase in only 66–86% of pairs.

That leaves one question open: **with a human-like reference, does typicality add anything over the raw log-likelihood it is built from?** This study answers it on the task the paper is about, inferring what a user wants from ambiguous requests, over whole dialogues. It also delivers what Xuan asked for on Nov 2: later turns scored as p(u_t | goal, history).

The study must end with one verdict per reference model: use raw log-likelihood, use typicality on top of it, or neither is better (the G5 decision rule in Phase 6).

| Q | Question |
|---|---|
| **G1 Inference** | How well does each scorer infer the goal, turn by turn? |
| **G2 Calibration** | Do per-turn Bayes factors match corpus rates: strong when a slot is first named in turn 1, weaker when it is named after the system asks, near zero on turns that carry no evidence? |
| **G3 Drift** | On turns that say nothing about a slot, how far does the cumulative log Bayes factor move? |
| **G4 Stability** | How much does the posterior move when the user's turns are rewritten without changing their meaning? |
| **G5 Verdict** | Does typicality beat raw log-likelihood from the same reference? Does plain clipping do as well? |

## 1. Rules

- Create branch `goal-inference` from `origin/simulator-study`. Reuse `lmcore.py`, `multiwoz/sim.py`, `sim_score.py`, `sim_pilot.py`, `sim_noise.py`, `typv0.py`, `claim1.py`, `rewrites.py`, `reference.py`, `jev_run.py`, and don't change their behavior. Where this plan needs a variation (for example a different normalization pool in `typv0`), add a parameter whose default keeps the old behavior.
- Same rules as the earlier plans:
  - preserve CRLF; stdlib-first; heavy dependencies as lazy optional extras
  - keys only via `keys.py`; never print, log, or commit a key
  - `python -m pytest` and a commit per phase
  - **[Frozen]** items change only via `docs/goal_inference_decisions.md` plus asking the user
  - if a count in Appendix A does not match yours, report the difference at the next STOP; don't adjust a frozen rule to make it match
- **STOP and ask the user** at five points:
  1. After Phase 1 (data, candidates, corpus reference, few-shot dialogues).
  2. After the format check (Phase 2).
  3. Before any API call: the rewrites (Phase 3) and Jev (Phase 5), each with a dry run that prints the number of calls.
  4. Before submitting any GPU job (Phase 4), with unit counts and a GPU-hour estimate measured on a smoke run.
  5. Before any optional part (Phase 7).
- Login node first: models are already downloaded at the revisions pinned in `configs/simulator.json`. Jobs set `HF_HUB_OFFLINE=1` and contain no `pip install`.
- Settings go in `configs/goal_inference.json`. Seed 20261115 for everything new; bootstrap 2,000 replicates over dialogues.

## 2. Frozen definitions

### 2.1 Dialogues and turns

- The 235 clean items of `data/multiwoz/derived/items.jsonl`.
- User turn t = 1 … T_d, with T_d = min(number of user turns, 6). That is 1,025 scored turns, 790 of them after turn 1.
- The history of turn t is every earlier MultiWOZ 2.2 turn, user and system, as logged. System turns are treated as given. The wizard never saw the goal, so p(system turn | goal, history) does not depend on the goal and cancels in every Bayes factor.
- **Subset:** the 60 dialogues of `data/multiwoz/derived/multiturn_pilot.jsonl` (266 scored turns).

### 2.2 Candidate goals

A goal gives each of food, area and pricerange either a value or "no preference". The candidate text is `hypotheses.explicit(goal)`, the frozen explicit template, so that candidates are mutually exclusive ("…The user has no preference about the type of food.").

For a dialogue with true goal G, the candidate set C_d is:

| Key | Goal | Count |
|---|---|---|
| `G` | the true goal | 1 |
| `drop:X` | G with constrained slot X set to no preference | one per constrained slot |
| `swap:X` | G with constrained slot X set to another value | one per constrained slot |
| `add:Y` | G with unconstrained slot Y set to a value | one per unconstrained slot |
| `broad` | no preference on all three slots | 1 |

- Sizes: 8 for a 3-slot goal, 7 for a 2-slot goal; 1,756 candidates and 7,690 (dialogue, turn, candidate) units in total.
- **Values for `swap:X` and `add:Y`:** drawn from `hypotheses.value_pools` with a generator seeded as `control_seed(item_id, kind)` does. Exclude the goal value, and exclude every value that slot takes in the dialogue's 2.2 state annotations at any turn. The distractor must be a value the dialogue never mentions. If a pool is exhausted, STOP.
- Internal keys never enter a text.
- **Prior:** uniform over C_d (primary). Sensitivity: a slot-level corpus prior, P(c) ∝ Π_slots (p_X if c constrains X, else 1 − p_X), with p_food = 3108/3927, p_area = 3104/3927, p_price = 3167/3927 from `reference_table.md`.

### 2.3 Turn labels

Computed with `corpus.turn_status(state, previous_user_state)` per (dialogue, turn, slot):

- `first_named`: the slot gets a value for the first time in this turn
- `silent_at_risk`: the slot has no value yet and gets none
- `silent_known`: the slot already has a value and this turn does not change it
- `renamed`, `dontcare`: reported, excluded from G2 and G3
- A turn is `search_silent` when no slot's value set changes.
- `value_conflict` (per candidate and turn): the turn gives slot X a value that differs from the candidate's value for X.

### 2.4 Reference models and formats

| Key · format | Role | Parts |
|---|---|---|
| `osim_mid` · S | primary reference, zero-shot | all, with samples |
| `qwen3_base` · FM | primary reference, few-shot (new format, §2.5) | all, with samples |
| `qwen25_inst` · S | instruct contrast (claim 1 over turns) | scoring only |
| `osim_final` · S | RL contrast | scoring only |
| `userlm` · S | cross-check | scoring only |

The multi-turn S context is exactly `sim_pilot.PilotFormat`: format S's system prompt with the candidate text, the fixed opener, then the real history in roles. Keep its `template_check` and `history_kept` self-checks.

### 2.5 Format FM: multi-turn few-shot for the base model [Frozen after Phase 2]

Plain text, no chat template.

- Header: `The following are conversations between customers and a restaurant booking assistant, each after a short description of the customer.`
- Each example: `\n\nCustomer description: {H_k}` then one line per turn, `\nCustomer: {u}` or `\nAssistant: {s}`, for the whole example dialogue.
- Query: `\n\nCustomer description: {H}`, the history lines, then `\nCustomer:`. Score ` {u_t}` plus the terminating `\n`. Samples stop at the first `\n`.
- Replace any newline inside an utterance with a space and count how often that happens.
- **Examples:** 4 complete dialogues, fixed for every query. Candidates are restaurant-only, slot-based, clean dialogues that are not among the 370 probe items, with 3–5 user turns and strictly alternating speakers. Seeded draw: two 2-slot and two 3-slot goals, at least one in which a constrained slot is first named in user turn 2. H_k is `explicit(goal)`.
- Report the candidate pool and the four chosen dialogues at the Phase 1 STOP. If the only candidates are `WOZ…` dialogues and their style differs visibly from the SNG dialogues, say so.
- **Caveat to carry into the report:** the examples show the base model how MultiWOZ users reveal constraints over turns. The chat models get no such examples.

### 2.6 Scorers

For unit (d, t, c), a scorer gives a per-turn score ℓ_t(c) on a log scale. The cumulative score is L_t(c) = Σ_{k ≤ t} ℓ_k(c), each ℓ_k computed with the **real** history. The posterior after turn t is softmax over C_d of L_t(c) + log prior.

1. **`raw`**: ℓ = −S(u_t | c, history), in nats.
2. **`clip`**: raw with bounded influence per turn. ℓ = max(−S − m_t, −κ), where m_t is the largest −S over C_d at that turn. κ = `clip_nats` of `configs/claims.json` (3.0); sensitivity 5.0 and 8.0. This is the cheap baseline from the Gaussian test, where clipping matched binning on robustness.
3. **`typ`**: centered typicality, the `cent1B` variant with 6 bins from `docs/typicality_v0_decision.md`, applied per turn.
   - Statistic: s(u | c) = S(u | c, history) − S(u | broad, history).
   - For each sample x of candidate c (drawn from the reference given c and the history), s(x | c) = S(x | c, history) − S(x | broad, history). μ_c and σ_c are the mean and SD (n − 1) over c's K samples.
   - **Pool [primary]:** the candidates of the same (dialogue, turn), uniform weights. H̄ = mean_c μ_c; σ̄² = mean_c [σ_c² + (μ_c − H̄)²]. T = (s − H̄)/σ̄.
   - Bins: one-sided, edges (−2, −1, 0, 1, 2). Score B: ℓ = −(H̄ + σ̄·c_b), c_b = (−2.5, −1.5, −0.5, 0.5, 1.5, 2.5).
   - Sensitivity: `typ_turnpool`, where the pool is every (dialogue, candidate) of the same turn index; and `typS`, the same recipe on the raw statistic S.
   - A cut-off sample (no end marker within 96 new tokens) is kept with its lower bound, as in Phase 2 of the simulator study.
4. **`jev`**: Jev's yes-probability for (candidate text, turn, history), read on the logit scale as `metrics.logit_score` does (primary) and on the log scale (sensitivity). One candidate per call. Prompt in §2.8.
5. **`ref`**: the corpus yardstick of §2.7. It is not a model; it shows what the slot annotations alone support.

`raw`, `clip` and `typ` exist for `osim_mid` and `qwen3_base`. `raw` and `clip` exist for the three scoring-only models.

### 2.7 Corpus reference for later turns (extends `reference.py`)

For slot X and user turn index t, among dialogues in which X is "at risk": X has no value before turn t and, where the goal has no preference on X, no `dontcare` before turn t either.

- a_t(X) = P(turn t gives X a value | the goal constrains X)
- b_t(X) = P(turn t gives X a value | the goal has no preference on X)

Rules:

- Populations as in the turn-1 reference: `match_clean` (the 235) for a_t; `pooled` (every dialogue with a slot-based restaurant goal, turns counted from the first restaurant turn) for b_t with +0.5 smoothing. Report both populations for both rates.
- Turn index t = 1, 2, and 3+ pooled.
- Per-turn reference log ratio for the pair (G, drop:X): ln(a_t/b_t) on a `first_named` turn, ln((1 − a_t)/(1 − b_t)) on a `silent_at_risk` turn, and 0 on a `silent_known` turn. The last value is assumption A1: once a slot has a value, later turns say nothing about whether it was constrained. State A1 in the report.
- `ref` as a scorer: per turn, for each at-risk slot X, candidate c contributes ln a_t if c constrains X and the turn names c's value, ln(a_t · 0.01) if the turn names another value, ln(1 − a_t) if the turn is silent; and ln b_t or ln(1 − b_t) if c has no preference on X. A `dontcare` turn counts as silent. Slots already valued contribute 0.
- Write `data/multiwoz/derived/gi_reference.json` and a readable `gi_reference.md` with intervals (bootstrap over dialogues).

Expected from my count on the clean set (to be re-estimated, not trusted): naming a slot is worth about 4.2 nats in turn 1 and about 1.3 in turn 2, because the system has often just asked and users with no preference pick a value anyway or accept an offer.

### 2.8 Jev prompt for later turns [Frozen]

- Turn 1: the existing `multiwoz_likely` pairwise question, unchanged, so cached records are reused when the candidate text matches.
- Turn t ≥ 2: the same template with this question:
  `Assume the user has the stated intention described in the state. Below is the conversation so far between the user and a restaurant booking assistant. Is the user likely to produce the following utterance as their next message when pursuing that intention? Conversation so far: {history, one line per turn as "User: …" / "Assistant: …"} Utterance: "<<UTTERANCE>>"`
- Add it as a new prompt version, `prompts/multiwoz_likely_mt/`, and leave the existing prompt files unchanged.
- `stated_intention` stays `find_restaurant`; the candidate text goes in `intention_description`; the internal key is never sent.
- `likely` wording only.

### 2.9 Rewrites of user turns

- Kinds: `para1`, `typo`, `zh`.
- Turn 1: reuse the accepted rewrites in `data/multiwoz/derived/rewrites.jsonl`.
- Turns 2 … T_d: generate through the existing `rewrites.py` pipeline (Claude, then the regex and NLI checks), unchanged, plus these validity checks for later turns:
  - every digit string and clock time of the original appears in the rewrite
  - every weekday name appears (in `zh`, its Chinese form is accepted)
  - every restaurant name from `ontology.json` (`restaurant-name`) that appears in the original appears unchanged in the rewrite
- A turn without an accepted rewrite of a kind keeps its original text. Report, per kind, the share of turns rewritten and the share of dialogues fully rewritten.
- Export 60 accepted later-turn rewrites to a CSV for the user to audit, as `rewrite-audit` does.
- **R1, turn-local [primary]:** each user turn is replaced by its rewrite when it is scored; the history stays original. Samples are reused, so every scorer applies. The cumulative score under kind r is Σ_k ℓ(rewrite_r(u_k) | c, original history).
- **R2, full history [sensitivity]:** the history also carries the rewritten user turns; system turns are unchanged. `para1` only, the 60-dialogue subset, `raw` and `clip` only, since `typ` would need new samples.
- **Noise floor:** rescore every R1 unit with the original utterance in the rewrite job's batch layout, as `sim_noise.py` does. The expected effects are about 1 nat and batch-layout noise is 0.15–0.5 nats, so report every stability number next to its noise floor.

## 3. Things that are easy to get wrong

- **The history is always the real one.** Never condition a later turn on text the model sampled. Under R1 the history is the original, not the rewritten one.
- **Distractor values must be absent from the dialogue** (§2.2). A `swap` or `add` value that the user later mentions turns a wrong candidate into a right one.
- **The typicality pool is local** to the (dialogue, turn). `typv0`'s pool was the 463 first-turn texts; don't reuse those statistics.
- **Jev's logit sum is not a log-likelihood.** Report it as the repo's reading of Jev and show the log-scale version beside it.
- **`broad` has s ≡ 0** for the human turn and for its own samples. That is correct, not a bug.
- **Sampling is the cost.** K = 32 for 7,690 units per sampled model. Shard by dialogue, resume from partial output, and keep within the 3-hour job limit.

## Phase 1: Data and reference (CPU, raw corpus)

Command: `python -m robust_likelihood.multiwoz gi-prepare`.

1. `data/multiwoz/derived/gi_turns.jsonl`: one row per (dialogue, turn) with the history, the human turn, and the turn labels per slot.
2. `data/multiwoz/derived/gi_candidates.jsonl`: one row per (dialogue, candidate) with key, goal, text, and for `swap`/`add` the drawn value.
3. `gi_reference.json` and `gi_reference.md` (§2.7).
4. `data/multiwoz/derived/gi_fewshot.json` (§2.5).
5. Check every count against Appendix A and write the comparison to `docs/goal_inference_decisions.md`.
6. Tests on the mini fixture for the candidate builder, the labels, the reference, and the distractor exclusion.

**STOP.** Show: the count comparison, the reference table by turn, five example dialogues with their candidate sets, and the four few-shot dialogues.

## Phase 2: Format check (small GPU or MPS run)

Command: `python -m robust_likelihood.multiwoz gi-check`.

- `qwen3_base` · FM: 8 samples at user turns 1, 2 and 3 for 6 seeded dialogues (three 2-slot, three 3-slot) under `G`. Flags: empty, cut off, more than one line, assistant voice. Boundary and exact-sampling checks as in `sim-check`.
- FM against F at turn 1: human-turn surprisal on 20 pairs under both formats. They need not agree; report the difference.
- `osim_mid` · S at turn 4 or later for the same dialogues, since the pilot stopped at turn 3.
- Write `docs/goal_inference_results/format_check.md`.

**STOP: the user reads the samples.** FM is frozen after this.

## Phase 3: Rewrites of later turns (API)

Command: `python -m robust_likelihood.multiwoz gi-rewrites [--dry-run]`.

- Dry run first: 790 turns × 3 kinds = 2,370 units before retries. **STOP with the count.**
- Then run, and write `data/multiwoz/derived/gi_rewrites.jsonl`, the acceptance rates per kind and check, and the audit CSV.

## Phase 4: Scoring and sampling (GPU)

Command: `python -m robust_likelihood.multiwoz gi-score --model <key> [--parts …] [--shard i/n] [--resume <dir>]`. Each run writes to `results/multiwoz/<UTC>-gi-<key>-<fmt>[-<i>of<n>]/`.

| Part | What | Models |
|---|---|---|
| `human` | S(u_t \| c, history) for all 7,690 units | all five |
| `samples` | K = 32 per unit, pure sampling with the Phase 2 settings of the simulator study, each with S under its own context and S under `broad` | `osim_mid`, `qwen3_base` |
| `urewrite` | R1: each accepted rewrite of the scored turn under every candidate, original history | all five |
| `null` | the noise floor of §2.9 | all five |
| `full` | R2 on the 60-dialogue subset, `para1` | all five |

- Add a `gi` stage to `scripts/sim_job.sh` or write `scripts/gi_job.sh` in the same style.
- Smoke run first: 5 dialogues, all parts, `osim_mid`. From its throughput, estimate GPU-hours per model.
- My rough estimate, from the pilot's 0.5–1 H100-hour for 1,206 units: 5–9 H100-hours for each sampled model and under 1 for each scoring-only model. Trust your measurement over this.

**STOP before submitting,** with the measured estimate and the shard plan.

## Phase 5: Jev (API)

Command: `python -m robust_likelihood.multiwoz gi-jev [--dry-run]`.

- All 7,690 units, original turns. Turn-1 units reuse the cache where the text matches.
- R1 with `para1` on the 60-dialogue subset: 2,003 units.
- Dry run prints both counts and the number of cache hits. **STOP before running.** If the user chooses the subset only, run the 2,003 original units and the 2,003 rewritten ones.

## Phase 6: Analysis and report

Commands: `gi-analyze`, then `gi-report` → `docs/goal_inference_results/summary.md`, `metrics.json`, `manifest.json`, figures.

Every metric is per scorer and reference model, with 95% dialogue-bootstrap intervals.

### G1, inference (by turn t = 1 … 6, and at the final turn T_d)

- **`logp_gold`:** mean log posterior of `G`. Primary.
- **`top1`:** share of dialogues whose argmax is `G`. Also the mean rank of `G`.
- **Slot decisions:** for each constrained slot, the share with L_t(G) > L_t(drop:X); for each unconstrained slot, the share with L_t(G) > L_t(add:Y); for each constrained slot, the share with L_t(G) > L_t(swap:X).
- **"Does the user care about this slot?":** AUROC of L_t(candidate with a value) − L_t(candidate without), with label = constrained, pooling (G, drop:X) for constrained slots and (add:Y, G) for unconstrained ones.
- Report `ref` in the same table. It is the level the annotations support: 17 dialogues have a constrained slot that the user never names within six turns, so no scorer should reach 100%.
- Report the strata separately: dialogues where every constrained slot is named in turn 1; dialogues with a slot first named later; the 17 with a slot never named; dialogues where an unconstrained slot gets a value.

### G2, calibration against the corpus

For pairs (G, drop:X), the per-turn difference Δℓ_t = ℓ_t(G) − ℓ_t(drop:X), grouped by X's label at turn t:

- `first_named` at turn 1, at turn 2, at turn 3+
- `silent_at_risk` at turn 1, at turn 2, at turn 3+
- `silent_known`

For each group: mean Δℓ_t, the reference value, and their difference (the Occam gap by turn). Also the **context ratio**: mean Δℓ for `first_named` at turn 2 divided by the mean at turn 1, next to the corpus ratio.

### G3, drift

On `silent_known` turns of pairs (G, drop:X), where the reference is 0 under A1:

- median and 90th percentile of |Δℓ_t|
- per dialogue and pair, the cumulative drift D = Σ Δℓ_t over those turns: mean (bias) and SD (random walk)
- **late flip rate:** the share of dialogues whose argmax at T_d differs from the argmax after the last turn that was not `search_silent`

### G4, stability (R1; R2 where it exists)

Per rewrite kind, next to the noise floor:

- per turn: median |change in Δℓ_t| over the pairs of G1
- cumulative: median |change in L_T(G) − L_T(c)|
- posterior: mean |change in P(G)| at turns 1, 2 and T_d
- argmax flip rate at turns 1, 2 and T_d

### Pre-registered predictions (failing them is a result)

| ID | Prediction |
|---|---|
| M1 | With `raw` from `osim_mid`, `top1` rises from turn 1 to turn 2 and is ≥ 80% at the final turn. |
| M2 | Context: for every reference model, the `raw` context ratio is below 0.8 and above the corpus ratio. Models discount a slot named after a question, but less than real users warrant. |
| M3 | Drift: with `osim_mid`, `raw` has a median \|Δℓ_t\| above 0.3 nats on `silent_known` turns; `typ` has a median of 0 and a smaller SD of cumulative drift. |
| M4 | Stability (claim 3): under `para1`, `typ` has a lower argmax flip rate at the final turn than `raw`, for both sampled references. |
| M5 | Instruct contrast: `raw` from `qwen25_inst` has a final `top1` at least 10 points below `osim_mid`'s and a per-turn paraphrase shift at least 3 times larger. |
| M6 | Jev: its Δℓ on `first_named` at turn 1 is below 50% of the corpus value, and its drift on `silent_known` turns is the smallest of all scorers. |
| M7 | Clipping: `clip` achieves at least half of `typ`'s reduction in final-turn argmax flips relative to `raw`. |

### G5, verdict [Frozen]

For each sampled reference (`osim_mid`, `qwen3_base`), compare `typ` with `raw` by paired dialogue bootstrap.

- **Stability gain:** under `para1` (R1), the final-turn argmax flip rate of `typ` is lower than `raw`'s by at least 3 percentage points, with a 95% interval that excludes 0; or the mean |change in P(G)| at the final turn is lower by at least 20% (relative), interval excluding 0.
- **Accuracy cost:** `logp_gold` of `typ` is worse than `raw`'s by more than 0.10 nats at the final turn or at turn 1, with the interval excluding 0.

| Stability gain | Accuracy cost | Verdict |
|---|---|---|
| yes | no | typicality adds: use it |
| yes | yes | trade-off: report both, no default |
| no | no | no difference: use raw log-likelihood |
| no | yes | typicality costs: use raw log-likelihood |

Apply the same rule to `clip` against `raw`. If `clip` gets the same verdict as `typ`, say plainly that bounded influence explains the gain and typicality's normalization adds nothing here.

### Report

`summary.md` leads with the G5 verdict table, then:

1. The prediction scorecard.
2. G1–G4, one section each, with one table and one figure.
3. The instruct contrast and the RL contrast (`qwen25_inst`, `osim_final`), for the wording of claim 1.
4. Jev against the model scorers.
5. A recommendation for pipeline v0: reference model, scorer, and prior.
6. Notes for Xuan, including assumption A1, the few-shot caveat, and every deviation from this plan.

**Figures** (`.png` + `.pdf` + `.csv`):

1. `logp_gold` and `top1` by turn, one line per scorer, one panel per reference model, with `ref`.
2. Per-turn Δℓ by label group against the corpus reference.
3. Cumulative drift per scorer (distribution over dialogues).
4. Argmax flip rate and mean |change in P(G)| by rewrite kind and scorer, with the noise floor.
5. Posterior of `G` by turn for 6 seeded dialogues, one line per scorer.

Commit `docs/goal_inference_results/` and the derived data files, without raw outputs.

## Phase 7 (optional; STOP before each)

- **Hypothesis wording:** one paraphrase of every candidate text on the 60-dialogue subset, with new samples for `typ`. The simulator study found the top hypothesis changing in 7–19% of items under hypothesis rewording, and this plan does not test that axis.
- **Jev on the whole prefix:** one call per (dialogue, turn, candidate) that shows the conversation so far and asks about all the user's messages together, on the subset. This tests whether a judge model accumulates evidence as well as the sum over turns does.

## Not in this plan

- An open-ended hypothesis space. Candidates here are the true goal's one-edit neighbours.
- The fail_info dialogues, in which the goal changes mid-dialogue.
- The implicit wording of candidates.
- A model that outputs the posterior directly.

## Appendix A: counts to verify

Computed with `corpus.turn_status` on the pinned corpus, for the 235 clean dialogues.

| Quantity | Expected |
|---|---|
| Dialogues that alternate USER, SYSTEM from the start | 235 |
| User turns per dialogue (3 / 4 / 5 / 6 / 7 / 8 / 10) | 40 / 102 / 61 / 25 / 5 / 1 / 1 |
| Scored turns with the cap of 6, by turn index 1–6 | 235 / 235 / 235 / 195 / 93 / 32 (1,025) |
| Candidates (124 two-slot × 7 + 111 three-slot × 8) | 1,756 |
| Units (dialogue, turn, candidate) | 7,690 |
| Subset: scored turns / units | 266 / 2,003 |
| Constrained slot-items (food / area / price) | 195 / 186 / 200 (581) |
| Unconstrained slot-items (food / area / price) | 40 / 49 / 35 (124) |
| Constrained slots first given a value at turn 1 / 2 / 3 / 4 / 5 | 430 / 119 / 11 / 3 / 1 |
| Constrained slots never given a value within the cap | 17, in 17 dialogues |
| `search_silent` turns by turn index 1–6 | 7 / 89 / 212 / 190 / 91 / 32 |
| `silent_known` pair-turns for (G, drop:X) | 1,763 |
| Unconstrained slots given a value (at turn 1 / 2 / 3 / 4 / 5) | 37 (2 / 25 / 8 / 1 / 1) |
| Turn-2 rate, constrained and at risk (food / area / price) | 46/53, 37/48, 36/50 |
| Turn-2 rate, unconstrained and at risk (food / area / price) | 8/38, 10/49, 7/34 |

## Appendix B: commands

```
git checkout -b goal-inference origin/simulator-study
python -m pip install -e ".[dev,figures,lm]"                                   # login node
python -m robust_likelihood.multiwoz gi-prepare                                # Phase 1 → STOP
python -m robust_likelihood.multiwoz gi-check                                  # Phase 2 → STOP
python -m robust_likelihood.multiwoz gi-rewrites --dry-run                     # Phase 3 → STOP, then without --dry-run
python -m robust_likelihood.multiwoz gi-score --model osim_mid --shard 1/4     # Phase 4, per model and shard (STOP before submit)
python -m robust_likelihood.multiwoz gi-jev --dry-run                          # Phase 5 → STOP, then without --dry-run
python -m robust_likelihood.multiwoz gi-analyze --runs results/multiwoz/*-gi-*
python -m robust_likelihood.multiwoz gi-report
```
