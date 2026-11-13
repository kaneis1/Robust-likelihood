# Plan: which simulator gives human text a reasonable probability?

Execution plan for Claude Code, repository `kaneis1/Robust-likelihood`. Save as `docs/simulator_study_plan.md`.

## 0. Purpose

Typicality v0 showed that Qwen2.5-7B-Instruct's own replies have surprisal of 2–11 nats, while real MultiWOZ first turns have a median of 101. Every typicality variant built on raw surprisal therefore put 100% of human pairs in its extreme bin.

Xuan's direction (Nov 2): run the estimation process with a model that assigns human messages reasonable probability, such as a base model or a human-simulator model.

This study compares candidate simulators on the 235 clean MultiWOZ items. It answers five questions:

| Q | Question |
|---|---|
| **Q1 Coverage** | Do human first turns fall inside the model's own sample distribution? |
| **Q2 Fidelity** | Does the model name slots at human rates? Does its raw log-likelihood reproduce the corpus Occam ratio? |
| **Q3 Robustness** | Are its Bayes factors still fragile under meaning-preserving rewrites? This is claim 1 with a human-like model. |
| **Q4 Typicality** | Do the typicality v0 variants regain signal when this model is the reference? |
| **Q5 What RL does** | OSim-8B-Mid vs the final OSim-8B: does task RL plus top-1-of-8 distillation narrow the distribution again? |

## 1. Rules

- Create branch `simulator-study` from `origin/claims-123`. It already has `lmcore.py`, `multiwoz/semantic.py`, `rewrites.py`, `claim1_lm.py`, and `typv0.py`; reuse them and don't change their behavior.
- Same rules as the earlier plans:
  - preserve CRLF; stdlib-first; heavy dependencies as lazy optional extras
  - keys only via `keys.py`
  - `python -m pytest` and a commit per phase
  - **[Frozen]** items change only via `docs/simulator_decisions.md` plus asking the user
- **STOP and ask the user** at three points:
  1. After the format check (Phase 1).
  2. Before submitting any GPU job.
  3. Before the optional multi-turn pilot.
- **Login node first.** Download every model there with `huggingface-cli download <id>` and record each model's exact revision hash. Job scripts set `HF_HUB_OFFLINE=1` and contain no `pip install`.
- This study needs no API calls. The rewrites from the claims plan are reused as-is.

## 2. Models [Frozen; verify each id loads before Phase 1]

| Key | Hugging Face id | Role | Prompt format |
|---|---|---|---|
| `qwen25_inst` | `Qwen/Qwen2.5-7B-Instruct` | the current reference, for continuity | S, plus `P0` (the exact Phase 8 prompt) |
| `qwen25_broad` | `Qwen/Qwen2.5-7B-Instruct` | can prompting fix it? (Xuan's alternative) | B |
| `qwen3_base` | `Qwen/Qwen3-8B-Base` | OSim's starting point, before midtraining | F (few-shot) |
| `qwen3_inst` | `Qwen/Qwen3-8B` (thinking off) | the instruct model OSim's RL experts start from | S |
| `osim_mid` | `cmu-lti/osim-8b-mid` | midtrained only | S, plus N (native) |
| `osim_final` | `cmu-lti/osim-8b` | midtraining, task RL, and expert distillation | S, plus N (native) |
| `userlm` | `microsoft/UserLM-8b` | another human simulator (Llama base) | S, plus N (native) |
| `osim_inst` (optional) | `sunweiwei/OSim-Inst-8B` | post-trained from instruct | S |

If `cmu-lti/osim-8b` doesn't resolve, try `cmu-lti/osim-8b-post`. If neither works, STOP.

Record each model card's license in the manifest. Both OSim cards and the UserLM card list MIT; this is research use.

**The ladder** `qwen3_base` → `osim_mid` → `osim_final` (all Qwen3 tokenizer) isolates midtraining from RL. `qwen3_inst` vs `osim_final` compares helpfulness tuning with behavioral tuning.

## 3. Prompt formats [Frozen after Phase 1]

H is the hypothesis text from `data/multiwoz/derived/probe.jsonl`; U is the human first turn.

- **S, simulator chat (primary for every chat model):**
  - system: `You are a user talking to a restaurant booking assistant. {H}`
  - user (the other party's opening line, the same for everyone): `Hello, welcome to the restaurant booking service. How can I help you?`
  - assistant: U, then end-of-turn

  The opener gives every chat model the same structure, and it cancels across hypotheses.
- **N, native:** OSim and UserLM follow their model card exactly, with the goal as the social-context/intent system prompt and **no opener**, if the card allows generating a first turn from the system prompt alone. Use `{H}` as the intent text. This is a sensitivity check against S.
- **B, broadened:** S with this system prompt:
  `You are a real person typing to a restaurant booking assistant. Real people write in many different ways: short or long, casual or formal, sometimes with typos. {H}`
  Keep this wording exactly. Don't mention leaving out details: that would change slot-naming rates by instruction.
- **F, few-shot (base model):** plain text, no chat template.
  - Header: `The following are first messages that customers typed to a restaurant booking assistant, each after a short description of the customer.`
  - Then 8 examples, each `\n\nCustomer: {H_k}\nMessage: {U_k}`.
  - Then `\n\nCustomer: {H}\nMessage:`, scoring ` {U}` plus the terminating `\n`.
  - Samples stop at the first `\n`.
  - **Examples:** 8 restaurant first turns from **non-probe** dialogues (MUL/PMUL dialogues in the train split whose first turn opens the restaurant domain). Seed 20261103. Cover 2- and 3-slot goals and include one turn that omits a constrained slot. H_k is rendered with the probe's `base()` template.
  - **Caveat (report it):** showing 8 human examples is an advantage the chat models don't get.
- **P0:** the exact Phase 8 prompt for `qwen25_inst`. Reproducing the earlier numbers checks the code path.
- **Qwen3 thinking:**
  - Call `apply_chat_template(..., enable_thinking=False)` whenever the template accepts it (`qwen3_inst`, and the OSim models if their template has thinking).
  - Assert that no sample contains a non-empty `<think>` block.
  - The scored prefix must include whatever empty think block the template inserts, identical across hypotheses.
- **Boundaries:**
  - Use `lmcore.end_of_turn_id` and `boundary_holds` for every model and format.
  - For F, the end marker is `\n`.
  - Log every boundary failure; if more than 1% of pairs fail, STOP.

## Phase 1: Format check (small GPU or MPS run)

For each model × format, draw 8 samples for each of 6 seeded goals (a mix of 2- and 3-slot goals) with the Phase 2 sampling settings. Write them to `docs/simulator_results/format_check.md`.

Automated flags per model:

- empty replies
- truncation (hit `max_new_tokens` with no end marker)
- assistant phrasing at the start ("How can I help", "Sure", "As an AI", "Certainly", "Here is")
- non-empty think blocks
- replies that answer as the restaurant ("We have…", "Our menu…")
- a median word count

Also confirm that P0 reproduces the earlier Qwen numbers on 20 pairs, within the 1-nat batch noise.

**STOP: the user reads the samples.** The user decides, per model, whether S or N becomes primary. Record the decision in `docs/simulator_decisions.md`; after that, the formats are frozen.

## Phase 2: Scoring and sampling (GPU, one job per model)

Command: `python -m robust_likelihood.multiwoz sim-score --model <key> --format <fmt>`. Each run writes to `results/multiwoz/<UTC>-sim-<key>-<fmt>/`.

1. **Human pairs:** S(U | H) for all 2,355 clean probe rows. Optionally add the 1,372 fail_info rows behind `--include-fail`.
2. **Samples:**
   - K = 64 per distinct hypothesis text in the clean probe; record the count, about 1,000.
   - Pure sampling, set **explicitly**: `do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=96`. Keep the `sampling_is_exact` self-check.
   - Save each sample's text, its own-prompt surprisal, and its surprisal under `broad|impl`, which the centered variants need.
3. **Rewrites** (the accepted ones from `rewrites.py`):
   - each U′ under every hypothesis of its item
   - each U under every accepted H′
4. **Hypothesis-rewrite samples, robustness subset only:** K = 64 per accepted H′ for the 60-item subset. Run this only for `osim_mid`, `osim_final`, and `qwen25_inst`, for the per-hypothesis and semantic typicality variants.
5. **Manifest:** model id and revision, format and full prompt text, settings, counts, truncation and boundary-failure rates, and timing.

**Estimate:** about 1.5–3 A100-hours per model and format; the total is about 15–25 GPU-hours. Write `scripts/sim_job.sh` with LSF (`bsub`) and PBS (`qsub`) placeholders and one model per job. **STOP before submitting**, with the per-model estimate.

## Phase 3: Metrics (`sim-analyze`)

Compute every metric per (model, format). Bootstrap over items, 2,000 replicates, seed 20261103.

### Q1, coverage

- **Human percentile.** For each pair, the share of H's 64 samples whose surprisal is ≥ S(U | H), i.e. how surprising the human turn is relative to the model's own replies.
  - Report the median, the share below 0.05, and the share **beyond all samples** (percentile 0).
- **Standardized gap.** (S(U|H) − μ_H)/σ_H: median and 90th percentile.
- **Surprisal per word,** human vs samples. This separates "longer" from "less likely".
- **Semantic coverage.** The share of human turns, under their gold G, whose slot tuple (regex extractor) appears among G's 64 samples. Also the median meaning-level probability P_sem.
- **Style.** For human turns and for samples: median words; share opening with a greeting; share containing a question; share with at least one non-dictionary token (a typo proxy); distinct-2. Report per model next to the human values.

### Q2, fidelity

- **Model slot-naming rates**, from samples with the regex extractor:
  - a_model(X): share of samples under G that name X, for X constrained
  - b_model(X): share of samples under `minus:X|impl` and `minus:X|expl`, separately, that name X
  - Report a and b next to the corpus values (a ≈ 0.73, b ≈ 0.007–0.015) and the model's own ln(a/b).
- **Raw log-likelihood Occam gap.** Run the existing `metrics.py` with scores = −S on the clean pairs: mean L_J vs corpus L_E for named and omitted single-slot pairs and full removals, explicit and implicit. Also the explicit-vs-implicit wording gap.

### Q3, robustness

The claim 1 metrics from `claim1.py`, recomputed with each model's −S:

- median |Δ log Bayes factor| and flip rate by rewrite kind
- ICC across hypothesis paraphrases, plus the within-to-between SD ratio
- posterior instability: SD of P(G) and argmax flip rate over 50 wording draws

### Q4, typicality

The six `typv0` variants plus centered/6-bins, recomputed with each model as the reference:

- share of human pairs in the extreme bin
- size principle kept (L_J ÷ L_E, explicit and implicit)
- contradiction log Bayes factor
- bin agreement under rewrites

### Q5, Mid vs final

For `osim_mid` vs `osim_final` (and `qwen3_base`, `qwen3_inst` as anchors):

- **Concentration:**
  - mean sample surprisal per hypothesis (an entropy estimate)
  - distinct sample texts out of 64
  - the share of the most frequent sample
- **Q1–Q4 side by side,** with paired bootstrap differences.

### Pre-registered predictions (failing them is a result)

| ID | Prediction |
|---|---|
| S1 | `osim_mid` has the best coverage: the highest median human percentile and the lowest share beyond all samples. The instruct models (`qwen25_inst`, `qwen3_inst`) have the worst. |
| S2 | RL narrows the distribution again: `osim_final`'s mean sample surprisal is ≥ 30% lower than `osim_mid`'s, and its human coverage is worse. |
| S3 | `osim_mid`'s a_model is within ±0.10 of the corpus a, and b_model within a factor of 3 of the corpus b. Instruct models have a_model ≥ 0.9 (they name every slot), which explains their Occam overshoot. |
| S4 | Claim 1 holds for every model: median \|Δ log Bayes factor\| under `para` and `zh` > 2 nats, including `osim_mid`. |
| S5 | With `osim_mid` as the reference, global one-sided typicality puts < 50% of human pairs in its extreme bin (vs 100% with `qwen25_inst`). |
| S6 | `osim_mid` beats `userlm` on coverage, and `qwen25_broad` improves on `qwen25_inst` but stays below `osim_mid`. |

## Phase 4: Report (`sim-report`)

Write `docs/simulator_results/summary.md` with one section per question, a prediction scorecard, a **recommendation** (which simulator to use for pipeline v0, and with which format), and notes for Xuan.

**Figures** (`.png` + `.pdf` + `.csv`):

1. Per model: box plots of human surprisal vs own-sample surprisal, on the same axis.
2. Per model: a_model and b_model next to the corpus values.
3. Raw log-likelihood L_J vs corpus L_E per model (named / omitted; explicit / implicit).
4. Median |Δ log Bayes factor| by rewrite kind per model.
5. Typicality extreme-bin share by variant × model.
6. Mid vs final: the concentration metrics, plus 10 example samples for the same goal from each.

Commit the results folder, without raw outputs.

## Phase 5 (optional; STOP before running): multi-turn pilot

Xuan asked for later turns to be scored given the goal and the history.

- For the S-format chat models, score user turns 2 and 3 of the 235 clean dialogues. The context is the real MultiWOZ history: earlier user turns and system turns, in their correct roles. The hypothesis set is unchanged.
- Q1 coverage only, with 32 samples per (dialogue, turn, hypothesis), on a 60-dialogue subset.
- This tests whether the coverage ranking holds once history is in the prompt.

## Appendix: commands

```
git checkout -b simulator-study origin/claims-123
python -m pip install -e ".[dev,figures,lm]"                      # login node
huggingface-cli download <each model id>                           # login node, record revisions
python -m robust_likelihood.multiwoz sim-check --all               # Phase 1 → STOP
python -m robust_likelihood.multiwoz sim-score --model osim_mid --format S     # Phase 2, per model (STOP before submit)
python -m robust_likelihood.multiwoz sim-analyze --runs results/multiwoz/*-sim-*
python -m robust_likelihood.multiwoz sim-report
```