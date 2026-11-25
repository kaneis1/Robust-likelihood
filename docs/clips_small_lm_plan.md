# Plan: CLIPS with small local language models as the utterance likelihood

Execution plan for Claude Code, repository `kaneis1/Robust-likelihood`. Save as `docs/clips_small_lm_plan.md`.

## 0. Purpose

CLIPS (Zhi-Xuan, Ying, Mansinghka & Tenenbaum, AAMAS 2024) infers a human's goal from their actions and one or more instructions. It scores each goal g by Bayesian inverse planning.

For language, the likelihood of an utterance u under a goal is a mixture over the commands c that the goal's plan could produce:

  P(u | g, s, π) = Σ_c P(u | c) · P(c | s, π)

- **The commands:** each c is a subset of the salient actions in the planned policy π, such as `(handover you me key1) where (iscolor key1 red)`. A subset holds at most 3 actions in mDKG and 4 in VH. In mDKG, each command is weighted by how many enumerated subsets produce it; in VH, the weights are uniform.
- **The language model term:** P(u | c) is a language model's probability of " u" plus the stop token "\n", after a prompt of few-shot command → utterance examples ending in `Input: {c}\nOutput:`. The prompt has 18 examples in mDKG and 12 in VH.
- **The models used:** `curie` (the pragmatic model in Doors, Keys & Gems), `text-davinci-002` (its literal listener), and `davinci-002` (VirtualHome). The paper chose base models "due to the more diffuse probabilities they provided".

**The problem:** OpenAI retired `curie` and `text-davinci-002` in January 2024, so the published results can no longer be rerun with the paper's own models.

**The question:** can a small local model compute P(u | c), and through it the likelihood of each goal, as well as the paper's models did? And how robust is the resulting goal posterior to the way the human phrases the instruction?

This is also the second dataset Xuan asked for on 17 Nov, a theory-of-mind task, and Xuan's own.

| Q | Question |
|---|---|
| **K1 Replacement** | With a local base model as P(u \| c), does CLIPS keep its goal accuracy and its agreement with human judgments? |
| **K2 Size and type** | How small can the model be? Do instruct models do worse, as the simulator study suggests? |
| **K3 Robustness** | How much does the goal posterior move under paraphrases, typos, filler and casual style of the instruction? |
| **K4 Planner** | Without the planner, does a small model that reads the goal and a text map do anywhere near as well? (optional) |

## 1. Rules

- **Branch:** create `clips-small-lm` from `origin/week-nov24`.
- **Where things go:**

  | What | Path |
  |---|---|
  | Python code | `src/robust_likelihood/clips/` |
  | Julia bridge scripts | `julia/clips/` |
  | Derived data | `data/clips/derived/` |
  | Results | `docs/clips_results/` |

- **Pinned external code:** clone `probcomp/CLIPS.jl` at commit `6536a2a86d6b4837481a00e630a2e302c90a5278` into `external/CLIPS.jl`, and add `external/` to `.gitignore`. Use its own `Project.toml` pins (Gen 0.4.6, PDDL 0.2.17, SymbolicPlanners 0.1.23, InversePlanning 0.0.1 from `cosilab/InversePlanning.jl#release-0.0.1`, GenGPT3 0.2). Never edit files under `external/`.
- **No OpenAI calls.** All language-model scores come from local models through the table described in §2.2.
- Same rules as the earlier plans:
  - stdlib-first Python, with heavy dependencies as lazy optional extras
  - keys only via `keys.py`; never print, log, or commit a key
  - `python -m pytest` and a commit per phase
  - **[Frozen]** items change only via `docs/clips_decisions.md` plus asking the user
- **STOP and ask the user** at five points:
  1. After Phase 0, if the human data are unavailable.
  2. After the record pass (Phase 1), with pair counts.
  3. Before GPU scoring (Phase 2), with an estimate.
  4. Before the API call for paraphrases (Phase 4).
  5. Before the optional Phase 6.
- Seed 20261126 for everything new. Bootstrap 2,000 replicates over problems.

## 2. Frozen definitions

### 2.1 Data

- **Doors, Keys & Gems (mDKG):**
  - 30 goal assistance problems (`doors-keys-gems/dataset/plans/observed`), each with one instruction
  - 4 goals: `(has human gem1)` … `gem4`
  - the paper's cost profile and rationality prior, from `experiments.jl`
- **VirtualHome (VH):**
  - 25 problems with 35 instructions in total; 10 problems have more than one
  - 6–12 goals per problem, from `virtual-home/dataset/goals`
- **Human judgments (mDKG):** goal and assistance ratings of 100 participants, on the paper's OSF page (https://osf.io/v8ru7/). Download them from the cluster. If that is blocked, **STOP**: ask the user to download them, or to ask Xuan for them.

### 2.2 The table mixture (the only change to CLIPS's model)

`julia/clips/table_mixture.jl` defines `TableMixture(table, role; temperature)`. It is a Gen generative function with the interface `GPT3Mixture` exposes to CLIPS: `simulate`, and `generate` with the output constrained.

- Given prompts p_1 … p_n with weights w_i and the observed output u, its score is logsumexp_i(table[(role, p_i, u)] / τ + log w_i).
- τ is the original mixture's temperature: 1.0, except 0.5 for VH's literal listener.
- **Record mode:** every (role, prompt, output) it is asked about is appended to a log, and the score is 0.
- **Replay mode:** a missing key is an error that names the key. Never fill it with a default.

The bridge scripts swap `pragmatic_gpt3_mixture` and `literal_gpt3_mixture` in both domains for `TableMixture` (roles `mdkg_pragmatic`, `mdkg_literal`, `vh_pragmatic`, `vh_literal`). They do this by rebinding the globals after `include`-ing CLIPS's `src` files, then run CLIPS's own `experiments.jl` loop unchanged. If a global can't be rebound cleanly, copy the experiment loop into `julia/clips/` and note every difference in the decisions file.

### 2.3 The language-model score [Frozen]

For each table key, the score is log P(" u" + "\n" | prompt), summed over the tokens after the prompt:

- the utterance with its leading space, as CLIPS adds it
- followed by one newline

Tokenize the prompt and the continuation separately and concatenate the ids, so the prompt boundary is exact. If the continuation is longer than 64 tokens (`max_tokens`), the score is −∞. Score the raw prompt text with no chat template for every model, instruct models included, as the paper scored `text-davinci-002`.

### 2.4 Models

| Key | Model | Why |
|---|---|---|
| `qwen3_0p6b` | Qwen/Qwen3-0.6B-Base | the student-sized backbone |
| `qwen3_1p7b` | Qwen/Qwen3-1.7B-Base | |
| `qwen3_4b` | Qwen/Qwen3-4B-Base | |
| `qwen3_8b` | Qwen/Qwen3-8B-Base | the simulator study's best reference; about curie's size |
| `smollm2_1p7b` | HuggingFaceTB/SmolLM2-1.7B | a second family at 1.7B |
| `qwen3_8b_inst` | Qwen/Qwen3-8B | instruct contrast for `qwen3_8b` |
| `qwen25_7b_inst` | Qwen/Qwen2.5-7B-Instruct | the claims-run instruct model |
| `osim_mid` (optional) | cmu-lti/osim-8b-mid | a human-utterance model |

Pin each model's revision in `configs/clips.json`. Use bf16 and the same batching rules as `lmcore.py`.

## Phase 0: Setup (CPU)

1. Write `scripts/clips_setup.sh`, which:
   - clones CLIPS.jl at the pinned commit
   - instantiates the Julia environment with its pins (Julia ≥ 1.6; use the cluster's module if there is one)
   - downloads the OSF human data into `data/clips/raw/`
2. **Smoke test:** run CLIPS's `testbed.jl` for mDKG problem `1.1.keys` and VH problem `1.1.stew_chicken`, with `TableMixture` in record mode. It must finish and log prompts.
3. Write `data/clips/derived/utterances.jsonl`: one row per instruction, with its domain, problem, time step, text and true goal.

## Phase 1: Record pass (CPU)

- Run the experiment loops of both domains in record mode, for every method that uses a language model:
  - CLIPS (actions + language)
  - language-only inverse planning
  - the literal listeners
- Write `data/clips/derived/pairs.jsonl`: the unique (role, prompt, output) keys, each with every (problem, goal, method) that used it.
- Check that the commands of each goal match what `enumerate_commands` produces from its plan. Report, per domain:
  - problems
  - goals per problem
  - mean and max commands per goal
  - unique prompts
  - unique pairs

**STOP** with the counts, and the time the record pass took.

## Phase 2: Scoring (GPU)

Command: `python -m robust_likelihood.clips score --model <key>`. It writes `data/clips/derived/scores/<key>.jsonl` with each key and its score.

- The few-shot examples are a long shared prefix. Cache its keys and values once per role and model, if that is simple in your setup. Otherwise batch as usual.
- **Determinism check:** rescore 200 random keys in another batch layout and report the largest difference.
- **Estimate:** at most a few thousand prompts of about 1,000 tokens each. That should be well under one H100-hour per model, but measure it.

**STOP before submitting.**

## Phase 3: Replay and evaluation (CPU)

For each model:
- Run both domains' experiment loops in replay mode with that model's table.
- Write CLIPS's own result files (`experiments_*.csv`, `inferences_per_timestep_*.csv`) under `results/clips/<key>/`.

Compute the paper's Table 1 metrics with CLIPS's own definitions, per domain, model and method:
- **P(g_true):** mean posterior of the true goal at the final step
- assistance precision and recall
- relative plan length and relative human cost (against CLIPS with `qwen3_8b`)
- **Goal correlation (mDKG):** correlation with the mean human goal ratings over (problem, goal)
- **Assistance correlation (mDKG):** correlation with the mean human assistance ratings

Report the paper's published numbers next to them. They come from curie and davinci-002 and cannot be rerun:

| Published (Table 1) | mDKG | VH |
|---|---|---|
| CLIPS, P(g_true) | 0.74 | 0.63 |
| Language-only IP, P(g_true) | 0.55 | 0.37 |
| Action-only IP, P(g_true) | 0.31 | 0.25 |
| CLIPS, goal correlation with humans | 0.93 | – |
| Humans, P(g_true) | 0.67 | – |

Action-only IP uses no language model, so it must reproduce the paper's number exactly up to the planner's randomness. Use it as the check that the setup is right, and **STOP if it misses by more than 0.02.**

## Phase 4: Robustness of the instruction (API for paraphrases, then GPU and CPU)

Perturb each of the 65 instructions (30 mDKG, 35 VH):

| Kind | How |
|---|---|
| `para1`, `para2` | two Claude paraphrases through the existing `rewrites.py` pipeline. Every protected word must be kept, and the NLI check must pass both ways. |
| `typo1`, `typo2` | `perturb.py`'s typo rules with a CLIPS protected list (below) |
| `fill1`, `fill2` | `perturb.py`'s filler layouts. Use only thanks, greeting and lead-in sentences that suit an instruction ("Thanks!", "Hey.", "Quick favour.") |
| `lower`, `nopunct` | as in `perturb.py` |

- **The CLIPS protected list:**
  - colours
  - object nouns: key(s), door(s), gem(s), and every VH object name in `virtual-home/dataset/problems/*.pddl`
  - numerals and number words
  - me, you, I, my, your
  - negations
- Add `perturb.protected_words(extra)` to `perturb.py`, with a default that keeps the old behavior, rather than copying the module.
- Valid units are those where every protected word is unchanged and the text changed. Report units per kind; short instructions will have few typo targets.
- **Scoring:** a perturbed instruction needs only new pairs: its problem's recorded prompts × the new output. Score them with every model, then replay with the instruction swapped. Paraphrases go through the API: **STOP with the count before calling.**
- **Metrics, per model and kind:**
  - median |Δ P(g_true)| at the final step
  - argmax flip rate
  - median |Δ log BF| between the true goal and its strongest rival, using the language term alone
  - for mDKG, the change in goal correlation with humans

## Phase 5: Report

`docs/clips_results/summary.md` opens with one sentence per question (K1 to K3). Then:
- a Table 1 replica per domain, all models as rows, and the published values
- P(g_true) and human correlation against model size, base and instruct marked
- the robustness table
- the prediction scorecard, and notes for Xuan

**Figures** (`.png` + `.pdf` + `.csv`):
1. P(g_true) by model, per domain and method.
2. Goal correlation with humans by model size.
3. |Δ P(g_true)| per perturbation kind and model.
4. Six example problems: the goal posterior under each model, original against paraphrase.

### Pre-registered predictions (failing them is a result)

| ID | Prediction |
|---|---|
| C1 | A local base model reproduces CLIPS: with `qwen3_8b`, mDKG P(g_true) is within 0.05 of 0.74, and goal correlation with humans is ≥ 0.85. |
| C2 | Size: `qwen3_1p7b` is within 0.05 of `qwen3_8b` on P(g_true) in both domains; `qwen3_0p6b` is not. |
| C3 | Instruct models do worse: `qwen3_8b_inst` has lower human goal correlation and lower P(g_true) than `qwen3_8b`. |
| C4 | Language still helps: CLIPS > language-only IP > action-only IP on P(g_true), for every base model of 1.7B or more. |
| C5 | Robustness: for base models of 1.7B or more, median \|Δ P(g_true)\| ≤ 0.10 under `para1` and ≤ 0.05 under `typo1`, with argmax flips ≤ 10%. |
| C6 | Instruct models are less robust: their argmax flip rate under `para1` is higher than their base counterparts'. |

## Phase 6 (optional; STOP before): no planner

- **The scorer:** the language model scores the instruction given a text description of the goal and the ASCII map of the state (`doors-keys-gems/src/ascii.jl`). The prompt is "Map: … Goal: the human wants gem 2. Instruction:". There are no commands and no planner. mDKG only.
- **The question:** does a small model need CLIPS's planner and commands to read the instruction pragmatically?
- **Prediction C7:** P(g_true) without the planner is below language-only IP for every model.

## Not in this plan

- **A trained student for P(u | c):**
  - Distill `qwen3_8b`'s scores into a 0.6B model, trained on synthetic utterances that the few-shot model generates from enumerated commands.
  - This is the CLIPS counterpart of Part D of `docs/week_nov24_plan.md`, and the cross-dataset test Xuan asked for.
  - Plan it after both have results.
- Online assistance, new problems, and sequential Monte Carlo over larger goal spaces.

## Appendix: commands

```
git checkout -b clips-small-lm origin/week-nov24
bash scripts/clips_setup.sh                                                   # Phase 0 (STOP if no human data)
julia --project=external/CLIPS.jl julia/clips/record.jl                       # Phase 1 → STOP
python -m robust_likelihood.clips score --model qwen3_8b                      # Phase 2, per model (STOP before)
julia --project=external/CLIPS.jl julia/clips/replay.jl --model qwen3_8b      # Phase 3, per model
python -m robust_likelihood.clips perturb && python -m robust_likelihood.clips score --model qwen3_8b --perturbed   # Phase 4
python -m robust_likelihood.clips report                                      # Phase 5
```