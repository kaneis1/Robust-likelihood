# Plan: evidence for the three claims

Execution plan for Claude Code, repository `kaneis1/Robust-likelihood`. Save as `docs/claims_plan.md` and run the steps in order.

## 0. The three claims and the four steps

The paper, in Xuan's abstract of Oct 26, needs three results:

| Claim | Statement |
|---|---|
| **C1** | Token log-probs are correct in a narrow sense, but fragile: sensitive to typos, paraphrase, language, length, and hypothesis rewording. |
| **C2** | Verbalized probabilities work in small discrete domains, but fail for high-dimensional language and disagree with generation probabilities. |
| **C3** | Semantic typicality fixes both. |

| Step | Mainly serves | Needs | Rough time |
|---|---|---|---|
| 1. Dice sweep | C2 (and C1 "correct") | 1 GPU or a ≥32 GB Mac (MPS) | 1–2 days |
| 2. Phase 8 + rewrite suite on MultiWOZ | C1, C2 | GPU node, API keys | about 1 week |
| 3. Typicality v0 from saved samples | C3 | outputs of step 2; **Xuan's decision** | 2–3 days + meeting |
| 4. Medical generator + trained estimator | C3 (trained, calibrated) | DDXPlus, GPU | 2–3 weeks |

**Assumption used throughout [Frozen].** The inference target is the user's **meaning**. Typos, paraphrase, language, and padding are nuisance variation. Under this assumption, a correct likelihood gives the same **log Bayes factor** between hypotheses for any two utterances with the same meaning. It also gives the same likelihood for any two wordings of the same hypothesis. Robustness is always measured on log Bayes factors, bins, or posteriors, **never on raw log p**: a longer string legitimately has a lower probability.

## 1. Rules

- Branch `claims-123` from `origin/multiwoz-size-principle`. `main` does not have Phase 8; if that branch has since been merged, branch from `main`.
- Same rules as `docs/multiwoz_size_principle_plan.md`:
  - preserve CRLF files and don't reformat them
  - stdlib-first, with heavy dependencies as lazy optional extras
  - keys only via `keys.py`, never printed or committed
  - `python -m pytest` and a commit at the end of every step
  - **[Frozen]** items change only via `docs/claims_decisions.md` plus asking the user
- **STOP and ask the user** at every point marked STOP. They mark API spend, GPU job submission, human review, and the typicality decision.
- **GPU jobs.** Install the extras and pre-download every model on the login node with `huggingface-cli download <id>`. Compute nodes may lack internet, so job scripts set `HF_HUB_OFFLINE=1` and contain no `pip install`. Write each job script with LSF (`bsub`) and PBS (`qsub`) placeholders, as `scripts/multiwoz_lm_job.sh` does.
- **Reference results.** Commit to `docs/<step>_results/`:
  - `summary.md`
  - `metrics.json`
  - figures as `.png` + `.pdf` + `.csv` of the plotted values
  - `manifest.json`: git commit, config, model ids and revisions, counts, timing

  Raw outputs stay in gitignored `results/`.

### Shared refactor (do first)

1. **Move, don't rewrite,** the general functions of `src/robust_likelihood/multiwoz/lm.py` into `src/robust_likelihood/lmcore.py`: `surprisals`, `scored_sequence`, `end_of_turn_id`, `sampling_config`, `sample_replies`, `sampling_is_exact`, `load_model`, `resolve_revision`, `pick_device`, `text_seed`. Generalize the prompt builders to take a list of chat messages.
   - `multiwoz/lm.py` imports from `lmcore`, and its behavior and tests stay unchanged.
2. **Add `src/robust_likelihood/verbal.py`:**
   - `local_verbal(model, tok, messages, samples=0)`: greedy by default. Every decoding option is set explicitly, as in `lm_probs.py`.
   - `api_verbal(provider, messages)`: through the existing transport, keys, and the `gpt`/`claude` endpoints and model ids in `configs/models.json`. If the existing request builders don't fit a free-form prompt, add a minimal chat-request function **without changing the existing builders**.
   - `parse_probability(text)`: accepts `0.3`, `30%`, `3e-5`, and `1/6`; returns `None` otherwise. Always keep the raw text.
3. **Model ids [Frozen; verify availability first]:**
   - primary LM: `Qwen/Qwen2.5-7B-Instruct`
   - second family: `allenai/OLMo-2-1124-7B-Instruct`
   - NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

   If a model can't be downloaded, STOP and ask; don't substitute silently.

---

## Step 1: Dice sweep

Package `src/robust_likelihood/dice/`, config `configs/dice.json`, CLI `python -m robust_likelihood.dice {build,score,analyze,report}`.

**Question.** Do verbalized probabilities track probability as the observation space grows from 6 outcomes to 6¹⁰, with exact ground truth? Token log-probs are the comparison.

### Design [Frozen]

| Hypothesis | Wording | True p(face) |
|---|---|---|
| `fair` | "a fair six-sided die" | 1/6 each |
| `even` | "a six-sided die that only ever lands on 2, 4, or 6, each equally often" | 1/3 for 2, 4, 6 |
| `loaded` | "a six-sided die loaded so that it lands on 6 half of the time, with the other faces equally likely" | 6: 1/2; others 1/10 |

- **k (rolls)** ∈ {1, 2, 3, 4, 6, 8, 10}. For each k and each generating hypothesis, draw 20 sequences with seed 20261027. That gives 420 sequences, each scored under all 3 hypotheses: **1,260 pairs**. The exact log-likelihood comes from the table.
- **Observation text:** the results as a comma-separated list, e.g. `3, 5, 2`.

### Scorers

- **S1: token log-prob** (Qwen and OLMo).
  - system: `You simulate dice rolls exactly as described.`
  - user: `The die: {wording}. Roll it {k} times and report the results in order as a comma-separated list of numbers. Reply with the list only.`
  - assistant: the list plus end-of-turn. Score with `lmcore`.
- **S2: verbalized, judgment framing.** This is how LLM-as-likelihood pipelines ask.
  `Someone rolled {wording} {k} times and reported the results in order: {list}. How likely is it that they would report exactly this sequence? Answer with a single probability between 0 and 1 and nothing else.`
- **S3: verbalized, computation framing (control).**
  `{Wording, capitalized} is rolled {k} times. What is the probability that the results, in order, are exactly {list}? Answer with a single number between 0 and 1 and nothing else.`
- **Where S2 and S3 run:**
  - local Qwen, greedy, on all 1,260 pairs
  - `claude` and `gpt` via the API, at temperature 0 where supported, on a 630-pair subset (10 sequences per (k, generator))

  **STOP after a 30-pair API pilot** and report the cost estimate.

### Normalization

- **k = 1:** verbalized probability of each of the 6 faces under each hypothesis, summed. True sum = 1.
- **k = 2:** all 36 sequences under `fair`, summed.
- **Token log-probs:** report Σ exp(log p) over the same sets. This should be ≤ 1, and close to 1 if the model puts its mass on correctly formatted lists.

### Metrics

1. **Slope** of log₁₀(score) against k, per scorer, on sequences scored under their own generating hypothesis. True slopes: `fair` −0.778, `even` −0.477 per roll.
2. **Accuracy at k = 1:** |verbalized − true|.
3. **Bayes factors:** predicted vs true log Bayes factor, e.g. `even` vs `fair` on all-even sequences, where the truth is k ln 2. Fit the slope.
4. **Normalization sums.**

### Figures

- log₁₀ P against k: truth, S1, S2, S3, one panel per model
- predicted vs true log Bayes factor
- normalization sums

### Pre-registered predictions

Report a pass/fail scorecard; failing is a result.

| ID | Prediction |
|---|---|
| D1 | S1 slope within a factor of 2 of the truth for `fair`, with R² > 0.9 ("correct in a narrow sense") |
| D2 | S2 judgment-framing slope \|slope\| < 0.15 per roll, i.e. under 20% of the true slope |
| D3 | S2 at k = 1 within ±0.10 of the truth (small domains work) |
| D4 | S2 sum over the 36 sequences at k = 2 > 2 |
| D5 | No prediction for S3. If S3 tracks the truth, the paper says verbalization works only when the model is told the generator and can do arithmetic, which never happens for language. |

**Acceptance:** sequences are reproducible from the seed; parse failures under 2% for local models; every raw answer is kept.

---

## Step 2: Phase 8 + rewrite suite on MultiWOZ

Extends `src/robust_likelihood/multiwoz/` with new modules `semantic.py`, `rewrites.py`, `claim1.py`, `verbalcheck.py`, and new CLI subcommands. The probe, the items, and the reference ratios come from the existing pipeline.

### 2a. Semantic content extractor (`semantic.py`)

- A deterministic regex extractor maps text to a slot tuple (food, area, pricerange). Each slot is a value, `dontcare`, or `none`.
- Vocabulary:
  - **food:** every value in restaurant goals and `ontology.json`
  - **area / price:** from `schema.json`
  - **synonyms [Frozen list, in code]:** centre/center/central/city centre → centre; cheap/inexpensive/budget → cheap; moderately priced/mid-range/reasonably priced → moderate; expensive/upscale/high-end/pricey → expensive; "any type of food", "any area", "any price", "don't care" phrasings → dontcare
- **Acceptance:** at least 95% per-slot agreement with the 2.2 turn-0 annotations on the 370 items. Report the confusion table; if agreement is below 95%, STOP.

### 2b. Rewrites (`rewrites.py`)

Applied to the **235 clean items**. Rewrite generation uses the `claude` provider at temperature 1, with frozen prompts and a recorded model id. If the API is unavailable, use local Qwen with the same prompts and record that.

**Utterance-side rewrites** (U → U′):

| Kind | How |
|---|---|
| `typo` | Programmatic, seeded: about one edit per 12 characters (adjacent swap or dropped letter). Never edit a slot-value word. |
| `pad` | Prepend one of 5 fixed neutral openers and append one of 5 fixed closers (seeded), e.g. "Hi there! Hope your day is going well." / "Thanks so much!" |
| `para1`, `para2` | LLM: "Rewrite this message so it means exactly the same thing. Keep every detail (cuisine, area, price, people, day, time) and add nothing." |
| `verbose` | LLM: "Rewrite this message to be about twice as long without adding any new information, preference, or detail." |
| `zh` | LLM translation into Chinese |

**Hypothesis-side rewrites** (H → H′), for every distinct hypothesis text in the clean probe:
- `hpara1`–`hpara3`: LLM paraphrases. Explicit-regime texts must keep their "no preference" statements.
- `hzh`: Chinese translation.

**Validity checks [Frozen].** A rewrite is accepted only if:
1. The regex slot tuple is unchanged. For U′ compare with U's annotated tuple; for H′ compare with the constrained slots. For `zh`, check the back-translation produced by the same LLM.
2. NLI entailment ≥ 0.8 in both directions between the original and the rewrite. For `zh`, use the back-translation.

Regenerate a rejected rewrite up to 2 times, then drop it, and report acceptance rates per kind.

**Human check:** export 100 random accepted rewrites (50 utterance-side, 50 hypothesis-side) to `docs/claim1_results/rewrite_audit.csv`. **STOP for the user's review.** Proceed only if at least 95% are judged meaning-preserving.

**Robustness subset:** a seeded 60-item subset of the clean set, used for expensive scorers.

**API budget:** about 3,500 calls including back-translations. **STOP after a 20-item pilot** with a cost estimate.

### 2c. GPU job A: language-model scoring (`scripts/claim1_job_a.sh`)

Primary model Qwen; extends `lm-score`. Each part writes its own jsonl.

1. **Phase 8 as already implemented** on the full probe (370 items, K = 64 samples per distinct hypothesis text). Keep `lm_samples.jsonl` with sample texts and per-sample surprisal S.
2. **Centering reference.** Score every probe utterance **and every sample** under `broad|impl` ("The user wants to find a restaurant."). This is needed for the centered score S(U|H) − S(U|broad|impl), i.e. domain-conditional PMI, and its reference distribution.
3. **Utterance-side rewrites:** score each accepted U′ under every hypothesis of its item. About 1,400 utterances × 10 hypotheses.
4. **Hypothesis-side rewrites:** score each original U under every accepted H′. About 9,400 pairs.
5. **Rewritten hypotheses, robustness subset only:** draw K = 64 samples under each accepted H′ text of the 60 items. This is needed for the per-hypothesis and semantic typicality variants in step 3.
6. **Second family:** repeat parts 3–4 with OLMo, scoring only (no sampling).

Estimated time: 4–8 GPU-hours on one A100. **STOP before submitting**; report the job script and the estimate.

### 2d. GPU job B: verbalized scores and self-consistency (`verbalcheck.py`)

Everything here is on the robustness subset (60 items × their hypotheses).

- **Exact-string framing:**
  `A user of a restaurant booking assistant has the following goal: {H}. What is the probability that this user's first message to the assistant is exactly: "{U}"? Answer with a single probability between 0 and 1 and nothing else.`
  Asked of local Qwen (greedy) on the original pairs plus every rewrite.
- **Meaning framing:** same prompt, ending instead with `…that this user's first message asks for exactly the following and nothing else: {slot description of U}?`, where the slot description is rendered from U's annotated tuple.
- **API:** `claude` and `gpt`, exact-string framing only, on original pairs plus `para1` and `hpara1`. That's about 1,800 calls per provider. **STOP before running.**
- **Jev:** the existing `run` path on `para1`, `zh`, `hpara1`, and `hzh` for the 60 items, about 2,400 calls. **STOP before running.**
- **Ground truths for C2:**
  - Exact-string generation probability: exp(−S(U|H)) from job A.
  - Meaning-level generation probability: P_sem(U|H) = (number of H's 64 samples with the same slot tuple as U + 0.5)/(64 + 1).
- **Normalization:** for 20 items, ask the exact-string question for 20 distinct samples drawn under G (from `lm_samples.jsonl`) and sum.

### 2e. Claim 1 and claim 2 analysis (`claim1.py`)

Scorers compared:
- token log-prob (Qwen, OLMo)
- **clipped** token log-prob: max(ℓ_H − max_H′ ℓ_H′, −3) nats [Frozen cap]
- centered token log-prob
- verbalized (local; API subset)
- Jev (subset)

Metrics:

1. **Correct in a narrow sense:** the Occam gap of Qwen's raw log-likelihood on clean pairs, using the existing `analyze --lm` machinery against the corpus ratios.
2. **Hypothesis rewording:**
   - ICC(1) of each scorer across {original, `hpara1`–`hpara3`, `hzh`} within each (item, semantic hypothesis)
   - the rewording-to-hypothesis ratio: SD within a semantic hypothesis ÷ SD between semantic hypotheses, for the same U
3. **Utterance rewrites:** |Δ log Bayes factor| between original and rewrite, for the pairs (G, minus:X), (G, broad), and (G, contra), reported:
   - absolute and relative to |log Bayes factor|
   - with the rate at which the preferred hypothesis flips
   - by rewrite kind
4. **Posterior instability:**
   - Per item, form a pool with one textual variant per semantic class from {G, minus:X (implicit), broad (implicit), contra}, with a uniform prior. Draw 50 times.
   - Report the SD of P(G) and the argmax flip rate.
5. **C2 self-consistency** (from 2d):
   - log₁₀ of the verbalized answer against log₁₀ exp(−S) and against log₁₀ P_sem, with rank correlation and median gap in orders of magnitude
   - normalization sums

Bootstrap CIs over items, 2,000 replicates, seed 20261027.

### Pre-registered predictions

| ID | Prediction |
|---|---|
| R1 | Raw log-likelihood Occam gap on named single-slot pairs is positive in both regimes and its L_J is at least half the corpus L_E ("correct") |
| R2 | ICC across hypothesis rewordings < 0.5 for token log-prob; translating a hypothesis to Chinese moves the median log Bayes factor by > 2 nats |
| R3 | Under `pad` and `typo`, the median \|Δ log Bayes factor\| < 0.5 nats even though raw log p moves a lot. Under `para` and `zh` it is > 1 nat. |
| R4 | Verbalized vs exact-string generation probability: median gap > 6 orders of magnitude; normalization sum over 20 samples > 1 |
| R5 | Verbalized vs meaning-level P_sem: rank correlation < 0.5 |

Commit `docs/claim1_results/`.

---

## Step 3: Typicality v0 from saved samples

Module `src/robust_likelihood/multiwoz/typv0.py`, config `configs/typicality_v0.json`. No new GPU work: everything is computed from step 2 outputs.

### Definitions [Frozen]

**Inputs.** For each hypothesis text h:
- K samples x_k with surprisal S(x_k|h)
- for every scored (U, h): S(U|h)

**Pool statistics.** The pool is the distinct hypothesis texts in the clean probe, with two weightings:
- uniform
- prior: each goal pattern weighted by its corpus frequency among the clean items

Compute:
- H̄ = Σ w(h) μ_h
- σ̄ = √Σ w(h)[σ_h² + (μ_h − H̄)²]

**Variants:**

| Id | Statistic s | Normalization | Bins | Score |
|---|---|---|---|---|
| `per2A` | S | per-hypothesis: T = (s − μ_h)/σ_h | two-sided on \|T\|, edges (1, 2) | A: log P_h(bin), add-one over h's samples. This is Xuan's current formula and Phase 8's `typicality_scores`. |
| `glob2A` | S | global: T = (s − H̄)/σ̄ | two-sided, edges (1, 2) | A |
| `glob1A` | S | global | one-sided, edges (−1, 0, 1) | A |
| `glob1B` | S | global | one-sided, edges (−1, 0, 1) | B: −(H̄ + σ̄·c_b), with c = (−1.5, −0.5, 0.5, 1.5); extreme bins capped |
| `cent1B` | S(·\|h) − S(·\|broad\|impl) | global, using the centered values of the samples | one-sided | B |
| `sem1B` | S_sem = −log P_sem | global, using the samples' S_sem | one-sided | B |

Under option A, P_h(b) is the share of h's own samples whose T falls in bin b.

Baselines: raw log-likelihood, clipped log-likelihood, verbalized, Jev.

Sensitivity:
- 3 bins with edges (−0.5, 0.5) and 6 bins with edges (−2, −1, 0, 1, 2)
- `sem1B` with factorized P_sem (product of per-slot frequencies), because K = 64 caps the joint P_sem at about 4.8 nats

### Metrics (per variant)

1. Occam gap against the corpus, named and omitted.
2. Information: contradiction AUROC and mean log Bayes factor G vs contra on clean pairs.
3. The step 2 robustness metrics. Add **bin agreement** under utterance rewrites: the share of (U′, h) whose bin equals that of (U, h).
4. **Pool sensitivity:** recompute H̄ and σ̄ with a random half of the pool (20 draws); report the change in log Bayes factors and the bin-flip rate.
5. **Dice check.** On the step 1 data, compute `per2A`, `glob2A`, `glob1A`, and `glob1B`, using exact entropies and SDs of each die. Each face of `fair` and `even` has constant surprisal, so `per2A` should tie the two on all-even sequences; that is the number-game argument, shown empirically.

### Predictions

| ID | Prediction |
|---|---|
| T1 | `per2A` erases the size principle: named-slot Occam gap ≤ −0.8·L_E |
| T2 | Option A variants tie or erase it; option B variants keep it: L_J ≥ 0.5·L_E |
| T3 | `sem1B` has hypothesis-rewording ICC > 0.9 and bin agreement > 0.9 under `para` and `zh`, where raw log-prob ICC < 0.5 |
| T4 | `cent1B` bin agreement under `pad` and `typo` > 0.9, above `glob1B` |

### Decision memo

Write `docs/typicality_v0_decision.md`: one table of variants × metrics, the dice check, and a recommendation. **STOP: the user takes it to Xuan.** Record the chosen variant, bin edges, and pool weighting in `configs/typicality_v0.json` as **[Frozen]**. Steps 4c onward use that choice.

---

## Step 4: Medical generator (trained estimator and calibration)

Package `src/robust_likelihood/medical/`, config `configs/medical.json`, results in `docs/medical_results/`.

**Why.** Here p(s|z) is **known exactly**, because the generator is defined by us with parameters estimated from DDXPlus. That makes the training labels and the calibration claim checkable. DDXPlus itself comes from a proprietary knowledge base whose probabilities are not released, so we do not claim to recover them.

### 4a. Data

- Download the **English** DDXPlus release (CC-BY-4.0) from figshare: `release_conditions.json`, `release_evidences.json`, and the train, validate, and test patient CSVs. The article is linked from the README of `github.com/mila-iqia/ddxplus`.
- Resolve file URLs through the figshare API and record sha256 values in `data/medical/checksum.json`. If access fails, STOP.
- Parse `EVIDENCES`. Map evidence codes to English question text from `release_evidences.json`.

### 4b. Generator spec, written to `docs/medical_generator_spec.md`, then STOP for review

1. **Pathologies Z:** the 12 pathologies with at least 5,000 train patients whose top-20 binary symptom sets overlap most (greedy max mean Jaccard). These are near misses by design. Print the list for the user to confirm.
2. **Symptom vocabulary E:** binary **symptom** evidences only (no antecedents, no categorical evidences), the top 20 by mutual information with the pathology among Z. Each gets a lay name, written by an LLM and checked by the user in the spec.
3. **Model [Frozen once approved]:**
   - p(e|z) = (count + 1)/(N_z + 2) from the train split
   - s ~ Π_e Bernoulli(p(e|z)), conditioned on s ≠ ∅
   - p(s|z) = Π_e p(e|z)^s_e (1 − p(e|z))^(1−s_e) / (1 − Π_e(1 − p(e|z))), exact
4. **Broad hypotheses:** 3 clinically sensible groups, e.g. upper respiratory, lower respiratory, ENT, proposed in the spec. p(s|group) = Σ_z p(z|group) p(s|z), with p(z|group) from train frequencies. This is the exact size-principle test.
5. **Hypothesis texts:** each pathology and group gets a lay description, plus 3 paraphrases and a Chinese translation, validated as in 2b.
6. **Splits:** by symptom set s. Train, validation, and test sets are disjoint in s.

### 4c. Corpus

- **Symptom sets:** per z, 1,000 train sets and 200 each for validation and test.
- **Verbalization:** each s becomes a first-person patient report written by local Qwen. Styles: train sets use `neutral`, `casual`, `typos`; validation and test sets use all five styles (`neutral`, `casual`, `typos`, `verbose`, `zh`).
  - The style is a nuisance variable, independent of z given s, by construction.
- **Round-trip check:** OLMo extracts the symptom set back from each report as JSON over the lay names of E. Keep only reports whose extracted set equals s. For `zh`, extract from the Chinese text directly.
- **Audit:** 300 reports through the `claude` API with the same extraction. **STOP:** report acceptance rates and the audit.
- **Ground truth per report:**
  - exact log p(s|h) for every hypothesis h (pathologies and groups)
  - the exact posterior over Z under a uniform prior

### 4d. Baselines on test

- token log-prob (Qwen): "a patient with {h} describes their symptoms" + report
- clipped token log-prob
- verbalized (local; API subset)
- typicality v0 (the frozen variant, computed from Qwen samples under each h; sem1B uses the round-trip extractor instead of regexes)

### 4e. Trained estimator (Xuan's training-algorithm contribution)

- **Teacher labels:** apply the frozen v0 rule to the **exact** semantic surprisal −log p(s|z). Compute H̄ and σ̄ over the hypothesis pool with exact entropies, via Monte Carlo with 100k draws per h. The result is a bin label for every (report, h).
- **Student:** `Qwen/Qwen2.5-1.5B-Instruct` with LoRA (r = 16) and a 4-way sequence-classification head over (hypothesis text, report), trained with cross-entropy for 3 epochs on the cluster. Sub-sample negative pairs to 1:3.
- **Three training regimes:**
  - **A, augmentation:** all styles of the same s, and all rewordings of h, share the label.
  - **B, canonicalization:** an LLM rewrites each report into a canonical sorted symptom list before the student sees it; trained on canonical text only.
  - **C, no augmentation:** neutral style and original hypothesis wording only.
- **STOP before training:** report data sizes and GPU-hours.

### 4f. Evaluation

| Property | Metric |
|---|---|
| Calibration | For each bin, the mean and spread of the true log p(s\|h) of reports assigned to it, against the bin's option B value: mean absolute error (nats) and Spearman correlation |
| Inference | KL to the exact posterior, top-1 agreement, and posterior ECE, on clean `neutral` reports and on each nuisance style |
| Robustness | The step 2 metrics: rewording ICC, \|Δ log Bayes factor\|, bin agreement, posterior instability, over styles and hypothesis rewordings |
| Size principle | Occam gap against the **exact** log p(s\|z)/p(s\|group) |
| Information cost | Clean top-1 and bits relative to exact, for every scorer, shown as a robustness-vs-information trade-off figure |

### Predictions

| ID | Prediction |
|---|---|
| M1 | Student A agrees with teacher bins ≥ 80% exactly and ≥ 98% within one bin, on test |
| M2 | Student A bin agreement across styles ≥ 0.9 and rewording ICC ≥ 0.9; token log-prob < 0.5 |
| M3 | Under nuisance styles, Student A's KL to the exact posterior is lower than token log-prob's; on clean `neutral` reports token log-prob may be better (information cost, reported) |
| M4 | Student A's size-principle Occam gap is within ±1 nat of exact; verbalized falls below −2 nats |
| M5 | A and B both beat C on robustness; no prediction on A vs B |

Final deliverable: `docs/medical_results/summary.md` with a "Notes for Xuan" section, plus one figure per claim:
- **C1:** rewording ICC and Bayes-factor instability by scorer
- **C2:** verbalized vs generation probability (dice sweep and MultiWOZ)
- **C3:** calibration plus robustness-vs-information

---

## Appendix: commands

```
git checkout -b claims-123 origin/multiwoz-size-principle
python -m pip install -e ".[dev,figures]"           # + ".[lm]" on the GPU login node
# Step 1
python -m robust_likelihood.dice build
python -m robust_likelihood.dice score --local                 # GPU or MPS
python -m robust_likelihood.dice score --api --pilot 30        # STOP: cost
python -m robust_likelihood.dice analyze && python -m robust_likelihood.dice report
# Step 2
python -m robust_likelihood.multiwoz semantic-check            # ≥95% agreement or STOP
python -m robust_likelihood.multiwoz rewrites --pilot 20       # STOP: cost
python -m robust_likelihood.multiwoz rewrites && python -m robust_likelihood.multiwoz rewrite-audit   # STOP: review
bsub/qsub scripts/claim1_job_a.sh ; bsub/qsub scripts/claim1_job_b.sh                               # STOP before each
python -m robust_likelihood.multiwoz claim1 --lm <job-a> --verbal <job-b>
# Step 3
python -m robust_likelihood.multiwoz typv0 --lm <job-a> --dice <dice-run>   # STOP: decision memo → Xuan
# Step 4
python -m robust_likelihood.medical download && python -m robust_likelihood.medical spec   # STOP: review spec
python -m robust_likelihood.medical generate                    # STOP: audit
python -m robust_likelihood.medical baselines ; python -m robust_likelihood.medical train   # STOP before train
python -m robust_likelihood.medical evaluate && python -m robust_likelihood.medical report
```