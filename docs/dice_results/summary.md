# Dice sweep: verbalized probabilities against exact likelihoods

Step 1 of `docs/claims_plan.md`. Three dice with known face probabilities generate roll sequences of
length k ∈ {1, 2, 3, 4, 6, 8, 10}; every likelihood is exact, and the observation space grows from 6
to 6¹⁰ outcomes. S1 is the token log-prob of the list as a simulated roll; S2 asks how likely a
reported sequence is (the LLM-as-likelihood framing); S3 asks for the probability as an arithmetic
question (control).

## 1. Setup and counts

- Sequences: 420; pairs (sequence × die): 1260; normalization queries: 54.
- API scorers see the 630-pair subset (10 sequences per cell) plus the normalization queries.
- api run `results/dice/20261029T080858Z-api`: claude claude-fable-5-1 (no temperature (rejected), 1368 calls); gpt gpt-6-astra (no temperature (rejected), 1368 calls)
- local run `results/dice/20261029T090607Z-local-olmo`: allenai/OLMo-2-1124-7B-Instruct at revision 470b1fba1ae0, bfloat16 on cuda:0, 1314 queries, scorers S1, 35.2 s
- local run `results/dice/20261029T085739Z-local-qwen`: Qwen/Qwen2.5-7B-Instruct at revision a09a35458c70, bfloat16 on cuda:0, 1314 queries, scorers S1, S2, S3, 504.5 s

| Model | Scorer | Queries | Parse failures | Zero answers |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | S1 | 1314 | 0 (0.0%) | 0 |
| Qwen2.5-7B-Instruct | S2 | 1314 | 0 (0.0%) | 142 |
| Qwen2.5-7B-Instruct | S3 | 1314 | 0 (0.0%) | 169 |
| OLMo-2-7B-Instruct | S1 | 1314 | 0 (0.0%) | 0 |
| Claude (API) | S2 | 684 | 3 (0.4%) | 116 |
| Claude (API) | S3 | 684 | 0 (0.0%) | 116 |
| GPT (API) | S2 | 684 | 0 (0.0%) | 116 |
| GPT (API) | S3 | 684 | 0 (0.0%) | 116 |

## 2. Slope of log10 score in k (own die)

True slopes: fair -0.778, even -0.477 per roll; `loaded` varies by sequence, so its truth is fitted on the same sequences.

| Model | Scorer | Die | Slope [95% CI] | Truth | Ratio [95% CI] | R² | n |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | S1 | fair | -2.752 [-3.010, -2.526] | -0.778 | 3.54 [3.25, 3.87] | 0.72 | 140/140 |
| Qwen2.5-7B-Instruct | S1 | even | -1.716 [-1.931, -1.490] | -0.477 | 3.60 [3.12, 4.05] | 0.52 | 140/140 |
| Qwen2.5-7B-Instruct | S1 | loaded | -1.367 [-1.660, -1.080] | -0.663 | 2.06 [1.69, 2.41] | 0.30 | 140/140 |
| Qwen2.5-7B-Instruct | S2 | fair | -0.220 [-0.232, -0.206] | -0.778 | 0.28 [0.26, 0.30] | 0.70 | 140/140 |
| Qwen2.5-7B-Instruct | S2 | even | -0.235 [-0.243, -0.226] | -0.477 | 0.49 [0.47, 0.51] | 0.94 | 140/140 |
| Qwen2.5-7B-Instruct | S2 | loaded | -0.306 [-0.334, -0.276] | -0.663 | 0.46 [0.41, 0.51] | 0.64 | 140/140 |
| Qwen2.5-7B-Instruct | S3 | fair | -0.311 [-0.321, -0.299] | -0.778 | 0.40 [0.38, 0.41] | 0.85 | 140/140 |
| Qwen2.5-7B-Instruct | S3 | even | -0.362 [-0.374, -0.350] | -0.477 | 0.76 [0.73, 0.78] | 0.94 | 140/140 |
| Qwen2.5-7B-Instruct | S3 | loaded | -0.366 [-0.379, -0.353] | -0.663 | 0.55 [0.52, 0.58] | 0.83 | 140/140 |
| OLMo-2-7B-Instruct | S1 | fair | -1.227 [-1.319, -1.134] | -0.778 | 1.58 [1.46, 1.70] | 0.82 | 140/140 |
| OLMo-2-7B-Instruct | S1 | even | -0.985 [-1.150, -0.813] | -0.477 | 2.06 [1.70, 2.41] | 0.61 | 140/140 |
| OLMo-2-7B-Instruct | S1 | loaded | -0.877 [-0.984, -0.781] | -0.663 | 1.32 [1.22, 1.44] | 0.70 | 140/140 |
| Claude (API) | S2 | fair | -0.778 [-0.778, -0.778] | -0.778 | 1.00 [1.00, 1.00] | 1.00 | 69/70 |
| Claude (API) | S2 | even | -0.477 [-0.477, -0.477] | -0.477 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| Claude (API) | S2 | loaded | -0.660 [-0.705, -0.613] | -0.661 | 1.00 [0.99, 1.00] | 0.88 | 70/70 |
| Claude (API) | S3 | fair | -0.778 [-0.778, -0.778] | -0.778 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| Claude (API) | S3 | even | -0.477 [-0.477, -0.477] | -0.477 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| Claude (API) | S3 | loaded | -0.661 [-0.710, -0.615] | -0.661 | 1.00 [1.00, 1.00] | 0.89 | 70/70 |
| GPT (API) | S2 | fair | -0.778 [-0.778, -0.778] | -0.778 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| GPT (API) | S2 | even | -0.477 [-0.477, -0.477] | -0.477 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| GPT (API) | S2 | loaded | -0.661 [-0.711, -0.612] | -0.661 | 1.00 [1.00, 1.00] | 0.89 | 70/70 |
| GPT (API) | S3 | fair | -0.778 [-0.778, -0.778] | -0.778 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| GPT (API) | S3 | even | -0.477 [-0.477, -0.477] | -0.477 | 1.00 [1.00, 1.00] | 1.00 | 70/70 |
| GPT (API) | S3 | loaded | -0.661 [-0.708, -0.612] | -0.661 | 1.00 [1.00, 1.00] | 0.89 | 70/70 |

## 3. Accuracy at k = 1

Each face under each die (18 queries).

| Model | Scorer | Mean abs error | Max abs error | Share within ±0.10 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | S1 | 0.178 | 0.580 | 0.50 |
| Qwen2.5-7B-Instruct | S2 | 0.024 | 0.167 | 0.89 |
| Qwen2.5-7B-Instruct | S3 | 0.012 | 0.067 | 1.00 |
| OLMo-2-7B-Instruct | S1 | 0.123 | 0.332 | 0.61 |
| Claude (API) | S2 | 0.000 | 0.000 | 1.00 |
| Claude (API) | S3 | 0.000 | 0.000 | 1.00 |
| GPT (API) | S2 | 0.000 | 0.000 | 1.00 |
| GPT (API) | S3 | 0.000 | 0.000 | 1.00 |

## 4. Bayes factors

Predicted against exact log Bayes factor over every sequence and die pair with both likelihoods
finite. The even-vs-fair column uses all-even sequences only, where the truth is k ln 2.
Impossible pairs: an odd face under the even die, whose exact probability is 0. Verbalized
scorers are scored by the share answered exactly 0; token log-probs by the median log p.

| Model | Scorer | Slope, all pairs [95% CI] | R² | Slope, even vs fair [95% CI] | Impossible pairs |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | S1 | 4.69 [4.25, 5.12] | 0.38 | 4.13 [3.45, 4.80] | median log p -56.4 (n = 229) |
| Qwen2.5-7B-Instruct | S2 | 0.42 [0.37, 0.46] | 0.31 | -0.08 [-0.13, -0.03] | 0.61 answered 0 (n = 229) |
| Qwen2.5-7B-Instruct | S3 | 0.17 [0.14, 0.21] | 0.09 | -0.19 [-0.28, -0.09] | 0.72 answered 0 (n = 229) |
| OLMo-2-7B-Instruct | S1 | 1.33 [1.16, 1.50] | 0.30 | 1.33 [0.97, 1.72] | median log p -24.3 (n = 229) |
| Claude (API) | S2 | 1.00 [1.00, 1.01] | 1.00 | 1.00 [1.00, 1.00] | 1.00 answered 0 (n = 113) |
| Claude (API) | S3 | 1.01 [1.00, 1.02] | 0.99 | 1.00 [1.00, 1.00] | 1.00 answered 0 (n = 113) |
| GPT (API) | S2 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 answered 0 (n = 113) |
| GPT (API) | S3 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 answered 0 (n = 113) |

## 5. Normalization sums

True sum = 1 for every set. Token log-probs should sum to at most 1.

| Model | Scorer | k=1 fair | k=1 even | k=1 loaded | k=2 fair (36) |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | S1 | 0.999 | 0.972 | 1 | 8.94e-07 |
| Qwen2.5-7B-Instruct | S2 | 1 | 0.958 | 0.854 | 1.46 |
| Qwen2.5-7B-Instruct | S3 | 1 | 1 | 1.21 | 1.03 |
| OLMo-2-7B-Instruct | S1 | 0.664 | 0.0483 | 0.572 | 0.879 |
| Claude (API) | S2 | 1 | 1 | 1 | 1 |
| Claude (API) | S3 | 1 | 1 | 1 | 1 |
| GPT (API) | S2 | 1 | 1 | 1 | 1 |
| GPT (API) | S3 | 1 | 1 | 1 | 1 |

## 6. Prediction scorecard

Failing a prediction is a result.

| ID | Model | Claim | Value | Result |
|---|---|---|---|---|
| D1 | OLMo-2-7B-Instruct | S1 fair slope within a factor of 2 of the truth, R² > 0.9 | r2 0.816, ratio 1.577 | fail |
| D1 | Qwen2.5-7B-Instruct | S1 fair slope within a factor of 2 of the truth, R² > 0.9 | r2 0.725, ratio 3.536 | fail |
| D2 | Claude (API) | S2 |fair slope| < 0.15 per roll | -0.778 | fail |
| D2 | GPT (API) | S2 |fair slope| < 0.15 per roll | -0.778 | fail |
| D2 | Qwen2.5-7B-Instruct | S2 |fair slope| < 0.15 per roll | -0.220 | fail |
| D3 | Claude (API) | S2 mean |error| at k = 1 ≤ 0.1 | 0.000 | pass |
| D3 | GPT (API) | S2 mean |error| at k = 1 ≤ 0.1 | 0.000 | pass |
| D3 | Qwen2.5-7B-Instruct | S2 mean |error| at k = 1 ≤ 0.1 | 0.024 | pass |
| D4 | Claude (API) | S2 sum over the 36 sequences at k = 2 > 2.0 | 1.000 | fail |
| D4 | GPT (API) | S2 sum over the 36 sequences at k = 2 > 2.0 | 1.000 | fail |
| D4 | Qwen2.5-7B-Instruct | S2 sum over the 36 sequences at k = 2 > 2.0 | 1.458 | fail |
| D5 | Claude (API) | S3 reported without a prediction | r2 1.000, ratio 1.000 | no prediction |
| D5 | GPT (API) | S3 reported without a prediction | r2 1.000, ratio 1.000 | no prediction |
| D5 | Qwen2.5-7B-Instruct | S3 reported without a prediction | r2 0.852, ratio 0.400 | no prediction |

## 7. Acceptance

- Sequences regenerate from seed 20261027 (`python -m robust_likelihood.dice build` is deterministic).
- Local parse failures under 2%: True.
- Every raw answer is kept in the run folders under `results/dice/`.

## Figures

- `fig_dice_slopes`: log10 P against k, exact and each scorer, one column per model.
- `fig_dice_bayes`: predicted against exact log Bayes factor.
- `fig_dice_normalization`: sums over the k = 1 and k = 2 outcome sets.

Operational definitions: D2 uses the fair die, |slope| < 0.15; D3 is the mean absolute error over
the 18 k = 1 queries, ≤ 0.1; D4 is the sum over the 36 k = 2 fair sequences, > 2.0.
Answers of exactly 0 and unparsed answers have no log score and are left out of the fits.
