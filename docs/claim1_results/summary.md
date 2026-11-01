# Claims 1 and 2 on MultiWOZ: rewrites, rewording, and verbalized probabilities

Step 2e of `docs/claims_plan.md`, from GPU jobs A and B, the Claude and GPT runs, and Jev. Robustness is measured on log Bayes factors and posteriors, never on raw log p. Brackets are 95% bootstrap intervals over items (2000 replicates, seed 20261027). Language-model and verbalized scores are in nats; Jev is in logits.

## 1. Scorers

| Scorer | Items | Scores | Notes |
|---|---|---|---|
| Qwen token log-prob | 235 | 24935 |  |
| Qwen token log-prob, 60-item subset | 60 | 6383 |  |
| Qwen clipped log-prob | 235 | 24935 |  |
| Qwen centered log-prob | 235 | 24935 |  |
| Qwen verbalized, exact string | 60 | 5041 | 1342 answers of 0 (no log score), 0 unparsed, 0 failed calls |
| Qwen verbalized, meaning | 60 | 4220 | 2163 answers of 0 (no log score), 0 unparsed, 0 failed calls |
| Claude verbalized, exact string | 60 | 1741 | 73 answers of 0 (no log score), 0 unparsed, 0 failed calls |
| GPT verbalized, exact string | 60 | 1406 | 408 answers of 0 (no log score), 0 unparsed, 0 failed calls |
| Jev, likely wording | 60 | 4649 |  |
| Jev, typical wording | 60 | 4649 |  |
| OLMo token log-prob | pending | – | OLMo scored the rewrites only; its original pairs are needed to compare with them (claim1-lm --model olmo --parts original). |

Noise floor. Job A rescored S(U | broad|impl) for all 235 utterances with the prompt Phase 8 used. 133 scores are identical; the rest differ because batches of another shape take other bfloat16 kernels: |difference| median 0.00, 90th percentile 0.83, max 1.40 nats. A log Bayes factor shift that compares an original with a rewrite spans two such runs, so shifts below about 1 nat are within this floor.

## 2. R1: the raw log-likelihood is correct in a narrow sense

Named single-slot pairs, mean over slots, against the MultiWOZ corpus log-ratio L_E.

| Regime | L_J (Qwen) | L_E (corpus) | Occam gap L_J − L_E |
|---|---|---|---|
| expl | 10.58 | 4.19 | 6.39 [4.75, 7.82] |
| impl | 16.23 | 4.19 | 12.04 [10.55, 13.30] |

## 3. Hypothesis rewording (R2)

ICC(1) of item-centered scores, groups = (item, hypothesis), members = the original text, three paraphrases, and a Chinese translation. SD ratio: the pooled SD across a hypothesis's rewordings over the SD across the item's hypotheses, for the same utterance (median over items). Shift: |score(H′) − score(H)|, the change in every log Bayes factor against H (median).

| Scorer | ICC(1) | SD ratio | hpara1 | hpara2 | hpara3 | hzh |
|---|---|---|---|---|---|---|
| Qwen token log-prob | 0.74 [0.72, 0.76] | 0.54 [0.51, 0.58] | 5.09 [4.63, 5.49] | 5.03 [4.65, 5.45] | 5.41 [5.07, 5.83] | 14.53 [13.56, 15.27] |
| Qwen token log-prob, 60-item subset | 0.73 [0.68, 0.77] | 0.55 [0.51, 0.60] | 5.44 [4.84, 6.14] | 5.40 [4.72, 6.02] | 5.60 [5.07, 6.11] | 15.10 [13.35, 17.15] |
| Qwen clipped log-prob | 0.38 [0.33, 0.42] | 0.49 [0.47, 0.59] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen centered log-prob | 0.74 [0.72, 0.76] | 0.54 [0.51, 0.58] | 5.09 [4.63, 5.49] | 5.03 [4.65, 5.45] | 5.41 [5.07, 5.83] | 14.53 [13.56, 15.27] |
| Qwen verbalized, exact string | 0.44 [0.37, 0.52] | 0.73 [0.59, 0.83] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen verbalized, meaning | 0.41 [0.37, 0.46] | 0.85 [0.76, 0.99] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Claude verbalized, exact string | 0.20 [0.11, 0.28] | 0.93 [0.87, 0.99] | 1.20 [0.69, 1.61] | – | – | – |
| GPT verbalized, exact string | 0.63 [0.55, 0.70] | 0.59 [0.56, 0.61] | 0.69 [0.00, 0.69] | – | – | – |
| Jev, likely wording | 0.95 [0.94, 0.96] | 0.19 [0.16, 0.22] | 0.12 [0.12, 0.13] | – | – | 0.16 [0.14, 0.17] |
| Jev, typical wording | 0.95 [0.94, 0.96] | 0.19 [0.15, 0.22] | 0.12 [0.10, 0.14] | – | – | 0.23 [0.19, 0.25] |

## 4. Utterance rewrites (R3)

|Δ log Bayes factor| between the original and the rewritten utterance, for G against minus:X, broad, and contra, pooled (median). Flip: the preferred hypothesis of the pair changes.

| Scorer | typo | pad | para1 + para2 | verbose | zh (Chinese) |
|---|---|---|---|---|---|
| Qwen token log-prob | 8.09 [7.50, 8.64] | 6.24 [5.85, 6.78] | 7.94 [7.26, 8.61] | 9.64 [8.91, 10.44] | 9.35 [8.52, 10.15] |
| Qwen token log-prob, 60-item subset | 7.52 [6.35, 8.35] | 7.01 [5.70, 8.18] | 7.42 [6.20, 8.62] | 9.90 [8.72, 12.06] | 9.96 [8.42, 11.78] |
| Qwen clipped log-prob | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen centered log-prob | 8.09 [7.50, 8.64] | 6.24 [5.85, 6.78] | 7.94 [7.26, 8.61] | 9.64 [8.91, 10.44] | 9.35 [8.52, 10.15] |
| Qwen verbalized, exact string | 0.00 [0.00, 0.69] | 0.00 [0.00, 0.69] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen verbalized, meaning | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Claude verbalized, exact string | – | – | 2.30 [1.90, 2.71] | – | – |
| GPT verbalized, exact string | – | – | 1.72 [1.61, 2.30] | – | – |
| Jev, likely wording | – | – | 0.15 [0.13, 0.18] | – | 0.22 [0.17, 0.25] |
| Jev, typical wording | – | – | 0.15 [0.13, 0.18] | – | 0.25 [0.19, 0.30] |

Flip rate of the preferred hypothesis:

| Scorer | typo | pad | para1 + para2 | verbose | zh (Chinese) |
|---|---|---|---|---|---|
| Qwen token log-prob | 0.17 [0.14, 0.19] | 0.12 [0.10, 0.13] | 0.15 [0.13, 0.17] | 0.19 [0.16, 0.21] | 0.19 [0.17, 0.22] |
| Qwen token log-prob, 60-item subset | 0.16 [0.12, 0.20] | 0.13 [0.09, 0.17] | 0.14 [0.11, 0.19] | 0.23 [0.17, 0.28] | 0.18 [0.13, 0.23] |
| Qwen clipped log-prob | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.01 [0.01, 0.02] | 0.01 [0.01, 0.02] | 0.01 [0.01, 0.02] |
| Qwen centered log-prob | 0.17 [0.14, 0.19] | 0.12 [0.10, 0.13] | 0.15 [0.13, 0.17] | 0.19 [0.16, 0.21] | 0.19 [0.17, 0.22] |
| Qwen verbalized, exact string | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] |
| Qwen verbalized, meaning | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Claude verbalized, exact string | – | – | 0.27 [0.21, 0.35] | – | – |
| GPT verbalized, exact string | – | – | 0.10 [0.06, 0.14] | – | – |
| Jev, likely wording | – | – | 0.01 [0.00, 0.03] | – | 0.04 [0.02, 0.06] |
| Jev, typical wording | – | – | 0.01 [0.00, 0.02] | – | 0.03 [0.01, 0.05] |

For contrast, the raw log p of the utterance under G moves by a median of 77.2 nats (typo), 55.7 nats (pad), 18.1 nats (para), 61.4 nats (verbose), 17.5 nats (zh) for Qwen.

## 5. Posterior instability

Per item, one textual variant per class of {G, minus:X, broad, contra} (implicit), uniform prior, 50 seeded draws.

| Scorer | Items | SD of P(G) | Argmax flip rate |
|---|---|---|---|
| Qwen token log-prob | 235 | 0.163 [0.139, 0.188] | 0.26 [0.22, 0.29] |
| Qwen token log-prob, 60-item subset | 60 | 0.157 [0.111, 0.204] | 0.28 [0.21, 0.35] |
| Qwen clipped log-prob | 235 | 0.120 [0.107, 0.134] | 0.32 [0.28, 0.37] |
| Qwen centered log-prob | 235 | 0.163 [0.139, 0.188] | 0.26 [0.22, 0.29] |
| Qwen verbalized, exact string | 60 | 0.060 [0.043, 0.079] | 0.23 [0.16, 0.31] |
| Qwen verbalized, meaning | 60 | 0.123 [0.098, 0.150] | 0.12 [0.07, 0.17] |
| Claude verbalized, exact string | 60 | 0.106 [0.090, 0.123] | 0.40 [0.32, 0.47] |
| GPT verbalized, exact string | 60 | 0.087 [0.070, 0.105] | 0.33 [0.25, 0.40] |
| Jev, likely wording | 60 | 0.007 [0.006, 0.009] | 0.12 [0.07, 0.19] |
| Jev, typical wording | 60 | 0.008 [0.006, 0.011] | 0.06 [0.03, 0.11] |

## 6. Claim 2: verbalized against generation probabilities (R4, R5)

log10 of the verbalized answer against log10 exp(−S) (Qwen's exact-string probability) and log10 P_sem (share of Qwen's 64 samples under H with U's annotated slots, (n + 0.5)/(K + 1)). Gap: median of log10 verbal − log10 truth, in orders of magnitude. Answers of 0 have no log and are left out.

| Verbalizer | Answers of 0 | ρ vs exp(−S) | gap vs exp(−S) | ρ vs P_sem | gap vs P_sem |
|---|---|---|---|---|---|
| Qwen verbalized, exact string | 1342 of 6383 | 0.40 [0.33, 0.46] | 46.8 [44.2, 49.5] | -0.03 [-0.07, 0.01] | 0.8 [0.8, 0.8] |
| Qwen verbalized, meaning | 2163 of 6383 | 0.04 [-0.03, 0.10] | 47.1 [44.7, 49.7] | 0.12 [0.04, 0.20] | 0.8 [0.8, 0.8] |
| Claude verbalized, exact string | 73 of 1814 | 0.12 [0.07, 0.18] | 40.9 [38.3, 43.4] | 0.24 [0.18, 0.30] | -0.6 [-0.6, -0.2] |
| GPT verbalized, exact string | 408 of 1814 | 0.15 [0.02, 0.27] | 39.4 [36.6, 41.9] | 0.00 [-0.08, 0.08] | -0.9 [-1.6, -0.9] |

Normalization: the verbalized probabilities of 20 distinct samples drawn under G|impl, summed per item. For comparison, Qwen's exact probabilities of the same samples, summed per item (median and largest over items); a sum above 1 is incoherent either way.

| Verbalizer | Items | Median sum | Share of items above 1 | Exact sum, median | Exact sum, largest |
|---|---|---|---|---|---|
| Qwen verbalized, exact string | 20 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5.6e-01 | 0.80 |
| Claude verbalized, exact string | 20 | 0.35 [0.32, 0.37] | 0.00 [0.00, 0.00] | 5.6e-01 | 0.80 |
| GPT verbalized, exact string | 20 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 5.6e-01 | 0.80 |

## 7. Prediction scorecard

Failing a prediction is a result.

| ID | Prediction | Value | Result |
|---|---|---|---|
| R1 | Qwen raw log-likelihood, named single-slot pairs: L_J > 0 in both regimes and L_J ≥ 0.5·L_E (mean over slots) | expl L_J 10.58, L_E 4.19, OG 6.39; impl L_J 16.23, L_E 4.19, OG 12.04 | pass |
| R2 | Qwen token log-prob: rewording ICC(1) < 0.5, and a Chinese hypothesis moves the log Bayes factor by a median > 2 nats | icc1 0.74; hzh_median_abs_shift 14.53 | fail |
| R3 | Qwen token log-prob: median |Δ log BF| < 0.5 nats under pad and typo, > 1 nat under para and zh | pad 6.24; typo 8.09; para 7.94; zh 9.35 | fail |
| R4 | Qwen verbalized (exact string) vs exp(−S): median gap > 6 orders of magnitude, and the 20-sample normalization sum > 1 (median over items) | median_gap_orders 46.78; median_norm_sum 1.00 | fail |
| R5 | Qwen verbalized (meaning) vs P_sem: Spearman < 0.5 | 0.12 | pass |

Figures: `fig_claim1_rewording`, `fig_claim1_utterance`, `fig_claim2_verbal` (.png, .pdf, and .csv of the plotted values). Interpretations of points the plan leaves open are in `docs/claims_decisions.md`.
