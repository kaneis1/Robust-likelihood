# Typicality v0 from saved samples

Step 3 of `docs/claims_plan.md`, computed from step 2's Qwen samples and scores with no new GPU work. Scores are in nats. Brackets in `metrics.json` are 95% bootstrap intervals over items (2000 replicates, seed 20261027).

Pool: 463 distinct hypothesis texts of the clean probe, all with 64 Phase 8 samples; 894 rewritten texts of the 60-item subset with 64 samples each; 24935 scored cells.

## 1. Where human utterances fall

Share of the 2,355 original (utterance, hypothesis) pairs in each bin, lowest T first (uniform weighting). Qwen's own samples under a hypothesis have S of a few nats; the human utterances have S near 100, so they sit in the top bin of every variant built on S. A variant whose pairs share one bin scores them alike, so its bin agreement and its stability say nothing about meaning.

| Variant | Bins (low → high) | Pool H, σ |
|---|---|---|
| per2A | 0.0% / 0.0% / 100.0% | –, – |
| glob2A | 0.0% / 0.0% / 100.0% | 4.50, 3.15 |
| glob1A | 0.0% / 0.0% / 0.0% / 100.0% | 4.50, 3.15 |
| glob1B | 0.0% / 0.0% / 0.0% / 100.0% | 4.50, 3.15 |
| cent1B | 0.0% / 0.3% / 2.3% / 97.5% | -56.30, 18.96 |
| sem1B | 0.0% / 14.8% / 3.7% / 81.5% | 0.24, 0.62 |

## 2. Variants against the metrics (uniform weighting)

Size principle: L_J/L_E and the Occam gap OG = L_J − L_E on named single-slot pairs (mean over slots); information: contradiction AUROC (G|impl vs ctrl:contra) and mean log Bayes factor G vs contra; robustness: hypothesis-rewording ICC(1) on the 60-item subset, median |shift| under a Chinese hypothesis, median |Δ log BF| under utterance rewrites, posterior argmax flip rate; bin agreement under utterance rewrites; pool sensitivity: bin-flip rate when the pool is halved (20 draws).

| Variant | L_J/L_E expl | L_J/L_E impl | OG expl | OG impl | contra AUROC | contra log BF | ICC (subset) | hzh shift | Δ pad | Δ typo | Δ para | Δ zh | flips | agree pad | agree typo | agree para | agree zh | pool flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per2A | -0.02 | -0.01 | -4.28 | -4.23 | 0.52 | 0.04 | 0.18 | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.66 | 1.00 | 1.00 | 1.00 | 1.00 | – |
| glob2A | -0.17 | 0.03 | -4.91 | -4.06 | 0.47 | -0.06 | 0.60 | 1.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| glob1A | -0.12 | 0.10 | -4.68 | -3.79 | 0.48 | -0.06 | 0.60 | 0.81 | 0.00 | 0.00 | 0.00 | 0.00 | 0.56 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| glob1B | 0.00 | 0.01 | -4.19 | -4.16 | 0.50 | 0.03 | – | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| cent1B | 0.35 | 0.40 | -2.71 | -2.52 | 0.54 | 1.50 | 0.48 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.98 | 0.97 | 0.95 | 0.97 | 0.00 |
| sem1B | 0.11 | 0.14 | -3.72 | -3.62 | 0.72 | 0.53 | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |

## 3. Baselines on the same metrics

Robustness columns are claim 1's (`docs/claim1_results/metrics.json`); the ICC is over all items the scorer covers. Verbalized scorers cover the 60-item subset, so their size-principle rows use it.

| Scorer | L_J/L_E expl | L_J/L_E impl | OG expl | OG impl | contra AUROC | contra log BF | ICC | hzh shift | Δ pad | Δ typo | Δ para | Δ zh | flips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen token log-prob | 2.52 | 3.87 | 6.39 | 12.04 | 0.62 | 14.58 | 0.74 | 14.53 | 6.24 | 8.09 | 7.94 | 9.35 | 0.26 |
| Qwen clipped log-prob | 0.08 | 0.23 | -3.86 | -3.24 | 0.65 | 0.90 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.32 |
| Jev, likely | 0.41 | 0.09 | -2.45 | -3.81 | 0.97 | 2.85 | 0.95 | 0.16 | – | – | 0.15 | 0.22 | 0.12 |
| Jev, typical | 0.46 | 0.10 | -2.27 | -3.76 | 0.96 | 3.23 | 0.95 | 0.23 | – | – | 0.15 | 0.25 | 0.06 |
| Qwen verbalized, exact (60 items) | -0.01 | 0.00 | -4.23 | -4.18 | 0.45 | -0.14 | 0.44 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.23 |
| Qwen verbalized, meaning (60 items) | – | 0.04 | – | -4.03 | 0.75 | 0.55 | 0.41 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| Claude verbalized (60 items) | 0.21 | 0.09 | -3.33 | -3.81 | 0.70 | 0.58 | 0.20 | – | – | – | 2.30 | – | 0.40 |
| GPT verbalized (60 items) | -0.48 | -0.01 | -6.20 | -4.25 | 0.51 | -0.58 | 0.63 | – | – | – | 1.72 | – | 0.33 |

## 4. Prior weighting

Each pool text weighted by how many clean items carry it in their probe, against uniform.

| Variant | Weighting | Pool H, σ | L_J/L_E expl | L_J/L_E impl | contra AUROC | pool flips |
|---|---|---|---|---|---|---|
| per2A | uniform | –, – | -0.02 | -0.01 | 0.52 | – |
| glob2A | uniform | 4.50, 3.15 | -0.17 | 0.03 | 0.47 | 0.00 |
| glob2A | prior | 5.31, 3.74 | -0.14 | 0.01 | 0.46 | 0.00 |
| glob1A | uniform | 4.50, 3.15 | -0.12 | 0.10 | 0.48 | 0.00 |
| glob1A | prior | 5.31, 3.74 | -0.14 | 0.06 | 0.47 | 0.00 |
| glob1B | uniform | 4.50, 3.15 | 0.00 | 0.01 | 0.50 | 0.00 |
| glob1B | prior | 5.31, 3.74 | 0.00 | 0.01 | 0.50 | 0.00 |
| cent1B | uniform | -56.30, 18.96 | 0.35 | 0.40 | 0.54 | 0.00 |
| cent1B | prior | -48.09, 25.32 | 1.00 | 1.54 | 0.62 | 0.08 |
| sem1B | uniform | 0.24, 0.62 | 0.11 | 0.14 | 0.72 | 0.00 |
| sem1B | prior | 0.30, 0.68 | 0.13 | 0.15 | 0.72 | 0.01 |

## 5. Sensitivity: bin edges and factorized P_sem

| Variant | Bins (low → high) | L_J/L_E expl | L_J/L_E impl | contra AUROC | ICC (subset) | agree para | agree zh |
|---|---|---|---|---|---|---|---|
| glob1A:3bins | 0% / 0% / 100% | -0.09 | 0.10 | 0.47 | 0.58 | 1.00 | 1.00 |
| glob1B:3bins | 0% / 0% / 100% | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| cent1B:3bins | 0% / 1% / 99% | 0.10 | 0.19 | 0.52 | 0.28 | 0.98 | 0.99 |
| sem1B:3bins | 0% / 18% / 82% | 0.07 | 0.07 | 0.72 | 0.64 | 1.00 | 1.00 |
| glob1A:6bins | 0% / 0% / 0% / 0% / 0% / 100% | -0.17 | 0.03 | 0.47 | 0.60 | 1.00 | 1.00 |
| glob1B:6bins | 0% / 0% / 0% / 0% / 0% / 100% | 0.00 | 0.01 | 0.50 | – | 1.00 | 1.00 |
| cent1B:6bins | 0% / 0% / 0% / 2% / 12% / 85% | 1.09 | 1.62 | 0.63 | 0.63 | 0.84 | 0.83 |
| sem1B:6bins | 0% / 0% / 15% / 4% / 0% / 81% | 0.18 | 0.21 | 0.72 | 0.67 | 1.00 | 1.00 |
| sem1B:factorized | 0% / 15% / 4% / 81% | 0.13 | 0.15 | 0.72 | 0.65 | 1.00 | 1.00 |

## 6. Dice check

Step 1's sequences, exact moments of each die (pool: the three dice, uniform), and option A's exact bin mass. On all-even sequences the truth favours `even` over `fair` by k ln 2. Impossible: a sequence with an odd face under `even`.

| Variant | Ties on all-even sequences | log BF at k = 2 (truth 1.39) | k = 6 (4.16) | k = 10 (6.93) | Impossible ruled out |
|---|---|---|---|---|---|
| per2A | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| glob2A | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| glob1A | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| glob1B | 0.00 | 0.87 | 4.10 | 9.59 | 0.00 |

## 7. Prediction scorecard

Failing a prediction is a result.

| ID | Prediction | Value | Result |
|---|---|---|---|
| T1 | per2A erases the size principle: named-slot Occam gap ≤ −0.8·L_E | expl OG -4.28; L_E 4.19; impl OG -4.23; L_E 4.19 | pass |
| T2 | Option A variants tie or erase the size principle (L_J < 0.5·L_E); option B variants keep it (L_J ≥ 0.5·L_E), named single-slot pairs, both regimes | L_J/L_E per2A expl -0.02; impl -0.01; glob2A expl -0.17; impl 0.03; glob1A expl -0.12; impl 0.10; glob1B expl 0.00; impl 0.01; cent1B expl 0.35; impl 0.40; sem1B expl 0.11; impl 0.14 | fail |
| T3 | sem1B: hypothesis-rewording ICC > 0.9 and bin agreement > 0.9 under para and zh, where raw log-prob ICC < 0.5 | icc1 0.65; bin_agreement para 1.00; zh 1.00; raw_logprob_icc1_subset 0.73 | fail (the premise does not hold: raw log-prob ICC is 0.73) |
| T4 | cent1B bin agreement under pad and typo > 0.9, above glob1B | cent1B pad 0.98; typo 0.97; glob1B pad 1.00; typo 1.00 | fail |

The recommendation is in `docs/typicality_v0_decision.md`.
