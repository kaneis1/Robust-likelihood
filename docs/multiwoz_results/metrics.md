# MultiWOZ size-principle metrics: `20261026T034616Z-full`

Jev scores are elicited yes-probabilities read on the logit scale; language-model scores are in nats. L_E comes from MultiWOZ naming rates of real users.

Scores per prompt version: {'multiwoz_likely': 3727, 'multiwoz_typical': 3727}. Clipped at 0.001: {'multiwoz_likely': 0, 'multiwoz_typical': 0}. Returned model ids: {'jev-1.13.0': 7454}. Bootstrap: 2000 replicates; clean items paired with the reference resample (verified).

## Occam gap, clean set, logit

L_J = l(narrow) − l(broader); L_E is the corpus log-ratio; OG = mean(L_J) − mean(L_E). The target is 0.

### likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.69 [4.05, 6.00] | 2.25 | -2.44 [-3.75, -1.79] | 0.54 | -4.14 [-5.46, -3.50] |
| single slot named: area | 138 | 3.89 [3.44, 4.55] | 0.89 | -3.00 [-3.66, -2.54] | 0.18 | -3.71 [-4.39, -3.25] |
| single slot named: price | 150 | 4.00 [3.48, 4.79] | 2.08 | -1.92 [-2.70, -1.39] | 0.43 | -3.57 [-4.36, -3.05] |
| single slot named: mean over slots | 430 | 4.19 | 1.74 | -2.45 [-3.01, -2.13] | 0.38 | -3.81 [-4.36, -3.49] |
| single slot omitted: food | 53 | -1.30 [-1.55, -1.09] | -1.34 | -0.04 [-0.31, 0.28] | -1.24 | 0.05 [-0.21, 0.36] |
| single slot omitted: area | 48 | -1.34 [-1.61, -1.11] | -0.60 | 0.74 [0.48, 1.04] | -0.50 | 0.84 [0.58, 1.13] |
| single slot omitted: price | 50 | -1.37 [-1.61, -1.16] | -1.00 | 0.38 [0.09, 0.70] | -0.86 | 0.51 [0.22, 0.83] |
| single slot omitted: mean over slots | 151 | -1.34 | -0.98 | 0.36 [0.24, 0.49] | -0.87 | 0.47 [0.35, 0.59] |
| full removal, every constrained slot named | 106 | 8.41 [7.78, 9.58] | 2.71 | -5.70 [-6.85, -5.05] | 0.07 | -8.33 [-9.49, -7.68] |
| full removal, some named | 122 | 5.99 [5.44, 6.93] | 1.35 | -4.64 [-5.70, -4.05] | -0.92 | -6.91 [-7.99, -6.28] |
| full removal, none named | 7 | -3.05 [-3.32, -2.82] | -1.49 | 1.55 [1.17, 2.03] | -1.32 | 1.73 [1.30, 2.22] |

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| full removal, 2 slots, 0 named | 5 | -2.66 [-2.90, -2.46] | -1.26 | 1.41 [1.02, 1.92] | -1.07 | 1.59 [1.12, 2.22] |
| full removal, 2 slots, 1 named | 13 | 2.87 [2.52, 3.47] | 0.59 | -2.28 [-3.06, -1.68] | -0.63 | -3.50 [-4.22, -3.05] |
| full removal, 2 slots, 2 named | 106 | 8.41 [7.78, 9.58] | 2.71 | -5.70 [-6.85, -5.05] | 0.07 | -8.33 [-9.49, -7.68] |
| full removal, 3 slots, 0 named | 2 | -4.01 [-4.37, -3.71] | -2.08 | 1.93 [1.56, 2.33] | -1.94 | 2.06 [1.75, 2.41] |
| full removal, 3 slots, 1 named | 13 | 1.82 [1.22, 2.86] | 0.26 | -1.56 [-2.75, -0.79] | -1.24 | -3.06 [-4.16, -2.39] |
| full removal, 3 slots, 2 named | 96 | 6.97 [6.38, 8.00] | 1.60 | -5.38 [-6.44, -4.79] | -0.92 | -7.89 [-8.94, -7.32] |

### typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.69 [4.05, 6.00] | 2.51 | -2.18 [-3.48, -1.52] | 0.64 | -4.05 [-5.36, -3.41] |
| single slot named: area | 138 | 3.89 [3.44, 4.55] | 1.00 | -2.89 [-3.55, -2.41] | 0.20 | -3.69 [-4.38, -3.24] |
| single slot named: price | 150 | 4.00 [3.48, 4.79] | 2.25 | -1.75 [-2.53, -1.22] | 0.46 | -3.54 [-4.33, -3.01] |
| single slot named: mean over slots | 430 | 4.19 | 1.92 | -2.27 [-2.84, -1.95] | 0.43 | -3.76 [-4.32, -3.44] |
| single slot omitted: food | 53 | -1.30 [-1.55, -1.09] | -2.02 | -0.72 [-1.03, -0.38] | -1.97 | -0.67 [-0.98, -0.33] |
| single slot omitted: area | 48 | -1.34 [-1.61, -1.11] | -0.96 | 0.38 [0.08, 0.73] | -0.95 | 0.39 [0.10, 0.73] |
| single slot omitted: price | 50 | -1.37 [-1.61, -1.16] | -1.48 | -0.10 [-0.45, 0.29] | -1.43 | -0.06 [-0.41, 0.32] |
| single slot omitted: mean over slots | 151 | -1.34 | -1.48 | -0.15 [-0.29, 0.00] | -1.45 | -0.11 [-0.25, 0.04] |
| full removal, every constrained slot named | 106 | 8.41 [7.78, 9.58] | 3.35 | -5.06 [-6.23, -4.40] | 0.06 | -8.35 [-9.52, -7.70] |
| full removal, some named | 122 | 5.99 [5.44, 6.93] | 1.27 | -4.72 [-5.78, -4.13] | -1.69 | -7.68 [-8.76, -7.04] |
| full removal, none named | 7 | -3.05 [-3.32, -2.82] | -2.13 | 0.92 [0.30, 1.63] | -2.08 | 0.97 [0.31, 1.70] |

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| full removal, 2 slots, 0 named | 5 | -2.66 [-2.90, -2.46] | -1.82 | 0.84 [0.06, 1.90] | -1.79 | 0.87 [0.05, 1.98] |
| full removal, 2 slots, 1 named | 13 | 2.87 [2.52, 3.47] | 0.39 | -2.49 [-3.36, -1.79] | -1.26 | -4.13 [-4.91, -3.58] |
| full removal, 2 slots, 2 named | 106 | 8.41 [7.78, 9.58] | 3.35 | -5.06 [-6.23, -4.40] | 0.06 | -8.35 [-9.52, -7.70] |
| full removal, 3 slots, 0 named | 2 | -4.01 [-4.37, -3.71] | -2.88 | 1.13 [0.80, 1.48] | -2.81 | 1.20 [0.88, 1.54] |
| full removal, 3 slots, 1 named | 13 | 1.82 [1.22, 2.86] | -0.04 | -1.87 [-3.05, -1.04] | -1.99 | -3.81 [-4.96, -3.05] |
| full removal, 3 slots, 2 named | 96 | 6.97 [6.38, 8.00] | 1.56 | -5.41 [-6.48, -4.81] | -1.71 | -8.68 [-9.75, -8.09] |

## Wording gap, clean set, logit

Δ = l(explicit) − l(implicit) for the same breadth. The target is 0.

| Scorer | n | Mean Δ (95%) | Share with abs(Δ) > 1 | Mean Δ, G | Mean Δ, minus:food | Mean Δ, minus:area | Mean Δ, minus:pricerange | Mean Δ, broad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| likely | 940 | -1.13 [-1.19, -1.07] | 0.51 | 0.13 | -1.16 | -0.46 | -1.14 | -2.29 |
| typical | 940 | -1.42 [-1.50, -1.34] | 0.52 | 0.03 | -1.35 | -0.58 | -1.31 | -3.00 |

## Monotonicity, clean set, logit

| Scorer | Wording | Single slot named: broader wins | Ties | Full removal, all named: broader wins | Ties |
| --- | --- | --- | --- | --- | --- |
| likely | explicit | 3/430 (0.01) | 1 | 3/106 (0.03) | 0 |
| likely | implicit | 17/430 (0.04) | 40 | 10/106 (0.09) | 12 |
| typical | explicit | 3/430 (0.01) | 2 | 3/106 (0.03) | 0 |
| typical | implicit | 10/430 (0.02) | 61 | 10/106 (0.09) | 34 |

## Controls, clean set, logit

| Scorer | Contradiction AUROC | G minus contra (95%) | by changed slot: food / area / price | G minus tie (95%) | G minus tie2 | All tie controls (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| likely | 0.965 | 2.85 [2.70, 3.01] | 3.32 / 2.52 / 2.65 | 0.07 [-0.06, 0.20] | 0.06 | 0.07 [-0.05, 0.19] |
| typical | 0.958 | 3.23 [3.04, 3.42] | 3.80 / 2.76 / 3.02 | 0.09 [-0.09, 0.26] | 0.06 | 0.09 [-0.08, 0.25] |

## Per-hypothesis bias, clean set, logit

Mean within-item centred score.

| Hypothesis type | likely | typical |
| --- | --- | --- |
| `G`, explicit | 0.99 | 1.14 |
| `G`, implicit | 0.69 | 0.72 |
| `broad`, explicit | -1.12 | -1.37 |
| `broad`, implicit | 1.17 | 1.63 |
| `ctrl:contra`, implicit | -2.14 | -2.47 |
| `ctrl:tie2`, implicit | -0.44 | -0.60 |
| `ctrl:tie`, implicit | 0.37 | 0.15 |
| `minus:area`, explicit | 0.14 | 0.09 |
| `minus:area`, implicit | 0.60 | 0.67 |
| `minus:food`, explicit | -0.53 | -0.59 |
| `minus:food`, implicit | 0.63 | 0.76 |
| `minus:pricerange`, explicit | -0.60 | -0.65 |
| `minus:pricerange`, implicit | 0.54 | 0.66 |

## Secondary analyses

### Log mapping, clean set: likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.69 [4.05, 6.00] | 0.79 | -3.90 [-5.21, -3.26] | 0.15 | -4.53 [-5.85, -3.90] |
| single slot named: area | 138 | 3.89 [3.44, 4.55] | 0.26 | -3.63 [-4.29, -3.17] | 0.03 | -3.86 [-4.52, -3.40] |
| single slot named: price | 150 | 4.00 [3.48, 4.79] | 0.74 | -3.26 [-4.05, -2.74] | 0.12 | -3.87 [-4.66, -3.36] |
| single slot named: mean over slots | 430 | 4.19 | 0.60 | -3.60 [-4.15, -3.28] | 0.10 | -4.09 [-4.64, -3.77] |
| single slot omitted: food | 53 | -1.30 [-1.55, -1.09] | -0.24 | 1.06 [0.84, 1.31] | -0.24 | 1.06 [0.85, 1.31] |
| single slot omitted: area | 48 | -1.34 [-1.61, -1.11] | -0.08 | 1.26 [1.03, 1.54] | -0.08 | 1.26 [1.04, 1.54] |
| single slot omitted: price | 50 | -1.37 [-1.61, -1.16] | -0.17 | 1.20 [0.98, 1.46] | -0.16 | 1.21 [1.00, 1.47] |
| single slot omitted: mean over slots | 151 | -1.34 | -0.16 | 1.17 [1.07, 1.29] | -0.16 | 1.18 [1.08, 1.30] |
| full removal, every constrained slot named | 106 | 8.41 [7.78, 9.58] | 0.81 | -7.60 [-8.75, -6.96] | -0.03 | -8.44 [-9.60, -7.79] |
| full removal, some named | 122 | 5.99 [5.44, 6.93] | 0.53 | -5.45 [-6.50, -4.86] | -0.18 | -6.16 [-7.23, -5.56] |
| full removal, none named | 7 | -3.05 [-3.32, -2.82] | -0.39 | 2.65 [2.24, 3.24] | -0.36 | 2.68 [2.26, 3.27] |

### Log mapping, clean set: typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.69 [4.05, 6.00] | 0.71 | -3.97 [-5.29, -3.33] | 0.15 | -4.54 [-5.86, -3.90] |
| single slot named: area | 138 | 3.89 [3.44, 4.55] | 0.23 | -3.66 [-4.32, -3.20] | 0.03 | -3.86 [-4.52, -3.40] |
| single slot named: price | 150 | 4.00 [3.48, 4.79] | 0.63 | -3.37 [-4.16, -2.85] | 0.11 | -3.89 [-4.68, -3.37] |
| single slot named: mean over slots | 430 | 4.19 | 0.52 | -3.67 [-4.22, -3.35] | 0.10 | -4.09 [-4.65, -3.78] |
| single slot omitted: food | 53 | -1.30 [-1.55, -1.09] | -0.28 | 1.02 [0.80, 1.27] | -0.28 | 1.02 [0.81, 1.26] |
| single slot omitted: area | 48 | -1.34 [-1.61, -1.11] | -0.09 | 1.25 [1.02, 1.53] | -0.09 | 1.25 [1.02, 1.52] |
| single slot omitted: price | 50 | -1.37 [-1.61, -1.16] | -0.21 | 1.16 [0.94, 1.42] | -0.21 | 1.16 [0.94, 1.42] |
| single slot omitted: mean over slots | 151 | -1.34 | -0.19 | 1.14 [1.04, 1.26] | -0.19 | 1.14 [1.04, 1.26] |
| full removal, every constrained slot named | 106 | 8.41 [7.78, 9.58] | 0.79 | -7.61 [-8.77, -6.99] | -0.04 | -8.44 [-9.61, -7.80] |
| full removal, some named | 122 | 5.99 [5.44, 6.93] | 0.46 | -5.53 [-6.57, -4.93] | -0.23 | -6.22 [-7.28, -5.62] |
| full removal, none named | 7 | -3.05 [-3.32, -2.82] | -0.50 | 2.55 [2.13, 3.13] | -0.50 | 2.55 [2.13, 3.14] |

### fail_info set, logit: likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 84 | 4.69 [4.05, 6.00] | 2.26 | -2.43 [-3.76, -1.77] | 0.79 | -3.90 [-5.22, -3.23] |
| single slot named: area | 81 | 3.89 [3.44, 4.55] | 0.60 | -3.29 [-4.03, -2.76] | -0.06 | -3.95 [-4.68, -3.45] |
| single slot named: price | 77 | 4.00 [3.48, 4.79] | 1.47 | -2.52 [-3.31, -1.94] | 0.17 | -3.83 [-4.61, -3.26] |
| single slot named: mean over slots | 242 | 4.19 | 1.45 | -2.75 [-3.32, -2.38] | 0.30 | -3.89 [-4.47, -3.55] |
| single slot omitted: food | 29 | -1.30 [-1.55, -1.09] | -1.36 | -0.06 [-0.34, 0.27] | -1.22 | 0.07 [-0.20, 0.42] |
| single slot omitted: area | 34 | -1.34 [-1.61, -1.11] | -0.48 | 0.86 [0.58, 1.15] | -0.38 | 0.96 [0.68, 1.25] |
| single slot omitted: price | 37 | -1.37 [-1.61, -1.16] | -0.86 | 0.51 [0.19, 0.89] | -0.73 | 0.65 [0.35, 0.99] |
| single slot omitted: mean over slots | 100 | -1.34 | -0.90 | 0.44 [0.28, 0.62] | -0.78 | 0.56 [0.41, 0.74] |
| full removal, every constrained slot named | 50 | 8.46 [7.83, 9.58] | 1.25 | -7.21 [-8.39, -6.39] | -1.29 | -9.74 [-10.91, -8.91] |
| full removal, some named | 81 | 5.82 [5.27, 6.81] | 1.15 | -4.68 [-5.76, -4.00] | -0.90 | -6.73 [-7.84, -6.03] |
| full removal, none named | 4 | -3.00 [-3.29, -2.76] | -0.99 | 2.01 [1.30, 2.76] | -0.70 | 2.29 [1.48, 3.11] |

### fail_info set, logit: typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 84 | 4.69 [4.05, 6.00] | 2.40 | -2.29 [-3.62, -1.63] | 0.83 | -3.85 [-5.18, -3.19] |
| single slot named: area | 81 | 3.89 [3.44, 4.55] | 0.61 | -3.28 [-4.01, -2.75] | -0.06 | -3.95 [-4.68, -3.45] |
| single slot named: price | 77 | 4.00 [3.48, 4.79] | 1.47 | -2.52 [-3.31, -1.93] | 0.13 | -3.86 [-4.65, -3.29] |
| single slot named: mean over slots | 242 | 4.19 | 1.50 | -2.70 [-3.27, -2.33] | 0.30 | -3.89 [-4.46, -3.54] |
| single slot omitted: food | 29 | -1.30 [-1.55, -1.09] | -2.05 | -0.76 [-1.07, -0.38] | -1.95 | -0.65 [-0.98, -0.25] |
| single slot omitted: area | 34 | -1.34 [-1.61, -1.11] | -0.85 | 0.49 [0.16, 0.84] | -0.80 | 0.54 [0.21, 0.88] |
| single slot omitted: price | 37 | -1.37 [-1.61, -1.16] | -1.34 | 0.03 [-0.36, 0.48] | -1.28 | 0.10 [-0.27, 0.52] |
| single slot omitted: mean over slots | 100 | -1.34 | -1.42 | -0.08 [-0.27, 0.14] | -1.34 | -0.00 [-0.20, 0.22] |
| full removal, every constrained slot named | 50 | 8.46 [7.83, 9.58] | 1.62 | -6.84 [-8.07, -5.96] | -1.59 | -10.04 [-11.28, -9.14] |
| full removal, some named | 81 | 5.82 [5.27, 6.81] | 1.06 | -4.77 [-5.86, -4.08] | -1.63 | -7.46 [-8.55, -6.75] |
| full removal, none named | 4 | -3.00 [-3.29, -2.76] | -1.51 | 1.49 [0.63, 2.61] | -1.45 | 1.55 [0.62, 2.61] |

### Reference V2 (a from `match_all_turn0`, b from `match_all_turn0`): likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 3.43 [2.45, 4.69] | 2.25 | -1.18 [-2.45, -0.20] | 0.54 | -2.88 [-4.16, -1.92] |
| single slot named: area | 138 | 4.62 [4.39, 4.84] | 0.89 | -3.73 [-3.98, -3.47] | 0.18 | -4.44 [-4.68, -4.19] |
| single slot named: price | 150 | 3.31 [2.36, 4.58] | 2.08 | -1.24 [-2.53, -0.26] | 0.43 | -2.88 [-4.15, -1.93] |
| single slot named: mean over slots | 430 | 3.79 | 1.74 | -2.05 [-2.84, -1.58] | 0.38 | -3.40 [-4.18, -2.94] |
| single slot omitted: food | 53 | -1.30 [-1.50, -1.12] | -1.34 | -0.04 [-0.30, 0.24] | -1.24 | 0.06 [-0.20, 0.33] |
| single slot omitted: area | 48 | -1.29 [-1.51, -1.12] | -0.60 | 0.70 [0.48, 0.94] | -0.50 | 0.79 [0.57, 1.02] |
| single slot omitted: price | 50 | -1.26 [-1.46, -1.09] | -1.00 | 0.26 [-0.00, 0.53] | -0.86 | 0.39 [0.13, 0.65] |
| single slot omitted: mean over slots | 151 | -1.28 | -0.98 | 0.31 [0.18, 0.44] | -0.87 | 0.41 [0.28, 0.55] |
| full removal, every constrained slot named | 106 | 7.47 [6.50, 9.10] | 2.71 | -4.77 [-6.42, -3.77] | 0.07 | -7.40 [-9.06, -6.42] |
| full removal, some named | 122 | 5.38 [4.58, 6.70] | 1.35 | -4.04 [-5.42, -3.18] | -0.92 | -6.30 [-7.71, -5.42] |
| full removal, none named | 7 | -2.94 [-3.15, -2.75] | -1.49 | 1.44 [1.07, 1.84] | -1.32 | 1.61 [1.21, 2.05] |

### Reference V2 (a from `match_all_turn0`, b from `match_all_turn0`): typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 3.43 [2.45, 4.69] | 2.51 | -0.92 [-2.19, 0.07] | 0.64 | -2.79 [-4.06, -1.82] |
| single slot named: area | 138 | 4.62 [4.39, 4.84] | 1.00 | -3.62 [-3.86, -3.36] | 0.20 | -4.43 [-4.67, -4.17] |
| single slot named: price | 150 | 3.31 [2.36, 4.58] | 2.25 | -1.07 [-2.36, -0.10] | 0.46 | -2.85 [-4.13, -1.91] |
| single slot named: mean over slots | 430 | 3.79 | 1.92 | -1.87 [-2.65, -1.41] | 0.43 | -3.36 [-4.13, -2.89] |
| single slot omitted: food | 53 | -1.30 [-1.50, -1.12] | -2.02 | -0.72 [-1.02, -0.39] | -1.97 | -0.67 [-0.97, -0.35] |
| single slot omitted: area | 48 | -1.29 [-1.51, -1.12] | -0.96 | 0.33 [0.07, 0.62] | -0.95 | 0.34 [0.09, 0.63] |
| single slot omitted: price | 50 | -1.26 [-1.46, -1.09] | -1.48 | -0.22 [-0.55, 0.11] | -1.43 | -0.17 [-0.50, 0.15] |
| single slot omitted: mean over slots | 151 | -1.28 | -1.48 | -0.20 [-0.37, -0.02] | -1.45 | -0.17 [-0.34, 0.01] |
| full removal, every constrained slot named | 106 | 7.47 [6.50, 9.10] | 3.35 | -4.13 [-5.78, -3.12] | 0.06 | -7.41 [-9.07, -6.43] |
| full removal, some named | 122 | 5.38 [4.58, 6.70] | 1.27 | -4.12 [-5.50, -3.27] | -1.69 | -7.07 [-8.48, -6.19] |
| full removal, none named | 7 | -2.94 [-3.15, -2.75] | -2.13 | 0.81 [0.20, 1.48] | -2.08 | 0.85 [0.23, 1.53] |

### Reference V3 (a from `pooled_first_restaurant_turn`, b from `pooled_first_restaurant_turn`): likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.46 [3.82, 5.78] | 2.25 | -2.21 [-3.52, -1.58] | 0.54 | -3.92 [-5.23, -3.29] |
| single slot named: area | 138 | 3.60 [3.17, 4.28] | 0.89 | -2.71 [-3.39, -2.27] | 0.18 | -3.42 [-4.10, -2.98] |
| single slot named: price | 150 | 3.69 [3.19, 4.49] | 2.08 | -1.62 [-2.40, -1.09] | 0.43 | -3.27 [-4.07, -2.76] |
| single slot named: mean over slots | 430 | 3.92 | 1.74 | -2.18 [-2.74, -1.87] | 0.38 | -3.54 [-4.10, -3.22] |
| single slot omitted: food | 53 | -0.87 [-0.91, -0.82] | -1.34 | -0.47 [-0.65, -0.27] | -1.24 | -0.38 [-0.55, -0.19] |
| single slot omitted: area | 48 | -0.80 [-0.84, -0.76] | -0.60 | 0.20 [0.06, 0.34] | -0.50 | 0.30 [0.16, 0.42] |
| single slot omitted: price | 50 | -0.80 [-0.84, -0.76] | -1.00 | -0.20 [-0.39, -0.01] | -0.86 | -0.07 [-0.25, 0.11] |
| single slot omitted: mean over slots | 151 | -0.82 | -0.98 | -0.16 [-0.27, -0.05] | -0.87 | -0.05 [-0.15, 0.05] |
| full removal, every constrained slot named | 106 | 7.87 [7.24, 9.00] | 2.71 | -5.16 [-6.33, -4.52] | 0.07 | -7.79 [-8.95, -7.14] |
| full removal, some named | 122 | 6.07 [5.54, 7.02] | 1.35 | -4.72 [-5.76, -4.14] | -0.92 | -6.99 [-8.04, -6.39] |
| full removal, none named | 7 | -1.88 [-1.95, -1.82] | -1.49 | 0.39 [0.13, 0.67] | -1.32 | 0.56 [0.26, 0.90] |

### Reference V3 (a from `pooled_first_restaurant_turn`, b from `pooled_first_restaurant_turn`): typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.46 [3.82, 5.78] | 2.51 | -1.95 [-3.25, -1.31] | 0.64 | -3.83 [-5.13, -3.20] |
| single slot named: area | 138 | 3.60 [3.17, 4.28] | 1.00 | -2.60 [-3.27, -2.16] | 0.20 | -3.41 [-4.08, -2.96] |
| single slot named: price | 150 | 3.69 [3.19, 4.49] | 2.25 | -1.45 [-2.22, -0.94] | 0.46 | -3.23 [-4.04, -2.73] |
| single slot named: mean over slots | 430 | 3.92 | 1.92 | -2.00 [-2.57, -1.69] | 0.43 | -3.49 [-4.05, -3.18] |
| single slot omitted: food | 53 | -0.87 [-0.91, -0.82] | -2.02 | -1.15 [-1.38, -0.90] | -1.97 | -1.10 [-1.34, -0.86] |
| single slot omitted: area | 48 | -0.80 [-0.84, -0.76] | -0.96 | -0.16 [-0.36, 0.05] | -0.95 | -0.15 [-0.34, 0.05] |
| single slot omitted: price | 50 | -0.80 [-0.84, -0.76] | -1.48 | -0.68 [-0.96, -0.40] | -1.43 | -0.64 [-0.91, -0.37] |
| single slot omitted: mean over slots | 151 | -0.82 | -1.48 | -0.66 [-0.82, -0.52] | -1.45 | -0.63 [-0.79, -0.48] |
| full removal, every constrained slot named | 106 | 7.87 [7.24, 9.00] | 3.35 | -4.52 [-5.69, -3.88] | 0.06 | -7.81 [-8.98, -7.16] |
| full removal, some named | 122 | 6.07 [5.54, 7.02] | 1.27 | -4.80 [-5.84, -4.21] | -1.69 | -7.76 [-8.83, -7.15] |
| full removal, none named | 7 | -1.88 [-1.95, -1.82] | -2.13 | -0.24 [-0.78, 0.37] | -2.08 | -0.20 [-0.78, 0.44] |

### Reference V4 (a from `pooled_no_woz`, b from `pooled_no_woz`): likely

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.45 [3.72, 6.37] | 2.25 | -2.20 [-4.11, -1.47] | 0.54 | -3.91 [-5.82, -3.18] |
| single slot named: area | 138 | 3.50 [2.98, 4.40] | 0.89 | -2.61 [-3.54, -2.06] | 0.18 | -3.32 [-4.25, -2.79] |
| single slot named: price | 150 | 3.38 [2.84, 4.29] | 2.08 | -1.30 [-2.23, -0.76] | 0.43 | -2.95 [-3.86, -2.42] |
| single slot named: mean over slots | 430 | 3.78 | 1.74 | -2.04 [-2.74, -1.70] | 0.38 | -3.39 [-4.08, -3.07] |
| single slot omitted: food | 53 | -0.75 [-0.79, -0.71] | -1.34 | -0.59 [-0.77, -0.39] | -1.24 | -0.50 [-0.68, -0.31] |
| single slot omitted: area | 48 | -0.68 [-0.72, -0.64] | -0.60 | 0.08 [-0.05, 0.23] | -0.50 | 0.18 [0.05, 0.31] |
| single slot omitted: price | 50 | -0.68 [-0.72, -0.64] | -1.00 | -0.32 [-0.51, -0.13] | -0.86 | -0.19 [-0.37, -0.01] |
| single slot omitted: mean over slots | 151 | -0.70 | -0.98 | -0.27 [-0.38, -0.17] | -0.87 | -0.17 [-0.27, -0.07] |
| full removal, every constrained slot named | 106 | 7.57 [6.90, 8.98] | 2.71 | -4.86 [-6.32, -4.17] | 0.07 | -7.50 [-8.95, -6.80] |
| full removal, some named | 122 | 5.94 [5.36, 7.14] | 1.35 | -4.60 [-5.85, -3.96] | -0.92 | -6.87 [-8.14, -6.21] |
| full removal, none named | 7 | -1.61 [-1.68, -1.55] | -1.49 | 0.12 [-0.14, 0.42] | -1.32 | 0.29 [-0.01, 0.65] |

### Reference V4 (a from `pooled_no_woz`, b from `pooled_no_woz`): typical

| Pairs | n | L_E (95%) | Explicit mean L_J | Explicit OG (95%) | Implicit mean L_J | Implicit OG (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| single slot named: food | 142 | 4.45 [3.72, 6.37] | 2.51 | -1.94 [-3.85, -1.21] | 0.64 | -3.82 [-5.73, -3.08] |
| single slot named: area | 138 | 3.50 [2.98, 4.40] | 1.00 | -2.50 [-3.43, -1.95] | 0.20 | -3.30 [-4.23, -2.77] |
| single slot named: price | 150 | 3.38 [2.84, 4.29] | 2.25 | -1.14 [-2.06, -0.58] | 0.46 | -2.92 [-3.83, -2.38] |
| single slot named: mean over slots | 430 | 3.78 | 1.92 | -1.86 [-2.56, -1.51] | 0.43 | -3.35 [-4.04, -3.02] |
| single slot omitted: food | 53 | -0.75 [-0.79, -0.71] | -2.02 | -1.27 [-1.51, -1.03] | -1.97 | -1.22 [-1.46, -0.99] |
| single slot omitted: area | 48 | -0.68 [-0.72, -0.64] | -0.96 | -0.28 [-0.47, -0.07] | -0.95 | -0.27 [-0.46, -0.07] |
| single slot omitted: price | 50 | -0.68 [-0.72, -0.64] | -1.48 | -0.80 [-1.08, -0.53] | -1.43 | -0.75 [-1.03, -0.49] |
| single slot omitted: mean over slots | 151 | -0.70 | -1.48 | -0.78 [-0.94, -0.63] | -1.45 | -0.75 [-0.90, -0.60] |
| full removal, every constrained slot named | 106 | 7.57 [6.90, 8.98] | 3.35 | -4.22 [-5.69, -3.50] | 0.06 | -7.51 [-8.96, -6.80] |
| full removal, some named | 122 | 5.94 [5.36, 7.14] | 1.27 | -4.68 [-5.92, -4.04] | -1.69 | -7.63 [-8.90, -6.96] |
| full removal, none named | 7 | -1.61 [-1.68, -1.55] | -2.13 | -0.51 [-1.04, 0.13] | -2.08 | -0.47 [-1.03, 0.19] |

## Repeat noise: `20261026T041814Z-repeats`

| Prompt version | Groups | Identical groups | Deterministic | Mean SD of l | Max range of l |
| --- | --- | --- | --- | --- | --- |
| multiwoz_likely | 302 | 33 | False | 0.058 | 0.410 |

## Prediction scorecard

| ID | Prediction | Value (95%) | Result |
| --- | --- | --- | --- |
| P1 | Explicit, named, single-slot OG within ±1.5 nats (mean over slots, likely) | -2.45 [-3.01, -2.13] | fail |
| P2 | Implicit, named, single-slot OG ≤ −2 nats (mean over slots, likely) | -3.81 [-4.36, -3.49] | pass |
| P3 | Implicit full removal with all constrained slots named: OG ≤ −6 nats (likely) | -8.33 [-9.49, -7.68] | pass |
| P4 | Wording gap mean ≤ −1 logit (explicit penalized more than implicit; likely) | -1.13 [-1.19, -1.07] | pass |
| P5 | Tie offset |mean l(G|impl) − l(ctrl:tie)| > 0.5 logit (likely) | 0.07 [-0.06, 0.20] | fail |
| P6 | Phase 8 language-model check | n/a | pending |
