# Week of 24 Nov: typos and length, hypothesis rewording, quantile bins, and a trained estimator

- **A. Typos and filler:** one typo or filler sentence costs 11–17 nats of log p but moves the Bayes factor only 8–13% for Qwen3-8B-Base and the simulators; the instruct models are 2–10 times less robust.
- **B. Hypothesis rewording:** the top hypothesis changes for 6.8% of items with raw log-likelihood and 15–16% with binned typicality; the bin edges, not the score, are the fragile part.
- **C. Quantile bins:** they lose to fixed edges under every rewrite (para1 bin agreement 66% against 82%), so the frozen rule trained on cent1B:6bins.
- **D. Trained estimator:** a 0.6B student matches the teacher (Spearman 0.92–0.94, no drop on unseen cuisines), keeps the size principle and the contradiction gap, scores about 8,000 times faster, and changes its top hypothesis less often than raw 8B log-likelihood; the untrained 0.6B log-probability is as accurate but less stable (D6 fails).

Inputs and commit: `manifest_abc.json` (git `303fb6c`). Intervals: item bootstrap, 2,000 replicates, seed 20261124. Plan: `docs/week_nov24_plan.md`; decisions and deviations: `docs/week_nov24_decisions.md`.

## Part A: typos, style and length

**Median Δ log p(U | h) per unit, nats** (95% item-bootstrap interval)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo1 | -14.22 [-14.64, -13.71] | -13.97 [-14.46, -13.46] | -16.50 [-16.98, -15.99] | -13.75 [-14.25, -13.08] | -25.97 [-26.73, -24.60] | -20.70 [-22.22, -18.81] |
| typo2 | -25.40 [-26.41, -24.75] | -26.21 [-26.83, -24.96] | -29.34 [-30.14, -28.38] | -24.56 [-25.77, -23.30] | -46.94 [-49.11, -45.30] | -36.91 [-38.12, -35.57] |
| typo4 | -44.56 [-46.16, -43.19] | -45.93 [-47.91, -45.08] | -50.34 [-51.49, -48.86] | -44.16 [-45.34, -42.83] | -74.79 [-80.34, -70.16] | -59.14 [-61.83, -57.64] |
| slottypo1 | -13.79 [-14.17, -13.24] | -13.38 [-14.01, -13.06] | -15.46 [-16.07, -14.77] | -12.83 [-13.56, -12.43] | -22.58 [-24.18, -20.96] | -19.89 [-20.75, -18.88] |
| lower | -3.47 [-3.69, -3.30] | -4.59 [-4.92, -4.39] | -2.64 [-2.87, -2.35] | -2.74 [-3.07, -2.52] | -22.18 [-23.14, -20.87] | -19.20 [-20.62, -17.80] |
| nopunct | -4.75 [-5.22, -4.49] | -5.15 [-5.76, -4.25] | -4.32 [-4.85, -3.33] | -1.85 [-2.64, -1.06] | -10.79 [-14.54, -7.72] | -14.50 [-16.38, -12.82] |
| noapos | -5.13 [-5.38, -4.91] | -5.33 [-5.85, -5.17] | -5.93 [-6.74, -5.18] | -4.59 [-5.14, -4.31] | -14.17 [-15.19, -13.37] | -9.09 [-10.29, -7.65] |
| fill1 | -11.11 [-11.86, -10.58] | -11.85 [-12.34, -11.36] | -14.68 [-15.53, -14.02] | -13.46 [-14.51, -12.60] | -14.41 [-16.20, -13.16] | -19.10 [-19.96, -18.23] |
| fill2 | -21.50 [-22.85, -19.42] | -21.65 [-23.05, -19.61] | -24.18 [-25.66, -22.25] | -22.03 [-23.34, -20.79] | -34.31 [-36.41, -31.61] | -42.01 [-46.03, -38.55] |
| fill4 | -42.85 [-44.36, -41.57] | -46.05 [-47.80, -44.96] | -52.55 [-54.19, -51.15] | -46.53 [-47.81, -45.51] | -77.25 [-80.24, -75.60] | -79.57 [-83.32, -76.60] |
| terse | 1.97 [0.49, 3.90] | 0.72 [-1.46, 2.44] | 0.72 [-1.06, 3.97] | 6.83 [4.41, 9.04] | 18.68 [13.45, 22.03] | 15.39 [11.77, 21.71] |
| terse_rx | 3.88 [1.92, 5.54] | 1.00 [-0.38, 2.27] | 1.56 [0.07, 3.30] | 9.06 [8.09, 11.78] | 20.80 [17.80, 24.32] | 16.95 [13.52, 22.17] |
| fixed | 6.54 [5.53, 7.45] | 7.68 [7.13, 8.21] | 8.84 [8.02, 9.93] | 7.09 [5.60, 7.39] | 17.78 [16.54, 23.37] | 7.29 [4.11, 12.73] |

**Median |Δ log BF|, nats** (95% item-bootstrap interval)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo1 | 0.46 [0.43, 0.50] | 0.48 [0.44, 0.53] | 1.08 [0.99, 1.17] | 1.10 [1.03, 1.19] | 2.17 [2.00, 2.32] | 4.10 [3.76, 4.53] |
| typo2 | 0.65 [0.60, 0.70] | 0.69 [0.65, 0.77] | 1.48 [1.33, 1.60] | 1.58 [1.47, 1.68] | 3.56 [3.31, 3.86] | 4.96 [4.51, 5.43] |
| typo4 | 0.98 [0.87, 1.09] | 1.00 [0.88, 1.12] | 1.68 [1.53, 1.83] | 2.13 [1.90, 2.25] | 5.73 [5.25, 6.25] | 6.08 [5.32, 6.57] |
| slottypo1 | 0.70 [0.65, 0.76] | 0.83 [0.74, 0.91] | 1.32 [1.21, 1.43] | 1.39 [1.30, 1.48] | 3.73 [3.43, 4.06] | 4.16 [3.72, 4.65] |
| lower | 0.46 [0.43, 0.50] | 0.51 [0.47, 0.55] | 0.85 [0.80, 0.92] | 0.99 [0.91, 1.07] | 2.50 [2.32, 2.73] | 3.33 [3.04, 3.58] |
| nopunct | 0.38 [0.37, 0.43] | 0.49 [0.43, 0.56] | 0.83 [0.71, 0.95] | 0.91 [0.83, 0.99] | 4.35 [3.87, 4.86] | 4.31 [3.85, 4.67] |
| noapos | 0.51 [0.45, 0.56] | 0.42 [0.38, 0.48] | 0.86 [0.75, 0.95] | 0.89 [0.75, 1.04] | 1.43 [1.27, 1.62] | 4.83 [4.28, 5.27] |
| fill1 | 0.43 [0.39, 0.47] | 0.43 [0.40, 0.47] | 0.72 [0.67, 0.79] | 0.73 [0.68, 0.78] | 3.04 [2.79, 3.31] | 2.64 [2.35, 2.93] |
| fill2 | 0.78 [0.72, 0.84] | 0.70 [0.65, 0.77] | 1.21 [1.12, 1.30] | 1.28 [1.22, 1.37] | 3.60 [3.30, 3.82] | 6.81 [6.29, 7.33] |
| fill4 | 0.95 [0.89, 1.01] | 1.06 [0.98, 1.14] | 1.91 [1.79, 2.04] | 1.68 [1.57, 1.79] | 5.29 [4.93, 5.66] | 6.50 [6.08, 7.03] |
| terse | 1.12 [0.99, 1.32] | 1.62 [1.36, 1.94] | 2.71 [2.36, 3.10] | 2.52 [2.14, 2.90] | 8.97 [7.99, 9.71] | 8.69 [7.34, 9.80] |
| terse_rx | 1.14 [1.07, 1.27] | 1.83 [1.68, 2.00] | 3.03 [2.75, 3.25] | 2.76 [2.53, 2.94] | 9.68 [8.92, 10.30] | 8.20 [7.60, 9.03] |
| fixed | 0.46 [0.30, 0.64] | 0.37 [0.27, 0.57] | 0.63 [0.25, 1.42] | 0.63 [0.41, 1.07] | 2.46 [1.50, 3.79] | 2.84 [1.91, 5.68] |

**Median |Δ log BF| ÷ |log BF|** (95% item-bootstrap interval)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo1 | 0.10 [0.09, 0.11] | 0.08 [0.07, 0.09] | 0.13 [0.12, 0.14] | 0.13 [0.12, 0.15] | 0.18 [0.16, 0.19] | 0.46 [0.42, 0.52] |
| typo2 | 0.14 [0.13, 0.16] | 0.11 [0.11, 0.12] | 0.17 [0.16, 0.19] | 0.20 [0.18, 0.22] | 0.28 [0.26, 0.31] | 0.56 [0.52, 0.62] |
| typo4 | 0.22 [0.19, 0.25] | 0.16 [0.15, 0.19] | 0.21 [0.19, 0.23] | 0.26 [0.23, 0.29] | 0.44 [0.39, 0.49] | 0.61 [0.54, 0.68] |
| slottypo1 | 0.16 [0.14, 0.18] | 0.15 [0.13, 0.17] | 0.17 [0.15, 0.18] | 0.17 [0.15, 0.18] | 0.32 [0.28, 0.34] | 0.47 [0.40, 0.53] |
| lower | 0.11 [0.10, 0.11] | 0.08 [0.08, 0.09] | 0.11 [0.10, 0.12] | 0.12 [0.11, 0.13] | 0.20 [0.18, 0.22] | 0.38 [0.33, 0.42] |
| nopunct | 0.08 [0.07, 0.10] | 0.08 [0.07, 0.09] | 0.10 [0.09, 0.12] | 0.11 [0.10, 0.13] | 0.39 [0.33, 0.44] | 0.49 [0.45, 0.54] |
| noapos | 0.11 [0.10, 0.14] | 0.07 [0.06, 0.08] | 0.10 [0.10, 0.11] | 0.10 [0.09, 0.12] | 0.12 [0.10, 0.14] | 0.58 [0.48, 0.73] |
| fill1 | 0.09 [0.08, 0.10] | 0.07 [0.06, 0.08] | 0.09 [0.08, 0.09] | 0.09 [0.08, 0.10] | 0.25 [0.22, 0.29] | 0.29 [0.26, 0.31] |
| fill2 | 0.16 [0.14, 0.17] | 0.11 [0.10, 0.12] | 0.14 [0.13, 0.15] | 0.15 [0.14, 0.17] | 0.29 [0.26, 0.32] | 0.78 [0.69, 0.87] |
| fill4 | 0.20 [0.18, 0.21] | 0.16 [0.15, 0.18] | 0.23 [0.22, 0.25] | 0.20 [0.18, 0.21] | 0.42 [0.38, 0.44] | 0.72 [0.63, 0.80] |
| terse | 0.28 [0.25, 0.32] | 0.32 [0.27, 0.37] | 0.40 [0.36, 0.45] | 0.35 [0.31, 0.39] | 0.79 [0.70, 0.84] | 0.98 [0.85, 1.10] |
| terse_rx | 0.27 [0.25, 0.29] | 0.30 [0.28, 0.33] | 0.39 [0.37, 0.42] | 0.34 [0.31, 0.36] | 0.77 [0.72, 0.81] | 0.92 [0.83, 0.99] |
| fixed | 0.08 [0.05, 0.12] | 0.06 [0.04, 0.07] | 0.05 [0.03, 0.13] | 0.08 [0.04, 0.11] | 0.15 [0.11, 0.28] | 0.57 [0.31, 0.90] |

**Share of Bayes factors that change sign** (95% item-bootstrap interval)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo1 | 1.9% [1.1%, 3.0%] | 2.6% [1.5%, 3.8%] | 2.4% [1.5%, 3.4%] | 2.7% [1.8%, 3.8%] | 6.7% [5.3%, 8.2%] | 16.1% [13.9%, 18.5%] |
| typo2 | 2.2% [1.3%, 3.3%] | 2.3% [1.5%, 3.2%] | 2.6% [1.7%, 3.5%] | 3.5% [2.5%, 4.6%] | 10.3% [8.5%, 12.1%] | 20.8% [18.2%, 23.5%] |
| typo4 | 3.7% [2.1%, 5.6%] | 3.6% [2.4%, 5.1%] | 3.4% [2.3%, 4.7%] | 5.0% [3.5%, 6.8%] | 13.2% [10.9%, 15.7%] | 21.8% [18.3%, 25.4%] |
| slottypo1 | 1.3% [0.7%, 2.0%] | 1.6% [1.0%, 2.2%] | 1.9% [1.2%, 2.6%] | 2.5% [1.8%, 3.4%] | 11.9% [10.0%, 13.8%] | 14.6% [12.1%, 17.0%] |
| lower | 1.0% [0.5%, 1.7%] | 1.6% [1.0%, 2.4%] | 2.2% [1.5%, 3.1%] | 2.2% [1.5%, 3.0%] | 5.8% [4.6%, 7.0%] | 12.8% [10.9%, 14.9%] |
| nopunct | 0.7% [0.2%, 1.2%] | 3.0% [2.0%, 4.2%] | 3.3% [2.1%, 4.6%] | 1.8% [1.1%, 2.6%] | 12.4% [10.3%, 14.5%] | 17.2% [14.7%, 19.8%] |
| noapos | 1.3% [0.5%, 2.4%] | 1.5% [0.4%, 2.8%] | 1.5% [0.6%, 2.6%] | 1.3% [0.3%, 2.5%] | 4.9% [2.9%, 7.2%] | 21.4% [16.8%, 26.2%] |
| fill1 | 1.6% [0.9%, 2.3%] | 1.4% [0.8%, 2.0%] | 2.0% [1.3%, 2.7%] | 1.9% [1.3%, 2.6%] | 9.2% [7.7%, 10.8%] | 8.8% [7.2%, 10.5%] |
| fill2 | 2.0% [1.3%, 2.8%] | 2.0% [1.3%, 2.9%] | 2.2% [1.5%, 3.0%] | 2.6% [1.7%, 3.5%] | 9.3% [7.8%, 11.0%] | 24.2% [21.3%, 26.9%] |
| fill4 | 1.9% [1.2%, 2.8%] | 3.1% [2.0%, 4.2%] | 3.6% [2.7%, 4.6%] | 3.5% [2.6%, 4.7%] | 12.5% [10.7%, 14.4%] | 24.8% [21.9%, 27.7%] |
| terse | 2.1% [1.1%, 3.5%] | 4.7% [2.9%, 6.7%] | 5.8% [3.6%, 8.4%] | 4.4% [2.6%, 6.4%] | 23.6% [20.2%, 27.1%] | 36.9% [32.0%, 41.6%] |
| terse_rx | 1.9% [1.2%, 2.7%] | 3.7% [2.7%, 4.9%] | 5.2% [3.9%, 6.6%] | 3.7% [2.7%, 4.8%] | 25.2% [22.5%, 27.8%] | 34.4% [31.0%, 37.8%] |
| fixed | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 4.0% [0.0%, 8.9%] | 18.0% [3.7%, 37.5%] |

**Posterior argmax flip rate** (95% item-bootstrap interval)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo1 | 3.8% [1.7%, 6.4%] | 5.1% [2.6%, 8.1%] | 4.3% [1.7%, 7.2%] | 4.3% [1.7%, 7.2%] | 10.2% [6.8%, 14.0%] | 20.0% [14.9%, 25.1%] |
| typo2 | 4.3% [1.7%, 7.3%] | 6.0% [3.4%, 9.4%] | 5.6% [2.6%, 8.7%] | 6.5% [3.4%, 9.6%] | 13.4% [9.4%, 17.9%] | 27.6% [22.0%, 33.6%] |
| typo4 | 4.7% [1.5%, 8.8%] | 9.4% [4.8%, 15.0%] | 4.7% [1.5%, 8.6%] | 9.4% [4.7%, 14.6%] | 18.0% [11.6%, 25.0%] | 32.8% [24.6%, 40.8%] |
| slottypo1 | 3.5% [1.3%, 6.1%] | 4.8% [2.2%, 7.8%] | 4.8% [2.2%, 7.8%] | 4.4% [2.2%, 7.4%] | 18.0% [13.1%, 23.2%] | 22.4% [16.7%, 27.8%] |
| lower | 2.6% [0.9%, 4.7%] | 3.4% [1.3%, 6.0%] | 3.9% [1.7%, 6.5%] | 4.3% [1.7%, 7.3%] | 7.3% [4.3%, 10.8%] | 15.5% [11.2%, 20.2%] |
| nopunct | 2.3% [0.5%, 4.5%] | 9.5% [5.9%, 13.4%] | 5.9% [3.1%, 9.3%] | 3.2% [1.3%, 5.8%] | 13.6% [9.0%, 18.4%] | 31.2% [25.2%, 37.3%] |
| noapos | 2.6% [0.0%, 6.7%] | 2.6% [0.0%, 7.2%] | 0.0% [0.0%, 0.0%] | 3.9% [0.0%, 8.8%] | 6.6% [1.4%, 13.0%] | 28.9% [18.8%, 39.4%] |
| fill1 | 3.8% [1.7%, 6.4%] | 3.8% [1.7%, 6.4%] | 5.1% [2.6%, 8.1%] | 3.0% [1.3%, 5.1%] | 11.9% [8.1%, 16.2%] | 16.2% [11.5%, 20.9%] |
| fill2 | 4.3% [2.1%, 7.2%] | 5.5% [3.0%, 8.5%] | 4.7% [2.1%, 7.7%] | 3.4% [1.3%, 6.0%] | 10.2% [6.8%, 14.0%] | 31.5% [25.5%, 37.4%] |
| fill4 | 3.8% [1.7%, 6.4%] | 7.7% [4.7%, 11.1%] | 7.2% [4.3%, 10.6%] | 6.0% [3.0%, 8.9%] | 17.9% [13.2%, 22.6%] | 32.3% [26.4%, 38.3%] |
| terse | 2.0% [0.0%, 5.2%] | 8.9% [3.8%, 14.8%] | 10.9% [5.1%, 17.4%] | 5.9% [1.9%, 10.9%] | 35.6% [26.3%, 44.9%] | 59.4% [49.6%, 69.7%] |
| terse_rx | 4.3% [1.8%, 7.3%] | 9.1% [5.7%, 13.0%] | 11.3% [7.4%, 15.5%] | 6.5% [3.5%, 10.0%] | 34.3% [28.7%, 40.3%] | 51.7% [45.3%, 58.2%] |
| fixed | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 16.7% [0.0%, 51.1%] | 16.7% [0.0%, 55.6%] |

**Unit costs** (median, nats; the slot row is a ratio)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| per typo, typo1 | -14.22 [-14.64, -13.71] | -13.97 [-14.46, -13.46] | -16.50 [-16.98, -15.99] | -13.75 [-14.25, -13.08] | -25.97 [-26.73, -24.60] | -20.70 [-22.22, -18.81] |
| per typo, typo4 | -11.14 [-11.54, -10.80] | -11.48 [-11.98, -11.27] | -12.59 [-12.87, -12.22] | -11.04 [-11.33, -10.71] | -18.70 [-20.09, -17.54] | -14.79 [-15.46, -14.41] |
| per sentence, fill1 | -11.11 [-11.86, -10.58] | -11.85 [-12.34, -11.36] | -14.68 [-15.53, -14.02] | -13.46 [-14.51, -12.60] | -14.41 [-16.20, -13.16] | -19.10 [-19.96, -18.23] |
| per sentence, fill4 | -10.71 [-11.09, -10.39] | -11.51 [-11.95, -11.24] | -13.14 [-13.55, -12.79] | -11.63 [-11.95, -11.38] | -19.31 [-20.06, -18.90] | -19.89 [-20.83, -19.15] |
| per added token, fill4 | -1.90 [-1.93, -1.86] | -2.03 [-2.07, -2.00] | -2.32 [-2.38, -2.26] | -2.08 [-2.10, -2.03] | -3.37 [-3.45, -3.28] | -3.54 [-3.64, -3.46] |
| per removed token, terse | -0.28 [-0.57, -0.15] | -0.10 [-0.30, 0.17] | -0.07 [-0.56, 0.30] | -0.92 [-1.19, -0.77] | -2.51 [-2.97, -2.10] | -2.43 [-3.02, -1.54] |
| named slot's log BF kept, slottypo1 | 1.03 [0.98, 1.07] | 1.16 [1.10, 1.22] | 1.05 [1.00, 1.12] | 1.06 [0.97, 1.09] | 0.60 [0.51, 0.67] | 1.16 [1.00, 1.27] |

**typo1 by operation**: median Δ log p, nats

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| delete | -11.27 | -11.23 | -13.59 | -11.27 | -22.72 | -14.58 |
| double | -14.30 | -13.60 | -16.43 | -14.29 | -25.38 | -21.38 |
| neighbour | -17.71 | -17.47 | -19.49 | -16.73 | -29.30 | -24.59 |
| swap | -14.56 | -14.20 | -17.00 | -13.12 | -27.15 | -20.78 |

**Dose slopes** (least squares on unit values, nats per typo or per sentence)

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| typo: abs_delta_log_bf | 0.18 [0.15, 0.21] | 0.23 [0.18, 0.27] | 0.23 [0.17, 0.30] | 0.42 [0.34, 0.50] | 1.34 [1.17, 1.52] | 0.66 [0.47, 0.86] |
| typo: delta_log_p | -10.02 [-10.35, -9.69] | -10.60 [-10.92, -10.26] | -10.95 [-11.32, -10.58] | -10.12 [-10.52, -9.72] | -16.05 [-17.01, -15.02] | -12.95 [-13.78, -12.12] |
| fill: abs_delta_log_bf | 0.18 [0.16, 0.21] | 0.22 [0.20, 0.24] | 0.45 [0.41, 0.49] | 0.33 [0.29, 0.37] | 0.75 [0.63, 0.87] | 1.30 [1.16, 1.45] |
| fill: delta_log_p | -10.49 [-10.79, -10.20] | -11.59 [-11.92, -11.26] | -12.93 [-13.32, -12.53] | -11.37 [-11.72, -11.03] | -20.95 [-21.51, -20.41] | -20.63 [-21.49, -19.81] |

**Length both ways**: median |Δ log BF|, `terse` next to the simulator study's `verbose`

| Kind | Qwen3-8B-Base · F | OSim-8B-Mid | OSim-8B | UserLM-8B | Qwen2.5-7B-Inst | Qwen3-8B |
|---|---|---|---|---|---|---|
| verbose (Phase 2, Claude rewrites) | 1.30 [1.20, 1.39] | 1.34 [1.20, 1.47] | 2.02 [1.84, 2.24] | 2.35 [2.17, 2.55] | 6.75 [6.10, 7.30] | 8.34 [7.65, 9.01] |
| terse | 1.12 [0.99, 1.32] | 1.62 [1.36, 1.94] | 2.71 [2.36, 3.10] | 2.52 [2.14, 2.90] | 8.97 [7.99, 9.71] | 8.69 [7.34, 9.80] |
| terse_rx | 1.14 [1.07, 1.27] | 1.83 [1.68, 2.00] | 3.03 [2.75, 3.25] | 2.76 [2.53, 2.94] | 9.68 [8.92, 10.30] | 8.20 [7.60, 9.03] |

**Typicality, Qwen3-8B-Base · F as the reference**: share of human pairs whose bin changes

| Variant | typo1 | typo2 | typo4 | slottypo1 | lower | nopunct | noapos | fill1 | fill2 | fill4 | terse | terse_rx | fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glob1A | 68.6% | 96.0% | 93.7% | 65.9% | 14.9% | 24.9% | 24.2% | 53.1% | 86.9% | 96.6% | 31.2% | 33.9% | 41.9% |
| glob1B | 68.6% | 96.0% | 93.7% | 65.9% | 14.9% | 24.9% | 24.2% | 53.1% | 86.9% | 96.6% | 31.2% | 33.9% | 41.9% |
| cent1B | 7.6% | 8.1% | 10.2% | 13.1% | 5.8% | 7.2% | 7.5% | 4.4% | 10.3% | 14.7% | 21.2% | 19.3% | 6.5% |
| cent1B:6bins | 10.5% | 12.4% | 17.4% | 16.6% | 8.1% | 10.0% | 10.0% | 6.9% | 14.1% | 19.5% | 25.0% | 23.7% | 6.5% |
| glob1Q | 90.7% | 99.8% | 99.6% | 89.0% | 24.8% | 42.4% | 40.7% | 79.2% | 96.6% | 99.8% | 43.7% | 46.5% | 53.2% |
| cent1Q | 20.6% | 26.4% | 35.0% | 28.8% | 17.6% | 20.5% | 18.9% | 13.8% | 27.2% | 33.8% | 43.8% | 44.7% | 19.4% |
| glob1Q:3q | 86.9% | 97.8% | 96.0% | 83.4% | 17.5% | 34.3% | 30.3% | 71.7% | 94.2% | 97.8% | 39.7% | 42.6% | 32.3% |
| cent1Q:3q | 13.4% | 16.6% | 21.3% | 20.1% | 11.1% | 12.5% | 12.4% | 9.3% | 17.9% | 24.2% | 34.9% | 35.5% | 8.1% |

Share moved into the extreme bin

| Variant | typo1 | typo2 | typo4 | slottypo1 | lower | nopunct | noapos | fill1 | fill2 | fill4 | terse | terse_rx | fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glob1A | 17.3% | 55.6% | 93.0% | 16.5% | 2.2% | 5.5% | 5.9% | 14.1% | 40.0% | 92.2% | 0.0% | 0.0% | 0.0% |
| glob1B | 17.3% | 55.6% | 93.0% | 16.5% | 2.2% | 5.5% | 5.9% | 14.1% | 40.0% | 92.2% | 0.0% | 0.0% | 0.0% |
| cent1B | 2.6% | 3.7% | 4.9% | 2.3% | 1.7% | 2.5% | 3.8% | 1.2% | 1.6% | 1.9% | 4.5% | 4.5% | 6.5% |
| cent1B:6bins | 1.9% | 3.3% | 5.8% | 1.8% | 0.5% | 0.5% | 1.8% | 1.8% | 2.4% | 2.6% | 1.4% | 0.8% | 0.0% |
| glob1Q | 0.0% | 0.4% | 2.9% | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.4% | 1.8% | 0.0% | 0.0% | 0.0% |
| cent1Q | 2.0% | 3.5% | 6.1% | 2.1% | 0.5% | 0.6% | 1.8% | 1.7% | 2.4% | 2.5% | 1.0% | 0.7% | 0.0% |
| glob1Q:3q | 13.2% | 48.5% | 93.8% | 13.6% | 2.2% | 4.6% | 5.5% | 11.3% | 36.4% | 90.3% | 0.0% | 0.0% | 0.0% |
| cent1Q:3q | 3.4% | 4.8% | 6.1% | 2.3% | 0.9% | 1.7% | 1.8% | 1.7% | 2.3% | 3.0% | 2.6% | 3.2% | 4.8% |

Median |Δ score|, nats

| Variant | typo1 | typo2 | typo4 | slottypo1 | lower | nopunct | noapos | fill1 | fill2 | fill4 | terse | terse_rx | fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glob1A | 0.62 | 0.97 | 1.25 | 0.58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 1.45 | 0.00 | 0.00 | 0.00 |
| glob1B | 19.66 | 19.66 | 19.66 | 19.66 | 0.00 | 0.00 | 0.00 | 19.66 | 19.66 | 39.32 | 0.00 | 0.00 | 0.00 |
| cent1B | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cent1B:6bins | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| glob1Q | 12.16 | 23.35 | 37.33 | 12.16 | 0.00 | 0.00 | 0.00 | 12.16 | 21.84 | 37.33 | 0.00 | 0.00 | 9.68 |
| cent1Q | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| glob1Q:3q | 14.75 | 28.37 | 86.92 | 14.75 | 0.00 | 0.00 | 0.00 | 13.61 | 28.37 | 86.92 | 0.00 | 0.00 | 0.00 |
| cent1Q:3q | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**A3: the six natural typos of "restaurant", corrected** (teacher bins under the gold hypothesis, n = 6, descriptive)

| Variant | Item | Bin with typo | Bin corrected | Score with typo | Score corrected |
|---|---|---|---|---|---|
| cent1B:6bins | SNG0458 | 1 | 1 | 18.13 | 18.13 |
| cent1B:6bins | SNG0472 | 3 | 3 | 6.30 | 6.30 |
| cent1B:6bins | SNG0494 | 3 | 3 | 6.30 | 6.30 |
| cent1B:6bins | SNG0524 | 3 | 4 | 6.30 | 0.39 |
| cent1B:6bins | SNG0575 | 3 | 3 | 6.30 | 6.30 |
| cent1B:6bins | SNG0580 | 3 | 3 | 6.30 | 6.30 |
| cent1Q | SNG0458 | 0 | 0 | 14.00 | 14.00 |
| cent1Q | SNG0472 | 2 | 2 | 4.51 | 4.51 |
| cent1Q | SNG0494 | 1 | 1 | 7.87 | 7.87 |
| cent1Q | SNG0524 | 2 | 2 | 4.51 | 4.51 |
| cent1Q | SNG0575 | 1 | 1 | 7.87 | 7.87 |
| cent1Q | SNG0580 | 2 | 2 | 4.51 | 4.51 |

**Batch layout.** The same original scored in different units' batches (max − min per pair, nats); co-batching keeps this out of every Δ:

| Model | Pairs | Median | p90 |
|---|---|---|---|
| Qwen3-8B-Base · F | 2355 | 0.000 | 0.000 |
| OSim-8B-Mid | 2355 | 0.000 | 0.000 |
| OSim-8B | 2355 | 0.000 | 0.000 |
| UserLM-8B | 2355 | 0.000 | 0.097 |
| Qwen2.5-7B-Inst | 2355 | 0.000 | 0.000 |
| Qwen3-8B | 2355 | 0.000 | 0.000 |

## Part B: hypothesis rewording

Hypothesis rewording was already measured on raw log-likelihood in the simulator study (`docs/simulator_results/`). This part adds the typicality side. Bin agreement and the global-normalization variants need no samples under the reworded text; glob1A does, so it has rewording rows only for the models with an `hsample` run.

| Model | Score | Argmax flip rate | SD of P(G) | Bin agreement (hpara1–3, hzh mean) | Size principle kept, reworded (mean of kinds) |
|---|---|---|---|---|---|
| Qwen3-8B-Base · F | raw | 6.8% [5.1%, 8.6%] | 0.094 [0.084, 0.103] | – | 1.11 |
| Qwen3-8B-Base · F | glob1A | 49.1% [45.1%, 53.0%] | 0.023 [0.021, 0.026] | 94.8% | 0.02 |
| Qwen3-8B-Base · F | glob1B | 7.0% [4.5%, 9.5%] | 0.059 [0.046, 0.072] | 94.8% | 1.06 |
| Qwen3-8B-Base · F | cent1B | 14.9% [12.0%, 17.9%] | 0.144 [0.129, 0.158] | 89.3% | 1.02 |
| Qwen3-8B-Base · F | cent1B:6bins | 16.2% [13.1%, 19.3%] | 0.148 [0.134, 0.162] | 85.9% | 1.09 |
| Qwen3-8B-Base · F | glob1Q | 2.8% [1.5%, 4.2%] | 0.060 [0.047, 0.074] | 93.5% | 0.86 |
| Qwen3-8B-Base · F | cent1Q | 18.6% [15.3%, 22.1%] | 0.126 [0.112, 0.138] | 74.0% | 1.18 |
| Qwen3-8B-Base · F | glob1Q:3q | 2.2% [1.0%, 3.6%] | 0.038 [0.028, 0.049] | 95.6% | 0.89 |
| Qwen3-8B-Base · F | cent1Q:3q | 17.5% [14.2%, 20.7%] | 0.151 [0.136, 0.165] | 79.3% | 1.34 |
| OSim-8B-Mid | raw | 16.8% [14.3%, 19.5%] | 0.166 [0.150, 0.181] | – | 1.45 |
| OSim-8B-Mid | glob1A | 46.5% [42.4%, 50.6%] | 0.033 [0.030, 0.036] | 91.0% | 0.09 |
| OSim-8B-Mid | glob1B | 6.1% [4.0%, 8.4%] | 0.092 [0.076, 0.108] | 90.8% | 1.59 |
| OSim-8B-Mid | cent1B | 14.0% [10.9%, 17.0%] | 0.152 [0.135, 0.169] | 84.5% | 1.50 |
| OSim-8B-Mid | cent1B:6bins | 14.2% [11.0%, 17.1%] | 0.153 [0.136, 0.170] | 84.4% | 1.50 |
| OSim-8B-Mid | glob1Q | 14.0% [11.0%, 17.1%] | 0.155 [0.138, 0.172] | 82.6% | 1.47 |
| OSim-8B-Mid | cent1Q | 5.2% [3.3%, 7.4%] | 0.076 [0.065, 0.087] | 68.4% | 1.70 |
| OSim-8B-Mid | glob1Q:3q | 10.8% [8.0%, 13.7%] | 0.130 [0.113, 0.147] | 88.1% | 2.75 |
| OSim-8B-Mid | cent1Q:3q | 5.0% [3.2%, 7.2%] | 0.076 [0.066, 0.088] | 73.7% | 1.75 |
| OSim-8B | raw | 14.7% [12.1%, 17.3%] | 0.147 [0.131, 0.163] | – | 1.88 |
| OSim-8B | glob1A | 54.1% [50.0%, 58.4%] | 0.050 [0.047, 0.054] | 95.2% | 0.07 |
| OSim-8B | glob1B | 9.6% [6.7%, 12.7%] | 0.062 [0.047, 0.077] | 94.7% | 0.31 |
| OSim-8B | cent1B | 12.3% [9.4%, 15.1%] | 0.128 [0.110, 0.146] | 91.5% | 1.18 |
| OSim-8B | cent1B:6bins | 15.1% [11.9%, 18.4%] | 0.149 [0.133, 0.165] | 74.6% | 1.81 |
| OSim-8B | glob1Q | 13.7% [10.7%, 16.8%] | 0.140 [0.123, 0.158] | 84.1% | 1.49 |
| OSim-8B | cent1Q | 17.8% [14.3%, 21.1%] | 0.167 [0.150, 0.182] | 68.3% | 1.60 |
| OSim-8B | glob1Q:3q | 10.4% [7.5%, 13.6%] | 0.068 [0.053, 0.084] | 92.3% | 1.18 |
| OSim-8B | cent1Q:3q | 18.1% [14.7%, 21.6%] | 0.161 [0.144, 0.177] | 83.5% | 2.58 |
| UserLM-8B | raw | 18.5% [15.7%, 21.4%] | 0.198 [0.180, 0.216] | – | 1.83 |
| UserLM-8B | glob1B | 6.2% [4.0%, 8.6%] | 0.062 [0.050, 0.075] | 92.0% | 1.89 |
| UserLM-8B | cent1B | 9.4% [6.9%, 12.0%] | 0.108 [0.095, 0.123] | 83.2% | 1.64 |
| UserLM-8B | cent1B:6bins | 9.4% [6.9%, 12.0%] | 0.108 [0.095, 0.123] | 83.2% | 1.64 |
| UserLM-8B | glob1Q | 15.8% [12.5%, 19.1%] | 0.148 [0.133, 0.165] | 83.9% | 1.88 |
| UserLM-8B | cent1Q | 11.5% [8.8%, 14.5%] | 0.129 [0.116, 0.143] | 63.0% | 1.65 |
| UserLM-8B | glob1Q:3q | 11.8% [8.9%, 14.7%] | 0.108 [0.092, 0.123] | 88.8% | 1.94 |
| UserLM-8B | cent1Q:3q | 6.4% [4.3%, 9.0%] | 0.072 [0.061, 0.085] | 78.7% | 0.72 |
| Qwen2.5-7B-Inst | raw | 22.8% [19.7%, 25.9%] | 0.171 [0.148, 0.195] | – | 2.43 |
| Qwen2.5-7B-Inst | glob1A | 22.6% [20.2%, 25.3%] | 0.048 [0.045, 0.052] | 100.0% | -0.08 |
| Qwen2.5-7B-Inst | glob1B | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen2.5-7B-Inst | cent1B | 3.3% [1.6%, 5.2%] | 0.037 [0.025, 0.049] | 97.8% | 0.09 |
| Qwen2.5-7B-Inst | cent1B:6bins | 15.6% [12.3%, 18.7%] | 0.124 [0.105, 0.142] | 89.9% | 0.62 |
| Qwen2.5-7B-Inst | glob1Q | 0.3% [0.0%, 1.0%] | 0.001 [0.000, 0.004] | 99.9% | 0.00 |
| Qwen2.5-7B-Inst | cent1Q | 16.8% [13.5%, 20.1%] | 0.129 [0.109, 0.149] | 87.0% | 0.46 |
| Qwen2.5-7B-Inst | glob1Q:3q | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen2.5-7B-Inst | cent1Q:3q | 6.0% [3.8%, 8.4%] | 0.063 [0.048, 0.080] | 95.7% | 0.26 |
| Qwen3-8B | raw | 42.0% [38.6%, 45.2%] | 0.250 [0.227, 0.273] | – | 1.05 |
| Qwen3-8B | glob1B | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen3-8B | cent1B | 1.5% [0.4%, 2.9%] | 0.009 [0.004, 0.015] | 99.6% | 0.03 |
| Qwen3-8B | cent1B:6bins | 9.4% [7.2%, 11.8%] | 0.087 [0.071, 0.105] | 95.4% | 0.42 |
| Qwen3-8B | glob1Q | 0.5% [0.0%, 1.4%] | 0.001 [0.000, 0.002] | 99.9% | 0.00 |
| Qwen3-8B | cent1Q | 24.1% [21.1%, 27.3%] | 0.184 [0.166, 0.203] | 85.2% | 0.78 |
| Qwen3-8B | glob1Q:3q | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen3-8B | cent1Q:3q | 2.1% [0.9%, 3.5%] | 0.016 [0.009, 0.024] | 99.2% | 0.11 |
| Qwen2.5-7B-Inst · B | raw | 18.2% [14.9%, 21.5%] | 0.130 [0.108, 0.151] | – | 2.64 |
| Qwen2.5-7B-Inst · B | glob1B | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen2.5-7B-Inst · B | cent1B | 3.4% [1.9%, 5.1%] | 0.041 [0.029, 0.054] | 98.1% | 0.14 |
| Qwen2.5-7B-Inst · B | cent1B:6bins | 13.5% [10.5%, 16.6%] | 0.113 [0.095, 0.131] | 91.9% | 0.87 |
| Qwen2.5-7B-Inst · B | glob1Q | 0.6% [0.0%, 1.3%] | 0.002 [0.000, 0.005] | 99.9% | 0.00 |
| Qwen2.5-7B-Inst · B | cent1Q | 13.7% [10.6%, 16.7%] | 0.112 [0.092, 0.131] | 89.4% | 0.64 |
| Qwen2.5-7B-Inst · B | glob1Q:3q | 0.0% [0.0%, 0.0%] | 0.000 [0.000, 0.000] | 100.0% | 0.00 |
| Qwen2.5-7B-Inst · B | cent1Q:3q | 4.4% [2.7%, 6.4%] | 0.057 [0.042, 0.072] | 96.9% | 0.32 |

## Part C: quantile bins

| Model | Variant | Extreme bin | Size kept | Contra log BF | para1 | typo | pad | verbose | zh | typo1 | fill2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B-Base · F | glob1A | 3.4% | 0.02 | 0.04 | 75.2% | 3.7% | 3.5% | 15.0% | 21.5% | 31.4% | 13.1% |
| Qwen3-8B-Base · F | glob1B | 3.4% | 1.10 | 5.17 | 75.2% | 3.7% | 3.5% | 15.0% | 21.5% | 31.4% | 13.1% |
| Qwen3-8B-Base · F | cent1B | 63.2% | 1.18 | 5.27 | 87.3% | 83.0% | 88.7% | 83.2% | 78.1% | 92.4% | 89.7% |
| Qwen3-8B-Base · F | cent1B:6bins | 4.4% | 1.26 | 5.94 | 81.8% | 76.1% | 83.9% | 75.4% | 72.9% | 89.5% | 85.9% |
| Qwen3-8B-Base · F | glob1Q | 0.0% | 1.11 | 4.53 | 58.4% | 0.3% | 0.0% | 6.4% | 8.0% | 9.3% | 3.4% |
| Qwen3-8B-Base · F | cent1Q | 4.9% | 1.27 | 5.94 | 66.0% | 58.6% | 69.6% | 55.3% | 52.8% | 79.4% | 72.8% |
| Qwen3-8B-Base · F | glob1Q:3q | 2.3% | 1.06 | 4.58 | 66.9% | 2.6% | 2.4% | 9.2% | 13.1% | 13.1% | 5.8% |
| Qwen3-8B-Base · F | cent1Q:3q | 43.5% | 1.47 | 7.03 | 77.1% | 72.4% | 80.7% | 70.9% | 61.7% | 86.6% | 82.1% |
| OSim-8B-Mid | glob1A | 20.6% | 0.11 | 0.35 | 66.5% | 20.7% | 20.5% | 25.6% | 44.6% | 33.8% | 25.6% |
| OSim-8B-Mid | glob1B | 20.6% | 1.75 | 7.15 | 66.5% | 20.7% | 20.5% | 25.6% | 44.6% | 33.8% | 25.6% |
| OSim-8B-Mid | cent1B | 3.4% | 1.79 | 6.62 | 84.2% | 79.1% | 90.4% | 81.2% | 60.7% | 91.3% | 93.0% |
| OSim-8B-Mid | cent1B:6bins | 0.0% | 1.80 | 6.66 | 84.1% | 79.1% | 90.3% | 81.0% | 60.5% | 91.3% | 93.0% |
| OSim-8B-Mid | glob1Q | 0.1% | 1.62 | 6.02 | 48.0% | 0.1% | 1.1% | 6.8% | 21.9% | 14.6% | 5.9% |
| OSim-8B-Mid | cent1Q | 1.1% | 1.30 | 4.58 | 74.5% | 72.8% | 79.7% | 66.4% | 47.7% | 86.0% | 84.2% |
| OSim-8B-Mid | glob1Q:3q | 17.8% | 3.34 | 11.10 | 55.6% | 17.8% | 17.7% | 21.1% | 31.3% | 19.7% | 18.2% |
| OSim-8B-Mid | cent1Q:3q | 8.8% | 1.33 | 4.65 | 78.3% | 77.1% | 83.3% | 71.0% | 54.6% | 88.2% | 87.3% |
| OSim-8B | glob1A | 88.9% | 0.16 | 0.28 | 87.2% | 88.8% | 88.9% | 87.5% | 88.4% | 89.0% | 89.0% |
| OSim-8B | glob1B | 88.9% | 0.51 | 2.02 | 87.2% | 88.8% | 88.9% | 87.5% | 88.4% | 89.0% | 89.0% |
| OSim-8B | cent1B | 80.6% | 1.62 | 6.21 | 89.8% | 89.7% | 91.8% | 89.8% | 85.2% | 93.1% | 94.0% |
| OSim-8B | cent1B:6bins | 33.4% | 2.16 | 8.41 | 80.4% | 77.9% | 81.8% | 73.1% | 60.2% | 87.3% | 87.3% |
| OSim-8B | glob1Q | 37.6% | 1.86 | 7.34 | 54.8% | 37.9% | 37.5% | 37.6% | 47.1% | 38.6% | 38.0% |
| OSim-8B | cent1Q | 12.4% | 1.88 | 7.35 | 67.8% | 65.0% | 68.8% | 60.0% | 45.1% | 78.4% | 77.7% |
| OSim-8B | glob1Q:3q | 87.7% | 1.85 | 7.48 | 85.2% | 87.6% | 87.6% | 86.2% | 87.1% | 87.7% | 87.7% |
| OSim-8B | cent1Q:3q | 68.3% | 3.28 | 12.04 | 82.7% | 79.4% | 83.5% | 81.6% | 73.6% | 88.3% | 89.0% |
| UserLM-8B | glob1A | 0.3% | 0.08 | 0.26 | 80.4% | 3.9% | 15.5% | 26.3% | 65.5% | 59.1% | 38.9% |
| UserLM-8B | glob1B | 0.3% | 2.51 | 8.84 | 80.4% | 3.9% | 15.5% | 26.3% | 65.5% | 59.1% | 38.9% |
| UserLM-8B | cent1B | 36.0% | 1.95 | 6.28 | 84.6% | 82.3% | 86.4% | 81.9% | 72.8% | 90.5% | 90.2% |
| UserLM-8B | cent1B:6bins | 0.0% | 1.95 | 6.28 | 84.6% | 82.3% | 86.4% | 81.9% | 72.8% | 90.5% | 90.2% |
| UserLM-8B | glob1Q | 0.0% | 2.41 | 8.56 | 64.9% | 0.2% | 1.4% | 9.6% | 43.7% | 33.8% | 13.1% |
| UserLM-8B | cent1Q | 5.1% | 1.82 | 6.21 | 71.6% | 68.2% | 73.1% | 68.7% | 55.3% | 81.1% | 78.9% |
| UserLM-8B | glob1Q:3q | 0.2% | 2.33 | 8.03 | 75.8% | 2.8% | 3.1% | 21.9% | 61.8% | 56.9% | 30.2% |
| UserLM-8B | cent1Q:3q | 10.4% | 0.95 | 3.77 | 86.3% | 83.6% | 85.5% | 82.8% | 72.6% | 89.7% | 88.8% |
| Qwen2.5-7B-Inst | glob1A | 100.0% | -0.01 | 0.02 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen2.5-7B-Inst | glob1B | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen2.5-7B-Inst | cent1B | 97.6% | 0.34 | 1.40 | 95.2% | 97.1% | 97.9% | 96.8% | 96.3% | 98.5% | 97.8% |
| Qwen2.5-7B-Inst | cent1B:6bins | 85.5% | 1.41 | 5.26 | 83.3% | 85.4% | 89.8% | 85.0% | 82.2% | 93.8% | 91.3% |
| Qwen2.5-7B-Inst | glob1Q | 99.9% | 0.01 | 0.03 | 99.8% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% |
| Qwen2.5-7B-Inst | cent1Q | 84.2% | 1.05 | 3.93 | 79.6% | 80.9% | 85.5% | 81.8% | 79.6% | 91.3% | 86.5% |
| Qwen2.5-7B-Inst | glob1Q:3q | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen2.5-7B-Inst | cent1Q:3q | 95.2% | 0.88 | 3.37 | 91.7% | 94.6% | 95.0% | 94.1% | 92.8% | 96.8% | 95.8% |
| Qwen3-8B | glob1A | 100.0% | -0.03 | 0.04 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen3-8B | glob1B | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen3-8B | cent1B | 99.8% | 0.02 | 0.09 | 99.3% | 99.8% | 99.7% | 99.8% | 99.7% | 99.7% | 99.7% |
| Qwen3-8B | cent1B:6bins | 96.6% | 0.54 | 2.11 | 93.6% | 95.3% | 92.1% | 95.2% | 94.0% | 97.3% | 92.4% |
| Qwen3-8B | glob1Q | 100.0% | 0.00 | 0.00 | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen3-8B | cent1Q | 86.3% | 1.14 | 4.84 | 84.2% | 83.9% | 75.2% | 82.6% | 80.2% | 91.6% | 77.1% |
| Qwen3-8B | glob1Q:3q | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen3-8B | cent1Q:3q | 99.7% | 0.09 | 0.34 | 99.1% | 99.5% | 99.1% | 99.4% | 99.2% | 99.6% | 99.0% |
| Qwen2.5-7B-Inst · B | glob1A | 100.0% | -0.07 | 0.09 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | – | – |
| Qwen2.5-7B-Inst · B | glob1B | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | – | – |
| Qwen2.5-7B-Inst · B | cent1B | 97.8% | 0.36 | 1.45 | 95.9% | 97.8% | 98.1% | 96.5% | 95.4% | – | – |
| Qwen2.5-7B-Inst · B | cent1B:6bins | 87.6% | 1.34 | 5.52 | 85.1% | 88.1% | 91.4% | 85.6% | 82.7% | – | – |
| Qwen2.5-7B-Inst · B | glob1Q | 99.9% | 0.01 | 0.04 | 99.8% | 99.9% | 99.9% | 99.9% | 99.9% | – | – |
| Qwen2.5-7B-Inst · B | cent1Q | 87.0% | 1.04 | 4.24 | 81.6% | 85.9% | 87.7% | 81.9% | 79.9% | – | – |
| Qwen2.5-7B-Inst · B | glob1Q:3q | 100.0% | 0.00 | 0.00 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | – | – |
| Qwen2.5-7B-Inst · B | cent1Q:3q | 96.7% | 0.79 | 3.06 | 94.1% | 96.2% | 97.0% | 94.1% | 93.1% | – | – |

Bin agreement columns: para1 to zh are the claims rewrites (Phase 2 scores), typo1 and fill2 are Part A's (co-batched).

**Training target (plan 5.1 rule).** cent1Q if its bin agreement under para1 and typo1 is within 2 points of cent1B:6bins or better and its size principle kept (impl, slot named) is in [0.5, 2]; otherwise cent1B:6bins. cent1Q: para1 66.0%, typo1 79.4%, size kept 1.27; cent1B:6bins: para1 81.8%, typo1 89.5%. **Rule's choice: cent1B:6bins.**

## Part D: a first trained typicality estimator

Teacher: cent1B:6bins with Qwen3-8B-Base · F as the reference (H -9.26, sigma 5.91). Held-out foods: african, european, french, mexican, spanish (17 items). Student: Qwen3-0.6B-Base with LoRA r 16.

**Training**

| Run | Training pairs | Best step | Best val loss | Minutes | Pairs/s (prediction) |
|---|---|---|---|---|---|
| C-mlp | 266,006 | 3000 | 0.516 | 32.8 | 1179 |
| R-lin | 266,006 | 8314 | 0.058 | 33.3 | 1098 |
| R-mlp | 266,006 | 8314 | 0.061 | 32.9 | 1176 |

**Agreement with the teacher** (human pairs; the synthetic columns are held-out samples)

| Run | Spearman | Seen food | Unseen food | Exact bin | Within one bin | Synthetic val | Synthetic, held-out food |
|---|---|---|---|---|---|---|---|
| C-mlp | 0.92 | 0.92 | 0.94 | 83.2% [81.3%, 85.1%] | 100.0% [99.9%, 100.0%] | 0.90 | 0.90 |
| R-lin | 0.94 | 0.94 | 0.96 | 87.3% [85.5%, 89.0%] | 100.0% [100.0%, 100.0%] | 0.91 | 0.93 |
| R-mlp | 0.94 | 0.94 | 0.95 | 87.6% [85.8%, 89.4%] | 100.0% [100.0%, 100.0%] | 0.90 | 0.93 |

**Inference on the human pairs** (size principle kept: L_J ÷ L_E, implicit regime; top-1 gives 1/k credit when the true goal ties k-way at the top, which binned scores often do)

| Score | Size kept, named | Size kept, omitted | Contra log BF | Mean log P(G) | Top-1 |
|---|---|---|---|---|---|
| C-mlp · expected | 1.44 | 1.45 | 6.49 | -1.52 [-1.73, -1.29] | 44.7% [38.3%, 51.1%] |
| C-mlp · argmax | 1.44 | 1.31 | 6.51 | -1.64 [-1.94, -1.34] | 53.4% [48.7%, 58.2%] |
| R-lin · cont | 1.27 | 1.17 | 5.78 | -1.28 [-1.45, -1.11] | 44.7% [38.3%, 51.1%] |
| R-lin · bin | 1.35 | 0.85 | 5.84 | -1.32 [-1.59, -1.07] | 55.7% [51.5%, 60.2%] |
| R-mlp · cont | 1.27 | 1.15 | 5.86 | -1.27 [-1.43, -1.10] | 44.3% [37.9%, 50.6%] |
| R-mlp · bin | 1.32 | 0.79 | 5.94 | -1.34 [-1.60, -1.09] | 54.6% [50.3%, 58.9%] |
| teacher (64 samples, 8B) | 1.26 | 1.02 | 5.94 | -1.53 [-1.80, -1.27] | 50.6% [46.3%, 54.9%] |
| raw log-lik, Qwen3-8B-Base · F | 1.23 | 1.27 | 5.79 | -1.40 [-1.58, -1.21] | 44.3% [37.9%, 50.6%] |
| raw log-lik, Qwen3-0.6B-Base · F, untrained | 1.02 | 0.83 | 4.75 | -1.18 [-1.35, -1.02] | 47.2% [40.9%, 53.6%] |
| Jev (likely) | 0.09 | 0.65 | 2.85 | -1.76 [-1.83, -1.68] | 31.6% [26.2%, 37.2%] |
| Jev (typical) | 0.10 | 1.09 | 3.23 | -2.07 [-2.18, -1.95] | 27.2% [22.6%, 32.2%] |

**Cost.** The teacher needs 7.1 s of 8B sampling and scoring per new hypothesis text; the students score C-mlp 1179, R-lin 1098, R-mlp 1176 pairs per second.

**Verdict (frozen rule): The pipeline works.**

## Prediction scorecard

| ID | Prediction | Result |
|---|---|---|
| T1 | median Delta log p <= -3 nats per typo at typo1, every model | pass |
| T2 | median \|Delta log BF\| <= 0.5 at typo1 for the human-like models, larger for qwen25_inst and qwen3_inst | fail |
| T3 | typo2, qwen3_base reference: glob1B changes bin for >= 20% of human pairs, cent1B:6bins for <= 10% | fail |
| T4 | human-like models keep >= 50% of the named slot's log BF under slottypo1 | pass |
| T5 | each added sentence costs >= 5 nats of log p; \|Delta log BF\| at fill4 <= 1 nat median for the human-like models | fail |
| T6 | lower, nopunct, noapos move \|Delta log BF\| by <= 0.5 nats median for the human-like models | fail |
| D1 | Spearman >= 0.7 between T_hat and T on human pairs, for at least one run | pass |
| D2 | size principle kept (implicit, slot named) between 0.5 and 2 | pass |
| D3 | contradiction log Bayes factor >= 3 nats | pass |
| D4 | bin agreement under para1 >= the teacher's | pass |
| D5 | >= 100 times more pairs per second than the teacher for a new hypothesis | pass |
| D6 | the 0.6B model's own log-prob is worse than the student on both \|size kept - 1\| and \|delta log BF\| under para | fail |
| D7 | Spearman drops by <= 0.1 on unseen-food items | pass |

## Notes for Xuan

1. **Typos and filler are expensive in log p and cheap for inference, with human-like models.**
   - One typo costs 14–17 nats of log p(U | h) for the base and simulator models, and 21–26 for the
     instruct models. Each further typo costs about 10 more.
   - It moves the Bayes factor by 8–13% of its size (median |Δ log BF| 0.5 nats for Qwen3-8B-Base and
     OSim-8B-Mid, 1.1 for OSim-8B and UserLM). 2–3% of Bayes factors change sign, and the posterior's
     top hypothesis changes for 4–5% of turns.
   - A filler sentence costs 11–15 nats, about 2 nats per added token, and moves the Bayes factor by
     7–9% at one sentence and 16–23% at four.
   - The instruct models are 2–10 times less robust: one typo moves Qwen3-8B's Bayes factor by 46% and
     changes its top hypothesis for 20% of turns.
   - So your point holds for the human-like models, mostly: they ignore much of what a reader ignores,
     though not all of it. Predictions T2, T5 and T6 set the bar at 0.5–1 nat for all four human-like
     models. Qwen3-8B-Base and OSim-8B-Mid meet it or miss by at most 0.06 nats; OSim-8B and UserLM miss
     by 0.35–0.9.
2. **A typo inside the slot value costs no evidence.** "itallian" keeps the full log Bayes factor of
   the named slot (ratio 1.03–1.16) for every model except Qwen2.5-7B-Instruct (0.60). The models read
   misspelled values as the values.
3. **Length moves inference both ways about equally.** The terse rewrites move the Bayes factor by about
   as much as the verbose ones did in the simulator study (1.1 against 1.3 nats for Qwen3-8B-Base; 2.7
   against 2.0 for OSim-8B). Only 101 of 235 terse rewrites passed the frozen NLI check, because a
   fragment ("thai restaurant west") does not entail the full sentence for the NLI model. The
   regex-only set (230 units) gives the same numbers, so the selection does not drive the result.
4. **Correcting a natural typo raises log p by 6.5–9 nats** for the base and simulator models and moves
   the Bayes factor by 0.4–0.6 nats; the teacher's bin changes for 1 of the 6 turns.
5. **Raw-surprisal typicality reacts to every edit; centered typicality mostly does not.** With
   Qwen3-8B-Base as the reference, glob1B changes bin for 69% of human pairs after one typo and 96% after
   two. cent1B:6bins changes for 10% and 12%, cent1B for 8%. T3 asked for at most 10% at two typos, so
   it fails narrowly; the contrast it predicts is large.
6. **Typicality is less stable than raw log-likelihood under hypothesis rewording (Part B).** For
   Qwen3-8B-Base, rewording a hypothesis changes the posterior's top hypothesis for 6.8% of items with
   raw log-likelihood and 15–16% with the centered variants. The bins quantize the score, so a small
   shift that crosses an edge moves the whole bin. Bin agreement under rewording is 86–89% and the size
   principle survives (1.0–1.1). For the instruct models typicality looks stable only because almost
   every pair sits in one bin.
7. **Quantile bins do not beat fixed edges (Part C).** cent1Q keeps the size principle and the
   contradiction gap as well as cent1B:6bins, but its bin agreement is lower under every rewrite
   (para1 66% against 82%, typo1 79% against 90%, hypothesis rewording 74% against 86%). Human pairs sit
   high in the samples' distribution: under cent1B:6bins 88% fall in two bins, while the quantile edges
   spread them over all six, and every extra edge among them is a place to flip. So the plan's rule
   chose cent1B:6bins as the training target.
8. **Every Δ in Part A is against an original scored in the same batch.** Rescoring the same original
   in different units' batches gives the same score for five of the six models (UserLM: p90 0.1 nats),
   so batch layout adds nothing here.
9. **A 0.6B student learns the teacher, and its scores do the inference job (Part D).**
   - Three Qwen3-0.6B-Base students (LoRA r 16, 266,006 synthetic pairs, 33 minutes each on one H100)
     reach Spearman 0.92–0.94 with the teacher on the 2,355 human pairs. 83–88% land in the teacher's
     bin, and every pair lands within one bin (one exception, for C-mlp).
   - Held-out cuisines cost nothing: Spearman on the 17 unseen-food items is 0.94–0.96 (D7).
   - Used as the likelihood, the students' continuous scores keep the size principle (1.27 named, 1.15–1.45
     omitted) and the contradiction gap (5.8–6.5 nats, teacher 5.9). Goal accuracy matches the teacher
     and raw 8B log-likelihood: mean log P(G) −1.27 to −1.52 against −1.53 and −1.40; top-1 44–45%.
   - They are more stable than anything else here. The top hypothesis changes for 1.3–2.2% of turns
     under para1 (raw 8B 5.2%, teacher 18.8%) and 0.9–2.1% under two filler sentences (4.3%, 18.7%).
     Under one typo they match raw 8B (1.7–4.7% against 3.8%). Under hypothesis rewording, R-lin and
     R-mlp match raw 8B (6.8–7.6%) and C-mlp is worse (10.3%).
   - Much of the gain over the teacher comes from the output being continuous. The students' own
     binned outputs flip about as often as the teacher (para1 14–17%). The fair comparison, the teacher's
     unbinned centred score, is not in this report.
   - Cost: the teacher needs 7.1 s of 8B sampling per new hypothesis text; a student scores about
     1,100 pairs per second, roughly 8,000 times faster (D5).
   - D6 fails. The untrained 0.6B model's own log-probability keeps the size principle better than the
     students (1.02 against 1.27) and has the best mean log P(G) of all (−1.18). Its weakness is
     stability: |Δ log BF| under paraphrase is 0.95 nats against 0.58–0.62 for the students. So what the
     student buys over its own backbone is robustness, not accuracy.
   - The frozen rule's verdict is "the pipeline works": D1–D3 pass for all three runs.
10. **Caveats.** First turns only; one seed; the restaurant domain; bfloat16; 6 natural typos (A3) is
   too few for statistics; T5's per-sentence threshold (≥ 5 nats) is met by far, its fill4 threshold is not.
   Part D: one training seed per run; the R runs were still improving at the last step (best val loss at
   step 8,314 of 8,314), while C-mlp peaked at step 3,000. The training target is clipped at T = 2.07, so
   R students flatten above it (figure 4). Top-1 for binned scores gives 1/k credit for k-way ties.
