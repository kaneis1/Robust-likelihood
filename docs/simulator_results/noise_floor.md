# Simulator study: noise floor of the utterance-rewrite shifts

Each primary run's urewrite part was scored again with every rewrite replaced by its original utterance: the same rows, order, batch size (32), bfloat16 and code path as Phase 2. Claim 1's |Δ log BF| from those scores is rescoring noise alone, aggregated exactly as the real shifts (median over the size-principle and contradiction pairs, 95% item-bootstrap intervals). Noise runs at commit `a408f62`; generated 2026-11-14T06:56:28Z.

A null row that lands in a batch of the same shape as its original reproduces the original bit for bit (see the last table), so about half the null pairs have no noise at all, and the overall noise median understates what a real rewrite, which always changes length, carries. The columns marked *changed* use only the null pairs whose scores moved: the conservative noise floor.

## para

| Model | Real median \|Δ log BF\| | Noise median, all | Noise median, changed | Noise p95, changed | Real pairs above changed p95 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 6.15 [5.63, 6.76] | 0.03 [0.00, 0.06] | 0.32 | 0.97 | 90% |
| Qwen2.5-7B-Inst · B (broadened) | 5.83 [5.37, 6.43] | 0.02 [0.00, 0.07] | 0.33 | 1.12 | 88% |
| Qwen3-8B · S | 6.97 [6.40, 7.60] | 0.02 [0.00, 0.09] | 0.46 | 1.74 | 86% |
| Qwen3-8B-Base · F (few-shot) | 0.95 [0.88, 1.03] | 0.00 [0.00, 0.03] | 0.18 | 0.68 | 62% |
| OSim-8B-Mid · S | 1.24 [1.12, 1.35] | 0.02 [0.00, 0.04] | 0.18 | 0.62 | 71% |
| OSim-8B · S | 1.85 [1.73, 2.01] | 0.01 [0.00, 0.05] | 0.21 | 0.83 | 75% |
| UserLM-8B · S | 1.85 [1.75, 2.02] | 0.01 [0.00, 0.03] | 0.15 | 0.51 | 85% |

## zh

| Model | Real median \|Δ log BF\| | Noise median, all | Noise median, changed | Noise p95, changed | Real pairs above changed p95 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 7.84 [7.19, 8.51] | 0.01 [0.00, 0.06] | 0.31 | 0.95 | 92% |
| Qwen2.5-7B-Inst · B (broadened) | 7.43 [6.78, 7.95] | 0.01 [0.00, 0.06] | 0.32 | 1.08 | 91% |
| Qwen3-8B · S | 8.66 [7.96, 9.63] | 0.01 [0.00, 0.09] | 0.47 | 1.77 | 88% |
| Qwen3-8B-Base · F (few-shot) | 1.46 [1.35, 1.56] | 0.00 [0.00, 0.02] | 0.16 | 0.69 | 74% |
| OSim-8B-Mid · S | 2.15 [1.98, 2.39] | 0.01 [0.00, 0.04] | 0.17 | 0.61 | 83% |
| OSim-8B · S | 2.66 [2.43, 2.92] | 0.01 [0.00, 0.05] | 0.21 | 0.80 | 82% |
| UserLM-8B · S | 3.10 [2.87, 3.38] | 0.01 [0.00, 0.03] | 0.14 | 0.49 | 92% |

## typo

| Model | Real median \|Δ log BF\| | Noise median, all | Noise median, changed | Noise p95, changed | Real pairs above changed p95 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 6.50 [5.96, 7.05] | 0.03 [0.00, 0.07] | 0.31 | 0.97 | 92% |
| Qwen2.5-7B-Inst · B (broadened) | 6.03 [5.57, 6.38] | 0.03 [0.00, 0.08] | 0.34 | 1.12 | 89% |
| Qwen3-8B · S | 6.76 [6.20, 7.40] | 0.03 [0.00, 0.10] | 0.46 | 1.76 | 86% |
| Qwen3-8B-Base · F (few-shot) | 1.08 [0.97, 1.17] | 0.00 [0.00, 0.03] | 0.18 | 0.68 | 64% |
| OSim-8B-Mid · S | 1.20 [1.10, 1.34] | 0.02 [0.00, 0.04] | 0.18 | 0.61 | 72% |
| OSim-8B · S | 2.12 [1.93, 2.31] | 0.02 [0.00, 0.06] | 0.21 | 0.83 | 79% |
| UserLM-8B · S | 2.35 [2.21, 2.52] | 0.01 [0.00, 0.03] | 0.15 | 0.51 | 89% |

## pad

| Model | Real median \|Δ log BF\| | Noise median, all | Noise median, changed | Noise p95, changed | Real pairs above changed p95 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 4.02 [3.78, 4.26] | 0.02 [0.00, 0.06] | 0.32 | 0.97 | 86% |
| Qwen2.5-7B-Inst · B (broadened) | 3.80 [3.55, 4.13] | 0.02 [0.00, 0.07] | 0.33 | 1.12 | 83% |
| Qwen3-8B · S | 7.17 [6.58, 7.77] | 0.02 [0.00, 0.09] | 0.46 | 1.72 | 86% |
| Qwen3-8B-Base · F (few-shot) | 0.90 [0.85, 0.96] | 0.00 [0.00, 0.02] | 0.17 | 0.68 | 61% |
| OSim-8B-Mid · S | 0.86 [0.80, 0.92] | 0.02 [0.00, 0.04] | 0.18 | 0.61 | 63% |
| OSim-8B · S | 1.54 [1.44, 1.67] | 0.02 [0.00, 0.05] | 0.21 | 0.83 | 70% |
| UserLM-8B · S | 1.84 [1.70, 1.94] | 0.01 [0.00, 0.03] | 0.15 | 0.51 | 85% |

## verbose

| Model | Real median \|Δ log BF\| | Noise median, all | Noise median, changed | Noise p95, changed | Real pairs above changed p95 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 6.75 [6.10, 7.30] | 0.02 [0.00, 0.06] | 0.32 | 0.98 | 92% |
| Qwen2.5-7B-Inst · B (broadened) | 6.55 [6.02, 7.06] | 0.01 [0.00, 0.06] | 0.33 | 1.12 | 91% |
| Qwen3-8B · S | 8.34 [7.65, 9.01] | 0.01 [0.00, 0.07] | 0.44 | 1.76 | 89% |
| Qwen3-8B-Base · F (few-shot) | 1.30 [1.20, 1.39] | 0.00 [0.00, 0.01] | 0.16 | 0.67 | 73% |
| OSim-8B-Mid · S | 1.34 [1.20, 1.47] | 0.01 [0.00, 0.04] | 0.18 | 0.61 | 74% |
| OSim-8B · S | 2.02 [1.84, 2.24] | 0.01 [0.00, 0.04] | 0.21 | 0.80 | 79% |
| UserLM-8B · S | 2.35 [2.17, 2.55] | 0.00 [0.00, 0.03] | 0.16 | 0.50 | 88% |

## Rescoring the human part in its own layout

The human pairs scored again in Phase 2's order and batches, against Phase 2's scores.

| Model | Pairs | Identical | Median \|gap\| (nats) | Max \|gap\| (nats) |
|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 2355 | 100.0% | 0.000 | 0.000 |
| Qwen2.5-7B-Inst · B (broadened) | 2355 | 100.0% | 0.000 | 0.000 |
| Qwen3-8B · S | 2355 | 100.0% | 0.000 | 0.000 |
| Qwen3-8B-Base · F (few-shot) | 2355 | 100.0% | 0.000 | 0.000 |
| OSim-8B-Mid · S | 2355 | 100.0% | 0.000 | 0.000 |
| OSim-8B · S | 2355 | 100.0% | 0.000 | 0.000 |
| UserLM-8B · S | 2355 | 100.0% | 0.000 | 0.000 |
