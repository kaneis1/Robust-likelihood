# CLIPS with small local language models as the utterance likelihood

- **K1 Replacement:** with a local base model as P(u | c), CLIPS keeps most of its accuracy. P(g_true) is 0.66–0.68 on mDKG (curie 0.74, humans 0.67) and 0.61–0.64 on VH (davinci-002 0.63). Goal correlation with humans is 0.83–0.88 (curie 0.93), so Qwen3-8B-Base misses both of C1's bars.
- **K2 Size and type:** size makes no difference down to 0.6B. Qwen3-0.6B-Base scores 0.68 on mDKG and 0.61 on VH, against 0.66 and 0.63 for 8B, and has the highest human correlation (0.88), so C2 fails in the opposite direction. Instruct tuning is not uniformly worse: Qwen3-8B's instruct version trails its base on mDKG (P(g_true) 0.64, r 0.79) and ties it on VH, while Qwen2.5-7B-Instruct is the best model on VH (0.69).
- **K3 Robustness:** rewording the instruction barely moves the posterior. The median |Δ P(g_true)| is at most 0.018 for every model and kind. Under a paraphrase or one typo, the base models change their top goal for 0–4% of instructions and the instruct models for 5–10%. Filler lowers the instruct models' mDKG correlation with humans by 0.16–0.24.

Plan: `docs/clips_small_lm_plan.md`; decisions and deviations: `docs/clips_decisions.md`. Intervals: bootstrap over problems (Table 1) or instructions (robustness), 2,000 replicates, seed 20261126. Table 1's definitions reproduce every cell of the paper's table from its own result files.

## K1 and K2: goal accuracy, assistance and agreement with humans

**Doors, Keys & Gems (mDKG): CLIPS**

|  | P(g_true) | Assist. prec. | Assist. rec. | Rel. plan length | Rel. human cost | Goal r | Assist. r |
|---|---|---|---|---|---|---|---|
| Humans (paper) | 0.67 (0.04) | 0.83 (0.02) | 0.85 (0.01) | – | – | – | – |
| CLIPS, paper (curie) | 0.74 (0.05) | 0.97 (0.03) | 0.97 (0.03) | 1.00 (0.00) | 1.00 (0.00) | 0.93 [0.88, 0.96] | 0.96 [0.93, 0.98] |
| Language-only IP, paper (curie) | 0.55 (0.05) | 0.90 (0.06) | 0.83 (0.06) | 1.26 (0.10) | 1.18 (0.07) | 0.74 [0.61, 0.85] | 0.83 [0.73, 0.93] |
| Action-only IP, paper (curie) | 0.31 (0.04) | 0.43 (0.09) | 0.40 (0.09) | 1.68 (0.15) | 1.46 (0.08) | 0.15 [-0.03, 0.35] | 0.45 [0.24, 0.63] |
| **CLIPS · Qwen3-0.6B-Base** | 0.68 (0.05) | 0.93 (0.05) | 0.93 (0.05) | 0.99 (0.01) | 0.99 (0.01) | 0.88 [0.81, 0.94] | 0.95 [0.90, 0.98] |
| **CLIPS · Qwen3-1.7B-Base** | 0.68 (0.05) | 0.93 (0.05) | 0.92 (0.05) | 0.99 (0.01) | 0.99 (0.01) | 0.88 [0.80, 0.93] | 0.93 [0.87, 0.97] |
| **CLIPS · Qwen3-4B-Base** | 0.66 (0.05) | 0.93 (0.05) | 0.92 (0.05) | 0.99 (0.01) | 0.99 (0.01) | 0.85 [0.76, 0.93] | 0.93 [0.87, 0.97] |
| **CLIPS · Qwen3-8B-Base** | 0.66 (0.05) | 0.90 (0.06) | 0.88 (0.06) | 1.00 (0.00) | 1.00 (0.00) | 0.83 [0.68, 0.93] | 0.91 [0.84, 0.96] |
| **CLIPS · SmolLM2-1.7B** | 0.67 (0.05) | 0.93 (0.05) | 0.92 (0.05) | 0.99 (0.01) | 0.99 (0.01) | 0.88 [0.80, 0.93] | 0.93 [0.87, 0.97] |
| **CLIPS · Qwen3-8B (instruct)** | 0.64 (0.06) | 0.87 (0.06) | 0.85 (0.06) | 1.02 (0.02) | 1.04 (0.04) | 0.79 [0.61, 0.92] | 0.87 [0.76, 0.95] |
| **CLIPS · Qwen2.5-7B-Instruct** | 0.67 (0.05) | 0.90 (0.06) | 0.88 (0.06) | 1.00 (0.00) | 1.00 (0.00) | 0.84 [0.71, 0.92] | 0.91 [0.84, 0.96] |
| **CLIPS · OSim-8B-Mid** | 0.66 (0.05) | 0.90 (0.06) | 0.88 (0.06) | 1.00 (0.00) | 1.00 (0.00) | 0.82 [0.67, 0.93] | 0.91 [0.84, 0.96] |

**Doors, Keys & Gems (mDKG): language-only IP and the literal listeners**

|  | P(g_true) | Assist. prec. | Assist. rec. | Rel. plan length | Rel. human cost | Goal r | Assist. r |
|---|---|---|---|---|---|---|---|
| Language-only IP, paper | 0.55 (0.05) | 0.90 (0.06) | 0.83 (0.06) | 1.26 (0.10) | 1.18 (0.07) | 0.74 [0.61, 0.85] | 0.83 [0.73, 0.93] |
| Language-only IP · Qwen3-0.6B-Base | 0.48 (0.05) | 0.90 (0.06) | 0.87 (0.06) | 1.18 (0.10) | 1.09 (0.05) | 0.64 [0.46, 0.77] | 0.84 [0.71, 0.94] |
| Language-only IP · Qwen3-1.7B-Base | 0.47 (0.05) | 0.87 (0.06) | 0.83 (0.06) | 1.20 (0.10) | 1.13 (0.06) | 0.62 [0.45, 0.76] | 0.78 [0.60, 0.92] |
| Language-only IP · Qwen3-4B-Base | 0.46 (0.05) | 0.87 (0.06) | 0.80 (0.07) | 1.29 (0.11) | 1.14 (0.06) | 0.60 [0.42, 0.75] | 0.81 [0.70, 0.90] |
| Language-only IP · Qwen3-8B-Base | 0.47 (0.05) | 0.87 (0.06) | 0.83 (0.06) | 1.24 (0.11) | 1.13 (0.06) | 0.59 [0.39, 0.76] | 0.84 [0.74, 0.93] |
| Language-only IP · SmolLM2-1.7B | 0.47 (0.05) | 0.90 (0.06) | 0.83 (0.06) | 1.26 (0.11) | 1.12 (0.06) | 0.64 [0.47, 0.77] | 0.83 [0.73, 0.92] |
| Language-only IP · Qwen3-8B (instruct) | 0.47 (0.05) | 0.87 (0.06) | 0.83 (0.06) | 1.22 (0.11) | 1.12 (0.06) | 0.59 [0.37, 0.76] | 0.84 [0.74, 0.93] |
| Language-only IP · Qwen2.5-7B-Instruct | 0.48 (0.05) | 0.87 (0.06) | 0.82 (0.07) | 1.24 (0.11) | 1.12 (0.06) | 0.61 [0.42, 0.77] | 0.83 [0.73, 0.92] |
| Language-only IP · OSim-8B-Mid | 0.46 (0.05) | 0.87 (0.06) | 0.82 (0.07) | 1.20 (0.11) | 1.10 (0.05) | 0.58 [0.37, 0.76] | 0.80 [0.66, 0.91] |
| Literal naive, paper | – | 0.58 (0.04) | 0.47 (0.02) | 1.54 (0.07) | 1.52 (0.05) | – | 0.54 [0.45, 0.61] |
| Literal naive · Qwen3-0.6B-Base | – | 0.54 (0.04) | 0.49 (0.03) | 1.66 (0.10) | 1.81 (0.09) | – | 0.46 [0.36, 0.55] |
| Literal naive · Qwen3-1.7B-Base | – | 0.56 (0.04) | 0.49 (0.03) | 1.63 (0.10) | 1.77 (0.09) | – | 0.48 [0.37, 0.57] |
| Literal naive · Qwen3-4B-Base | – | 0.59 (0.04) | 0.48 (0.03) | 1.62 (0.10) | 1.74 (0.09) | – | 0.48 [0.37, 0.58] |
| Literal naive · Qwen3-8B-Base | – | 0.56 (0.04) | 0.49 (0.03) | 1.68 (0.11) | 1.80 (0.11) | – | 0.47 [0.37, 0.57] |
| Literal naive · SmolLM2-1.7B | – | 0.53 (0.04) | 0.50 (0.03) | 1.74 (0.11) | 1.89 (0.10) | – | 0.47 [0.35, 0.56] |
| Literal naive · Qwen3-8B (instruct) | – | 0.55 (0.05) | 0.51 (0.05) | 1.61 (0.11) | 1.67 (0.09) | – | 0.43 [0.32, 0.53] |
| Literal naive · Qwen2.5-7B-Instruct | – | 0.51 (0.05) | 0.43 (0.04) | 1.54 (0.08) | 1.60 (0.08) | – | 0.44 [0.32, 0.56] |
| Literal naive · OSim-8B-Mid | – | 0.55 (0.04) | 0.47 (0.03) | 1.70 (0.11) | 1.83 (0.10) | – | 0.46 [0.34, 0.56] |
| Literal efficient, paper | – | 0.65 (0.08) | 0.54 (0.07) | 1.55 (0.11) | 1.58 (0.10) | – | 0.47 [0.24, 0.66] |
| Literal efficient · Qwen3-0.6B-Base | – | 0.48 (0.07) | 0.25 (0.05) | 1.60 (0.10) | 1.76 (0.10) | – | 0.33 [0.09, 0.53] |
| Literal efficient · Qwen3-1.7B-Base | – | 0.54 (0.08) | 0.25 (0.05) | 1.60 (0.11) | 1.76 (0.10) | – | 0.28 [0.05, 0.48] |
| Literal efficient · Qwen3-4B-Base | – | 0.44 (0.08) | 0.28 (0.06) | 1.62 (0.12) | 1.75 (0.11) | – | 0.32 [0.08, 0.52] |
| Literal efficient · Qwen3-8B-Base | – | 0.53 (0.08) | 0.29 (0.06) | 1.62 (0.12) | 1.79 (0.12) | – | 0.37 [0.16, 0.54] |
| Literal efficient · SmolLM2-1.7B | – | 0.56 (0.07) | 0.17 (0.03) | 1.61 (0.11) | 1.81 (0.11) | – | 0.31 [0.06, 0.54] |
| Literal efficient · Qwen3-8B (instruct) | – | 0.52 (0.08) | 0.42 (0.07) | 1.65 (0.14) | 1.74 (0.13) | – | 0.37 [0.15, 0.55] |
| Literal efficient · Qwen2.5-7B-Instruct | – | 0.50 (0.08) | 0.41 (0.07) | 1.53 (0.10) | 1.62 (0.10) | – | 0.37 [0.15, 0.57] |
| Literal efficient · OSim-8B-Mid | – | 0.46 (0.08) | 0.27 (0.06) | 1.59 (0.11) | 1.77 (0.11) | – | 0.37 [0.17, 0.54] |

**VirtualHome (VH): CLIPS**

|  | P(g_true) | Assist. prec. | Assist. rec. | Rel. plan length | Rel. human cost | Goal r | Assist. r |
|---|---|---|---|---|---|---|---|
| CLIPS, paper (davinci-002) | 0.63 (0.04) | 0.87 (0.04) | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) | – | – |
| Language-only IP, paper (davinci-002) | 0.37 (0.05) | 0.59 (0.05) | 0.96 (0.03) | 1.33 (0.07) | 1.35 (0.08) | – | – |
| Action-only IP, paper (davinci-002) | 0.25 (0.03) | 0.61 (0.07) | 0.84 (0.07) | 1.30 (0.07) | 1.33 (0.07) | – | – |
| **CLIPS · Qwen3-0.6B-Base** | 0.61 (0.05) | 0.85 (0.05) | 1.00 (0.00) | 1.01 (0.02) | 1.01 (0.02) | – | – |
| **CLIPS · Qwen3-1.7B-Base** | 0.62 (0.05) | 0.85 (0.05) | 1.00 (0.00) | 1.01 (0.02) | 1.01 (0.02) | – | – |
| **CLIPS · Qwen3-4B-Base** | 0.64 (0.04) | 0.86 (0.04) | 1.00 (0.00) | 1.00 (0.01) | 0.99 (0.01) | – | – |
| **CLIPS · Qwen3-8B-Base** | 0.63 (0.05) | 0.87 (0.04) | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) | – | – |
| **CLIPS · SmolLM2-1.7B** | 0.63 (0.04) | 0.88 (0.04) | 1.00 (0.00) | 0.99 (0.01) | 0.99 (0.01) | – | – |
| **CLIPS · Qwen3-8B (instruct)** | 0.63 (0.06) | 0.82 (0.05) | 1.00 (0.00) | 1.05 (0.03) | 1.05 (0.03) | – | – |
| **CLIPS · Qwen2.5-7B-Instruct** | 0.69 (0.05) | 0.86 (0.05) | 0.96 (0.04) | 1.03 (0.05) | 1.03 (0.06) | – | – |
| **CLIPS · OSim-8B-Mid** | 0.64 (0.05) | 0.87 (0.04) | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) | – | – |

**VirtualHome (VH): language-only IP and the literal listeners**

|  | P(g_true) | Assist. prec. | Assist. rec. | Rel. plan length | Rel. human cost | Goal r | Assist. r |
|---|---|---|---|---|---|---|---|
| Language-only IP, paper | 0.37 (0.05) | 0.59 (0.05) | 0.96 (0.03) | 1.33 (0.07) | 1.35 (0.08) | – | – |
| Language-only IP · Qwen3-0.6B-Base | 0.36 (0.05) | 0.57 (0.06) | 0.94 (0.03) | 1.37 (0.09) | 1.40 (0.10) | – | – |
| Language-only IP · Qwen3-1.7B-Base | 0.38 (0.05) | 0.59 (0.05) | 0.92 (0.05) | 1.26 (0.06) | 1.29 (0.08) | – | – |
| Language-only IP · Qwen3-4B-Base | 0.38 (0.05) | 0.59 (0.05) | 0.94 (0.03) | 1.27 (0.08) | 1.30 (0.09) | – | – |
| Language-only IP · Qwen3-8B-Base | 0.37 (0.05) | 0.58 (0.05) | 0.94 (0.03) | 1.30 (0.08) | 1.33 (0.09) | – | – |
| Language-only IP · SmolLM2-1.7B | 0.37 (0.05) | 0.58 (0.05) | 0.92 (0.05) | 1.26 (0.07) | 1.29 (0.08) | – | – |
| Language-only IP · Qwen3-8B (instruct) | 0.37 (0.06) | 0.55 (0.06) | 0.88 (0.06) | 1.34 (0.08) | 1.39 (0.10) | – | – |
| Language-only IP · Qwen2.5-7B-Instruct | 0.45 (0.06) | 0.62 (0.06) | 0.86 (0.06) | 1.20 (0.07) | 1.24 (0.08) | – | – |
| Language-only IP · OSim-8B-Mid | 0.38 (0.05) | 0.59 (0.05) | 0.94 (0.03) | 1.30 (0.08) | 1.34 (0.09) | – | – |
| Literal listener, paper | – | 0.64 (0.06) | 0.70 (0.07) | 1.48 (0.08) | 1.54 (0.09) | – | – |
| Literal listener · Qwen3-0.6B-Base | – | 0.66 (0.06) | 0.72 (0.06) | 1.62 (0.10) | 1.72 (0.11) | – | – |
| Literal listener · Qwen3-1.7B-Base | – | 0.68 (0.06) | 0.70 (0.07) | 1.58 (0.10) | 1.68 (0.11) | – | – |
| Literal listener · Qwen3-4B-Base | – | 0.68 (0.07) | 0.70 (0.07) | 1.62 (0.11) | 1.72 (0.12) | – | – |
| Literal listener · Qwen3-8B-Base | – | 0.67 (0.07) | 0.72 (0.07) | 1.60 (0.11) | 1.69 (0.12) | – | – |
| Literal listener · SmolLM2-1.7B | – | 0.63 (0.06) | 0.69 (0.07) | 1.64 (0.10) | 1.74 (0.11) | – | – |
| Literal listener · Qwen3-8B (instruct) | – | 0.68 (0.07) | 0.71 (0.07) | 1.60 (0.11) | 1.69 (0.12) | – | – |
| Literal listener · Qwen2.5-7B-Instruct | – | 0.72 (0.07) | 0.71 (0.07) | 1.56 (0.11) | 1.66 (0.12) | – | – |
| Literal listener · OSim-8B-Mid | – | 0.67 (0.07) | 0.70 (0.07) | 1.58 (0.10) | 1.68 (0.11) | – | – |

**Setup check: action-only IP uses no language model** (plan: stop if it misses by more than 0.02; the last column is the largest per-problem gap against the paper's own result file)

| Model | Domain | Problems | P(g_true) | Published | Largest gap | Check |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen3-0.6B-Base | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| Qwen3-1.7B-Base | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen3-1.7B-Base | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| Qwen3-4B-Base | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen3-4B-Base | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| Qwen3-8B-Base | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen3-8B-Base | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| SmolLM2-1.7B | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| SmolLM2-1.7B | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| Qwen3-8B (instruct) | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen3-8B (instruct) | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| Qwen2.5-7B-Instruct | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| Qwen2.5-7B-Instruct | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |
| OSim-8B-Mid | mdkg | 30 | 0.314 | 0.31 | 4.9e-10 | pass |
| OSim-8B-Mid | vh | 25 | 0.250 | 0.25 | 0.0e+00 | pass |

## K3: robustness to the phrasing of the instruction

Both domains pooled (65 instructions; the n row gives the valid units per kind). Each Δ is between the original and the perturbed instruction scored in the same batch. P(g_true) is CLIPS's at the final step; log BF is between the true goal and its strongest rival, from the language term alone.

**Median |Δ P(g_true)|**

| Model | para1 | para2 | typo1 | typo2 | fill1 | fill2 | lower | nopunct |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.006 | 0.005 | 0.008 | 0.018 | 0.009 | 0.006 | 0.005 | 0.003 |
| Qwen3-1.7B-Base | 0.003 | 0.003 | 0.003 | 0.007 | 0.010 | 0.006 | 0.002 | 0.002 |
| Qwen3-4B-Base | 0.003 | 0.003 | 0.003 | 0.008 | 0.009 | 0.006 | 0.002 | 0.003 |
| Qwen3-8B-Base | 0.004 | 0.004 | 0.001 | 0.007 | 0.005 | 0.004 | 0.002 | 0.002 |
| SmolLM2-1.7B | 0.004 | 0.004 | 0.003 | 0.011 | 0.016 | 0.012 | 0.003 | 0.003 |
| Qwen3-8B (instruct) | 0.002 | 0.000 | 0.000 | 0.007 | 0.001 | 0.005 | 0.001 | 0.001 |
| Qwen2.5-7B-Instruct | 0.000 | 0.000 | 0.001 | 0.003 | 0.000 | 0.003 | 0.001 | 0.001 |
| OSim-8B-Mid | 0.003 | 0.004 | 0.003 | 0.005 | 0.008 | 0.002 | 0.002 | 0.003 |
| n (units) | 61 | 63 | 48 | 26 | 65 | 65 | 65 | 65 |

**Posterior argmax flips**

| Model | para1 | para2 | typo1 | typo2 | fill1 | fill2 | lower | nopunct |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 1.6% | 1.6% | 2.1% | 7.7% | 6.2% | 6.2% | 1.5% | 1.5% |
| Qwen3-1.7B-Base | 1.6% | 3.2% | 4.2% | 11.5% | 6.2% | 4.6% | 0.0% | 0.0% |
| Qwen3-4B-Base | 3.3% | 1.6% | 0.0% | 7.7% | 1.5% | 0.0% | 1.5% | 0.0% |
| Qwen3-8B-Base | 3.3% | 1.6% | 0.0% | 7.7% | 1.5% | 0.0% | 3.1% | 0.0% |
| SmolLM2-1.7B | 3.3% | 3.2% | 2.1% | 7.7% | 1.5% | 0.0% | 0.0% | 0.0% |
| Qwen3-8B (instruct) | 8.2% | 7.9% | 8.3% | 11.5% | 4.6% | 10.8% | 1.5% | 1.5% |
| Qwen2.5-7B-Instruct | 4.9% | 6.3% | 10.4% | 15.4% | 1.5% | 10.8% | 6.2% | 3.1% |
| OSim-8B-Mid | 1.6% | 1.6% | 0.0% | 3.8% | 3.1% | 0.0% | 0.0% | 0.0% |
| n (units) | 61 | 63 | 48 | 26 | 65 | 65 | 65 | 65 |

**Median |Δ log BF|, language term (nats)**

| Model | para1 | para2 | typo1 | typo2 | fill1 | fill2 | lower | nopunct |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.02 | 0.02 | 0.02 | 0.04 | 0.06 | 0.06 | 0.02 | 0.02 |
| Qwen3-1.7B-Base | 0.03 | 0.03 | 0.01 | 0.02 | 0.11 | 0.08 | 0.02 | 0.04 |
| Qwen3-4B-Base | 0.05 | 0.04 | 0.01 | 0.06 | 0.09 | 0.07 | 0.02 | 0.01 |
| Qwen3-8B-Base | 0.03 | 0.02 | 0.01 | 0.03 | 0.04 | 0.03 | 0.01 | 0.01 |
| SmolLM2-1.7B | 0.03 | 0.04 | 0.02 | 0.11 | 0.05 | 0.07 | 0.02 | 0.02 |
| Qwen3-8B (instruct) | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.01 | 0.01 |
| Qwen2.5-7B-Instruct | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| OSim-8B-Mid | 0.03 | 0.03 | 0.02 | 0.06 | 0.06 | 0.02 | 0.02 | 0.02 |
| n (units) | 61 | 63 | 48 | 26 | 65 | 65 | 65 | 65 |

**mDKG: change in goal correlation with humans** (perturbed minus original, over the problems with a valid unit of the kind)

| Model | para1 | para2 | typo1 | typo2 | fill1 | fill2 | lower | nopunct |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | -0.006 | -0.001 | -0.014 | -0.043 | -0.004 | -0.010 | -0.002 | -0.000 |
| Qwen3-1.7B-Base | -0.005 | 0.002 | -0.007 | -0.001 | -0.035 | -0.035 | -0.007 | -0.000 |
| Qwen3-4B-Base | -0.023 | -0.002 | -0.004 | -0.014 | -0.030 | -0.028 | -0.001 | -0.001 |
| Qwen3-8B-Base | -0.001 | 0.006 | 0.004 | -0.002 | -0.012 | -0.015 | 0.012 | 0.003 |
| SmolLM2-1.7B | 0.001 | 0.002 | -0.002 | 0.003 | -0.002 | 0.000 | -0.003 | 0.002 |
| Qwen3-8B (instruct) | 0.026 | 0.027 | -0.080 | -0.034 | -0.030 | -0.162 | -0.087 | -0.093 |
| Qwen2.5-7B-Instruct | 0.017 | 0.008 | 0.003 | -0.036 | 0.013 | -0.243 | -0.021 | 0.020 |
| OSim-8B-Mid | 0.004 | 0.002 | 0.001 | -0.002 | -0.007 | -0.005 | 0.011 | 0.001 |

**Valid perturbed instructions per kind**

| Domain | para1 | para2 | typo1 | typo2 | fill1 | fill2 | lower | nopunct |
|---|---|---|---|---|---|---|---|---|
| mdkg | 28 | 29 | 29 | 15 | 30 | 30 | 30 | 30 |
| vh | 33 | 34 | 19 | 11 | 35 | 35 | 35 | 35 |

## Prediction scorecard

| ID | Prediction | Result |
|---|---|---|
| C1 | qwen3_8b: mDKG P(g_true) within 0.05 of 0.74 and goal correlation >= 0.85 | fail |
| C2 | qwen3_1p7b within 0.05 of qwen3_8b on P(g_true) in both domains; qwen3_0p6b is not | fail |
| C3 | qwen3_8b_inst below qwen3_8b on mDKG goal correlation and on P(g_true) in both domains | pass |
| C4 | CLIPS > language-only IP > action-only IP on P(g_true), every base model of 1.7B or more, both domains | pass |
| C5 | base models of 1.7B or more: median \|Δ P(g_true)\| <= 0.10 under para1 and <= 0.05 under typo1, argmax flips <= 10% (both domains pooled) | pass |
| C6 | qwen3_8b_inst flips the argmax under para1 more often than qwen3_8b (both domains pooled) | pass |

## Notes for Xuan

1. **The setup reproduces the paper wherever the language model plays no part.**
   - Action-only IP gives P(g_true) 0.314 on mDKG and 0.250 on VH for every model. The largest per-problem gap against the paper's own result files is 5e-10.
   - Table 1's definitions reproduce every cell of the paper's table from the OSF files (a test).
   - Two crashes in CLIPS's experiment code needed guarded fixes. Each is a no-op wherever the original code ran, so neither changes a result the original could produce:
     - (9) VH: taking the top 5 commands failed when fewer than 5 passed the 0.99 cumulative filter.
     - (10) mDKG: a missing best action in the assistance planner is now taken as wait. The same function already does this for the human's moves.
   - Both are listed in `loops.py` and `docs/clips_decisions.md`. `external/` is unchanged.
2. **On mDKG, CLIPS with local models matches humans, not curie. On VH it matches davinci-002.**
   - **mDKG, CLIPS:**
     - P(g_true) is 0.66–0.68 for the base models, against 0.74 for curie and 0.67 for humans.
     - Assistance precision is 0.90–0.93 and recall 0.88–0.93, against 0.97 and 0.97.
     - Goal correlation with humans is 0.83–0.88, against 0.93. With 30 problems the intervals are wide, and the base models' intervals reach 0.93.
   - **mDKG, language-only IP:** the gap here is larger. P(g_true) is 0.46–0.48 against 0.55, and goal correlation 0.58–0.64 against 0.74. So the difference from curie sits in the language term, and the planner absorbs part of it.
   - **VH:** every row matches the paper:
     - CLIPS: 0.61–0.64 against 0.63
     - language-only IP: 0.36–0.38 against 0.37
     - assistance precision: 0.85–0.88 against 0.87
   - **Literal listeners:** they recall less with local models. "Literal efficient" recall on mDKG is 0.17–0.42, against 0.54.
3. **There is no size effect from 0.6B to 8B (C2 fails).**
   - P(g_true) stays within 0.03 across the four Qwen3 base sizes in both domains. SmolLM2-1.7B matches them (0.67 and 0.63).
   - Human correlation falls slightly with size: 0.88 at 0.6B, 0.85 at 4B, 0.83 at 8B. The intervals overlap.
   - Our reading: the instructions are short and mostly name the object or colour. So scoring an utterance against candidate commands needs lexical matching more than world knowledge, and a 0.6B model already does that. The optional Phase 6 (no planner, no commands) would test this.
4. **One of the two instruct models is slightly worse (C3 passes narrowly).**
   - Qwen3-8B (instruct) against its base: mDKG P(g_true) 0.64 against 0.66, and r 0.79 against 0.83. On VH the gap is 0.6284 against 0.6290. That difference of 0.0006 is noise, so the "both domains" half of C3 holds in letter only.
   - Qwen2.5-7B-Instruct is the best model on VH: 0.69, with language-only IP 0.45 against 0.37. It is level with the base models on mDKG.
   - OSim-8B-Mid, the simulator, is within 0.01 of Qwen3-8B-Base on every headline number.
   - So the simulator study's finding that instruct models are worse does not carry over to accuracy here. It does carry over to robustness (note 6).
5. **Rewording barely moves the posterior (C5 passes with room to spare).**
   - For base models of 1.7B or more, median |Δ P(g_true)| is 0.001–0.004 under a paraphrase or one typo; the bars were 0.10 and 0.05. Top-goal flips are at most 4.2%.
   - The largest medians are 0.018 (Qwen3-0.6B-Base, two typos) and 0.016 (SmolLM2-1.7B, one filler sentence).
   - Two typos is the hardest kind: base models flip for 8–12% of units. Only 26 instructions have room for two typos, so that is 2–3 instructions.
   - The language-only Bayes factor moves very little: median |Δ log BF| is 0.01–0.05 nats for base models under a paraphrase. Our reading: a rewording shifts log p(u | c) about equally for every candidate command, and that common shift cancels in the Bayes factor between goals.
6. **Instruct models hold steady on most instructions and swing on a few (C6 passes).**
   - Qwen3-8B (instruct) changes its top goal for 5 of 61 paraphrases, against 2 for its base. Both instruct models move P(g_true) by more than 0.1 for 9–12 units per kind, against about 4 for Qwen3-8B-Base.
   - Their median |Δ log BF| is 0.00, because the language term does not move at all for 33–41% of units. But the 90th percentile is 1.6–1.7 nats, against 0.49 for the base. The medians hide this, so the flip rates are the measure to read.
   - A two-sentence filler lowers their mDKG human correlation by 0.16 (Qwen3-8B) and 0.24 (Qwen2.5-7B). Base models lose at most 0.035.
   - The counts are small: C6 rests on 5 flips against 2.
7. **Cost.**
   - **Scoring:** 7,295 (prompt, instruction) pairs per model. All 8 models took 23 minutes on one H100 NVL MIG 3g.47gb slice, and the perturbed instructions took 57 minutes more.
   - **Replays:** these run on CPU. The planner dominates: the slowest mDKG problems take about an hour each in a single thread.
8. **Not run:** Phase 6, which tests whether a small model can read the instruction without the planner and commands (prediction C7). The plan leaves it optional.
