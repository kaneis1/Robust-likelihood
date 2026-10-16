# Gaussian typicality test

Run `results/gaussian/<run-folder>`. Preset `full`, base seed 20261019, commit `<commit>`. Copy the numbers below from that run's `summary.md` and `figures/`; do not recompute them by hand.

These are exact Gaussian computations on simulated observations. They are not language-model scores.

## 1. Question

1. Does posterior inference on typicality, or on binned typicality, still converge to the true hypothesis as observations accumulate?
2. How much information is lost relative to conditioning on x, from a single observation and over many?
3. How should bin boundaries be chosen?
4. Is binned typicality more robust than the exact likelihood, and against which kinds of misspecification?

## 2. Setup

Every hypothesis is x | z ~ N(mu_z, sigma2_z I_d). The standardized typicality is T = (q - d) / sqrt(2d), with q = ||x - mu_z||^2 / sigma2_z; positive T means less probable than a typical draw. Under the hypothesis being scored, q is chi-square(d), so T and every bin mass are the same for all hypotheses of one dimension.

Families: mean-shift (20 hypotheses, sigma2 = 1, delta = r (2d)^(1/4)), scale (common mean, sigma2 in 0.25, 0.5, 1, 2, 4), and localization (441 grid points in coordinates 1-2 of R^d).

Scorers: `exact`, `tempered_0.5`, `tempered_0.1`, `clipped` (cap matched to `bins_abs3`), `typ_normal` (-T^2/2), `typ_density` (chi-square density of q), and binned typicality: `bins_abs2` (|T| 1.96), `bins_abs3` (|T| 1, 2), `bins_signed5` (T -2, -1, 1, 2), the equal-mass control, and the E5 edges. Bin masses are exact chi-square masses.

State the preset, the trials per cell, the wall-clock time, and whether the working tree was clean for this run.

## 3. Convergence and information loss

Figures `e1_convergence.png`, `e1_collapse.png`, `e1_single_observation_bits.png`.

Give the n90 table and the single-observation bits(1) table from `summary.md`. Say where the typicality scorers need more observations than `exact`, whether n90 collapses across d when plotted against r, and how much of the bits(1) gap to `exact` comes from reducing x to T rather than from binning.

## 4. Sharp versus broad hypotheses

Figure `e2_consistency.png`.

Give the predicted-versus-observed limit table for `typ_normal`, `typ_density`, and the binned scorers. Say at which d the posterior concentrates on a hypothesis broader than the truth, and by how much. Note any cell where binned scorers tie the broad hypotheses instead of choosing one.

## 5. Nuisance dimensions

Figures `e3_heatmap_d2.png`, `e3_heatmap_d100.png`, `e3_heatmap_d1000.png`, `e3_nuisance_dimensions.png`.

Give p_true(20) and loc_error(20) at each d for `exact`, `typ_normal`, `bins_abs3`, and the two content oracles. State the identity checks: the exact posterior across d, and `typ_content_oracle` against `typ_normal` at d = 2.

## 6. Robustness

Figure `e4_robustness.png` and the C1, C2, C5 table in `summary.md`.

Report `clipped` next to every bin scheme. Say which contaminations move the exact posterior to a wrong hypothesis, which ones bins absorb, and where bins only stall at the prior. Note that tempering leaves the argmax, and so the accuracy, of the exact likelihood unchanged.

## 7. Bin choice

Figure `e5_tradeoff.png`.

Give the chosen edges for K = 2, 3, 4 with their calibration bits(10) and acc(40), and the held-out comparison with `bins_abs2`, `bins_abs3`, and the equal-mass schemes. Say whether the optimized edges beat the named ones off the calibration cell.

## 8. Predictions

Copy the checklist from `summary.md`: P1 to P7 with PASS, FAIL, or INCONCLUSIVE and the numbers. For every FAIL, give the cells that failed and say whether the premise of the prediction held in those cells. A failed prediction is a result. Do not change the method, bins, or grids after seeing it.

## 9. Questions for Xuan

- Hypothesis-dependent bins are a different event for each hypothesis, so P(z | bin) is not a likelihood of one shared observation. E2 shows the practical consequence, inconsistency at small d. Which justification is intended: generalized Bayes, coarsening, or something else?
- If `clipped` matches bins on robustness (P6b), the case for typicality must rest on something other than robustness, for example normalization across hypotheses and learnability. Does she agree?
- E3 shows typicality cannot tell content dimensions from nuisance dimensions. For language, that is the form-versus-content problem from her notes. Should the regressor's input be a content representation?
