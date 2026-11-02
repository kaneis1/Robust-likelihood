# Typicality v0: which variant to freeze

Decision memo for step 3 of `docs/claims_plan.md`, 2026-11-02. Numbers are from
`docs/typicality_v0_results/` (generated tables, intervals, and every sensitivity run); all are
computed from step 2's Qwen2.5-7B samples and scores, with no new model runs.

## The question

Typicality v0 replaces a likelihood with a binned typicality score. Six frozen variants differ in
three choices: the statistic (raw surprisal S, centered S − S(·|broad), or semantic
−log P_sem), the normalization (per hypothesis, or global over the pool of hypotheses), and the
score of a bin (A: the share of the hypothesis's own samples in the bin; B: a fixed value,
−(H + σ·c_b), from the pool's mean and spread). Steps 4c onward use the choice made here.

## What decides it: human utterances are outside the model's typical set

Qwen's own samples under a hypothesis have a surprisal of 2–11 nats (pool mean 4.5, SD 3.2).
The human utterances have a median of 101 nats under the same hypotheses, 38 standard
deviations out. So every variant built on raw S puts **100% of the 2,355 human (utterance,
hypothesis) pairs in its top bin**. Those variants are perfectly stable under every rewrite
because nothing can move, and they carry no information about the utterance at all.

| Variant | Share of human pairs in the top bin | Size principle, L_J/L_E (expl, impl) | Contradiction log BF (nats) | Contradiction AUROC | Rewording ICC (60 items) | Bin agreement under rewrites | Pool sensitivity (bin flips) |
|---|---|---|---|---|---|---|---|
| `per2A` (current formula) | 100% | −0.02, −0.01 | 0.04 | 0.52 | 0.18 | 1.00 | – |
| `glob2A` | 100% | −0.17, 0.03 | −0.06 | 0.47 | 0.60 | 1.00 | 0.00 |
| `glob1A` | 100% | −0.12, 0.10 | −0.06 | 0.48 | 0.60 | 1.00 | 0.00 |
| `glob1B` | 100% | 0.00, 0.01 | 0.03 | 0.50 | – | 1.00 | 0.00 |
| `cent1B` | 97.5% | 0.35, 0.40 | 1.50 | 0.54 | 0.48 | 0.95–0.98 | 0.003 |
| `sem1B` | 81.5% | 0.11, 0.14 | 0.53 | 0.72 | 0.65 | 1.00* | 0.002 |
| **`cent1B`, 6 bins** | **85.4%** | **1.09, 1.62** | **6.24** | **0.63** | **0.63** | **0.81–0.88** | **0.010** |
| `cent1B`, prior weighting | 89.3% | 1.00, 1.54 | 6.77 | 0.62 | 0.59 | 0.86–0.92 | 0.079 |
| Raw token log-prob (baseline) | – | 2.52, 3.87 | 14.58 | 0.62 | 0.73 | – | – |
| Jev, likely (baseline) | – | 0.41, 0.09 | 2.85 | 0.97 | 0.95 | – | – |

L_J/L_E is the share of the corpus size-principle ratio a scorer reproduces on named
single-slot pairs (1 = matches real users). Bin agreement is the share of (rewritten utterance,
hypothesis) pairs that keep their bin, over pad, typo, paraphrase, verbose, and Chinese
utterance rewrites (the range across the five kinds). \*`sem1B` keeps the utterance's annotated slots by construction, so its agreement is 1
without being measured. The contradiction AUROC is the frozen unpaired definition, which favours
scorers on a fixed scale (Jev); the paired log Bayes factor is the fairer column for the others.

Under the preferred hypothesis of each size-principle and contradiction pair, `cent1B` with 6
bins flips in 0.2–0.3% of rewrites; raw token log-prob flips in 12–19% (claim 1).

## The dice check shows why `per2A` cannot work

On step 1's dice (exact entropies and spreads), every option A variant gives `fair` and `even`
exactly the same score on every all-even sequence, where the truth favours `even` by k ln 2: a
die whose faces all have the same surprisal makes every possible sequence perfectly typical.
That is the number-game argument, shown with exact numbers. `glob1B` tracks the truth (log BF
0.87, 4.10, 9.59 at k = 2, 6, 10 against 1.39, 4.16, 6.93) but never rules out an impossible
sequence, because option B caps the extreme bins at a finite score.

## Recommendation

Freeze **`cent1B` with 6 bins: centered statistic, global normalization over the pool, edges
(−2, −1, 0, 1, 2), score B with c = (−2.5, −1.5, −0.5, 0.5, 1.5, 2.5), uniform weighting.**

- It is the only configuration that keeps the size principle at the corpus's strength (Occam
  gap +0.38 nats explicit, interval −0.98 to 1.57; +2.58 implicit), keeps the contradiction
  signal (6.2 nats), and is stable (bin agreement 0.81–0.88 under utterance rewrites and
  0.87–0.91 under hypothesis rewrites, preferred-hypothesis flips under 0.3%, rewording ICC 0.63
  against 0.73 for raw log-prob on the same items).
- Uniform rather than prior weighting: at the frozen 4 bins, prior weighting reaches a similar
  size principle (1.00, 1.54), but halving the pool flips 7.9% of its bins against 0.3% under
  uniform weighting; with 6 bins and uniform weighting the rate is 1.0%.
- Centering is what makes the statistic usable: subtracting the broad hypothesis's surprisal
  removes the part of S that depends on the utterance alone (length, typos, greetings), which
  is also why it is stable under pad and typo.

For step 4 the statistic is the exact semantic surprisal −log p(s|z), so the rule carries over as
the centered semantic surprisal −log p(s|z) + log p(s|broad group), normalized and binned as
above. The K = 64 floor that weakens `sem1B` here (79% of pairs have no matching sample) does
not arise with exact probabilities.

## Caveats

- One sampler (Qwen, temperature 1, K = 64). A sampler closer to human text would put fewer
  pairs in the top bin and could change the ranking.
- Even the recommended variant puts 85% of human pairs in its top bin; its information comes
  from the 15% below it.
- Pre-registered predictions: T1 passes (`per2A` erases the size principle); T2 fails (at the
  frozen 4-bin edges, no option B variant keeps L_J ≥ 0.5·L_E); T3 fails (`sem1B` rewording ICC
  0.65, below 0.9; its premise, raw ICC < 0.5, also does not hold at 0.73); T4 fails (`cent1B`
  agreement 0.97–0.98 is high but `glob1B`'s is 1.00, only because nothing in `glob1B` moves).

## Questions for Xuan

1. Freeze `cent1B` with the 6-bin edges and uniform weighting, as recommended?
2. If a meaning-level statistic matters more than the size principle, `sem1B` is the
   alternative: the best contradiction AUROC of the typicality variants (0.72) and stable by
   construction, but it keeps only 11–14% of the size principle here.
3. For step 4: is the centered semantic surprisal the intended teacher statistic, with "broad"
   defined as the clinical group that contains the pathology?

The choice is recorded in `configs/typicality_v0.json` (`"chosen"`) as **[Frozen]** once agreed.
