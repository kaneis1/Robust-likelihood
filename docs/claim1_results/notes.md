# Claims 1 and 2 on MultiWOZ: what the numbers say

Hand-written notes on `summary.md` (generated) for the C1 and C2 write-up. Runs: 2026-11-01/02.
Scorecard: R1 pass, R2 fail, R3 fail, R4 fail, R5 pass. OLMo is still pending (see the end).

## C1: token log-probs point the right way, then overshoot and wobble

- **Right direction, wrong size (R1 passes).** On named single-slot pairs Qwen prefers the
  narrower hypothesis by 10.6 nats (explicit) and 16.2 nats (implicit), against a corpus
  log-ratio of 4.2. The size principle is there, at 2.5–4 times its empirical strength.
- **No rewrite is harmless (R3 fails, in the opposite direction from the prediction).** The
  plan expected `pad` and `typo` to cancel in a Bayes factor. They do not: median
  |Δ log BF| is 6.2 nats under `pad`, 8.1 under `typo`, 7.9 under paraphrase, 9.6 under
  `verbose`, and 9.4 under translation, and the preferred hypothesis of the pair flips in
  12–19% of pairs. The added or misspelled text costs a different number of nats under
  each hypothesis. Example, SNG01165: the padding costs 59.9 nats under G and 67.8 under
  broad, so the G-vs-broad log BF moves from −60.8 to −52.9. Rescoring noise (about 1 nat at
  the 90th percentile, from bfloat16 kernels) is far smaller than these shifts.
- **Rewording the hypothesis moves every Bayes factor against it.** A paraphrase moves the
  score by a median of about 5 nats, a Chinese translation by 14.5 nats, more than the whole
  corpus Occam gap. R2 still fails on its first half: ICC(1) is 0.74, because the
  hypotheses of an item differ even more than their rewordings (SD ratio 0.54).
- **Clipping and centering do not fix it.** Centering cancels in every Bayes factor, so it is
  identical to the raw score there. Clipping at −3 nats zeroes the Bayes factor shifts only
  because most pairs sit beyond the cap before and after a rewrite; it does so by discarding
  the comparison, and its posterior argmax flips the most of the language-model scorers
  (32% of draws).
- **Posterior.** Drawing one wording per hypothesis class flips Qwen's argmax in 26% of draws;
  the SD of P(G) is 0.16.

## C2: verbalized probabilities track neither the string nor the meaning

- **Not the string.** For the human utterances, verbalized answers sit a median 40–47 orders
  of magnitude above Qwen's exp(−S) (Claude 40.9, GPT 39.4, Qwen 46.8), with rank correlation
  0.04–0.40.
- **Not the meaning either (R5 passes).** Against P_sem, the share of Qwen's samples with the
  utterance's slots, Spearman is 0.12 for Qwen's meaning framing and −0.03 to 0.24 elsewhere.
- **Qwen's answers are mostly a constant.** "0.05" is its modal answer, and 21% (exact) and 34%
  (meaning) of answers are 0. Rewording barely moves them (median shift 0, flip rate about 0),
  which is robustness by being uninformative, not by tracking meaning. Claude's and GPT's
  answers do vary: a paraphrase moves their log Bayes factors by a median 2.3 and 1.7 nats
  and flips the preferred hypothesis in 27% and 10% of pairs.
- **R4 fails only on normalization, and that test turned out weaker than designed.** Its
  premise was that 20 samples' exact probabilities sum to far less than 1. Qwen's sampling
  under G is so concentrated that the same 20 distinct samples hold a median 56% of its
  probability (80% at most). Qwen's 20 verbal answers sum to exactly 1.00 (twenty times 0.05),
  Claude's to 0.35, GPT's to about 0. On Qwen's own samples a verbal 0.05 is close to the
  exact value (about 0.03 each); on human utterances it is 47 orders too high. The C2 gap
  therefore measures how far human text sits from the model's distribution as much as it
  measures verbalization error, which the write-up should say.

## Jev

- By far the most stable: ICC(1) 0.95, SD ratio 0.19, Bayes factor shifts of 0.15 logits under
  paraphrase and 0.22–0.25 under translation (likely and typical wordings), flips in 1–4% of
  pairs. Its posterior over hypotheses
  barely moves (SD of P(G) under 0.01), consistent with the earlier MultiWOZ finding that its
  breadth penalty is a fraction of the corpus ratio: stable, and compressed.

## Open

- **OLMo.** Job A scored OLMo on the rewrites only (plan part 6), so it has no original pairs
  to compare with. `claim1-lm --model olmo --parts original` is a run of about two GPU-minutes;
  then rerun `claim1` with that folder added.
- **Coverage.** The API verbalizers saw `para1` and `hpara1` only, and Jev `para1`, `zh`,
  `hpara1`, and `hzh`, as the plan specifies; their other cells are "not run", not zero.
