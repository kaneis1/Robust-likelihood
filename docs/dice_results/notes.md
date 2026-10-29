# Dice sweep: what the numbers say

Hand-written notes on `summary.md` (generated) for the C2 write-up. All runs: 2026-10-29.

## The API models compute; the local model does not

- `claude-fable-5-1` and `gpt-6-astra` return the exact sequence probability under **both**
  framings. Their slopes in k equal the truth (−0.778 fair, −0.477 even), k = 1 error is 0,
  every normalization sum is 1, and every impossible pair (an odd face under the even die)
  is answered 0. The judgment framing (S2) is not "how likely LLM-as-likelihood pipelines
  are"; for these models it is a probability question they solve. Claude writes the
  arithmetic before the number on most answers.
- So D2 and D4 fail for the API models, in the direction that verbalization works. Their
  S3 also tracks the truth, which is D5's case: the dice domain hands the model the
  generator and a calculator-sized computation. This is the condition the plan says never
  holds for language, and the MultiWOZ step (2d) is where the claim has to be shown.
- Qwen2.5-7B-Instruct verbalizes like the plan expected, but not past the thresholds:
  S2 slope −0.22 per roll (28% of the truth; D2 needs under 0.15), k = 2 sum 1.46 (D4 needs
  over 2), k = 1 mean error 0.024 (D3 passes). On all-even sequences its S2 even-vs-fair
  log Bayes factor has slope −0.08 against the true k ln 2: it does not prefer the even die
  at all, and 39% of impossible pairs get a nonzero probability.

## Token log-probs: a format penalty, not a likelihood

- Qwen's S1 slope is 3.5× the truth and R² is 0.72, so D1 fails. A diagnostic on the GPU
  (greedy and sampled replies, and per-token log-probs) shows why: Qwen writes lists as
  `3,4` with no space. The frozen observation text `3, 5, 2` puts a space after each comma;
  the first space costs Qwen 13 to 16 nats, after which further spaces cost nothing. That
  one-time penalty at every k ≥ 2 dominates the slope and makes the k = 2 sum 9e-7.
- Qwen is also a poor fair die at k = 1: P(3) = 0.54, P(4) = 0.33, P(2) = 0.12, and almost
  nothing on 1, 5, 6 (mean |error| 0.18).
- OLMo-2 uses the spaced format, so its slope is 1.6× the truth (within D1's factor of 2)
  but R² is 0.82, under 0.9: D1 fails on fit alone.
- Both points are C1 material: the "correct in a narrow sense" token likelihood is
  correct only for the surface form the model would write, and a meaning-preserving change
  of separator moves it by more than the whole likelihood of a 10-roll sequence.

## Open choice

A sensitivity run that scores S1 with the no-space format (`3,5,2`) would separate the
format penalty from the face distribution. It is one GPU minute and changes no frozen
item; it has not been run.
