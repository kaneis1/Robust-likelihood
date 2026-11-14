## Recommendation

**For pipeline v0, use Qwen3-8B-Base with the few-shot format F as the reference model, and
OSim-8B-Mid with format S as the zero-shot second reference.** Do not use an instruct model, under
any prompt, or the RL-trained OSim-8B as a likelihood reference.

Why Qwen3-8B-Base · F:
- **Coverage is the best of all models.** The median human percentile is 0.45, so a human turn is about
  as surprising as the model's typical sample. 0.2% of pairs lie beyond all 64 samples.
- **Corpus fidelity is the closest.** It names constrained slots at 0.69 (corpus 0.74). Its raw Occam
  gap is 5.2 vs 4.2 nats for a named slot and −1.7 vs −1.3 for an omitted one; Phase 8's was 16.2 and −23.7.
- **It is the most robust and its samples are clean.** Median |Δ log BF| is 1.0 nats under
  paraphrase. 0.4% of samples are empty and none are cut off.
- **Typicality works again with it as the reference.** glob1A puts 3.4% of human pairs in its extreme bin
  (Qwen: 100%) and keeps the size principle (L_J ÷ L_E 1.10).

Its weaknesses:
- **It over-names left-out slots.** It names a slot that the hypothesis leaves out in 6.3% of samples
  (corpus 1.2%).
- **It sees eight real MultiWOZ first turns,** which the chat models do not. For a MultiWOZ pipeline
  that is a legitimate design choice (the examples come from non-probe training dialogues). It is still an
  advantage in this comparison, and another domain needs its own examples.

Why OSim-8B-Mid · S as the second reference:
- **It gives human first turns the most probability of any model.** That is 1.70 nats per word, against
  1.80 for F and 6.5–7.2 for the instruct models.
- **It covers human turns without in-context examples.** The median percentile is 0.22, and 0.7% of
  pairs lie beyond all samples.
- **Its b is at the corpus rate** (0.007 vs 0.012), and its omitted-slot Occam term matches (−1.1 vs −1.3).

Its weakness is that it under-names constrained slots (a = 0.33 vs 0.74). Its samples are often
vaguer than the goal ("I'm looking for a restaurant to eat at.").

UserLM-8B · S has the best zero-shot coverage numbers: no pair lies beyond all samples. Part of that
comes from a very broad distribution:
- 7.1% of its samples run past 96 tokens into unrelated text.
- 6.5% are empty turns.
- 37% contain a word outside the MultiWOZ vocabulary.
- Its raw Occam gap is 2.2 times the corpus value.

It is a reasonable third model to check results against, not a reference.

## Notes for Xuan

1. **The direction holds.** Base and simulator models give human first turns reasonable probability:
   1.7–2.1 nats per word, against 6.5–7.2 for Qwen2.5-7B-Instruct and Qwen3-8B. Human turns fall inside
   the samples of the base model, OSim-8B-Mid and UserLM (0–0.7% beyond all 64), against 96–100% for the
   instruct models. OSim-8B, the RL model, sits between them at 38%.
2. **Prompting does not fix an instruct model.** The broadened prompt B ("real people write in many
   different ways…") moves Qwen2.5's share beyond all samples only from 99.8% to 96.1%.
3. **RL narrows the distribution again (S2 holds).** OSim-8B's samples are half as surprising as
   OSim-8B-Mid's (10.7 vs 21.0 nats), and 38% of human pairs fall beyond all its samples (Mid: 0.7%).
   The final model starts from Qwen3-8B, the instruct model, while Mid starts from the base model, so this
   mixes RL with the starting point. Measured from its own start, OSim training widens Qwen3-8B a great
   deal: the share beyond all samples falls from 100% to 38%.
4. **No model matches the corpus on both slot-naming rates.** The instruct models name almost every
   constrained slot (a = 0.92–0.99), which is the Occam overshoot found earlier. The base model is
   closest on a, and OSim-8B-Mid and UserLM are closest on b. The corpus rates are a = 0.74 and b = 0.012.
5. **Claim 1 is much weaker with human-like models (S4 fails).** The median |Δ log BF| under
   paraphrase is 1–2 nats for the base and simulator models, up to 3 under translation into Chinese.
   For the instruct models it is 6–8 nats.
   - The large shifts in claim 1 are a property of the instruct models' narrow distributions.
   - **The 1–2 nats are real, not rescoring noise** ([noise_floor.md](noise_floor.md)). The check
     rescored every rewrite pair with the original utterance put back. The resulting noise in
     |Δ log BF| has a median of 0.15–0.21 nats for the base and simulator models (0.31–0.47 for the
     instruct models). That counts only pairs whose batch layout changed; the others reproduce their
     scores bit for bit.
   - For the base and simulator models, 62–85% of real paraphrase shifts lie above the noise's 95th
     percentile.
   - Noise only adds to |Δ|, so S4's failure is not a noise artifact. The noise-free shifts are, if
     anything, a little smaller.
   - Rescoring in the same layout reproduces Phase 2 exactly, so the runs are deterministic. The
     noise comes only from batch layout (padding width and batch size in bfloat16).
   - The claims draft should state claim 1 for the likelihoods it was measured with, or restate it with
     these results: about 1–2 nats under paraphrase with human-like models, and 6–8 with instruct models.
6. **Typicality regains signal with a human-like reference (S5 holds).** With the base model, OSim-8B-Mid
   or UserLM as the reference, glob1A puts 0.3–21% of human pairs in its extreme bin, against 100% with
   Qwen. The size principle survives (L_J ÷ L_E 1.1–2.5, against 0.0). The semantic variant sem1B stays at
   62–87% for every model: its limit is the regex slot tuple, not the model.
7. **What failed in the scorecard.**
   - S1 and S6: OSim-8B-Mid is third on coverage, behind the few-shot base model and UserLM.
   - S3: OSim-8B-Mid's a is far below the corpus rate.
   - S4: as in note 5.
   - The other half of S6 holds: the broadened prompt helps Qwen2.5 a little and stays far below
     OSim-8B-Mid.
8. **Caveats.**
   - One seed; first turns only; the restaurant domain; 64 samples per hypothesis; bfloat16 on H100 MIG
     slices.
   - Format F's eight examples.
   - The N sensitivity runs agree with S for OSim-8B-Mid (median percentile 0.27 vs 0.22). Under N,
     OSim-8B often answers as the assistant (36% of samples), and UserLM drifts off topic (37% cut off).
     So S was the right primary format.
9. **The ranking holds at later turns (Phase 5 pilot, [pilot/pilot.md](pilot/pilot.md)).** User turns 2
   and 3 of 60 dialogues were given the real history in the prompt.
   - The coverage order is the same as at turn 1, except that OSim-8B-Mid and UserLM swap at turn 3
     within their intervals.
   - The instruct models stay at a median percentile of 0.
   - OSim-8B-Mid's median percentile rises from 0.19 to 0.38.
   - OSim-8B's share beyond all samples falls from 52% to 15–17%, so RL's narrowing is mostly a
     first-turn effect.
10. **Next.** Nothing from the plan is pending. A multi-turn version of the few-shot format F
    would be needed to test Qwen3-8B-Base on later turns.
