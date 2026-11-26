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
