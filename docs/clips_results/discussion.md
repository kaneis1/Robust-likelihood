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
   - Our reading: the instructions are short and mostly name the object or colour. So scoring an utterance against candidate commands needs lexical matching more than world knowledge, and a 0.6B model already does that. The planner does the reasoning about which key opens the way to which gem. Phase 6 (note 8) agrees with this: without the planner's commands, no model can read the instruction at all.
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
8. **Without the planner, every model is at chance (Phase 6; C7 passes).**
   - **Setup.** The model scores the instruction given an ASCII map of the state when it was spoken, plus a legend, what each agent holds and "Goal: the human wants gem k." There are no commands and no planner. P(g | u, map) is the softmax of the four goals' scores.
   - **P(g_true):** 0.23–0.25 for all eight models, with chance at 0.25. The intervals are tight, for example [0.23, 0.27] for Qwen3-0.6B-Base.
   - **Top-1 accuracy:** 0.10–0.33.
   - **Goal correlation with humans:** between −0.11 and 0.09, and every interval includes 0.
   - **Against language-only IP:** every model is 0.22–0.23 below it, and every paired interval excludes 0.
   - **The scores do depend on the goal.** The four goals' scores span a median 0.4–1.1 nats. But the models favour particular gem numbers whatever the instruction says (mostly gems 1 and 2; gems 2 and 3 for Qwen3-8B instruct), not the gem the instruction points to.
   - So with these models, reading "can you pass me the red key?" as "the human wants gem 2" needs CLIPS's planner and commands. The language model contributes the match between an instruction and a command, not the spatial inference. This is why size barely matters (note 3).
   - **Caveats.**
     - The map prompt is zero-shot, as the plan specifies. A few-shot version, or a chat model asked directly, might do better, but would no longer be the plan's P(u | goal, map).
     - The four scores of one instruction were scored as one batch. Scored in another layout, a single score moves by up to 0.68 nats (median 0.08–0.12).
