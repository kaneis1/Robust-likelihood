## What the pilot shows

**The coverage ranking holds once the history is in the prompt.**
- Turn 2 orders the five models exactly as turn 1 does.
- At turn 3 only the top two swap: OSim-8B-Mid has a median percentile of 0.38 and UserLM-8B 0.31, and
  their intervals overlap.
- The gap between the simulators and the instruct models stays wide. Qwen3-8B is beyond all 32
  samples for 100% of pairs at every turn. Qwen2.5-7B-Instruct falls from 99.8% to 85–86%, and its
  median percentile stays at 0.

**OSim-8B-Mid covers later turns better than first turns.**
- Its median percentile rises from 0.19 to 0.38 (paired difference +0.19 [0.06, 0.28] at both turns).
- At most 5% of pairs lie beyond all samples.
- Human and sample surprisal per word both rise, to about 2.3–2.5 nats, and stay level with each
  other. Later turns are harder to predict for the model and for the human alike.

**UserLM-8B is steady and its samples are cleaner.**
- Its median percentile is 0.31–0.38 at every turn, and at most 1.3% of pairs lie beyond all samples.
- With a history, fewer samples run past 96 tokens (7.2% at turn 1, 1.8–2.0% later) and fewer are
  empty (6.1% at turn 1, 2.3–2.6% later).
- More samples end the conversation with `<|endconversation|>`: 3.0–4.4%, against 2.1% at turn 1.

**The RL model's narrowness is mostly a first-turn effect.**
- OSim-8B's share of pairs beyond all samples falls from 52% at turn 1 to 17% and 15%.
- Its median percentile rises from 0.00 to 0.16–0.19.
- Its samples spread out: their median surprisal per word goes from 0.68 to 1.6–1.9 nats.
- It is still below OSim-8B-Mid at every turn. So the Phase 3 finding that RL narrows the
  distribution (S2) holds, but it is weaker once the user has a conversation to answer.

**Instruct models drift into the assistant's voice.**
- With a conversation to continue, Qwen2.5-7B-Instruct opens like an assistant in 5.9% (turn 2) and
  10.6% (turn 3) of its samples, for example "Let me check the details for both…".
- Those samples do not explain its lower share beyond all samples: without them the share at turns
  2–3 is 86.4% instead of 85.3%. Rather, its samples spread a little: their surprisal per word goes
  from 0.18 to 0.57 nats.

**For pipeline v0** this supports the Phase 4 recommendation. OSim-8B-Mid · S is the zero-shot
reference with the best coverage at later turns, and UserLM-8B stays a cross-check.

**Caveats.**
- 60 dialogues, 32 samples, one seed, and turns 2–3 only.
- The hypothesis still describes only the search constraints. Booking details in a human turn
  (people, day, time) are the user's own, for every model alike.
- Qwen3-8B-Base · F was not run: its few-shot examples are first turns.
- A model's samples and its human turns share the same history, so coverage here says nothing about
  how a simulator would steer a dialogue on its own.
