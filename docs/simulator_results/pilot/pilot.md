# Simulator study, Phase 5: multi-turn pilot

User turns 2 and 3 of 60 clean dialogues (seeded draw from the 235), scored and sampled by the S-format chat models given the hypothesis and the real history: format S's system prompt and opener, then every earlier turn in its role. Every hypothesis of the dialogue's first-turn set, 32 samples per (dialogue, turn, hypothesis), pure sampling. Turn 1 is the Phase 2 run on the same dialogues with each hypothesis's first 32 samples, so every turn has the same sample count. Brackets are 95% item-bootstrap intervals over the dialogues. Pilot runs at commit `c14a94c`; generated 2026-11-14T06:30:29Z.

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

## Coverage by turn, all hypotheses

Median human percentile (the share of the context's own samples at least as surprising as the human turn; 0.5 is a typical sample), then the share of pairs beyond all samples.

| Model | Median pct, turn 1 | Median pct, turn 2 | Median pct, turn 3 | Beyond all, turn 1 | Beyond all, turn 2 | Beyond all, turn 3 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 99.8% [99.5%, 100.0%] | 84.9% [79.1%, 89.8%] | 85.7% [79.2%, 91.2%] |
| Qwen3-8B · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] | 99.8% [99.5%, 100.0%] | 100.0% [100.0%, 100.0%] |
| OSim-8B-Mid · S | 0.19 [0.16, 0.25] | 0.38 [0.28, 0.47] | 0.38 [0.28, 0.47] | 1.8% [0.7%, 3.1%] | 5.0% [1.7%, 9.1%] | 4.3% [1.7%, 7.5%] |
| OSim-8B · S | 0.00 [0.00, 0.03] | 0.16 [0.09, 0.25] | 0.19 [0.12, 0.28] | 52.2% [44.7%, 59.9%] | 17.1% [11.0%, 23.8%] | 15.3% [9.1%, 22.4%] |
| UserLM-8B · S | 0.31 [0.28, 0.34] | 0.38 [0.31, 0.47] | 0.31 [0.25, 0.44] | 0.0% [0.0%, 0.0%] | 1.3% [0.0%, 3.9%] | 0.7% [0.2%, 1.3%] |

## Coverage by turn, gold hypothesis

Median human percentile (the share of the context's own samples at least as surprising as the human turn; 0.5 is a typical sample), then the share of pairs beyond all samples.

| Model | Median pct, turn 1 | Median pct, turn 2 | Median pct, turn 3 | Beyond all, turn 1 | Beyond all, turn 2 | Beyond all, turn 3 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] | 86.7% [76.7%, 95.0%] | 86.7% [76.7%, 95.0%] |
| Qwen3-8B · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] |
| OSim-8B-Mid · S | 0.28 [0.19, 0.38] | 0.41 [0.22, 0.47] | 0.38 [0.25, 0.47] | 0.0% [0.0%, 0.0%] | 1.7% [0.0%, 5.0%] | 3.3% [0.0%, 8.3%] |
| OSim-8B · S | 0.03 [0.03, 0.06] | 0.12 [0.06, 0.25] | 0.16 [0.09, 0.28] | 30.0% [18.3%, 41.7%] | 20.0% [10.0%, 31.7%] | 16.7% [8.3%, 26.7%] |
| UserLM-8B · S | 0.44 [0.34, 0.50] | 0.34 [0.31, 0.56] | 0.31 [0.25, 0.47] | 0.0% [0.0%, 0.0%] | 1.7% [0.0%, 5.0%] | 1.7% [0.0%, 5.0%] |

## Ranking by turn

Ordered by median percentile, then share beyond all samples (all hypotheses), as in Phase 3.

| Turn | Order (best first) | Pairs ordered differently from turn 1 |
|---|---|---|
| 1 | UserLM-8B · S > OSim-8B-Mid · S > OSim-8B · S > Qwen2.5-7B-Inst · S > Qwen3-8B · S | – |
| 2 | UserLM-8B · S > OSim-8B-Mid · S > OSim-8B · S > Qwen2.5-7B-Inst · S > Qwen3-8B · S | none |
| 3 | OSim-8B-Mid · S > UserLM-8B · S > OSim-8B · S > Qwen2.5-7B-Inst · S > Qwen3-8B · S | UserLM-8B · S / OSim-8B-Mid · S |

## Change from turn 1

Paired item-bootstrap differences, later turn minus turn 1, all hypotheses.

| Model | Δ median pct, turn 2 | Δ median pct, turn 3 | Δ beyond all, turn 2 | Δ beyond all, turn 3 |
|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | -14.9% [-20.7%, -10.0%] | -14.1% [-20.7%, -8.7%] |
| Qwen3-8B · S | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | -0.2% [-0.5%, 0.0%] | 0.0% [0.0%, 0.0%] |
| OSim-8B-Mid · S | 0.19 [0.06, 0.28] | 0.19 [0.06, 0.28] | 3.2% [-0.5%, 7.5%] | 2.5% [-0.2%, 6.0%] |
| OSim-8B · S | 0.16 [0.06, 0.22] | 0.19 [0.12, 0.28] | -35.2% [-46.5%, -23.0%] | -37.0% [-46.2%, -27.4%] |
| UserLM-8B · S | 0.06 [-0.03, 0.16] | 0.00 [-0.06, 0.12] | 1.3% [0.0%, 3.9%] | 0.7% [0.2%, 1.3%] |

## Surprisal per word under the gold hypothesis

Median nats per word.

| Model | human, turn 1 | samples, turn 1 | human, turn 2 | samples, turn 2 | human, turn 3 | samples, turn 3 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 6.27 | 0.18 | 6.67 | 0.57 | 6.43 | 0.57 |
| Qwen3-8B · S | 6.30 | 0.09 | 6.72 | 0.25 | 7.24 | 0.24 |
| OSim-8B-Mid · S | 1.67 | 1.50 | 2.34 | 2.31 | 2.50 | 2.47 |
| OSim-8B · S | 1.92 | 0.68 | 2.51 | 1.62 | 2.56 | 1.89 |
| UserLM-8B · S | 2.13 | 2.28 | 2.66 | 2.64 | 2.99 | 2.89 |

## Sample health by turn

Cut off: no end-of-turn token within 96 new tokens. The flags are Phase 1's heuristics.

| Model | Turn | Samples | Cut off | Empty | Assistant opening | Restaurant voice | Think block | Other flag token |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Inst · S | 1 | 7424 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Qwen2.5-7B-Inst · S | 2 | 19296 | 1.3% | 0.0% | 5.9% | 0.2% | 0.0% | 0.0% |
| Qwen2.5-7B-Inst · S | 3 | 19296 | 0.7% | 0.0% | 10.6% | 0.1% | 0.0% | 0.0% |
| Qwen3-8B · S | 1 | 7424 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Qwen3-8B · S | 2 | 19296 | 0.6% | 0.0% | 0.1% | 0.9% | 0.0% | 0.0% |
| Qwen3-8B · S | 3 | 19296 | 0.4% | 0.0% | 1.3% | 0.2% | 0.0% | 0.0% |
| OSim-8B-Mid · S | 1 | 7424 | 0.1% | 0.0% | 0.3% | 0.0% | 0.0% | 0.0% |
| OSim-8B-Mid · S | 2 | 19296 | 0.0% | 0.0% | 2.9% | 0.0% | 0.0% | 0.0% |
| OSim-8B-Mid · S | 3 | 19296 | 0.0% | 0.0% | 3.1% | 0.0% | 0.0% | 0.0% |
| OSim-8B · S | 1 | 7424 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| OSim-8B · S | 2 | 19296 | 0.0% | 0.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| OSim-8B · S | 3 | 19296 | 0.0% | 0.0% | 1.1% | 0.0% | 0.0% | 0.0% |
| UserLM-8B · S | 1 | 7424 | 7.2% | 6.1% | 0.2% | 0.3% | 0.0% | 2.1% |
| UserLM-8B · S | 2 | 19296 | 2.0% | 2.3% | 1.0% | 0.1% | 0.0% | 3.0% |
| UserLM-8B · S | 3 | 19296 | 1.8% | 2.6% | 1.2% | 0.1% | 0.0% | 4.4% |

## Example: SNG01728, user turn 2

Gold hypothesis: The user wants to find a moderately priced restaurant serving turkish food in the centre of town.

- *user*: I am looking for a restaurant. The restaurant should be in the centre and should be in the moderate price range
- *system*: Hello. What type of food are you looking for? There are 21 locations that fit these criteria.
- **human user turn 2**: I am looking for one that serves Turkish food.

Qwen2.5-7B-Inst · S (human S 29.1, percentile 0.00):
- I am in the mood for Turkish food. Can you suggest a place that serves Turkish cuisine?
- I'm in the mood for Turkish food. Can you suggest a restaurant that serves Turkish cuisine and is located in the centre of town?
- I'm looking for a Turkish restaurant that's moderately priced and located in the center of town. Could you provide me with any options that match this description?

Qwen3-8B · S (human S 34.4, percentile 0.00):
- I'm looking for Turkish food.
- I'm looking for Turkish food.
- I'm looking for Turkish food.

OSim-8B-Mid · S (human S 11.8, percentile 0.25):
- Turkish.
- Yeah, it could be anything
- I am interested in Turkish food.

OSim-8B · S (human S 12.1, percentile 0.12):
- Turkish.
- Yeah, can you give me the one that serve Turkish food?
- I am interested in Turkish food

UserLM-8B · S (human S 13.7, percentile 0.66):
- I am looking for the brazilian rodízio but it can be that not in the centre of the town. Hence the requirement of moderately price restaurant
- I am looking for a restaurant. The restaurant should be in the centre and should be in the moderate price range. The food can be sentarian but preferably for africant or turkish food
- the restaurants has to serve turkish food

## Example: SNG01728, user turn 3

Gold hypothesis: The user wants to find a moderately priced restaurant serving turkish food in the centre of town.

- *user*: I am looking for a restaurant. The restaurant should be in the centre and should be in the moderate price range
- *system*: Hello. What type of food are you looking for? There are 21 locations that fit these criteria.
- *user*: I am looking for one that serves Turkish food.
- *system*: I have Anatolia and Efes Restaurant. Do you have a preference?
- **human user turn 3**: No, I just need the address of one of them.

Qwen2.5-7B-Inst · S (human S 52.6, percentile 0.00):
- I don't mind which one. Can you tell me more about these restaurants, like their ratings and whether they have a reservation service?
- Let me check the details for both Anatolia and Efes Restaurant to see which one offers Turkish cuisine. Upon checking, both Anatolia and Efes Restaurant serve Turkish food. Which one would you prefer, or would you like information on both?
- I would prefer Anatolia. Can you provide me with more information about it, such as their specialties or any special offers? Also, do you need any more details to help you book a table?

Qwen3-8B · S (human S 36.1, percentile 0.00):
- Yes, I am looking for a restaurant that serves Turkish food. I would like to book a table at Anatolia or Efes Restaurant. Do you have availability for the next few days?
- Anatolia and Efes Restaurant both serve Turkish food, and they are both located in the centre of town with a moderate price range. Would you like more details about either of them, or would you prefer to book one of them directly?
- Yes, I would like to book a table at Anatolia.

OSim-8B-Mid · S (human S 19.0, percentile 0.41):
- Yes I have a preference. I do not want Anatolia. I do not like seafood.
- Anatolia works for me
- What is the type of atmosphere in Efes Restaurant?

OSim-8B · S (human S 18.3, percentile 0.38):
- Yes, please book one for Anatolia
- Anatolia works for me
- What is the type of atmosphere like at Efes Restaurant?

UserLM-8B · S (human S 28.1, percentile 0.44):
- Looking at the price lists on the websites, I would choose Efes Restaurant for this requirement
- Please consider the ratings as well
- <|endconversation|>assistant I'm sorry, it seems I have misunderstood your request. Do you have a particular restaurant in mind for your booking?
