# Alarm families and toy likelihoods

The 13-item pilot stays in `templates/synthetic_alarm_v2_report.md`. Prompt template `v2` and hypothesis definitions `synthetic_v2` were frozen before the held-out families. Development used eight families, two in each category. One development sentence repeated a pilot utterance, so the cache stored it under the old item id. That sentence was reworded, the development split was rerun, and the other 32 families were then sent unchanged.

Held-out run `families-holdout-20261002T073109Z`: 2,088 pairwise replies, all parsed. Toy run `toy-20261002T093933Z`: 81 replies. Models `jev-1.13.0`, `gpt-6-astra`, and `claude-fable-5-1`.

These model scores are elicited judgments. The toy targets are defined rules of a made-up generator, not frequencies from real users.

A family passes when every scored wording passes. A wording passes when the supported hypothesis is above 0.5 and strictly above every rejected hypothesis. Ambiguous wordings have no gold winner. On ordered sentences, the complete plan is the score that counts.

## Held-out families

Eight families in each category. Each family has a canonical sentence, a paraphrase, and a minor typo. Ambiguity families also have a clarified trio, and only that trio is scored.

| Category | Jev | GPT Astra | Claude Fable |
| --- | --- | --- | --- |
| Mentioned action stays unsupported | 3/8 | 8/8 | 7/8 |
| Negation and correction | 6/8 | 8/8 | 8/8 |
| Clarified request | 4/8 | 8/8 | 7/8 |
| Complete plan, with partials rejected | 2/8 | 3/8 | 2/8 |

The mean change in the supported score across the three wordings of a family is at most 0.07. Paraphrase and typo rarely move the winning score.

The plan misses are mostly partial plans that remain at or above 0.5. The uttered order still scores above the reversed order. Mean margin, correct plan minus reversed plan: Jev 0.49, GPT 0.96, Claude 0.84. Mean highest rejected-plan score: Jev 0.55, GPT 0.45, Claude 0.44.

Clarification raises the score of the operation named afterward. Mean rise from the ambiguous wording to the clarified wording: Jev 0.48, GPT 0.61, Claude 0.69.

Jev is the model that still assigns high support to a rejected alternative: mean highest rejected score 0.47 on mentions, 0.25 on negation, and 0.41 on clarified requests. GPT and Claude keep those means below 0.13.

## Toy likelihoods

The rules were given in the prompt. Three equivalent wordings were used. For "Please change my alarm" the targets are 0.60, 0.30, and 0.10 under Create, Disable, and Query. "Please stop my alarm" is 0.10, 0.60, and 0.30. "Please check my alarm" is 0.30, 0.10, and 0.60.

| Model | Mean absolute error | Maximum absolute error | Ranking matches the rules |
| --- | --- | --- | --- |
| Claude Fable | 0 | 0 | 9/9 |
| GPT Astra | 0 on the 8 complete cells | 0 | 8/8 complete cells |
| Jev | 0.047 | 0.11 | 9/9 |

Claude reproduces every target under all three rule wordings. GPT does the same on every cell that parsed. One GPT reply, Create given "Please stop my alarm" under the reversed rule wording, is missing from the saved responses, so that cell is incomplete. Jev keeps the ranking and stays within about 0.11 of each target. Its three scores for one sentence sum to about 1.14 rather than 1.

## Reading

On the held-out families, the top plan is usually the uttered order, and meaning-preserving edits barely move the supported score. The pass rule still fails when a mentioned action, a negated action, or a partial plan remains above 0.5. That is common for Jev, and it is the usual reason an ordered family fails for all three models.

When the emission probabilities are stated as rules, Claude and GPT return those numbers. Jev returns the right order with a small upward bias. Exact answers on a stated table do not by themselves show that the alarm scores are calibrated likelihoods.
