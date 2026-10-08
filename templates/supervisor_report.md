# Supervisor report

Run `20260927T151455Z`. Models `jev-1.13.0`, `gpt-4.1-2025-04-14`, and `claude-sonnet-4-5-20250929`. Fifteen MASSIVE training originals, each kept as the original wording, a paraphrase, a minor typo, and a repetition: 60 rows and 720 requests. Of those, 476 Jev and GPT successes were copied from the earlier cache. The 244 new HTTP calls are the 240 Claude requests plus the 4 GPT classification outputs that were invalid before and came back invalid again. Meaning-change controls stay for the next run.

These outputs are elicited plausibility judgments, not verified P(e|h).

## Comparison table

The 60 rows are four wordings of 15 originals, so these point metrics are clustered. Figures are from `metrics.json` for this run. A tie counts as incorrect. A classification distribution that misses the sum check is excluded and left as returned. Rank, margin, and a pairwise winner require all three hypothesis scores.

| | Jev | GPT | Claude |
| --- | --- | --- | --- |
| Classification accuracy | 1.00 (60/60) | 0.964 (54/56) | 0.933 (56/60) |
| Classification NLL | 0.0115 | 0.0720 | 0.230 |
| Classification Brier | 0.00272 | 0.0292 | 0.120 |
| Classification ties | 0 | 1 | 0 |
| Invalid classification outputs | 0/60 | 4/60 | 0/60 |
| Pairwise replies parsed | 180/180 | 180/180 | 26/180 |
| Complete pairwise groups | 60/60 | 60/60 | 0/60 |
| Pairwise winner accuracy | 1.00 (60/60) | 0.967 (58/60) | unavailable |
| Mean correct-intent rank | 1.00 | 1.017 | unavailable |
| Mean pairwise margin | 0.642 | 0.766 | unavailable |
| Ranking flips versus the original | 0/45 | 2/45 | 0 comparable groups |
| Yes/no flips at 0.5 | 5/135 | 3/135 | 0/17 parsed pairs |

On the 56 classification rows both Jev and GPT scored, Jev is correct on all 56 and GPT on 54. Claude is outside that paired block. The disagreement list remains Jev versus GPT, so its count is still 4. There were no transport failures, and every planned request was attempted.

Jev's unique winner is unchanged on all 45 variant-versus-original comparisons. GPT's unique winner changes on 2 of 45, both on example 650: one unique-to-tie flip and one change from query to set. Jev crosses the 0.5 yes/no line 5 times. Three of those start from an original `alarm_set` score of exactly 0.5 on example 4929 (0.50 to 0.53, 0.34, and 0.44). The other two are example 650 repetition (`alarm_set` 0.26 to 0.68, winner stays query) and example 5120 repetition (`alarm_remove` 0.60 to 0.44). GPT's three crossings are all the `alarm_set` score on example 650 (0.40 to 0.60, 0.70, and 0.85). On the paraphrase, query stays ahead of set (0.70 versus 0.60), so that crossing leaves the unique winner in place.

Claude's 26 parsed pairwise replies are each a single hypothesis. Every example is missing at least two of the three scores, so there is no winner, rank, or margin. The other 154 replies place `plausibility_yes_probability` inside a JSON code fence and then continue with prose, so they are stored as invalid. The 17 yes/no comparisons are only the hypothesis pairs where both the original and the variant happened to parse. Zero flips in that subset is not a stability result for the full pilot.

## Disagreement examples

All four rows in `disagreements.jsonl` are example 650. The gold label is `alarm_query`. The sentence is "i need to set an alarm how many do i have set": it asks to set an alarm and asks how many are already set. On the original wording and on the paraphrase, both Jev and GPT's pairwise winner is query. GPT's paraphrase classification is valid and also selects query (0.7 on query, 0.3 on set). Claude's classification of all four 650 wordings is query, so it agrees with the gold label on this example.

1. Classification, minor typo: "i need to set an alram how many do i have set". Jev assigns 0.99 to query and 0.01 to set. GPT assigns 0.60 to set and 0.40 to query, so its unique prediction is `alarm_set`. The spelling change leaves both clauses in place.

2. Classification, repetition of the original sentence. Jev still predicts query: query falls from 0.99 to 0.79 and set rises to 0.21. GPT returns 0.5 and 0.5, recorded as `tie`.

3. Pairwise, minor typo. Jev's yes probabilities are query 0.95, set 0.27, remove 0.11, so the unique winner is query. GPT ties query and set at 0.7 (remove 0.2). GPT's original pairwise winner was query (0.7 versus set 0.4), so this row is a unique-winner-to-tie flip.

4. Pairwise, repetition. Jev keeps query first (0.93 over set 0.68). Set rises through 0.5 and the winner stays query. GPT puts set at 0.85 and query at 0.70, so the unique winner changes from query to set.

GPT's classification of the original 650 wording is absent from that list. The returned probabilities are `alarm_set` 0.6, `alarm_query` 0.9, `alarm_remove` 0.1 (sum 1.6), so the output is invalid. The same four invalid GPT classification bodies returned on the retry in this run.

Three of those four invalid GPT outputs are example 4929, "turn on my first scheduled alarm", together with its typo and its repetition. Those triples are (0.1, 0.2, 0.1), (0.1, 0.1, 0.1), and (0.1, 0.1, 0.0). The paraphrase classification is valid (0.8 on set). Jev's pairwise winner on every 4929 wording is `alarm_set`. Claude's pairwise replies for 4929 are all among the 154 invalid outputs.

Claude's four classification misses are the same example, and they are stable across wording. Gold is `alarm_set`. On the original, the paraphrase, and the typo, Claude assigns 0.90 to `alarm_remove` and 0.05 to set. On the repetition it assigns 0.80 to remove and 0.05 to set. Jev assigns 0.91, 0.91, 0.86, and 0.93 to set on those four wordings.

## Recommendation

For this alarm pilot, Jev is the more usable instrument. Every classification and pairwise group parsed, every unique winner matched the gold label, and paraphrase, typo, and repetition left that winner in place. GPT matches the gold label on 54 of 56 valid classification rows and on 58 of 60 pairwise groups. The Jev–GPT gap sits in example 650, whose gold query label shares the sentence with an explicit set request, and in four malformed GPT classification distributions, three of them from example 4929.

Claude classification parsed on all 60 rows and matches the gold label on 56. The four misses are one original, 4929, predicted as remove on every wording. That is a stable wrong label. Claude pairwise scores are not usable in this run: 154 of 180 replies are invalid, and no example has all three hypothesis scores. The 0 flips among 17 parsed pairs do not rank Claude.

The comparison uses accuracy, coverage, and whether the unique winner changes. Mean margin and crossings of 0.5 are separate quantities. GPT's mean margin is 0.766 and Jev's is 0.642. Jev crosses 0.5 five times and GPT three times, while Jev's unique winner stays fixed.

The sample is 15 originals. This comparison covers these alarm wordings. Meaning-change controls are already approved; scoring them against the updated label is the next evidence to collect before a stronger claim. A pairwise score for Claude needs a reply that is only the JSON probability.

The four Jev–GPT inspection notes were carried forward on 2026-09-28 because those stored scores are unchanged. Revise a note if a second reading of the sentence differs.
