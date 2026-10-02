# Synthetic alarm follow-up

Run `synthetic-v2-20261002T061206Z`. Models `jev-1.13.0`, `gpt-6-astra`, and `claude-fable-5-1`. Thirteen reviewed sentences. Prompt template `v2`, hypothesis definitions `synthetic_v2`, dataset `synthetic_alarm_v2`, evaluation rule `synthetic_eval_v2`. The run made 219 pairwise requests. Nothing was left unattempted, and every reply parsed. There were no transport failures.

These outputs are elicited plausibility judgments, not verified P(e|h).

The earlier 22-item run stays in `templates/synthetic_alarm_report.md`. This follow-up uses different sentences and different hypothesis text, so a difference from that run is not evidence about the new wording alone.

A supported hypothesis must be above 0.5 and strictly above every rejected hypothesis on that row. A rejected hypothesis at or above 0.5 is a miss. Bare "Cancel my alarm." is stored and is not a pass or a failure. On the two ordered sentences, the four complete-plan hypotheses are the primary score. The five atomic scores on those sentences are a secondary diagnostic and do not decide the row.

## Comparison

| Test | Jev | GPT Astra | Claude Fable |
| --- | --- | --- | --- |
| Explicit current operation, including polite requests and cancel-by-deleting | 7/7 | 7/7 | 7/7 |
| "Turn off my alarm." is disable, and delete stays below 0.5 | pass | pass | miss |
| Quoted earlier instruction is query, and the quoted action stays below 0.5 | 0/2 | 2/2 | 2/2 |
| Complete plan, uttered order, with both partials rejected | 2/2 | 2/2 | 2/2 |
| Bare "Cancel my alarm." | not scored | not scored | not scored |

The explicit operations are "Create a new alarm.", "Could you create a new alarm?", "Enable my existing alarm.", "Disable my alarm, but keep it saved.", "Delete my alarm entirely.", "Could you delete my alarm entirely?", and "Cancel it by deleting the saved alarm." Query stays below 0.5 on both polite requests for every model. The highest query score on those two sentences is 0.06.

## What each test showed

Turn off. Jev and GPT give disable 0.98 and 1.0, with delete at 0.39 and 0.20. Claude also ranks disable first, at 0.97, and gives delete 0.60. Delete is a rejected hypothesis on this sentence, so Claude misses the row even though disable wins.

Quoted instruction. "Did I say 'delete my alarm entirely'?" and "Did I say 'disable my alarm, but keep it saved'?" have query as the unique winner for every model, at 0.96, 0.99, and 0.95 or 0.93. GPT and Claude keep the quoted action below 0.5. Jev does not. On the delete quote, delete is 0.65 and disable is 0.58. On the disable quote, disable is 0.86. Those two rows are the only Jev–GPT disagreements. The winners match. The disagreement is that Jev still supports the mentioned action.

Complete plans. "Delete the morning alarm, and after that create one for nine." supports delete-then-create and rejects create-then-delete, only-delete, and only-create. The reversed sentence swaps the two complete plans and still rejects both partials. Every model passes both rows. The uttered order is the unique winner, between 0.93 and 0.99. The reversed complete plan and both "only" partials stay below 0.5. The highest rejected plan score is Jev's only-delete at 0.35 on the reversed sentence.

Atomic diagnostic on those two sentences. Both requested actions are above 0.5 for every model. GPT ties create and delete at 0.99 on both sentences. Claude ranks the first-mentioned action higher: delete 0.90 then create 0.80 on the forward sentence, and create 0.70 then delete 0.65 on the reversed sentence, with disable below 0.5. Jev also supports both actions, and disable is 0.64 on the forward sentence and 0.82 on the reversed sentence. Those atomic scores do not change the plan result.

Bare cancel. Jev's highest score is delete at 0.92, and disable is also 0.52. Claude's highest score is delete at 0.90, with disable at 0.15. GPT puts no hypothesis above 0.5. Its highest stored score is delete at 0.10.

## Reading

On this set, the scores separate a complete current request from a partial request and from the reversed order. All three models pick the uttered complete plan and keep both "only" plans below 0.5. Polite "Could you ...?" is scored as the operation it asks for.

The remaining misses are about a mentioned action and about "turn off." Jev still gives the quoted action a score at or above 0.5 while ranking query first. Claude ranks "Turn off my alarm." as disable and also leaves delete at 0.60. Bare cancel stays an ambiguity probe: Jev supports delete and disable together, Claude's highest score is delete, and GPT does not place any hypothesis above 0.5.
