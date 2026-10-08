# Retrospective audit of the synthetic alarm run

This note sits beside `templates/synthetic_alarm_report.md` for run `synthetic-20261001T150250Z`. That report is unchanged. Nothing here rescores those stored replies.

The follow-up uses the same prompt template, `v2`, and the same three models. It changes the hypothesis definitions to `synthetic_v2`, the items to dataset `synthetic_alarm_v2`, and the evaluation rules to `synthetic_eval_v2`. A difference from the old run is a difference under clarified hypotheses and new sentences. It is not, by itself, an improvement caused by the new sentences.

These outputs remain elicited plausibility judgments, not verified P(e|h).

## Definitions used from here on

`intend_query` requests information about an alarm or about previous alarm instructions, and it does not issue an alarm-changing operation. The old description said the user wants an answer "about an alarm or about something the user said." It did not say that asking whether an instruction was said is different from issuing the quoted instruction. The follow-up states that difference.

A polite question that asks for an operation is still that operation. "Could you delete my alarm entirely?" currently requests deletion.

| Expression | Treatment |
| --- | --- |
| "Create a new alarm" | Create |
| "Enable my existing alarm" | Enable |
| "Disable my alarm, but keep it saved" | Disable |
| "Delete my alarm entirely" | Delete |
| "Turn off my alarm" | Disable under the documented task convention |
| "Cancel my alarm" | Ambiguous unless the utterance specifies disabling or deletion |

Assigning "cancel" to delete in a support map does not remove the ordinary-language ambiguity. Bare "Cancel my alarm." is kept as an unscored probe. "Cancel it by deleting the saved alarm." is the scored delete sentence.

An audit outcome may be `not_tested`. Order was stored on the old two-action rows and was not scored by the atomic yes-probabilities. That is an evaluation gap. State questions were stored and were not sent, so they stay outside the failure count.

## What remains interpretable in the old run

Meaning preserved and meaning reversed stay interpretable. Those sentences are enable, or an enable sentence reversed to an off sentence. Every model kept `intend_enable` on the preserved pairs and moved the winner from enable to disable when the sentence reversed. The off sentences use "turn off" and "disable," which the new convention still treats as disable. They do not use bare "cancel."

Negation and correction stay interpretable as "no alarm-changing operation was issued." All three models kept `intend_query` above 0.5 and the four changing actions below 0.5, including on "Don't cancel the seven o'clock alarm; tell me whether it is enabled." Those passes do not show that the model resolved "cancel," because both disable and delete were rejected together.

Quoted commands split.

- Claude's three passes stay interpretable. The quoted action stayed below 0.5 and query was first. Under the broadened query, asking whether an instruction was said is the request.
- GPT on "Did I tell you to 'delete my morning alarm' before?" still has an interpretable miss: query wins at 0.99, and the rejected delete score is 0.75. The old rule counts a rejected hypothesis at or above 0.5 as a miss. The new query wording still rejects issuing the quoted delete.
- Jev on "Did I say 'cancel my alarm' earlier?" has disable at 0.56. Bare "cancel" is now an ambiguity, so that disable score is not a settled failure.
- Jev on "Was it me who said 'turn off my alarm'?" has disable at 0.5. "Turn off" is disable under the convention, and the quote is not a current request, so the old at-or-above-0.5 rule still marks a miss. Query is first, at 0.95.

Multiple commands do not show an order failure. Both requested actions were supported on every sentence, and the stored order was never a hypothesis. That result is `not_tested` for order. The extra `intend_disable` scores on "Cancel my seven o'clock alarm, then set one for eight." (0.70, 0.90, and 0.80) use bare "cancel," so they are not settled failures. Jev's disable score of 0.75 on "First remove my noon alarm, then set an alarm for one." uses "remove," which this audit does not assign. The three Jev–GPT disagreements in this family are unique-winner versus tie. They are not evidence about order.

Underspecified evidence had no gold winner and no correct probability. It contributes no failure. The state questions `alarm_is_enabled` and `alarm_exists` were not submitted.

MASSIVE-lump classification, where it parsed, matched the collapsed labels. It still does not show that enable was separated from create, or disable from delete.

## Follow-up materials

The reviewed items are `data/synthetic/alarm_v2/`. Thirteen rows: eight scored atomic operations, one unscored bare-cancel probe, two quotation mentions paired with the explicit delete and disable sentences, and two complete plans that reverse each other.

For "Delete the morning alarm, and after that create one for nine." the supported hypothesis is deleting the morning alarm and then creating a new alarm for 9:00. The reversed order is rejected. "Only deleting the morning alarm" and "only creating a new alarm for 9:00" are rejected. The reversed utterance supports the other complete plan and rejects the first. Complete-plan scoring is the primary test. Atomic scores on those two sentences are diagnostic.

`synthetic_eval_v2` keeps the old support line for scored rows: a supported hypothesis is above 0.5 and strictly above every rejected hypothesis in that row's scored set. A rejected hypothesis at or above 0.5 is a miss. Bare cancel has no such set. Qualitative words are not turned into probabilities.

A later probability check, if run, asks two separate questions about one stated event: the probability that the event occurs, and whether that probability is unlikely, neutral, or likely. Numerical agreement and qualitative consistency are separate. That check does not say what a 0.75 plausibility score meant on a quoted alarm sentence.
