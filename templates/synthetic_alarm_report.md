# Synthetic alarm extension

Run `synthetic-20261001T150250Z`. Models `jev-1.13.0`, `gpt-6-astra`, and `claude-fable-5-1`. Twenty-two authored sentences, scored with hypothesis set `synthetic_v1` and prompt version `v2`. The run made 360 requests: 330 pairwise scores and 30 MASSIVE-lump classifications. Nothing was left unattempted, and there were no transport failures.

These outputs are elicited plausibility judgments, not verified P(e|h).

This set is not part of the MASSIVE pilot. The pilot report stays in `templates/supervisor_report.md` and in `results/20261001T120936Z` was not rewritten. Enabling, disabling, creating, and deleting are separate intended actions here. The MASSIVE labels still lump enable with create (`alarm_set`) and disable with delete (`alarm_remove`). A classification that follows that lump is reported on its own line. It is not the synthetic gold label.

State questions, including whether an alarm is currently enabled, were stored and were not sent to the models. They have no truth label.

A supported hypothesis must be above 0.5 and strictly above every rejected hypothesis. A rejected hypothesis at or above 0.5 is a miss. No item has a made-up numeric target.

## Comparison

| Test | Jev | GPT Astra | Claude Fable |
| --- | --- | --- | --- |
| Meaning preserved, winner stays `intend_enable` | 3/3 | 3/3 | 3/3 |
| Mean absolute score change on those pairs | 0.080 | 0.007 | 0.016 |
| Meaning reversed, enable falls and disable rises | 2/2 | 2/2 | 2/2 |
| Negation and correction | 3/3 | 3/3 | 3/3 |
| Quoted command | 1/3 | 2/3 | 3/3 |
| Multiple commands, both actions and no extra action | 1/3 | 2/3 | 2/3 |
| Underspecified evidence | no accuracy | no accuracy | no accuracy |
| MASSIVE-lump classification | 10/10 | 10/10 | 7/7 parsed |

Claude's three invalid outputs are classification replies for "Turn my seven o'clock alarm on.", "Turn on my first scheduled alarm.", and "Disable the seven o'clock alarm." The pairwise scores for those sentences still parsed, and the pairs they belong to passed.

## What each test showed

Meaning preserved. "Enable my first scheduled alarm." and "Please turn on my first scheduled alarm.", plus the weekday and seven-o'clock variants, keep `intend_enable` as the unique winner for every model. Score changes are small. The largest mean absolute change is Jev's 0.080.

Meaning reversed. "Turn on my first scheduled alarm." to "Turn off my first scheduled alarm.", and "Enable the seven o'clock alarm." to "Disable the seven o'clock alarm.", move the winner from `intend_enable` to `intend_disable` for every model. Enable falls and disable rises. This is not MASSIVE control `4929__meaning_change`, which stores the same surface change as `alarm_set` to `alarm_remove`.

Negation and correction. On all three sentences, including "Don't cancel the seven o'clock alarm; tell me whether it is enabled.", every model keeps `intend_query` above 0.5 and keeps create, enable, disable, and delete below 0.5. The sentence asks whether the alarm is enabled. It does not say that the alarm is enabled, and that state was not scored.

Quoted command. Claude keeps the quoted action below 0.5 on all three sentences, with `intend_query` first. GPT does so for the cancel quote and the turn-off quote. On "Did I tell you to 'delete my morning alarm' before?" GPT still gives `intend_delete` 0.75 while `intend_query` is 0.99, so query wins and the rejected delete action stays supported. Jev passes only "Did I tell you to 'delete my morning alarm' before?" On "Did I say 'cancel my alarm' earlier?" delete is 0.45, below the line, but disable is 0.56. On "Was it me who said 'turn off my alarm'?" disable is exactly 0.5. Query is first in both cases, at 0.96 and 0.95.

Multiple commands. The two requested actions are create and delete, in that stored order. Their probabilities are not required to sum to 1. All three models support both actions on every sentence. The misses are an extra `intend_disable` score at or above 0.5: all three models on "Cancel my seven o'clock alarm, then set one for eight." (disable 0.70, 0.90, and 0.80), and Jev on "First remove my noon alarm, then set an alarm for one." (disable 0.75). "Delete the morning alarm, and after that create one for nine." passes for every model.

The three Jev–GPT disagreements are all in this family. They are unique-winner versus tie, not a swap to a different action. On the seven-o'clock sentence both actions are high and disable is also high. On the morning sentence both actions pass and GPT ties create with delete at 0.99. On the noon sentence GPT ties the two requested actions and passes; Jev ranks delete first and also supports disable.

Underspecified evidence. "What about my seven o'clock alarm?" has no gold winner and no correct probability. Bare, every model gives `intend_query` the highest score, and Jev and GPT also leave the action scores above 0.5. The top gap is 0.12 for Jev, 0.23 for GPT, and 0.35 for Claude. After the context "Is my seven o'clock alarm enabled?", the action scores fall and query stays near the top (Jev and Claude 0.96 and 0.90, GPT 0.99). After "Cancel my seven o'clock alarm.", delete and disable rise relative to enable. That is the sensitivity the item was written to show.

## Reading

On this authored set, all three models keep an enable paraphrase stable and swap enable for disable when the sentence reverses. They treat the negation sentences as questions. Quoted commands and two-action sentences are where support spreads: a quoted delete can still receive a high delete score, and "cancel, then set" can also support disable. The bare seven-o'clock question does not pick one action. Adding a context sentence moves the action scores.

The MASSIVE-lump classification, where it parsed, matched `alarm_set` for the enable sentences and `alarm_remove` for the disable sentences. That agrees with the old collapsed labels. It does not by itself show that the model separated enable from create, or disable from delete. The pairwise scores are the evidence for that separation.
