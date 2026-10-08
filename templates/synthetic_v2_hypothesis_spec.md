# Hypothesis content for synthetic_alarm_v2

Prompt template `v2`. Hypothesis definitions `synthetic_v2`. Dataset `synthetic_alarm_v2`. Evaluation rules `synthetic_eval_v2`.

Each hypothesis names the action, the object, the arguments, and the order. A missing field is part of the hypothesis text, not something filled in later. Atomic scores on a two-action sentence stay a secondary diagnostic. The primary test is the complete plan.

## Atomic operations

| Hypothesis | Action | Object | Arguments | Order |
| --- | --- | --- | --- | --- |
| `intend_create` | create | a new alarm | none beyond the utterance | one action, issued now |
| `intend_enable` | enable | an existing alarm | turn it on | one action, issued now |
| `intend_disable` | disable | an existing alarm | turn it off and keep it saved | one action, issued now |
| `intend_delete` | delete | an existing alarm | remove it entirely | one action, issued now |
| `intend_query` | request information | an alarm, or a previous alarm instruction | no alarm-changing operation | one request, issued now |

"Turn off my alarm" is scored as `intend_disable` by the task convention. "Cancel my alarm" has no action assignment. "Could you delete my alarm entirely?" has action delete, issued now.

## Complete plans for the morning-alarm pair

The forward utterance is "Delete the morning alarm, and after that create one for nine." The reversed utterance exchanges the two steps.

| Hypothesis | Action | Object | Arguments | Order | Forward | Reversed |
| --- | --- | --- | --- | --- | --- | --- |
| `plan_delete_then_create` | delete, then create | the morning alarm; a new alarm | entirely; for 9:00 | delete the morning alarm, then create a new alarm for 9:00 | supported | rejected |
| `plan_create_then_delete` | create, then delete | a new alarm; the morning alarm | for 9:00; entirely | create a new alarm for 9:00, then delete the morning alarm | rejected | supported |
| `plan_only_delete` | delete | the morning alarm | entirely | that one action | rejected | rejected |
| `plan_only_create` | create | a new alarm | for 9:00 | that one action | rejected | rejected |

The single-action hypotheses say "only." Without that word, a single action would still be compatible with the full request.
