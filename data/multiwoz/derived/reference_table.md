# MultiWOZ reference rates

Written by `python -m robust_likelihood.multiwoz prepare` from the pinned MultiWOZ 2.1 goals and 2.2 turn annotations. These are rates of real users, not model scores.

## Selection

| Quantity | Count |
| --- | --- |
| Dialogues in 2.1 / in 2.2 | 10,438 / 10,437 |
| With a restaurant goal / restaurant-only / SNG restaurant-only | 4,692 / 1,311 / 442 |
| Name-based (excluded): clean / fail_info | 44 / 28 |
| Slot-based: clean / fail_info | 235 / 135 |
| Clean items by constrained slots: 2 / 3 | 124 / 111 |
| Items with no named constrained slot | 11 |

Clean set, turn-0 slot status. `dontcare` is its own status, not a named value.

| Slot | Constrained | Named (mismatch) | Omitted | No-pref | No-pref named | No-pref dontcare |
| --- | --- | --- | --- | --- | --- | --- |
| food | 195 | 142 (0) | 53 | 40 | 1 | 1 |
| area | 186 | 138 (3) | 48 | 49 | 0 | 0 |
| price | 200 | 150 (1) | 50 | 35 | 1 | 0 |

## Populations

The pooled population is the first restaurant turn of every dialogue, any prefix, with a slot-based restaurant goal: 3,932 such goals, of which 5 never reach a restaurant turn and 0 are missing from 2.2.

| Population | Dialogues | Slot | Named / constrained | a | No-pref named / n | Value mismatch among named |
| --- | --- | --- | --- | --- | --- | --- |
| `match_clean_turn0` | 235 | food | 142 / 195 | 0.728 | 1 / 40 | 0.000 |
| `match_clean_turn0` | 235 | area | 138 / 186 | 0.742 | 0 / 49 | 0.022 |
| `match_clean_turn0` | 235 | price | 150 / 200 | 0.750 | 1 / 35 | 0.007 |
| `match_all_turn0` | 370 | food | 226 / 308 | 0.734 | 1 / 62 | 0.004 |
| `match_all_turn0` | 370 | area | 219 / 301 | 0.728 | 0 / 69 | 0.087 |
| `match_all_turn0` | 370 | price | 227 / 314 | 0.723 | 1 / 56 | 0.079 |
| `pooled_first_restaurant_turn` | 3,927 | food | 1,809 / 3,108 | 0.582 | 5 / 819 | 0.144 |
| `pooled_first_restaurant_turn` | 3,927 | area | 1,730 / 3,104 | 0.557 | 12 / 823 | 0.077 |
| `pooled_first_restaurant_turn` | 3,927 | price | 1,758 / 3,167 | 0.555 | 10 / 760 | 0.065 |
| `pooled_no_woz` | 3,251 | food | 1,418 / 2,684 | 0.528 | 3 / 567 | 0.023 |
| `pooled_no_woz` | 3,251 | area | 1,353 / 2,692 | 0.503 | 8 / 559 | 0.094 |
| `pooled_no_woz` | 3,251 | price | 1,379 / 2,753 | 0.501 | 8 / 498 | 0.081 |

## Primary reference: a from `match_clean_turn0`, b from `pooled_first_restaurant_turn` with +0.5 smoothing

| Slot | a (Wilson 95%) | b (bootstrap 95%) | Named: ln(a/b) (95%) | Omitted: ln((1−a)/(1−b)) (95%) |
| --- | --- | --- | --- | --- |
| food | 0.728 [0.662, 0.786] | 0.00671 [0.00182, 0.01262] | 4.69 [4.05, 6.00] | -1.30 [-1.55, -1.09] |
| area | 0.742 [0.675, 0.799] | 0.01517 [0.00786, 0.02347] | 3.89 [3.44, 4.55] | -1.34 [-1.61, -1.11] |
| price | 0.750 [0.686, 0.805] | 0.01380 [0.00616, 0.02279] | 4.00 [3.48, 4.79] | -1.37 [-1.61, -1.16] |

All three slots named and removed: L_E = 12.57 nats [11.64, 14.25].

## Sensitivity variants

| Variant | a from | b from | Slot | a | b | Named | Omitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary | `match_clean_turn0` | `pooled_first_restaurant_turn` | food | 0.728 | 0.00671 | 4.69 | -1.30 |
| primary | `match_clean_turn0` | `pooled_first_restaurant_turn` | area | 0.742 | 0.01517 | 3.89 | -1.34 |
| primary | `match_clean_turn0` | `pooled_first_restaurant_turn` | price | 0.750 | 0.01380 | 4.00 | -1.37 |
| V2 | `match_all_turn0` | `match_all_turn0` | food | 0.734 | 0.02381 | 3.43 | -1.30 |
| V2 | `match_all_turn0` | `match_all_turn0` | area | 0.728 | 0.00714 | 4.62 | -1.29 |
| V2 | `match_all_turn0` | `match_all_turn0` | price | 0.723 | 0.02632 | 3.31 | -1.26 |
| V3 | `pooled_first_restaurant_turn` | `pooled_first_restaurant_turn` | food | 0.582 | 0.00671 | 4.46 | -0.87 |
| V3 | `pooled_first_restaurant_turn` | `pooled_first_restaurant_turn` | area | 0.557 | 0.01517 | 3.60 | -0.80 |
| V3 | `pooled_first_restaurant_turn` | `pooled_first_restaurant_turn` | price | 0.555 | 0.01380 | 3.69 | -0.80 |
| V4 | `pooled_no_woz` | `pooled_no_woz` | food | 0.528 | 0.00616 | 4.45 | -0.75 |
| V4 | `pooled_no_woz` | `pooled_no_woz` | area | 0.503 | 0.01518 | 3.50 | -0.68 |
| V4 | `pooled_no_woz` | `pooled_no_woz` | price | 0.501 | 0.01703 | 3.38 | -0.68 |

## Reading

a must match the probe population: turn 0 of single-domain restaurant dialogues. b is a rare-event rate, so it comes from the pooled corpus, where the matched population would have only 0 or 1 events per slot.

The slot-level ratio ignores which value was named. Under a no-preference goal a specific value is less likely still, so the true likelihood ratio is at least a/b.

Intervals are percentile bootstrap intervals over 2,000 replicates that resample dialogues within each population; the a interval in the primary table is Wilson's.
