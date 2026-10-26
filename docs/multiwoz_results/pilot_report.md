# MultiWOZ pilot: Jev on seeded clean items

Run `20261025T185518Z-pilot`. Requested `jev-1.13.0`; prompt versions `multiwoz_likely`, `multiwoz_typical`. 20 clean items drawn with seed 20261026. Scores are Jev's elicited yes-probabilities, read as l = logit(clip(s, 0.001, 1 - 0.001)); they are not likelihoods.

## Checks

| Check | Result | Target | Pass |
| --- | --- | --- | --- |
| Success rate | 398/398 (100.0%) | at least 99% | yes |
| Returned model id | jev-1.13.0 (398) | constant | yes |
| Median l(G\|impl) − l(ctrl:contra), likely (20 items) | 2.78 | above 2 | yes |
| Median l(G\|impl) − l(ctrl:contra), typical (20 items) | 2.73 | above 2 | yes |

Scores at or beyond the clip bounds: 0 of 398. HTTP requests: 398; cache hits: 0; failures: 0.

## Five items

For each item, the first table lists every hypothesis with Jev's score s and its logit l. The second compares Jev's log-ratio L_J = l(narrow) − l(broader) with the corpus log-ratio L_E for the same removal, by wording, as in the 25 October pilot. Positive values favour the narrower goal. For a three-slot goal the explicit and implicit goal texts are the same, so one score serves both.

### SNG01184

> Can you help me find a good thai restaurant in the west?

Goal: food thai, area west, price expensive. Turn 0: food named, area named, price omitted.

| Hypothesis | Text | likely s | likely l | typical s | typical l |
| --- | --- | --- | --- | --- | --- |
| `G|impl` | The user wants to find an expensive restaurant serving thai food in the west of town. | 0.680 | 0.75 | 0.620 | 0.49 |
| `minus:food|impl` | The user wants to find an expensive restaurant in the west of town. | 0.550 | 0.20 | 0.550 | 0.20 |
| `minus:food|expl` | The user wants to find an expensive restaurant in the west of town. The user has no preference about the type of food. | 0.280 | -0.94 | 0.270 | -0.99 |
| `minus:area|impl` | The user wants to find an expensive restaurant serving thai food. | 0.560 | 0.24 | 0.490 | -0.04 |
| `minus:area|expl` | The user wants to find an expensive restaurant serving thai food. The user has no preference about the area. | 0.480 | -0.08 | 0.460 | -0.16 |
| `minus:pricerange|impl` | The user wants to find a restaurant serving thai food in the west of town. | 0.930 | 2.59 | 0.970 | 3.48 |
| `minus:pricerange|expl` | The user wants to find a restaurant serving thai food in the west of town. The user has no preference about the price range. | 0.940 | 2.75 | 0.970 | 3.48 |
| `broad|impl` | The user wants to find a restaurant. | 0.920 | 2.44 | 0.960 | 3.18 |
| `broad|expl` | The user wants to find a restaurant. The user has no preference about the type of food, the area, or the price range. | 0.610 | 0.45 | 0.670 | 0.71 |
| `ctrl:contra` | The user wants to find an expensive restaurant serving steakhouse food in the west of town. | 0.080 | -2.44 | 0.070 | -2.59 |
| `ctrl:tie` | The user wants to find a moderately priced restaurant serving thai food in the west of town. | 0.930 | 2.59 | 0.960 | 3.18 |

| Removed | Turn 0 | Wording | L_J likely | L_J typical | L_E |
| --- | --- | --- | --- | --- | --- |
| food | named | explicit | 1.70 | 1.48 | 4.69 |
| food | named | implicit | 0.55 | 0.29 | 4.69 |
| area | named | explicit | 0.83 | 0.65 | 3.89 |
| area | named | implicit | 0.51 | 0.53 | 3.89 |
| price | omitted | explicit | -2.00 | -2.99 | -1.37 |
| price | omitted | implicit | -1.83 | -2.99 | -1.37 |
| all | 2 of 3 named | explicit | 0.31 | -0.22 | 7.20 |
| all | 2 of 3 named | implicit | -1.69 | -2.69 | 7.20 |

### SNG01744

> I am looking for a place to dine. The restaurant should be in the cheap price range and should serve vietnamese food.

Goal: food vietnamese, price cheap. Turn 0: food named, price named.

| Hypothesis | Text | likely s | likely l | typical s | typical l |
| --- | --- | --- | --- | --- | --- |
| `G|impl` | The user wants to find a cheap restaurant serving vietnamese food. | 0.890 | 2.09 | 0.960 | 3.18 |
| `G|expl` | The user wants to find a cheap restaurant serving vietnamese food. The user has no preference about the area. | 0.920 | 2.44 | 0.960 | 3.18 |
| `minus:food|impl` | The user wants to find a cheap restaurant. | 0.890 | 2.09 | 0.940 | 2.75 |
| `minus:food|expl` | The user wants to find a cheap restaurant. The user has no preference about the type of food or the area. | 0.350 | -0.62 | 0.390 | -0.45 |
| `minus:pricerange|impl` | The user wants to find a restaurant serving vietnamese food. | 0.900 | 2.20 | 0.950 | 2.94 |
| `minus:pricerange|expl` | The user wants to find a restaurant serving vietnamese food. The user has no preference about the area or the price range. | 0.450 | -0.20 | 0.540 | 0.16 |
| `broad|impl` | The user wants to find a restaurant. | 0.900 | 2.20 | 0.960 | 3.18 |
| `broad|expl` | The user wants to find a restaurant. The user has no preference about the type of food, the area, or the price range. | 0.230 | -1.21 | 0.220 | -1.27 |
| `ctrl:contra` | The user wants to find a moderately priced restaurant serving vietnamese food. | 0.800 | 1.39 | 0.860 | 1.82 |

| Removed | Turn 0 | Wording | L_J likely | L_J typical | L_E |
| --- | --- | --- | --- | --- | --- |
| food | named | explicit | 3.06 | 3.63 | 4.69 |
| food | named | implicit | 0.00 | 0.43 | 4.69 |
| price | named | explicit | 2.64 | 3.02 | 4.00 |
| price | named | implicit | -0.11 | 0.23 | 4.00 |
| all | 2 of 2 named | explicit | 3.65 | 4.44 | 8.68 |
| all | 2 of 2 named | implicit | -0.11 | 0.00 | 8.68 |

### SNG01896

> I am looking for a restaurant. The restaurant should be in the south and should be in the cheap price range

Goal: area south, price cheap. Turn 0: area named, price named.

| Hypothesis | Text | likely s | likely l | typical s | typical l |
| --- | --- | --- | --- | --- | --- |
| `G|impl` | The user wants to find a cheap restaurant in the south of town. | 0.890 | 2.09 | 0.960 | 3.18 |
| `G|expl` | The user wants to find a cheap restaurant in the south of town. The user has no preference about the type of food. | 0.910 | 2.31 | 0.960 | 3.18 |
| `minus:area|impl` | The user wants to find a cheap restaurant. | 0.870 | 1.90 | 0.950 | 2.94 |
| `minus:area|expl` | The user wants to find a cheap restaurant. The user has no preference about the type of food or the area. | 0.610 | 0.45 | 0.750 | 1.10 |
| `minus:pricerange|impl` | The user wants to find a restaurant in the south of town. | 0.850 | 1.73 | 0.940 | 2.75 |
| `minus:pricerange|expl` | The user wants to find a restaurant in the south of town. The user has no preference about the type of food or the price range. | 0.370 | -0.53 | 0.410 | -0.36 |
| `broad|impl` | The user wants to find a restaurant. | 0.880 | 1.99 | 0.950 | 2.94 |
| `broad|expl` | The user wants to find a restaurant. The user has no preference about the type of food, the area, or the price range. | 0.320 | -0.75 | 0.320 | -0.75 |
| `ctrl:contra` | The user wants to find a moderately priced restaurant in the south of town. | 0.670 | 0.71 | 0.680 | 0.75 |

| Removed | Turn 0 | Wording | L_J likely | L_J typical | L_E |
| --- | --- | --- | --- | --- | --- |
| area | named | explicit | 1.87 | 2.08 | 3.89 |
| area | named | implicit | 0.19 | 0.23 | 3.89 |
| price | named | explicit | 2.85 | 3.54 | 4.00 |
| price | named | implicit | 0.36 | 0.43 | 4.00 |
| all | 2 of 2 named | explicit | 3.07 | 3.93 | 7.89 |
| all | 2 of 2 named | implicit | 0.10 | 0.23 | 7.89 |

### SNG02067

> Hi, I am trying to find a restaurant serving Italian cuisine, something expensive please. What would you suggest?

Goal: food italian, price expensive. Turn 0: food named, price named.

| Hypothesis | Text | likely s | likely l | typical s | typical l |
| --- | --- | --- | --- | --- | --- |
| `G|impl` | The user wants to find an expensive restaurant serving italian food. | 0.930 | 2.59 | 0.970 | 3.48 |
| `G|expl` | The user wants to find an expensive restaurant serving italian food. The user has no preference about the area. | 0.930 | 2.59 | 0.970 | 3.48 |
| `minus:food|impl` | The user wants to find an expensive restaurant. | 0.910 | 2.31 | 0.960 | 3.18 |
| `minus:food|expl` | The user wants to find an expensive restaurant. The user has no preference about the type of food or the area. | 0.630 | 0.53 | 0.700 | 0.85 |
| `minus:pricerange|impl` | The user wants to find a restaurant serving italian food. | 0.910 | 2.31 | 0.960 | 3.18 |
| `minus:pricerange|expl` | The user wants to find a restaurant serving italian food. The user has no preference about the area or the price range. | 0.620 | 0.49 | 0.670 | 0.71 |
| `broad|impl` | The user wants to find a restaurant. | 0.900 | 2.20 | 0.960 | 3.18 |
| `broad|expl` | The user wants to find a restaurant. The user has no preference about the type of food, the area, or the price range. | 0.300 | -0.85 | 0.320 | -0.75 |
| `ctrl:contra` | The user wants to find an expensive restaurant serving korean food. | 0.110 | -2.09 | 0.090 | -2.31 |

| Removed | Turn 0 | Wording | L_J likely | L_J typical | L_E |
| --- | --- | --- | --- | --- | --- |
| food | named | explicit | 2.05 | 2.63 | 4.69 |
| food | named | implicit | 0.27 | 0.30 | 4.69 |
| price | named | explicit | 2.10 | 2.77 | 4.00 |
| price | named | implicit | 0.27 | 0.30 | 4.00 |
| all | 2 of 2 named | explicit | 3.43 | 4.23 | 8.68 |
| all | 2 of 2 named | implicit | 0.39 | 0.30 | 8.68 |

### SNG0454

> I am looking for an Italian restaurant in the centre of town, could you provide me with some options to choose from please?

Goal: food italian, area centre, price cheap. Turn 0: food named, area named, price omitted.

| Hypothesis | Text | likely s | likely l | typical s | typical l |
| --- | --- | --- | --- | --- | --- |
| `G|impl` | The user wants to find a cheap restaurant serving italian food in the centre of town. | 0.800 | 1.39 | 0.810 | 1.45 |
| `minus:food|impl` | The user wants to find a cheap restaurant in the centre of town. | 0.560 | 0.24 | 0.610 | 0.45 |
| `minus:food|expl` | The user wants to find a cheap restaurant in the centre of town. The user has no preference about the type of food. | 0.220 | -1.27 | 0.290 | -0.90 |
| `minus:area|impl` | The user wants to find a cheap restaurant serving italian food. | 0.720 | 0.94 | 0.760 | 1.15 |
| `minus:area|expl` | The user wants to find a cheap restaurant serving italian food. The user has no preference about the area. | 0.580 | 0.32 | 0.670 | 0.71 |
| `minus:pricerange|impl` | The user wants to find a restaurant serving italian food in the centre of town. | 0.930 | 2.59 | 0.970 | 3.48 |
| `minus:pricerange|expl` | The user wants to find a restaurant serving italian food in the centre of town. The user has no preference about the price range. | 0.940 | 2.75 | 0.970 | 3.48 |
| `broad|impl` | The user wants to find a restaurant. | 0.910 | 2.31 | 0.970 | 3.48 |
| `broad|expl` | The user wants to find a restaurant. The user has no preference about the type of food, the area, or the price range. | 0.530 | 0.12 | 0.660 | 0.66 |
| `ctrl:contra` | The user wants to find a cheap restaurant serving italian food in the east of town. | 0.330 | -0.71 | 0.370 | -0.53 |
| `ctrl:tie` | The user wants to find a moderately priced restaurant serving italian food in the centre of town. | 0.920 | 2.44 | 0.960 | 3.18 |

| Removed | Turn 0 | Wording | L_J likely | L_J typical | L_E |
| --- | --- | --- | --- | --- | --- |
| food | named | explicit | 2.65 | 2.35 | 4.69 |
| food | named | implicit | 1.15 | 1.00 | 4.69 |
| area | named | explicit | 1.06 | 0.74 | 3.89 |
| area | named | implicit | 0.44 | 0.30 | 3.89 |
| price | omitted | explicit | -1.37 | -2.03 | -1.37 |
| price | omitted | implicit | -1.20 | -2.03 | -1.37 |
| all | 2 of 3 named | explicit | 1.27 | 0.79 | 7.20 |
| all | 2 of 3 named | implicit | -0.93 | -2.03 | 7.20 |
