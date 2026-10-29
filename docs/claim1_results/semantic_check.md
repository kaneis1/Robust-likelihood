# Semantic extractor check

Regex slot extractor against the MultiWOZ 2.2 turn-0 annotations of the 370 items.
Acceptance: at least 95% agreement on every slot.

| Slot | Agreement |
|---|---|
| food | 98.9% |
| area | 98.6% |
| pricerange | 97.0% |

Passes: yes.

## Confusion (annotated → extracted)

`value->same` and `value->other` split named slots by whether the value agrees.

| Cell | food | area | pricerange |
|---|---|---|---|
| dontcare->dontcare | 1 | 0 | 0 |
| none->dontcare | 0 | 1 | 0 |
| none->none | 141 | 147 | 141 |
| none->value | 1 | 3 | 1 |
| value->none | 1 | 1 | 7 |
| value->other | 2 | 0 | 3 |
| value->same | 224 | 218 | 218 |

## Disagreements

| Item | Slot | Annotated | Extracted | Utterance |
|---|---|---|---|---|
| SNG01425 | area | none | centre | Is it possible to get any Greek food in the centre part of town? |
| SNG01720 | pricerange | expensive | none | I'm looking for somewhere to eat in the centre part of town, somewhere nice, price is no object. |
| SNG02321 | pricerange | moderate | expensive | I'm looking for a restaurant, somewhere not too expensive, but not bottom of the barrel either, something in the mid range. |
| SNG0459 | pricerange | moderate | none | I'm looking for a restaurant in the centre of town with a modest price range. Can you recommend one? |
| SNG0465 | food | european | traditional | I'm looking for a restaurant that serves traditional european food, that's not too expensive or too cheap. |
| SNG0465 | pricerange | moderate | expensive | I'm looking for a restaurant that serves traditional european food, that's not too expensive or too cheap. |
| SNG0530 | pricerange | expensive | none | Do you have any recommendations for a restaurant in the south area in a high price range? |
| SNG0552 | food | none | british | Are there any expensive British restaurants in town? |
| SNG0556 | pricerange | none | cheap | Hi, I'm looking for a cheap austrian restaurant. |
| SNG0561 | area | none | centre | I am looking for a restaurant in the centre that serves african food. |
| SNG0565 | food | american, north american | traditional american | I'm in the mood for a traditional American food restaurant tonight and looking to treat myself to some expensive fine dining. Can you help me find a perfect place? |
| SNG0574 | area | none | dontcare | Know anywhere that has european food? |
| SNG0610 | area | none | centre | Hi there. I'm hoping you can help me. I am looking to try molecular gastronomy food. Might there be a restaurant like this in the centre of town? |
| SNG0653 | pricerange | expensive | moderate | I'm looking for a greek restaurant that is moderately expensive. |
| SNG0687 | pricerange | expensive | none | I want to eat someplace with unusual food. Price is no object. What can you suggest? |
| SNG0696 | pricerange | expensive | none | What fancy restaurants are there on the east side of town? |
| SNG0748 | area | east | none | I am looking for an Eastern restaurant, medium price |
| SNG0748 | pricerange | moderate | none | I am looking for an Eastern restaurant, medium price |
| SNG1268 | food | gastropod, gastropub | none | AM looking for a place to dine. The restaurant should be in the south and should serve gastropod food. |
| SNG1329 | pricerange | moderate | none | Hi, I need a place to eat in the centre for a medium price range. What do you have that you can suggest? |

## Hypothesis texts

Every distinct probe hypothesis text (983) against the slots it was built from:
100.0% extract exactly.

