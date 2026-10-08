INTENTS = ("alarm_set", "alarm_query", "alarm_remove")

MEANING_PRESERVING = ("original", "paraphrase", "minor_typo", "repetition")

# polite_restatement is a schema value only. The pilot writer does not emit it.
SCHEMA_TRANSFORMATIONS = MEANING_PRESERVING + ("polite_restatement", "meaning_change")

PROBABILITY_SUM_TOLERANCE = 1e-3
NLL_CLIP_FLOOR = 1e-6
YES_THRESHOLD = 0.5

QUANTITY = "These outputs are elicited plausibility judgments, not verified P(e|h)."

CONNECTIVITY_UTTERANCES = (
    "set an alarm for tomorrow morning at seven",
    "what time is my next alarm",
    "cancel the alarm I set for tonight",
)
CONNECTIVITY_QUESTION = "Does this text mention an alarm?"

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 520, 529})
