"""Token log P(e|h) over observation tokens only.

Calling this module is not a completed baseline and never returns a placeholder score.
"""


def log_probability_evidence_given_hypothesis(evidence_tokens, hypothesis: str) -> float:
    """log P(e|h) over observation tokens only."""
    raise NotImplementedError("not implemented")
