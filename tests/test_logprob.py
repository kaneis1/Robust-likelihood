import pytest

from robust_likelihood.logprob import log_probability_evidence_given_hypothesis
from robust_likelihood.storage import repo_root


def test_log_probability_is_not_implemented():
    with pytest.raises(NotImplementedError, match="^not implemented$"):
        log_probability_evidence_given_hypothesis(["set", "an", "alarm"], "alarm_set")


def test_supervisor_report_keeps_quantity_statement():
    text = (repo_root() / "templates" / "supervisor_report.md").read_text(encoding="utf-8")
    assert "These outputs are elicited plausibility judgments, not verified P(e|h)." in text
    assert "## Comparison table" in text
    assert "## Disagreement examples" in text
    assert "## Recommendation" in text
    assert "20260927T151455Z" in text
    assert "claude-sonnet-4-5-20250929" in text
