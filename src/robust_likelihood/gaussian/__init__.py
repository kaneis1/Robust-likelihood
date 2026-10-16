"""Test 1: typicality-based Bayesian inference on isotropic Gaussian hypotheses.

The observation x is replaced by its standardized typicality under each hypothesis,
optionally binned, and the posterior is computed from that. Every quantity has a
closed form here, so a failure belongs to the method and not to estimation.

Run as ``python -m robust_likelihood.gaussian``. The plan is docs/gaussian_test_plan.md.
"""
