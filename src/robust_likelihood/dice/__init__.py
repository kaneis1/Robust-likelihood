"""Dice sweep: verbalized probabilities and token log-probs against exact likelihoods.

Three dice with known face probabilities generate roll sequences of length k from 1
to 10, so the observation space grows from 6 to 6^10 outcomes while every likelihood
stays exact. Step 1 of docs/claims_plan.md.
"""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.storage import read_json

CONFIG_FILE = Path("configs") / "dice.json"
DATA_DIR = Path("data") / "dice"
RESULTS_DIR = Path("results") / "dice"
DOCS_DIR = Path("docs") / "dice_results"


def load_config(root: Path) -> dict:
    return read_json(root / CONFIG_FILE)
