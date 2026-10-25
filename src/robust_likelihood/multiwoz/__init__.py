"""MultiWOZ size-principle test: elicited scores against corpus likelihood ratios.

A hypothesis that constrains a slot the utterance names should beat a hypothesis
with no preference about that slot, by a ratio measured from real users. Jev's
scores are elicited judgments; the corpus rates are the reference.
"""

from __future__ import annotations

from pathlib import Path

from robust_likelihood.storage import read_json

CONFIG_FILE = Path("configs") / "multiwoz_size_principle.json"
RAW_DIR = Path("data") / "multiwoz" / "raw"
DERIVED_DIR = Path("data") / "multiwoz" / "derived"
CHECKSUM_FILE = Path("data") / "multiwoz" / "checksum.json"
RESULTS_DIR = Path("results") / "multiwoz"


def load_config(root: Path) -> dict:
    return read_json(root / CONFIG_FILE)


def jev_model_id(root: Path) -> str:
    """The pinned Jev id lives in configs/models.json and is not duplicated here."""
    return read_json(root / "configs" / "models.json")["models"]["jev"]
