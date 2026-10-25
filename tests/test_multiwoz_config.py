import json

from robust_likelihood.multiwoz import jev_model_id, load_config
from robust_likelihood.storage import repo_root


def test_config_pins_the_corpus_and_reads_jev_from_models_json():
    root = repo_root()
    config = load_config(root)
    assert config["multiwoz"]["commit"] == "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b"
    assert config["slots"] == ["food", "area", "pricerange"]
    assert config["stated_intention"] == "find_restaurant"
    assert config["prompt_versions"] == ["multiwoz_likely", "multiwoz_typical"]
    assert "jev-" not in json.dumps(config)
    assert jev_model_id(root) == "jev-1.13.0"
