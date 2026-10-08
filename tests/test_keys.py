import json
from pathlib import Path

import pytest

from robust_likelihood.keys import (
    MissingAPIKey,
    UnrecognizedKeyLabels,
    load_keys,
    parse_key_file,
)


def test_environment_keys_do_not_need_the_file(tmp_path):
    keys = load_keys(
        ("jev", "gpt"),
        environ={"JEV_API_KEY": "jev-from-env", "OPENAI_API_KEY": "gpt-from-env"},
        key_file=tmp_path / "missing.txt",
    )
    assert keys.jev == "jev-from-env"
    assert keys.gpt == "gpt-from-env"
    assert "jev-from-env" not in repr(keys)


def test_file_fallback_uses_known_labels(tmp_path):
    path = tmp_path / "API_key.txt"
    path.write_text(
        "Jev api:\njev-from-file\n\nClaude api key :\nclaude-from-file\n\nOpen api key:\ngpt-from-file\n",
        encoding="utf-8",
    )
    keys = load_keys(("jev", "gpt", "claude"), environ={}, key_file=path)
    assert keys.jev == "jev-from-file"
    assert keys.gpt == "gpt-from-file"
    assert keys.claude == "claude-from-file"


def test_unknown_label_reports_the_name_only(tmp_path):
    path = tmp_path / "API_key.txt"
    path.write_text("Mystery token:\nsuper-secret-value\n", encoding="utf-8")
    with pytest.raises(UnrecognizedKeyLabels) as caught:
        parse_key_file(path.read_text(encoding="utf-8"))
    message = str(caught.value)
    assert "Mystery token" in message
    assert "super-secret-value" not in message


def test_typesafe_env_name_fills_jev():
    keys = load_keys(("jev",), environ={"TYPESAFE_API_KEY": "typesafe-value"})
    assert keys.require("jev") == "typesafe-value"


def test_decision_api_uses_the_openai_key(tmp_path):
    path = tmp_path / "API_key.txt"
    path.write_text("Open api key:\ngpt-from-file\n", encoding="utf-8")
    keys = load_keys(("gpt", "gpt_decision"), environ={}, key_file=path)
    assert keys.require("gpt_decision") == "gpt-from-file"
    assert "gpt-from-file" not in repr(keys)


def test_missing_key_names_the_provider():
    with pytest.raises(MissingAPIKey, match="gpt"):
        load_keys(("gpt",), environ={}, key_file=Path("does-not-exist.txt"))


def test_loading_keys_does_not_call_the_network(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    path = tmp_path / "API_key.txt"
    path.write_text("Open api key:\ngpt-local\n", encoding="utf-8")
    load_keys(("gpt",), environ={}, key_file=path)
