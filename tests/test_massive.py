import hashlib

import pytest

from robust_likelihood.massive import canonical_line, dataset_sha256, sha256_lines
from robust_likelihood.storage import read_json, repo_root


def test_canonical_jsonl_hash_includes_sorted_keys_and_newlines():
    line = canonical_line({"b": 1, "a": 2})
    assert line == '{"a":2,"b":1}'
    assert sha256_lines([[line]]) == hashlib.sha256(b'{"a":2,"b":1}\n').hexdigest()


def test_published_checksum_matches_local_splits():
    directory = repo_root() / "data" / "massive" / "en-US"
    checksum = read_json(directory / "checksum.json")
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            pytest.skip("MASSIVE split is not hydrated")
    assert dataset_sha256(directory) == checksum["sha256"]
