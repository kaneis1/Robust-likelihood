import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import pytest

from robust_likelihood.multiwoz.download import (
    V21_MEMBERS,
    V22_DIALOGUES,
    ChecksumMismatch,
    combined_digest,
    download,
    sha256_bytes,
)
from robust_likelihood.storage import read_json

COMMIT = "0" * 40


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("MultiWOZ_2.1/data.json", json.dumps({"SNG0001.json": {}, "MUL0002.json": {}}))
        for member in V21_MEMBERS[1:]:
            bundle.writestr(f"MultiWOZ_2.1/{member}", f"{member}\n")
        bundle.writestr("__MACOSX/MultiWOZ_2.1/._data.json", "resource fork")
    return buffer.getvalue()


def _v22_bytes(relative: str) -> bytes:
    return json.dumps([{"dialogue_id": relative}]).encode("utf-8")


class FakeFetcher:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        prefix = f"https://raw.githubusercontent.com/example/multiwoz/{COMMIT}/"
        assert url.startswith(prefix)
        return self.files[url[len(prefix):]]


def _setup():
    files = {"data/MultiWOZ_2.1.zip": _zip_bytes(), "data/MultiWOZ_2.2/schema.json": b"[]"}
    for relative in V22_DIALOGUES:
        files[f"data/MultiWOZ_2.2/{relative}"] = _v22_bytes(relative)
    combined = combined_digest({relative: sha256_bytes(_v22_bytes(relative)) for relative in V22_DIALOGUES})
    config = {
        "multiwoz": {
            "repo": "example/multiwoz",
            "commit": COMMIT,
            "zip_sha256": sha256_bytes(files["data/MultiWOZ_2.1.zip"]),
            "v22_dialogues_combined_sha256": combined,
        }
    }
    return files, config


def test_combined_digest_hashes_sorted_hex_strings():
    hashes = {"train/b.json": "BB", "dev/a.json": "aa", "test/c.json": "cc"}
    expected = hashlib.sha256(b"aaccbb").hexdigest()
    assert combined_digest(hashes) == expected
    assert sorted(V22_DIALOGUES)[0] == "dev/dialogues_001.json"
    assert len(V22_DIALOGUES) == 21


def test_download_verifies_and_is_idempotent(tmp_path):
    files, config = _setup()
    fetcher = FakeFetcher(files)
    fixed = datetime(2026, 10, 26, tzinfo=timezone.utc)
    summary = download(tmp_path, config, fetcher=fetcher, now=fixed)
    assert len(fetcher.urls) == 23
    assert summary["dialogues"] == {"v21": 2, "v22": 21}
    raw = tmp_path / "data" / "multiwoz" / "raw"
    assert (raw / "MultiWOZ_2.1" / "data.json").is_file()
    assert not (raw / "MultiWOZ_2.1" / "._data.json").exists()
    assert not list(raw.rglob("*.part"))
    checksum = read_json(tmp_path / "data" / "multiwoz" / "checksum.json")
    assert checksum["commit"] == COMMIT
    assert checksum["v22_dialogues_combined_sha256"] == config["multiwoz"]["v22_dialogues_combined_sha256"]
    assert checksum["files"]["MultiWOZ_2.2/schema.json"] == sha256_bytes(b"[]")
    assert checksum["written_at"] == "2026-10-26T00:00:00Z"

    again = FakeFetcher(files)
    later = datetime(2026, 10, 27, tzinfo=timezone.utc)
    assert download(tmp_path, config, fetcher=again, now=later)["fetched"] == []
    assert again.urls == []
    assert read_json(tmp_path / "data" / "multiwoz" / "checksum.json")["written_at"] == "2026-10-26T00:00:00Z"

    (raw / "MultiWOZ_2.2" / "dev" / "dialogues_002.json").write_bytes(b"tampered")
    repair = FakeFetcher(files)
    assert download(tmp_path, config, fetcher=repair, now=later)["fetched"] == ["MultiWOZ_2.2/dev/dialogues_002.json"]
    assert len(repair.urls) == 1


def test_wrong_zip_hash_leaves_nothing_in_place(tmp_path):
    files, config = _setup()
    config["multiwoz"]["zip_sha256"] = "f" * 64
    with pytest.raises(ChecksumMismatch, match="MultiWOZ_2.1.zip"):
        download(tmp_path, config, fetcher=FakeFetcher(files))
    raw = tmp_path / "data" / "multiwoz" / "raw"
    assert not (raw / "MultiWOZ_2.1.zip").exists()
    assert not (tmp_path / "data" / "multiwoz" / "checksum.json").exists()


def test_wrong_combined_digest_writes_no_checksum(tmp_path):
    files, config = _setup()
    files["data/MultiWOZ_2.2/train/dialogues_017.json"] = b"[]"
    with pytest.raises(ChecksumMismatch, match="combined"):
        download(tmp_path, config, fetcher=FakeFetcher(files))
    assert not (tmp_path / "data" / "multiwoz" / "checksum.json").exists()
