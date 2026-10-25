"""Pinned MultiWOZ 2.1 goals and 2.2 turn annotations.

Files are fetched by commit from raw.githubusercontent.com with urllib and never
cloned, so no checkout setting can rewrite their line endings. The 2.1 zip must
match its pinned sha256. The 21 2.2 dialogue files are checked by one combined
digest: sort their paths relative to MultiWOZ_2.2/, concatenate their lowercase
hex sha256 strings with no separator, and hash that ASCII string.

checksum.json records every file's sha256. A rerun keeps each file whose hash
already matches and fetches only what is missing or different.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.massive import sha256_file
from robust_likelihood.multiwoz import CHECKSUM_FILE, RAW_DIR
from robust_likelihood.storage import read_json, write_json

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
ZIP_SOURCE = "data/MultiWOZ_2.1.zip"
ZIP_NAME = "MultiWOZ_2.1.zip"
V21_DIR = "MultiWOZ_2.1"
V21_MEMBERS = ("data.json", "ontology.json", "valListFile.txt", "testListFile.txt")
V22_SOURCE = "data/MultiWOZ_2.2"
V22_DIR = "MultiWOZ_2.2"
V22_DIALOGUES = (
    tuple(f"dev/dialogues_{index:03d}.json" for index in (1, 2))
    + tuple(f"test/dialogues_{index:03d}.json" for index in (1, 2))
    + tuple(f"train/dialogues_{index:03d}.json" for index in range(1, 18))
)
V22_OTHER = ("schema.json",)
COMBINED_RULE = (
    "sha256 of the lowercase hex sha256 strings of the 21 dialogues_*.json files, "
    "concatenated with no separator in sorted order of their paths relative to MultiWOZ_2.2/"
)
EXPECTED_DIALOGUES = {"v21": 10438, "v22": 10437}


class ChecksumMismatch(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combined_digest(hashes: dict[str, str]) -> str:
    joined = "".join(hashes[path].lower() for path in sorted(hashes))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _matches(path: Path, expected: str | None) -> bool:
    return expected is not None and path.is_file() and sha256_file(path) == expected


def _install(path: Path, data: bytes, expected: str | None) -> str:
    """Check the bytes, then write a temporary file and rename it into place."""
    digest = sha256_bytes(data)
    if expected is not None and digest != expected:
        raise ChecksumMismatch(f"{path.name} sha256 {digest} does not match {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)
    return digest


def _extract(archive: Path, member: str, path: Path, expected: str | None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, partial.open("wb") as sink:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            sink.write(chunk)
    found = digest.hexdigest()
    if expected is not None and found != expected:
        partial.unlink()
        raise ChecksumMismatch(f"{member} sha256 {found} does not match {expected}")
    partial.replace(path)
    return found


def count_dialogues(raw: Path) -> dict[str, int]:
    v21 = json.loads((raw / V21_DIR / "data.json").read_text(encoding="utf-8"))
    v22 = 0
    for relative in V22_DIALOGUES:
        v22 += len(json.loads((raw / V22_DIR / relative).read_text(encoding="utf-8")))
    return {"v21": len(v21), "v22": v22}


def download(root: Path, config: dict, *, fetcher=fetch, now: datetime | None = None) -> dict:
    pinned = config["multiwoz"]
    raw = root / RAW_DIR
    checksum_path = root / CHECKSUM_FILE
    previous = read_json(checksum_path) if checksum_path.is_file() else {}
    if previous.get("commit") != pinned["commit"]:
        previous = {}
    known = previous.get("files", {})

    def url(path: str) -> str:
        return RAW_URL.format(repo=pinned["repo"], commit=pinned["commit"], path=path)

    hashes: dict[str, str] = {}
    fetched: list[str] = []
    archive = raw / ZIP_NAME
    if not _matches(archive, pinned["zip_sha256"]):
        _install(archive, fetcher(url(ZIP_SOURCE)), pinned["zip_sha256"])
        fetched.append(ZIP_NAME)
    hashes[ZIP_NAME] = pinned["zip_sha256"]
    for member in V21_MEMBERS:
        name = f"{V21_DIR}/{member}"
        if _matches(raw / name, known.get(name)):
            hashes[name] = known[name]
            continue
        hashes[name] = _extract(archive, name, raw / name, known.get(name))
        fetched.append(name)
    for relative in V22_DIALOGUES + V22_OTHER:
        name = f"{V22_DIR}/{relative}"
        if _matches(raw / name, known.get(name)):
            hashes[name] = known[name]
            continue
        hashes[name] = _install(raw / name, fetcher(url(f"{V22_SOURCE}/{relative}")), known.get(name))
        fetched.append(name)
    combined = combined_digest({relative: hashes[f"{V22_DIR}/{relative}"] for relative in V22_DIALOGUES})
    if combined != pinned["v22_dialogues_combined_sha256"]:
        raise ChecksumMismatch(
            f"MultiWOZ 2.2 combined sha256 {combined} does not match {pinned['v22_dialogues_combined_sha256']}"
        )
    unchanged = not fetched and previous.get("files") == hashes
    dialogues = previous.get("dialogues") if unchanged else count_dialogues(raw)
    payload = {
        "repo": pinned["repo"],
        "commit": pinned["commit"],
        "source_url": url(""),
        "files": hashes,
        "zip_sha256": pinned["zip_sha256"],
        "v22_dialogues_combined_sha256": combined,
        "combined_digest_rule": COMBINED_RULE,
        "dialogues": dialogues,
    }
    if {key: previous.get(key) for key in payload} == payload:
        payload["written_at"] = previous["written_at"]
    else:
        payload["written_at"] = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(checksum_path, payload)
    return {
        "fetched": fetched,
        "dialogues": dialogues,
        "v22_dialogues_combined_sha256": combined,
        "checksum": str(checksum_path),
        "verified": True,
    }
