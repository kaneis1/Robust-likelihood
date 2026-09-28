"""English MASSIVE snapshot.

The raw jsonl files are gitignored. checksum.json records the archive hash and the
canonical dataset hash: each JSON object re-serialized with sort_keys, compact
separators, and UTF-8, concatenated in train / validation / test order with newlines.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from robust_likelihood.storage import read_json

ARCHIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SPLIT_NAMES = ("train", "validation", "test")
PARTITION_TO_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


class DatasetDrift(RuntimeError):
    pass


def canonical_line(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(groups: list[list[str]]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        for line in group:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def read_canonical_lines(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lines.append(canonical_line(json.loads(line)))
    return lines


def dataset_sha256(directory: Path) -> str:
    groups = [read_canonical_lines(directory / f"{name}.jsonl") for name in SPLIT_NAMES]
    return sha256_lines(groups)


def _member_name(locale: str) -> str:
    return f"1.1/data/{locale}.jsonl"


def extract_splits(archive: Path, locale: str) -> dict[str, list[str]]:
    member_name = _member_name(locale)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile(member_name)
        if member is None:
            names = [item.name for item in tar.getmembers() if item.name.endswith(f"{locale}.jsonl")]
            raise FileNotFoundError(f"Missing {member_name}; found {names}")
        raw = member.read()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        split = PARTITION_TO_SPLIT.get(obj.get("partition"))
        if split is None:
            raise ValueError(f"Unexpected partition {obj.get('partition')!r}")
        splits[split].append(canonical_line(obj))
    return splits


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def ensure_archive(archive: Path, expected_sha: str, url: str = ARCHIVE_URL) -> None:
    if archive.exists() and sha256_file(archive) == expected_sha:
        return
    _download(url, archive)
    digest = sha256_file(archive)
    if digest != expected_sha:
        raise RuntimeError(f"Archive sha256 {digest} does not match {expected_sha}")


def install_splits(directory: Path, splits: dict[str, list[str]]) -> list[str]:
    written = []
    for name in SPLIT_NAMES:
        path = directory / f"{name}.jsonl"
        current = read_canonical_lines(path)
        if current:
            if current != splits[name]:
                raise DatasetDrift(f"{name}.jsonl does not match the official {name} split")
            continue
        text = "".join(line + "\n" for line in splits[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        written.append(name)
    return written


def verify_directory(directory: Path, checksum: dict) -> None:
    digest = dataset_sha256(directory)
    expected = checksum["sha256"]
    if digest != expected:
        raise RuntimeError(f"Dataset sha256 {digest} does not match {expected}")
    rows = checksum.get("rows", {})
    for name in SPLIT_NAMES:
        actual = len(read_canonical_lines(directory / f"{name}.jsonl"))
        if name in rows and actual != rows[name]:
            raise RuntimeError(f"{name} has {actual} rows, checksum expects {rows[name]}")


def download(root: Path, locale: str = "en-US") -> dict:
    directory = root / "data" / "massive" / locale
    directory.mkdir(parents=True, exist_ok=True)
    checksum_path = directory / "checksum.json"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing {checksum_path}")
    checksum = read_json(checksum_path)
    archive = root / "data" / "massive" / "amazon-massive-dataset-1.1.tar.gz"
    ensure_archive(archive, checksum["archive_sha256"])
    splits = extract_splits(archive, locale)
    written = install_splits(directory, splits)
    verify_directory(directory, checksum)
    return {
        "locale": locale,
        "wrote": written,
        "rows": {name: len(splits[name]) for name in SPLIT_NAMES},
        "sha256": checksum["sha256"],
        "verified": True,
    }
