import json
import os
import time
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("ROBUST_LIKELIHOOD_ROOT")
    if override:
        return Path(override)
    start = Path(__file__).resolve()
    for candidate in start.parents:
        if (candidate / "configs" / "models.json").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repository root")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    delay = 0.25
    for attempt in range(12):
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
