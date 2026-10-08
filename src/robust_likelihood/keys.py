import os
from dataclasses import dataclass
from pathlib import Path


class UnrecognizedKeyLabels(Exception):
    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        super().__init__("Unrecognized key labels: " + ", ".join(self.labels))


class MissingAPIKey(Exception):
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Missing API key for {provider}")


_LABELS = {
    "jev api": "jev",
    "jev": "jev",
    "jev api key": "jev",
    "typesafe": "jev",
    "typesafe api": "jev",
    "typesafe api key": "jev",
    "open api key": "gpt",
    "open api": "gpt",
    "openai": "gpt",
    "openai api key": "gpt",
    "claude api key": "claude",
    "claude": "claude",
    "anthropic": "claude",
    "anthropic api key": "claude",
}

_ENV = {
    "jev": ("JEV_API_KEY", "TYPESAFE_API_KEY"),
    "gpt": ("OPENAI_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
}


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


@dataclass
class APIKeys:
    jev: str | None = None
    gpt: str | None = None
    claude: str | None = None

    def __repr__(self) -> str:
        flags = {
            "jev": self.jev is not None,
            "gpt": self.gpt is not None,
            "claude": self.claude is not None,
        }
        return f"APIKeys({flags})"

    def require(self, provider: str) -> str:
        value = getattr(self, _key_slot(provider))
        if not value:
            raise MissingAPIKey(provider)
        return value


def parse_key_file(text: str) -> dict[str, str]:
    """Parse a local key file. Values are never included in errors."""
    found: dict[str, str] = {}
    unknown: list[str] = []
    seen_labels: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        index += 1
        if not raw:
            continue
        if ":" not in raw:
            unknown.append(raw)
            continue
        label, _, rest = raw.partition(":")
        label_name = label.strip()
        value = rest.strip()
        if not value:
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index >= len(lines):
                unknown.append(label_name)
                continue
            value = lines[index].strip()
            index += 1
        provider = _LABELS.get(_normalize_label(label_name))
        if provider is None:
            unknown.append(label_name)
            continue
        if provider in seen_labels:
            unknown.append(label_name)
            continue
        seen_labels[provider] = label_name
        found[provider] = value
    if unknown:
        raise UnrecognizedKeyLabels(unknown)
    return found


def default_key_file(root: Path) -> Path | None:
    for name in ("API_key.txt", "Api_key.txt"):
        path = root / name
        if path.is_file():
            return path
    return None


def _key_slot(provider: str) -> str:
    if provider == "gpt_decision":
        return "gpt"
    return provider


def load_keys(
    providers: tuple[str, ...] | list[str],
    *,
    environ: dict[str, str] | None = None,
    root: Path | None = None,
    key_file: Path | None = None,
) -> APIKeys:
    """Load keys from the environment, then from a local file.

    Reading this file does not call the network. Key values are not logged.
    """
    env = environ if environ is not None else os.environ
    keys = APIKeys()
    missing: list[str] = []
    for provider in providers:
        slot = _key_slot(provider)
        if getattr(keys, slot):
            continue
        for name in _ENV[slot]:
            value = env.get(name, "").strip()
            if value:
                setattr(keys, slot, value)
                break
        else:
            missing.append(provider)
    if not missing:
        return keys
    path = key_file
    if path is None and root is not None:
        path = default_key_file(root)
    if path is None or not path.is_file():
        raise MissingAPIKey(missing[0])
    parsed = parse_key_file(path.read_text(encoding="utf-8"))
    for provider in missing:
        slot = _key_slot(provider)
        if getattr(keys, slot):
            continue
        value = parsed.get(slot, "").strip()
        if not value:
            raise MissingAPIKey(provider)
        setattr(keys, slot, value)
    return keys
