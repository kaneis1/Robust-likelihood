"""python -m robust_likelihood.multiwoz download"""

from __future__ import annotations

import argparse
import json
import sys

from robust_likelihood.multiwoz import load_config
from robust_likelihood.multiwoz.download import EXPECTED_DIALOGUES, ChecksumMismatch, download
from robust_likelihood.storage import repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robust_likelihood.multiwoz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ChecksumMismatch, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    root = repo_root()
    config = load_config(root)
    if args.command == "download":
        summary = download(root, config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["dialogues"] != EXPECTED_DIALOGUES:
            print(f"Dialogue counts differ from Appendix A: expected {EXPECTED_DIALOGUES}", file=sys.stderr)
            return 1
        return 0
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
