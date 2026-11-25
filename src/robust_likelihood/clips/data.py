"""The instructions of CLIPS's two datasets, read the way CLIPS.jl's plan_io.jl reads them (stdlib only).

A plan file holds one PDDL action per line; a line starting with ';' is an utterance, said after the
actions before it (its time step is the number of actions so far). mDKG plan ids are the file names
("1.1.keys"); the true goal is the gem the completed plan picks up last. VirtualHome plan ids are the
first two fields ("1.1" from "1.1.stew_chicken.pddl"), and the third field is the true goal's label.
mDKG also has the simplified utterances its literal listener reads (dataset/utterances/literal).
"""

from __future__ import annotations

import re
from pathlib import Path

MDKG = "doors-keys-gems"
VH = "virtual-home"
MDKG_PATTERN = re.compile(r"^(\d+)\w?\.(\d+)\..*")
VH_PATTERN = re.compile(r"^(\d+)\.(\d+)\.([\w\d]+)")


def parse_plan(text: str) -> tuple[list[str], list[str], list[int]]:
    actions, annotations, times = [], [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0] == ";":
            annotations.append(line[1:].strip())
            times.append(len(actions))
        else:
            actions.append(line)
    return actions, annotations, times


def mdkg_sort_key(name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in MDKG_PATTERN.match(name).groups())


def utterance_rows(clips_root: Path) -> list[dict]:
    """One row per instruction: domain, plan id, problem, index, time step, text, true goal."""
    rows = []
    folder = clips_root / MDKG / "dataset"
    names = sorted((path.stem for path in (folder / "plans" / "observed").glob("*.pddl")
                    if MDKG_PATTERN.match(path.stem)), key=mdkg_sort_key)
    for name in names:
        actions, annotations, times = parse_plan((folder / "plans" / "observed" / f"{name}.pddl").read_text(encoding="utf-8"))
        completed, _, _ = parse_plan((folder / "plans" / "completed" / f"{name}.pddl").read_text(encoding="utf-8"))
        last = re.fullmatch(r"\(pickup (\w+) (\w+)\)", completed[-1])
        if last is None:
            raise ValueError(f"{name}: the completed plan does not end in a pickup")
        literal_path = folder / "utterances" / "literal" / f"{name}.txt"
        literal = literal_path.read_text(encoding="utf-8").strip() if literal_path.exists() else None
        for index, (text, time) in enumerate(zip(annotations, times)):
            rows.append({"domain": "mdkg", "plan_id": name, "problem_id": re.match(r"(\d+\w?)", name).group(1),
                         "index": index, "timestep": time, "observed_actions": len(actions), "text": text,
                         "literal_text": literal, "true_goal": f"(has {last.group(1)} {last.group(2)})"})
    folder = clips_root / VH / "dataset"
    found = []
    for path in (folder / "plans" / "observed").glob("*.pddl"):
        match = VH_PATTERN.match(path.name)
        if match:
            found.append((f"{match.group(1)}.{match.group(2)}", match.group(3), path))
    for name, goal, path in sorted(found, key=lambda entry: tuple(int(part) for part in entry[0].split("."))):
        actions, annotations, times = parse_plan(path.read_text(encoding="utf-8"))
        for index, (text, time) in enumerate(zip(annotations, times)):
            rows.append({"domain": "vh", "plan_id": name, "problem_id": name.split(".")[0], "index": index,
                         "timestep": time, "observed_actions": len(actions), "text": text, "literal_text": None,
                         "true_goal": goal})
    return rows
