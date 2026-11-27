"""Phase 6 of docs/clips_small_lm_plan.md (optional): the instruction read without CLIPS's planner.

The language model scores the instruction given the ASCII map of the state in which it was spoken and a
text description of the goal. There are no commands and no planner, and only mDKG is used:

    P(g | u, map) ∝ P(u | map, "Goal: the human wants gem g.")      (uniform prior over the four gems)

**States.** Each state is the problem's ASCII grid (doors-keys-gems/dataset/problems/<id>.txt, read the
way ascii.jl's ascii_to_pddl reads it) after the observed actions that come before the instruction.
Those actions are simulated with the domain's effects. Every action is checked against its
precondition, and the grid parse is checked against the problem's .pddl file. ascii.jl only parses
grids, so the map is drawn here:
- walls, keys, locked doors and agents use ascii.jl's letters;
- gem k (the k-th gem in reading order, as ascii_to_pddl names them) is drawn as the digit k, so
  "gem 2" in the goal points to a cell of the map;
- a legend line names the letters;
- a line says what each agent holds.

**Scores.** The four goal prompts of one instruction differ only in the goal's digit and are scored as one
batch (batch layout alone moves a bf16 score by up to about 0.5 nats; docs/clips_decisions.md). The
score is Phase 2's: log P(" " + instruction + "\\n" | prompt), with no chat template. A layout check
scores every sequence again, in reversed order with batch size 7.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

MDKG = Path("external/CLIPS.jl/doors-keys-gems/dataset")
OUT = Path("data/clips/derived/noplanner")
SCORES = Path("data/clips/derived/scores_noplanner")
KEY_COLOURS = {"r": "red", "b": "blue", "y": "yellow", "e": "green", "p": "pink"}  # ascii.jl's key_dict
DOOR_COLOURS = {letter.upper(): colour for letter, colour in KEY_COLOURS.items()}  # ascii.jl's door_dict
COLOUR_LETTERS = {colour: letter for letter, colour in KEY_COLOURS.items()}
GEM_COLOURS = ("red", "yellow", "blue", "green")  # ascii.jl's gem_colors, by gem number
LEGEND = ("Legend: W wall, . floor, h human, m robot, r y b e p keys (red, yellow, blue, green, pink), "
          "R Y B E P locked doors of those colours, 1 2 3 4 gems.")
GOALS = 4
LAYOUT = "one batch per instruction: its four goal prompts"


@dataclass
class Grid:
    width: int
    height: int
    walls: set[tuple[int, int]]
    objects: dict[str, dict]  # name -> {"kind", "colour", "x", "y"}; x, y = -1 once picked up
    agents: dict[str, tuple[int, int]]
    locked: set[str]
    held: dict[str, list[str]] = field(default_factory=lambda: {"human": [], "robot": []})


def parse_ascii(text: str) -> Grid:
    """ascii_to_pddl's reading: rows split on newlines (empty ones dropped), each stripped; objects named
    by kind in reading order (door1, key1, gem1, ...)."""
    rows = [row.strip() for row in text.replace("\r\n", "\n").split("\n") if row != ""]
    walls, objects, agents, locked = set(), {}, {}, set()
    counts = {"door": 0, "key": 0, "gem": 0}
    for y, row in enumerate(rows, start=1):
        for x, char in enumerate(row, start=1):
            if char == ".":
                continue
            if char == "W":
                walls.add((x, y))
            elif char in DOOR_COLOURS or char in KEY_COLOURS or char in "gG":
                kind = "door" if char in DOOR_COLOURS else "key" if char in KEY_COLOURS else "gem"
                counts[kind] += 1
                name = f"{kind}{counts[kind]}"
                colour = (DOOR_COLOURS.get(char) or KEY_COLOURS.get(char) or GEM_COLOURS[counts[kind] - 1])
                objects[name] = {"kind": kind, "colour": colour, "x": x, "y": y}
                if kind == "door":
                    locked.add(name)
            elif char == "h":
                agents["human"] = (x, y)
            elif char == "m":
                agents["robot"] = (x, y)
            else:
                raise ValueError(f"unknown map character {char!r} at ({x}, {y})")
    return Grid(max(len(row) for row in rows), len(rows), walls, objects, agents, locked)


def check_against_pddl(grid: Grid, pddl: str) -> None:
    """The grid must give the problem file's walls, objects, colours, locks and agent positions."""
    size = re.search(r"new-bit-matrix false (\d+) (\d+)", pddl)
    if (int(size.group(1)), int(size.group(2))) != (grid.height, grid.width):
        raise ValueError(f"size {size.groups()} differs from the grid's {(grid.height, grid.width)}")
    walls = {(int(x), int(y)) for y, x in re.findall(r"set-index walls true (\d+) (\d+)", pddl)}
    if walls != grid.walls:
        raise ValueError(f"walls differ: {sorted(walls ^ grid.walls)}")
    locations: dict[str, dict[str, int]] = {}
    for axis, name, value in re.findall(r"\(= \((x|y)loc (\w+)\) (\d+)\)", pddl):
        locations.setdefault(name, {})[axis] = int(value)
    colours = dict(re.findall(r"\(iscolor (\w+) (\w+)\)", pddl))
    expected = {name: (spec["x"], spec["y"]) for name, spec in grid.objects.items()}
    expected.update(grid.agents)
    found = {name: (place["x"], place["y"]) for name, place in locations.items()}
    if found != expected:
        raise ValueError(f"positions differ: {sorted(set(found.items()) ^ set(expected.items()))}")
    if colours != {name: spec["colour"] for name, spec in grid.objects.items()}:
        raise ValueError("colours differ")
    if set(re.findall(r"\(locked (\w+)\)", pddl)) != grid.locked:
        raise ValueError("locked doors differ")


def blocked(grid: Grid, x: int, y: int) -> bool:
    if not (1 <= x <= grid.width and 1 <= y <= grid.height) or (x, y) in grid.walls:
        return True
    return any(grid.objects[door]["x"] == x and grid.objects[door]["y"] == y for door in grid.locked)


def adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def apply(grid: Grid, action: str) -> None:
    """One action of the domain, its precondition checked (turn-taking is not tracked)."""
    name, *args = action.strip("()").split()
    agent = args[0]
    x, y = grid.agents[agent]
    moves = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    if name in moves:
        dx, dy = moves[name]
        if blocked(grid, x + dx, y + dy):
            raise ValueError(f"{action}: ({x + dx}, {y + dy}) is blocked")
        grid.agents[agent] = (x + dx, y + dy)
    elif name == "wait":
        return
    elif name == "pickup":
        item = grid.objects[args[1]]
        if (item["x"], item["y"]) != (x, y):
            raise ValueError(f"{action}: the item is not under the agent")
        item["x"] = item["y"] = -1
        grid.held[agent].append(args[1])
    elif name == "unlock":
        key, door = args[1], args[2]
        target = grid.objects[door]
        if key not in grid.held[agent] or door not in grid.locked or not adjacent((x, y), (target["x"], target["y"])) \
                or grid.objects[key]["colour"] != target["colour"]:
            raise ValueError(f"{action}: precondition fails")
        grid.held[agent].remove(key)
        grid.locked.discard(door)
    elif name == "handover":
        other, item = args[1], args[2]
        if item not in grid.held[agent] or not adjacent((x, y), grid.agents[other]):
            raise ValueError(f"{action}: precondition fails")
        grid.held[agent].remove(item)
        grid.held[other].append(item)
    else:
        raise ValueError(f"unknown action {action}")


def describe(grid: Grid, name: str) -> str:
    spec = grid.objects[name]
    return f"gem {name[3:]}" if spec["kind"] == "gem" else f"the {spec['colour']} {spec['kind']}"


def render(grid: Grid) -> tuple[list[str], str]:
    """The map's rows (unlocked doors drawn as floor) and the line of what each agent holds."""
    cells = [["." for _ in range(grid.width)] for _ in range(grid.height)]
    for x, y in grid.walls:
        cells[y - 1][x - 1] = "W"
    for name, spec in grid.objects.items():
        if spec["x"] < 0 or (spec["kind"] == "door" and name not in grid.locked):
            continue
        letter = {"door": COLOUR_LETTERS.get(spec["colour"], "?").upper(), "key": COLOUR_LETTERS.get(spec["colour"]),
                  "gem": name[3:]}[spec["kind"]]
        cells[spec["y"] - 1][spec["x"] - 1] = letter
    for agent, letter in (("human", "h"), ("robot", "m")):
        x, y = grid.agents[agent]
        cells[y - 1][x - 1] = letter
    holds = " ".join(f"The {agent} has {' and '.join(describe(grid, item) for item in grid.held[agent]) or 'nothing'}."
                     for agent in ("human", "robot"))
    return ["".join(row) for row in cells], holds


def observed_actions(text: str) -> tuple[list[str], list[int]]:
    """A plan file's actions and, per annotation, how many actions come before it (plan_io.jl's parse_plan)."""
    actions, marks = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            marks.append(len(actions))
        else:
            actions.append(line)
    return actions, marks


def prompt_text(rows: list[str], holds: str, goal: int) -> str:
    return "\n".join([LEGEND, "Map:", *rows, holds, f"Goal: the human wants gem {goal}.", "Instruction:"])


def build(root: Path, out: Path | None = None) -> dict:
    """states.jsonl, prompts.jsonl and pairs.jsonl for every mDKG instruction (in out, default OUT)."""
    from robust_likelihood.clips.record import prompt_id

    utterances = [row for row in read_jsonl(root / "data" / "clips" / "derived" / "utterances.jsonl")
                  if row["domain"] == "mdkg"]
    states, prompts, pairs = [], {}, []
    for row in utterances:
        problem = row["problem_id"]
        grid = parse_ascii((root / MDKG / "problems" / f"{problem}.txt").read_text(encoding="utf-8"))
        check_against_pddl(grid, (root / MDKG / "problems" / f"{problem}.pddl").read_text(encoding="utf-8"))
        actions, marks = observed_actions((root / MDKG / "plans" / "observed" / f"{row['plan_id']}.pddl")
                                          .read_text(encoding="utf-8"))
        if marks[row["index"]] != row["observed_actions"]:
            raise ValueError(f"{row['plan_id']}: the instruction comes after {marks[row['index']]} actions, "
                             f"not {row['observed_actions']}")
        for action in actions[:row["observed_actions"]]:
            apply(grid, action)
        lines, holds = render(grid)
        true_goal = int(re.fullmatch(r"\(has human gem(\d)\)", row["true_goal"]).group(1))
        output = " " + row["text"].strip()
        ids = []
        for goal in range(1, GOALS + 1):
            text = prompt_text(lines, holds, goal)
            ids.append(prompt_id(text))
            prompts[ids[-1]] = text
            pairs.append({"role": "noplanner", "plan_id": row["plan_id"], "goal": goal, "prompt_id": ids[-1],
                          "output": output})
        states.append({"plan_id": row["plan_id"], "problem_id": problem, "observed_actions": row["observed_actions"],
                       "map": lines, "holds": holds, "true_goal": true_goal, "text": row["text"],
                       "prompt_ids": ids})
    out = out or root / OUT
    write_jsonl(out / "states.jsonl", states)
    write_jsonl(out / "prompts.jsonl", [{"prompt_id": key, "prompt": text} for key, text in sorted(prompts.items())])
    write_jsonl(out / "pairs.jsonl", pairs)
    return {"instructions": len(states), "prompts": len(prompts), "pairs": len(pairs),
            "holding": sum("has nothing. The robot has nothing." not in state["holds"] for state in states)}


def score_noplanner(root: Path, key: str, *, device: str | None = None, tokenizer=None, model=None) -> dict:
    """GPU: every instruction's four goal prompts as one batch, then the layout check."""
    import torch
    import transformers

    from robust_likelihood.clips.score import encode, load_clips_config
    from robust_likelihood.lmcore import load_model, pick_device, surprisals
    from robust_likelihood.multiwoz.jev_run import git_state

    config = load_clips_config(root)
    spec, settings = config["models"][key], config["score"]
    clock, started = time.monotonic(), datetime.now(timezone.utc)
    prompts = {row["prompt_id"]: row["prompt"] for row in read_jsonl(root / OUT / "prompts.jsonl")}
    states = read_jsonl(root / OUT / "states.jsonl")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if model is None:
        model = load_model(spec["id"], spec["revision"], settings["dtype"], pick_device(device))
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad = pad if pad is not None else 0
    found, sequences, uneven = [], [], 0
    for state in states:
        output = " " + state["text"].strip()
        encoded = [encode(tokenizer, prompts[found_id], output, int(settings["max_tokens"]))
                   for found_id in state["prompt_ids"]]
        if any(item is None for item in encoded):
            raise ValueError(f"{state['plan_id']}: the instruction is longer than max_tokens")
        uneven += len({len(item[0]) for item in encoded}) > 1
        values = surprisals(model, encoded, len(encoded), pad)
        found.append({"plan_id": state["plan_id"], "prompt_ids": state["prompt_ids"], "output": output,
                      "scores": [-value for value in values]})
        sequences += encoded
    again = surprisals(model, list(reversed(sequences)), 7, pad)
    first = [score for row in found for score in row["scores"]]
    gaps = [abs(-value - score) for value, score in zip(again, reversed(first))]
    write_jsonl(root / SCORES / f"{key}.jsonl", found)
    manifest = {"kind": "clips-score-noplanner", "model_key": key, "model": spec["id"], "revision": spec["revision"],
                "dtype": settings["dtype"], "device": str(next(model.parameters()).device), "git": git_state(root),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                "instructions": len(found), "sequences": len(sequences), "layout": LAYOUT,
                "instructions_with_uneven_prompt_lengths": uneven,
                "layout_check": {"layout": "reversed order, batch size 7", "max_abs_difference": max(gaps),
                                 "median_abs_difference": statistics.median(gaps)},
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_seconds": round(time.monotonic() - clock, 1)}
    write_json(root / SCORES / f"{key}.manifest.json", manifest)
    return {"model": key, "instructions": len(found), "uneven": uneven, "layout_max_abs": max(gaps),
            "elapsed_seconds": manifest["elapsed_seconds"]}


def posterior(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def run_noplanner(root: Path, models: list[str], out: Path) -> dict:
    """metrics_phase6.json: per model, P(g_true) and goal correlation without the planner, against
    language-only IP and CLIPS from the replays (paired over problems)."""
    from robust_likelihood.clips.evaluate import (ProblemBoot, fnum, mean_block, method_rows, pearson, read_humans,
                                                  run_rows)

    clock = time.monotonic()
    config = read_json(root / "configs" / "clips.json")
    replicates, seed = config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    states = {row["plan_id"]: row for row in read_jsonl(root / OUT / "states.jsonl")}
    plans = sorted(states)
    boot = ProblemBoot(plans, replicates, seed)
    humans_path = root / "data" / "clips" / "raw" / "results" / "doors-keys-gems" / "mean_human_responses.csv"
    humans = read_humans(humans_path) if humans_path.exists() else None
    result: dict = {"problems": len(plans), "chance": 1 / GOALS, "layout": LAYOUT, "models": {}}
    for model in models:
        path = root / SCORES / f"{model}.jsonl"
        if not path.exists():
            continue
        rows = {row["plan_id"]: row for row in read_jsonl(path)}
        probs = {plan: posterior(rows[plan]["scores"]) for plan in plans}
        p_true = {plan: probs[plan][states[plan]["true_goal"] - 1] for plan in plans}
        top = {plan: float(max(range(GOALS), key=lambda k: probs[plan][k]) == states[plan]["true_goal"] - 1)
               for plan in plans}
        block: dict = {"p_true": mean_block(p_true, plans, boot), "top1": mean_block(top, plans, boot),
                       "uniform_like": sum(max(probs[plan]) - min(probs[plan]) < 0.05 for plan in plans)}
        if humans:
            cells = {plan: {f"goal_probs_{k + 1}": probs[plan][k] for k in range(GOALS)} for plan in plans}

            def correlation(sample):
                x = [cells[plan][f"goal_probs_{k}"] for plan in sample for k in range(1, GOALS + 1)]
                y = [fnum(humans[plan][f"goal_probs_{k}"]) for plan in sample for k in range(1, GOALS + 1)]
                return pearson(x, y)

            block["goal_correlation"] = {"r": correlation(plans), "ci95": boot.interval(correlation)}
        replay, _ = run_rows(root, model, "mdkg")
        if replay:
            methods = method_rows(replay, "mdkg")
            for name in ("language_only", "clips"):
                theirs = {plan: fnum(row["true_goal_probs"]) for plan, row in methods[name].items()}
                if set(theirs) != set(plans):
                    continue
                diff = {plan: p_true[plan] - theirs[plan] for plan in plans}
                block[f"minus_{name}"] = mean_block(diff, plans, boot)
                block[f"{name}_p_true"] = statistics.fmean(theirs.values())
        block["probs"] = {plan: probs[plan] for plan in plans}
        result["models"][model] = block
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_phase6.json", result)
    write_json(out / "manifest_phase6.json", {
        "kind": "clips-noplanner", "models": sorted(result["models"]), "states": (OUT / "states.jsonl").as_posix(),
        "scores": SCORES.as_posix(), "bootstrap": {"replicates": replicates, "seed": seed},
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1)})
    return result
