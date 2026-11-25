"""Phase 1 of docs/clips_small_lm_plan.md: the record pass's keys, which goal used each, and the counts.

Inputs, per domain, from scripts/clips_job.sh record:
- results/clips/record/<domain>.jsonl: every (role, prompt, output) a language-model mixture was asked,
  once per key, with the plan and method (literal or pragmatic) that first asked it.
- results/clips/record/<domain>.particles.jsonl: after pragmatic inference, each particle's goal index,
  cost profile, and the prompts its utterance model built at each utterance time (enumerate_commands
  from that particle's plan, one prompt per lifted command).

pairs.jsonl has one row per unique key with every (plan, goal, method) that uses it; the prompt is named by
its id (prompt_id: the first 16 hex digits of its sha256), and prompts.jsonl holds each prompt's text once. A literal-listener
key has no goal (the literal listener reads commands, not goals); a pragmatic key's goals are the
particles whose prompts include it.

The check: every pragmatic key of the record log is some particle's prompt, and every particle prompt is
a pragmatic key of the record log; per particle the prompts are distinct (enumerate_commands' output
after unique!), and their weights sum to 1.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from pathlib import Path

from robust_likelihood.storage import read_jsonl

ROLES = {"mdkg": ("mdkg_literal", "mdkg_pragmatic"), "vh": ("vh_literal", "vh_pragmatic")}


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def split_prompts(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """pairs with prompt ids, and the distinct prompts; refuses a (vanishingly unlikely) id collision."""
    prompts: dict[str, str] = {}
    pairs = []
    for row in rows:
        found = prompt_id(row["prompt"])
        if prompts.setdefault(found, row["prompt"]) != row["prompt"]:
            raise ValueError(f"prompt id collision at {found}")
        pairs.append({key: value for key, value in row.items() if key != "prompt"} | {"prompt_id": found})
    return pairs, [{"prompt_id": found, "prompt": text} for found, text in sorted(prompts.items())]


def build_pairs(domain: str, record: list[dict], particles: list[dict]) -> tuple[list[dict], dict]:
    literal_role, pragmatic_role = ROLES[domain]
    uses: dict[tuple, set] = defaultdict(set)
    for row in record:
        key = (row["role"], row["prompt"], row["output"])
        if row["role"] == literal_role:
            uses[key].add((row["context"].get("plan_id", ""), None, "literal"))
    pragmatic_keys = {(row["role"], row["prompt"], row["output"]) for row in record if row["role"] == pragmatic_role}
    particle_keys = set()
    commands, goals_per_plan, problems = [], defaultdict(set), set()
    duplicate_prompts, weight_errors = 0, 0
    for row in particles:
        problems.add(row["plan_id"])
        goals_per_plan[row["plan_id"]].add(row["goal"])
        prompts = row["prompts"]
        commands.append(len(prompts))
        duplicate_prompts += len(prompts) - len(set(prompts))
        weight_errors += abs(sum(row["probs"]) - 1) > 1e-9
        for prompt in prompts:
            key = (pragmatic_role, prompt, row["output"])
            particle_keys.add(key)
            uses[key].add((row["plan_id"], row["goal"], "pragmatic"))
    rows = [{"role": role, "prompt": prompt, "output": output,
             "uses": [{"plan_id": plan, "goal": goal, "method": method} for plan, goal, method in sorted(
                 found, key=lambda use: (use[0], -1 if use[1] is None else use[1], use[2]))]}
            for (role, prompt, output), found in sorted(uses.items())]
    literal_plans = {row["context"].get("plan_id") for row in record if row["role"] == literal_role}
    counts = {
        "problems": len(problems | literal_plans),
        "goals_per_problem": {"mean": statistics.fmean(len(found) for found in goals_per_plan.values()),
                              "min": min(len(found) for found in goals_per_plan.values()),
                              "max": max(len(found) for found in goals_per_plan.values())} if goals_per_plan else None,
        "commands_per_goal": {"mean": statistics.fmean(commands), "max": max(commands)} if commands else None,
        "unique_prompts": len({row["prompt"] for row in rows}),
        "unique_pairs": len(rows),
        "pairs_by_role": {role: sum(1 for row in rows if row["role"] == role) for role in ROLES[domain]},
        "check": {"pragmatic_keys_without_particle": len(pragmatic_keys - particle_keys),
                  "particle_prompts_not_recorded": len(particle_keys - pragmatic_keys),
                  "duplicate_prompts_within_a_particle": duplicate_prompts,
                  "particles_with_weights_not_summing_to_1": weight_errors},
    }
    counts["check"]["passes"] = bool(rows) and bool(particles) and not any(counts["check"].values())
    return rows, counts


def run_pairs(root: Path, domains: tuple[str, ...] = ("mdkg", "vh"), suffix: str = "") -> tuple[list[dict], dict]:
    rows, counts = [], {}
    for domain in domains:
        base = root / "results" / "clips" / "record" / f"{domain}{suffix}"
        found, counts[domain] = build_pairs(domain, read_jsonl(base.with_suffix(".jsonl")),
                                            read_jsonl(Path(f"{base}.particles.jsonl")))
        rows += found
    return rows, counts
