"""Simulator study, Phase 5 (docs/simulator_study_plan.md): the multi-turn pilot.

Xuan asked for later turns to be scored given the goal and the history. For each S-format chat model
(configs/simulator.json phase5), user turns 2 and 3 of a seeded subset of the 235 clean dialogues,
under every hypothesis of the dialogue's first-turn set, unchanged:

- The context is format S with the real history: the system prompt with H, the opener, then every
  earlier MultiWOZ turn in its role (user turns in the simulated human's role, system turns in the
  other role), as the model's chat template renders a conversation. The model writes the next user
  turn; the human's real turn is scored as the reply, then the end-of-turn token.
- 32 samples per (dialogue, turn, hypothesis), pure sampling as in Phase 2, each with S under its own
  context and S_lower when cut off. Hypotheses of a dialogue that share a text share their samples.
- Q1 coverage only (sim_pilot_report.py), with turn 1 from the Phase 2 runs on the same dialogues.

write_pilot_plan (CPU, raw corpus) writes the dialogues and their histories to the phase5 file;
sim_pilot (GPU) runs one model. torch and transformers are imported only inside sim_pilot.
"""

from __future__ import annotations

import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.corpus import load_turns
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.sim import (
    SimFormat,
    chat_messages,
    generation_config,
    load_sim_config,
    other_role,
    use_config,
)
from robust_likelihood.multiwoz.sim_score import sample_block, tally
from robust_likelihood.storage import read_jsonl, write_json, write_jsonl

PILOT_FORMATS = ("S", "B", "N")
PROGRESS_EVERY = 100


class Context(NamedTuple):
    """A hypothesis text and the turns before the scored one, as (speaker, utterance) pairs."""

    text: str
    history: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------- the dialogues (pure, then raw corpus)


def alternates(turns: list[dict], last_turn: int) -> bool:
    """The dialogue reaches user turn last_turn, its turns alternating USER, SYSTEM from the start."""
    wanted = 2 * last_turn - 1
    return len(turns) >= wanted and all(
        turn["speaker"] == ("USER" if index % 2 == 0 else "SYSTEM") for index, turn in enumerate(turns[:wanted]))


def pilot_rows(turns_by_id: dict[str, list[dict]], items: list[dict], count: int, seed: int,
               user_turns: list[int]) -> list[dict]:
    """count clean dialogues drawn with seed among those reaching the last user turn, then one row per
    (dialogue, user turn): the history before it and the human's turn. User turn n is turn 2(n-1)."""
    clean = sorted(item["item_id"] for item in items if item["set"] == "clean")
    usable = [item_id for item_id in clean if alternates(turns_by_id[item_id], max(user_turns))]
    chosen = sorted(random.Random(seed).sample(usable, count))
    rows = []
    for item_id in chosen:
        turns = turns_by_id[item_id]
        for turn in sorted(user_turns):
            index = 2 * (turn - 1)
            rows.append({"item_id": item_id, "turn": turn,
                         "history": [{"speaker": found["speaker"], "utterance": found["utterance"]}
                                     for found in turns[:index]],
                         "utterance": turns[index]["utterance"]})
    return rows


def write_pilot_plan(root: Path, raw: Path, derived: Path | None = None) -> dict:
    config = load_sim_config(root)
    phase5 = config["phase5"]
    derived = derived or root / DERIVED_DIR
    items = read_jsonl(derived / "items.jsonl")
    turns = load_turns(raw)
    rows = pilot_rows(turns, items, int(phase5["dialogues"]), int(config["seed"]), list(phase5["user_turns"]))
    first = {item["item_id"]: item["utterance"] for item in items}
    mismatched = [row["item_id"] for row in rows if row["history"][0]["utterance"] != first[row["item_id"]]]
    if mismatched:
        raise ValueError(f"turn 0 of the raw dialogue is not the probe's first turn for {sorted(set(mismatched))}")
    write_jsonl(root / phase5["file"], rows)
    clean = [item for item in items if item["set"] == "clean"]
    usable = sum(1 for item in clean if alternates(turns[item["item_id"]], max(phase5["user_turns"])))
    chosen = {row["item_id"] for row in rows}
    sizes = {}
    for item in clean:
        if item["item_id"] in chosen:
            sizes[str(item["n_constrained"])] = sizes.get(str(item["n_constrained"]), 0) + 1
    return {"file": phase5["file"], "clean": len(clean), "usable": usable, "dialogues": len(chosen),
            "rows": len(rows), "by_n_constrained": sizes}


def pilot_plan(rows: list[dict], probe: list[dict], dialogues: set[str] | None = None) -> list[dict]:
    """One unit per (dialogue, turn, distinct clean hypothesis text of the dialogue), with its probe keys."""
    texts: dict[str, dict[str, list[str]]] = {}
    for row in probe:
        if row["set"] == "clean":
            texts.setdefault(row["item_id"], {}).setdefault(row["text"], []).append(row["hyp_key"])
    units = []
    for row in rows:
        if dialogues is not None and row["item_id"] not in dialogues:
            continue
        history = tuple((turn["speaker"], turn["utterance"]) for turn in row["history"])
        for text, keys in sorted(texts[row["item_id"]].items()):
            units.append({"item_id": row["item_id"], "turn": row["turn"], "text": text, "hyp_keys": sorted(keys),
                          "utterance": row["utterance"], "context": Context(text, history)})
    return units


# ---------------------------------------------------------------- the format with history


class PilotFormat(SimFormat):
    """A chat format with the real history after its opening messages. Every hypothesis argument of
    SimFormat (and of draw and sample_surprisals) is a Context here."""

    def __init__(self, tokenizer, spec: dict, fmt: str, prompts: dict):
        if fmt not in PILOT_FORMATS:
            raise ValueError(f"the pilot extends a chat format ({', '.join(PILOT_FORMATS)}), not {fmt}")
        super().__init__(tokenizer, spec, fmt, prompts)

    def messages(self, context: Context) -> list[dict]:
        human, other = self.spec["speaker_role"], other_role(self.spec["speaker_role"])
        return chat_messages(self.fmt, self.spec, self.prompts, context.text) + [
            {"role": human if speaker == "USER" else other, "content": utterance}
            for speaker, utterance in context.history]

    def template_check(self, context: Context | None = None) -> bool:
        """After the history, the template still closes the simulated human's turn with the end token."""
        messages = self.messages(context or Context("H")) + [{"role": self.spec["speaker_role"], "content": "xyzzy"}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, **self.spec.get("template_kwargs", {}))
        return f"xyzzy{self.spec['end_token']}" in rendered

    def history_kept(self, context: Context) -> bool:
        """Every earlier turn appears in the prefix, verbatim and in order."""
        text, position = self.prefix_text(context), 0
        for _, utterance in context.history:
            found = text.find(utterance, position)
            if found < 0:
                return False
            position = found + len(utterance)
        return True


# ---------------------------------------------------------------- Phase 5 run (GPU)


def sim_pilot(
    root: Path,
    key: str,
    fmt: str | None = None,
    *,
    batch_size: int = 32,
    dtype: str = "bfloat16",
    device: str | None = None,
    label: str | None = None,
    dialogues: set[str] | None = None,
    samples: int | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    tokenizer=None,
    model=None,
    now: datetime | None = None,
) -> dict:
    """One model of the pilot. dialogues and samples shrink a run for smoke tests; tokenizer and model may
    be passed in (tests), otherwise they are loaded at the pinned revision."""
    from robust_likelihood.lmcore import load_model, pick_device, sampling_config, sampling_is_exact, surprisals, text_seed

    config = load_sim_config(root)
    phase5 = config["phase5"]
    fmt = fmt or phase5["format"]
    if key not in phase5["models"]:
        raise ValueError(f"the pilot runs {phase5['models']}, not {key}")
    spec = config["models"][key]
    if fmt not in spec["formats"]:
        raise ValueError(f"{key} has no format {fmt}")
    derived = derived or root / DERIVED_DIR
    units = pilot_plan(read_jsonl(root / phase5["file"]), read_jsonl(derived / "probe.jsonl"), dialogues)
    if not units:
        raise ValueError("no pilot units: write the plan first (sim-pilot-plan)")
    settings, seed = config["sampling"], int(config["seed"])
    count = int(samples or phase5["samples"])
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    import torch
    import transformers

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    pilot_format = PilotFormat(tokenizer, spec, fmt, config["prompts"])
    failed = [unit for unit in units if not pilot_format.boundary_holds(unit["context"], unit["utterance"])]
    lost = [unit for unit in units if not pilot_format.history_kept(unit["context"])]
    rate = len(failed) / len(units)
    checks = {"template_check": pilot_format.template_check(units[-1]["context"]), "history_kept": not lost,
              "boundary_pairs": len(units), "boundary_failures": len(failed), "boundary_failure_rate": rate}
    run_dir = (results_root or root / RESULTS_DIR) / (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-sim-pilot-{key}-{fmt}" + (f"-{label}" if label else ""))
    turns = sorted({unit["turn"] for unit in units})
    manifest = {
        "kind": "sim-pilot", "quantity": "Surprisal S in nats of a simulator model, later user turns given the history.",
        "model_key": key, "format": fmt, "model": spec["id"], "revision": spec["revision"],
        "license": spec.get("license"), "dtype": dtype, "git": git_state(root), "prompts": config["prompts"],
        "plan_file": phase5["file"], "dialogues": len({unit["item_id"] for unit in units}), "user_turns": turns,
        "units": len(units), "samples_per_context": count, "seed": seed,
        "seed_rule": "torch.manual_seed(int(sha256(f'{seed}:pilot|{item_id}|{turn}|{text}').hexdigest()[:8], 16))",
        "prefix_example": pilot_format.prefix_text(units[-1]["context"]), "end_token": spec.get("end_token"),
        "checks": checks, "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "stage_seconds": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    if failed:
        write_jsonl(run_dir / "boundary_failures.jsonl", [
            {"item_id": unit["item_id"], "turn": unit["turn"], "text": unit["text"], "utterance": unit["utterance"]}
            for unit in failed])
    if (not checks["template_check"] or lost
            or rate > float(config["format_check"]["boundary_failure_limit"])):
        raise RuntimeError(f"Phase 5 self-check failed before scoring: {checks}")

    if model is None:
        model = load_model(spec["id"], spec["revision"], dtype, pick_device(device))
    model.eval()
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else pilot_format.end
    generation = generation_config(pilot_format, settings, pad)
    checks["sampling_is_exact"] = sampling_is_exact(
        model, pilot_format.prefix_ids(units[0]["context"]),
        use_config(model, sampling_config(settings, pilot_format.end, pad)))
    manifest.update({
        "device": str(next(model.parameters()).device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "sampling": {name: getattr(generation, name) for name in (
            "do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "eos_token_id",
            "pad_token_id")},
    })
    write_json(run_dir / "manifest.json", manifest)
    if not checks["sampling_is_exact"]:
        raise RuntimeError(f"Phase 5 self-check failed: {checks}")

    stage = time.monotonic()
    sequences = [pilot_format.scored(unit["context"], unit["utterance"]) for unit in units]
    values = surprisals(model, sequences, batch_size, pad)
    write_jsonl(run_dir / "pilot_scores.jsonl", [
        {"item_id": unit["item_id"], "turn": unit["turn"], "hyp_key": hyp_key, "text": unit["text"], "S": value,
         "tokens_scored": len(tokens) - first, "prefix_tokens": first}
        for unit, value, (tokens, first) in zip(units, values, sequences) for hyp_key in unit["hyp_keys"]])
    manifest["stage_seconds"]["human"] = round(time.monotonic() - stage, 1)
    write_json(run_dir / "manifest.json", manifest)

    stage = time.monotonic()
    rows = []
    for done, unit in enumerate(units, 1):
        block = sample_block(model, pilot_format, unit["context"], count, generation,
                             text_seed(seed, f"pilot|{unit['item_id']}|{unit['turn']}|{unit['text']}"), batch_size,
                             pad, None)
        rows.extend({"item_id": unit["item_id"], "turn": unit["turn"], "text": unit["text"],
                     **{name: value for name, value in row.items() if name != "S_broad"}} for row in block)
        if done % PROGRESS_EVERY == 0:
            manifest["progress"] = {"units_done": done, "seconds": round(time.monotonic() - stage, 1)}
            write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "pilot_samples.jsonl", rows)
    manifest["samples"] = tally(rows)
    manifest["stage_seconds"]["samples"] = round(time.monotonic() - stage, 1)
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    manifest.pop("progress", None)
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "units": len(units), "checks": checks, "stage_seconds": manifest["stage_seconds"],
            "samples": manifest["samples"]}
