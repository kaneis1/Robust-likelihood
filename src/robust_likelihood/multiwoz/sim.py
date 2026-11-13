"""Simulator study (docs/simulator_study_plan.md): prompt formats, few-shot examples, and the Phase 1
format check. Later phases (scoring, analysis, report) build on the same formats.

A format turns a hypothesis text H into a scored prefix, and a human first turn U into the scored
reply [Frozen after Phase 1, configs/simulator.json]:

- S: system "You are a user talking to a restaurant booking assistant. {H}", the other party's
  opener, then U as the simulated human's turn. The human speaks in the role the model generates:
  the assistant role for assistant-tuned models and OSim, the user role for UserLM, whose chat
  template only ever opens a user turn; the opener takes the other role.
- B: S with the broadened system prompt.
- N: the native form, the goal as the only system message and no opener.
- F: plain few-shot text for the base model, scoring " {U}" and the terminating newline.
- P0: the exact Phase 8 prompt (multiwoz/lm.py), to check the code path against earlier numbers.

The prefix of a chat format is the chat template's generation prompt (with the model's template
options, such as enable_thinking=False for Qwen3) plus the template's separator before the content
(a newline for UserLM, whose header is not followed by one). The reply is U, then the end-of-turn
token. torch and transformers are imported only where a model runs.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.corpus import (
    SLOTS,
    constrained_slots,
    dialogue_key,
    first_goal,
    goal_set,
    load_goals,
    restaurant_state,
    slot_match,
    slot_values,
    turn_status,
)
from robust_likelihood.multiwoz.hypotheses import base
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

CONFIG_FILE = Path("configs") / "simulator.json"
CHAT_FORMATS = ("S", "B", "N", "P0")
ASSISTANT_OPENINGS = ("how can i help", "sure", "as an ai", "certainly", "here is")
RESTAURANT_VOICE = re.compile(r"(?:^|[.!?]\s+)(?:we have|our menu)\b", re.IGNORECASE)
THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)


def load_sim_config(root: Path) -> dict:
    return read_json(root / CONFIG_FILE)


# ---------------------------------------------------------------- formats (pure)


def other_role(speaker_role: str) -> str:
    return "user" if speaker_role == "assistant" else "assistant"


def chat_messages(fmt: str, spec: dict, prompts: dict, hypothesis: str) -> list[dict]:
    """The messages before the simulated human's first turn."""
    if fmt == "P0":
        from robust_likelihood.multiwoz.lm import SYSTEM_PROMPT, USER_PROMPT

        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT.format(text=hypothesis)}]
    if fmt == "N":
        return [{"role": "system", "content": prompts["N_system"].format(H=hypothesis)}]
    if fmt in ("S", "B"):
        system = prompts["S_system" if fmt == "S" else "B_system"].format(H=hypothesis)
        return [{"role": "system", "content": system},
                {"role": other_role(spec["speaker_role"]), "content": prompts["opener"]}]
    raise ValueError(f"{fmt} is not a chat format")


def fewshot_prefix(prompts: dict, examples: list[dict], hypothesis: str) -> str:
    text = prompts["F_header"]
    for example in examples:
        text += prompts["F_example"].format(H=example["H"], U=example["U"])
    return text + prompts["F_query"].format(H=hypothesis)


def flags(text: str, raw: str, truncated: bool, flag_tokens: tuple[str, ...] = ()) -> dict:
    """Phase 1's automated checks on one sample; raw is the decoded reply with special tokens kept."""
    opening = text.strip().lower()
    think = [match.group(1).strip() for match in THINK.finditer(raw)]
    return {
        "empty": not text.strip(),
        "truncated": truncated,
        "assistant_opening": opening.startswith(ASSISTANT_OPENINGS),
        "restaurant_voice": bool(RESTAURANT_VOICE.search(text.strip())),
        "think_block": any(think),
        "flag_tokens": [token for token in flag_tokens if token in raw],
        "words": len(text.split()),
    }


# ---------------------------------------------------------------- few-shot examples (raw corpus, CPU)


def fewshot_candidates(raw: Path, excluded: set[str]) -> list[dict]:
    """Train-split MUL/PMUL dialogues whose first user turn opens the restaurant domain and nothing else."""
    goals = load_goals(raw)
    found = []
    for path in sorted((raw / "MultiWOZ_2.2" / "train").glob("dialogues_*.json")):
        for dialogue in json.loads(path.read_text(encoding="utf-8")):
            key = dialogue_key(dialogue["dialogue_id"])
            if not key.startswith(("MUL", "PMUL")) or key in excluded or key not in goals:
                continue
            turn = dialogue["turns"][0]
            if turn["speaker"] != "USER":
                continue
            others = [frame for frame in turn["frames"] if frame["service"] != "restaurant"]
            state = restaurant_state(turn)
            if not state or state.get("active_intent", "NONE") == "NONE":
                continue
            if any(frame["state"]["active_intent"] != "NONE" or frame["state"]["slot_values"] for frame in others):
                continue
            goal = goals[key]["goal"]
            if goal_set(goal) != "clean":
                continue
            g_first = first_goal(goal)
            constrained = constrained_slots(g_first)
            if len(constrained) not in (2, 3):
                continue
            status = turn_status(state)
            values = {slot: slot_values(state, slot) for slot in SLOTS}
            match = slot_match(g_first, constrained, status, values)
            if any(status[slot] == "dontcare" for slot in constrained) or any(
                    status[slot] == "named" and not match[slot] for slot in constrained):
                continue
            if any(status[slot] != "omitted" for slot in SLOTS if slot not in constrained):
                continue
            found.append({"dialogue_id": key, "n_constrained": len(constrained),
                          "status": {slot: status[slot] for slot in constrained},
                          "H": base({slot: g_first[slot] for slot in constrained}),
                          "U": " ".join(turn["utterance"].split())})
    return found


def select_fewshot(candidates: list[dict], seed: int, count: int = 8) -> list[dict]:
    """Half 2-slot and half 3-slot turns, drawn at random within each size, so naming and omission follow
    the corpus (a 3-slot first turn that names all three slots is almost absent: 1 of 228 candidates),
    then shuffled. At least one turn omits a constrained slot."""
    candidates = sorted(candidates, key=lambda found: found["dialogue_id"])
    two = [found for found in candidates if found["n_constrained"] == 2]
    three = [found for found in candidates if found["n_constrained"] == 3]
    rng = random.Random(seed)
    chosen = rng.sample(two, count // 2) + rng.sample(three, count - count // 2)
    rng.shuffle(chosen)
    if not any("omitted" in found["status"].values() for found in chosen):
        raise ValueError("no few-shot example omits a constrained slot; the plan asks for at least one")
    return chosen


def write_fewshot(root: Path, raw: Path, derived: Path | None = None) -> dict:
    config = load_sim_config(root)
    derived = derived or root / DERIVED_DIR
    probe_items = {row["item_id"] for row in read_jsonl(derived / "items.jsonl")}
    candidates = fewshot_candidates(raw, probe_items)
    examples = select_fewshot(candidates, int(config["seed"]), int(config["fewshot"]["examples"]))
    payload = {"seed": int(config["seed"]), "candidates": len(candidates),
               "rule": "train-split MUL/PMUL first user turns opening only the restaurant domain, clean goals with "
                       "2 or 3 constrained slots, named values matching the goal; 4 two-slot and 4 three-slot turns "
                       "drawn at random within each size, shuffled",
               "examples": examples}
    write_json(root / config["fewshot"]["file"], payload)
    return {"file": config["fewshot"]["file"], "candidates": len(candidates),
            "examples": [(found["dialogue_id"], found["n_constrained"], found["status"]) for found in examples]}


# ---------------------------------------------------------------- formats on a tokenizer


class SimFormat:
    """Prefix, reply, and end of one (model, format) on its tokenizer."""

    def __init__(self, tokenizer, spec: dict, fmt: str, prompts: dict, examples: list[dict] | None = None):
        self.tokenizer, self.spec, self.fmt, self.prompts = tokenizer, spec, fmt, prompts
        self.examples = examples or []
        if fmt == "F":
            self.end = None
            self.end_text = spec.get("end_text", "\n")
        else:
            self.end = tokenizer.convert_tokens_to_ids(spec["end_token"])
            if self.end is None or self.end == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(f"{spec['end_token']} is not a token of {spec['id']}")
        self.flag_tokens = tuple(spec.get("flag_tokens", ()))
        self._texts: dict[str, str] = {}
        self._ids: dict[str, list[int]] = {}

    def messages(self, hypothesis: str) -> list[dict]:
        return chat_messages(self.fmt, self.spec, self.prompts, hypothesis)

    def prefix_text(self, hypothesis: str) -> str:
        if hypothesis not in self._texts:
            if self.fmt == "F":
                self._texts[hypothesis] = fewshot_prefix(self.prompts, self.examples, hypothesis)
            else:
                rendered = self.tokenizer.apply_chat_template(
                    self.messages(hypothesis), add_generation_prompt=True, tokenize=False,
                    **self.spec.get("template_kwargs", {}))
                self._texts[hypothesis] = rendered + self.spec.get("separator", "")
        return self._texts[hypothesis]

    def prefix_ids(self, hypothesis: str) -> list[int]:
        """The prefix tokens, cached per hypothesis; callers copy before extending."""
        if hypothesis not in self._ids:
            self._ids[hypothesis] = self.encode(self.prefix_text(hypothesis))
        return self._ids[hypothesis]

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

    def reply_text(self, utterance: str) -> str:
        return f" {utterance}{self.end_text}" if self.fmt == "F" else utterance

    def scored(self, hypothesis: str, utterance: str, prefix: list[int] | None = None) -> tuple[list[int], int]:
        """Tokens and the first scored index: prefix, reply, and the end token of a chat format."""
        prefix = prefix if prefix is not None else self.prefix_ids(hypothesis)
        reply = self.encode(self.reply_text(utterance))
        return prefix + reply + ([] if self.end is None else [self.end]), len(prefix)

    def boundary_holds(self, hypothesis: str, utterance: str) -> bool:
        """Tokenizing prefix and reply apart gives the ids of tokenizing them together."""
        whole = self.encode(self.prefix_text(hypothesis) + self.reply_text(utterance))
        tokens, _ = self.scored(hypothesis, utterance)
        return whole + ([] if self.end is None else [self.end]) == tokens

    def template_check(self) -> bool:
        """The template closes the simulated human's turn with the configured end token."""
        if self.fmt == "F":
            return True
        messages = self.messages("H") + [{"role": self.spec["speaker_role"], "content": "xyzzy"}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, **self.spec.get("template_kwargs", {}))
        return f"xyzzy{self.spec['end_token']}" in rendered


def generation_config(sim_format: SimFormat, settings: dict, pad: int):
    """Pure sampling, every option explicit. F stops at the first newline (a stopping rule, not a warper)."""
    from robust_likelihood.lmcore import sampling_config, with_settings

    if sim_format.fmt == "F":
        eos = sim_format.tokenizer.eos_token_id
        return with_settings(sampling_config(settings, eos, pad), stop_strings=[sim_format.end_text])
    return sampling_config(settings, sim_format.end, pad)


def use_config(model, config):
    """Make config the model's own generation_config, as multiwoz.lm does. Otherwise transformers
    (4.50 on) fills every option left at its global default (temperature 1, top_p 1, repetition
    penalty 1) from the model's generation_config, which for Qwen3-8B and UserLM-8b means
    temperature 0.6 and top_p below 1."""
    model.generation_config = config
    return config


def draw(model, sim_format: SimFormat, hypothesis: str, count: int, config, seed: int) -> list[dict]:
    """count samples, each with its text, the raw decoded reply, the reply tokens (chat formats, and F
    samples cut off at max_new_tokens), and whether it ended."""
    import torch

    from robust_likelihood.lmcore import with_settings

    tokenizer = sim_format.tokenizer
    prefix = sim_format.prefix_ids(hypothesis)
    device = next(model.parameters()).device
    use_config(model, config)
    torch.manual_seed(seed)
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    extra = {"tokenizer": tokenizer} if sim_format.fmt == "F" else {}
    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            generation_config=with_settings(config, num_return_sequences=count), **extra)
    found = []
    for row in output[:, len(prefix):].tolist():
        if sim_format.fmt == "F":
            raw = tokenizer.decode(row, skip_special_tokens=False)
            eos = tokenizer.eos_token or ""
            cut = [index for index in (raw.find(sim_format.end_text), raw.find(eos) if eos else -1) if index >= 0]
            ended = bool(cut)
            text = raw[: min(cut)] if cut else raw
            text = text[1:] if text.startswith(" ") else text
            # An ended F sample is rescored from its text; one cut off keeps its tokens for a lower bound.
            found.append({"text": text, "raw": raw, "tokens": None if ended else row, "ended": ended})
        else:
            ended = sim_format.end in row
            tokens = row[: row.index(sim_format.end)] if ended else row
            found.append({"text": tokenizer.decode(tokens, skip_special_tokens=True),
                          "raw": tokenizer.decode(tokens, skip_special_tokens=False), "tokens": tokens, "ended": ended})
    return found


def sample_surprisals(model, sim_format: SimFormat, hypothesis: str, samples: list[dict], batch_size: int, pad: int):
    """S of each ended sample under its own prefix: its token ids for chat formats (as Phase 8 scored
    samples), its text rescored for F, whose newline stop can fall inside a token."""
    from robust_likelihood.lmcore import scored_sequence, surprisals

    prefix = sim_format.prefix_ids(hypothesis)
    sequences, where = [], []
    for index, sample in enumerate(samples):
        if not sample["ended"]:
            continue
        if sim_format.fmt == "F":
            sequences.append(sim_format.scored(hypothesis, sample["text"], prefix))
        else:
            sequences.append(scored_sequence(prefix, sample["tokens"], sim_format.end))
        where.append(index)
    values = surprisals(model, sequences, batch_size, pad) if sequences else []
    found = [None] * len(samples)
    for index, value in zip(where, values):
        found[index] = value
    return found


# ---------------------------------------------------------------- Phase 1: format check (GPU)


def check_goals(items: list[dict], probe: list[dict], seed: int, two: int, three: int) -> list[dict]:
    rng = random.Random(seed)
    clean = sorted((item for item in items if item["set"] == "clean"), key=lambda item: item["item_id"])
    chosen = rng.sample([item for item in clean if item["n_constrained"] == 2], two) + rng.sample(
        [item for item in clean if item["n_constrained"] == 3], three)
    goal_text = {row["item_id"]: row["text"] for row in probe if row["hyp_key"] == "G|impl"}
    return [{"item_id": item["item_id"], "n_constrained": item["n_constrained"], "H": goal_text[item["item_id"]],
             "U": item["utterance"]} for item in chosen]


def combos(config: dict, keys: list[str] | None) -> list[tuple[str, str]]:
    keys = keys or list(config["models"])
    return [(key, fmt) for key in keys for fmt in config["models"][key]["formats"]]


def format_check(
    root: Path,
    keys: list[str] | None,
    phase8: Path,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    batch_size: int = 16,
    derived: Path | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    from robust_likelihood.lmcore import (
        load_model,
        pick_device,
        prefix_ids,
        sampling_config,
        sampling_is_exact,
        surprisals,
        text_seed,
    )

    config = load_sim_config(root)
    settings, check = config["sampling"], config["format_check"]
    seed = int(config["seed"])
    derived = derived or root / DERIVED_DIR
    items = read_jsonl(derived / "items.jsonl")
    probe = read_jsonl(derived / "probe.jsonl")
    clean_probe = sorted((row for row in probe if row["set"] == "clean"), key=lambda row: (row["item_id"], row["hyp_key"]))
    goals = check_goals(items, probe, seed, int(check["goals_2slot"]), int(check["goals_3slot"]))
    boundary_pairs = random.Random(seed).sample(clean_probe, int(check["boundary_pairs"]))
    examples = json.loads((root / config["fewshot"]["file"]).read_text(encoding="utf-8"))["examples"]
    phase8_rows = [row for row in read_jsonl(phase8 / "lm_scores.jsonl") if row["set"] == "clean"]
    p0_pairs = random.Random(seed + 1).sample(sorted(phase8_rows, key=lambda row: (row["item_id"], row["hyp_key"])),
                                              int(check["p0_pairs"]))
    text_of = {(row["item_id"], row["hyp_key"]): (row["text"], row["utterance"]) for row in probe}
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    run_dir = (results_root or root / RESULTS_DIR) / f"{started.strftime('%Y%m%dT%H%M%SZ')}-sim-check"
    manifest = {"kind": "sim-check", "git": git_state(root), "config": config, "goals": goals,
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "models": {}, "checks": {}}
    write_json(run_dir / "manifest.json", manifest)
    wanted = combos(config, keys)
    by_model: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, fmt in wanted:
        spec = config["models"][key]
        by_model.setdefault((spec["id"], spec["revision"]), []).append((key, fmt))
    device = pick_device(device)
    rows = []
    for (model_id, revision), mine in by_model.items():
        stage = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = load_model(model_id, revision, dtype, device)
        model.eval()
        manifest["models"][model_id] = {"revision": revision, "load_seconds": round(time.monotonic() - stage, 1)}
        for key, fmt in mine:
            spec = config["models"][key]
            sim_format = SimFormat(tokenizer, spec, fmt, config["prompts"], examples)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
                sim_format.end if sim_format.end is not None else tokenizer.eos_token_id)
            generation = generation_config(sim_format, settings, pad)
            # The newline stop of F is a stopping rule, not a logits processor, so the check runs without it.
            first_prefix = sim_format.encode(sim_format.prefix_text(goals[0]["H"]))
            exact = sampling_is_exact(model, first_prefix, use_config(model, sampling_config(
                settings, sim_format.end if sim_format.end is not None else tokenizer.eos_token_id, pad)))
            boundary_failures = [(row["item_id"], row["hyp_key"]) for row in boundary_pairs
                                 if not sim_format.boundary_holds(row["text"], row["utterance"])]
            found = {"model": key, "format": fmt, "model_id": model_id, "revision": revision,
                     "prefix_example": sim_format.prefix_text(goals[0]["H"]),
                     "template_check": sim_format.template_check(), "sampling_is_exact": exact,
                     "boundary_pairs": len(boundary_pairs), "boundary_failures": len(boundary_failures),
                     "boundary_failed_pairs": boundary_failures[:10]}
            human = surprisals(model, [sim_format.scored(goal["H"], goal["U"]) for goal in goals], batch_size, pad)
            for goal, human_s in zip(goals, human):
                samples = draw(model, sim_format, goal["H"], int(check["samples"]), generation,
                               text_seed(seed, f"{key}|{fmt}|{goal['H']}"))
                values = sample_surprisals(model, sim_format, goal["H"], samples, batch_size, pad)
                for index, (sample, value) in enumerate(zip(samples, values)):
                    rows.append({"model": key, "format": fmt, "item_id": goal["item_id"], "sample": index,
                                 "text": sample["text"], "S": value, "human_S": human_s,
                                 **flags(sample["text"], sample["raw"], not sample["ended"], sim_format.flag_tokens)})
            if fmt == "P0":
                reference = {(row["item_id"], row["hyp_key"]): row["S"] for row in p0_pairs}
                standard = [prefix_ids(tokenizer, sim_format.messages(text_of[pair][0])) for pair in reference]
                ours = [sim_format.encode(sim_format.prefix_text(text_of[pair][0])) for pair in reference]
                rescored = surprisals(model, [sim_format.scored(*text_of[pair]) for pair in reference], batch_size, pad)
                gaps = [abs(value - reference[pair]) for pair, value in zip(reference, rescored)]
                found["p0"] = {"pairs": len(gaps), "prefix_matches_phase8_code": standard == ours,
                               "max_abs_gap_nats": max(gaps), "median_abs_gap_nats": statistics.median(gaps),
                               "within_tolerance": max(gaps) <= float(check["p0_tolerance_nats"])}
            manifest["checks"][f"{key}|{fmt}"] = found
            write_json(run_dir / "manifest.json", manifest)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(run_dir / "samples.jsonl", rows)
    manifest.update({"finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "format_check.md").write_text(format_check_markdown(manifest, rows), encoding="utf-8", newline="\n")
    return {"run_dir": str(run_dir), "combos": len(wanted), "samples": len(rows)}


def _median(values: list[float]):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def format_check_markdown(manifest: dict, rows: list[dict]) -> str:
    """Flags per (model, format), then every sample next to the human turn of its goal."""
    limit = float(manifest["config"]["format_check"]["boundary_failure_limit"])
    lines = ["# Simulator study, Phase 1: format check", "",
             f"Run {manifest['started_at']}, commit `{(manifest['git'] or {}).get('commit')}`. "
             f"{len(manifest['goals'])} seeded goals × {manifest['config']['format_check']['samples']} samples per "
             "(model, format), pure sampling (temperature 1, top_p 1, top_k 0, repetition penalty 1, 96 new tokens). "
             "S is surprisal in nats under the sample's own prompt; human S is the real first turn under the same "
             "prompt.", "",
             "**Decision needed per model: which format becomes primary (S or N).** Then the formats are frozen.", "",
             "| Model | Format | Empty | Truncated | Assistant opening | Restaurant voice | Think block | Flag tokens | "
             "Median words | Median sample S | Median human S | Boundary failures | Exact sampling | Template |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, check in manifest["checks"].items():
        mine = [row for row in rows if f"{row['model']}|{row['format']}" == name]
        count = len(mine)

        def share(field):
            return f"{sum(1 for row in mine if row[field])}/{count}"

        failures = check["boundary_failures"] / max(1, check["boundary_pairs"])
        human = _median([row["human_S"] for row in mine])
        sample = _median([row["S"] for row in mine])
        lines.append(
            f"| {check['model']} | {check['format']} | {share('empty')} | {share('truncated')} | "
            f"{share('assistant_opening')} | {share('restaurant_voice')} | {share('think_block')} | "
            f"{sum(1 for row in mine if row['flag_tokens'])}/{count} | {_median([row['words'] for row in mine])} | "
            f"{'–' if sample is None else f'{sample:.1f}'} | {'–' if human is None else f'{human:.1f}'} | "
            f"{check['boundary_failures']}/{check['boundary_pairs']}{' **STOP**' if failures > limit else ''} | "
            f"{'yes' if check['sampling_is_exact'] else '**no**'} | {'ok' if check['template_check'] else '**fails**'} |")
    p0 = [check["p0"] for check in manifest["checks"].values() if "p0" in check]
    if p0:
        found = p0[0]
        lines += ["", f"P0 reproduces Phase 8 on {found['pairs']} pairs: prefix identical to the Phase 8 code "
                      f"{'yes' if found['prefix_matches_phase8_code'] else '**no**'}; |S − S_phase8| median "
                      f"{found['median_abs_gap_nats']:.3f}, max {found['max_abs_gap_nats']:.3f} nats "
                      f"({'within' if found['within_tolerance'] else '**outside**'} the 1-nat batch noise)."]
    for name, check in manifest["checks"].items():
        mine = [row for row in rows if f"{row['model']}|{row['format']}" == name]
        lines += ["", f"## {check['model']} · {check['format']}", "", "Prompt for the first goal:", "", "```",
                  check["prefix_example"].rstrip("\n"), "```"]
        for goal in manifest["goals"]:
            samples = [row for row in mine if row["item_id"] == goal["item_id"]]
            human = samples[0]["human_S"] if samples else None
            lines += ["", f"**{goal['item_id']}** ({goal['n_constrained']} slots): {goal['H']}",
                      f"- *human* (S {'–' if human is None else f'{human:.1f}'}): {goal['U']}"]
            for row in samples:
                marks = [label for label, field in (("empty", "empty"), ("truncated", "truncated"),
                                                    ("assistant", "assistant_opening"),
                                                    ("restaurant", "restaurant_voice"), ("think", "think_block"))
                         if row[field]] + row["flag_tokens"]
                text = " ".join(row["text"].split()) or "∅"
                score = "–" if row["S"] is None else f"{row['S']:.1f}"
                lines.append(f"- {row['sample']} (S {score}){' [' + ', '.join(marks) + ']' if marks else ''}: {text}")
    return "\n".join(lines) + "\n"
