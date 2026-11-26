"""Phase 4 of docs/clips_small_lm_plan.md: perturbed instructions and the table pairs they need.

Kinds, per instruction (30 mDKG, 35 VirtualHome):
- typo1, typo2: robust_likelihood.multiwoz.perturb's typo rules (nested doses, interior edits, no edit that
  spells a real word), on words of at least four letters that are not protected.
- fill1, fill2: the instruction plus a thanks sentence; fill2 also opens with a greeting or lead-in. The
  sentences are short and contain no protected word.
- lower, nopunct: as in perturb.py.
- para1, para2: two Claude paraphrases (run_paraphrases, through rewrites.py's API client and NLI model).

The protected list (plan Phase 4): colours; key(s), door(s), gem(s) and every VirtualHome object and type
name in virtual-home/dataset/problems/*.pddl, without trailing digits, with plurals, and the heads of the
compound ones (fork, knife, glass, board); numerals and number
words; me, you, I, my, your; negations. A unit is valid when its text changed and its protected words, as
a lower-cased multiset after expanding contractions ("I'll" counts as "i", "don't" as "not"), are the
original's; a paraphrase must also pass the NLI check both ways (rewrites.NLI_THRESHOLD).

Scoring (plan Phase 4, docs/clips_decisions.md): a perturbed instruction needs its utterance's recorded
prompts x the new output. Batch layout moves a score by up to about 0.5 nats, so each (utterance, prompt)
is one batch holding the original output and every valid perturbed output of that utterance; each delta
is read against the original's score from the same batch.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from robust_likelihood.clips.record import prompt_id
from robust_likelihood.multiwoz import perturb as mw

NON_API_KINDS = ("typo1", "typo2", "fill1", "fill2", "lower", "nopunct")
API_KINDS = ("para1", "para2")
KINDS = API_KINDS + NON_API_KINDS
COLOURS = ("red", "blue", "yellow", "green", "purple", "orange", "pink", "white", "black", "grey", "gray", "brown")
MDKG_OBJECTS = ("key", "keys", "door", "doors", "gem", "gems")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve")
PRONOUNS = ("me", "you", "i", "my", "your")
# The everyday heads of VirtualHome's compound type names ("cutleryfork1 - cutleryfork"), which the
# instructions use ("Can you get two forks?").
COMPOUND_HEADS = {"cutleryfork": "fork", "cutleryknife": "knife", "chefknife": "knife", "waterglass": "glass",
                  "wineglass": "glass", "cuttingboard": "board"}
NEGATIONS = ("not", "no", "never", "nothing", "none", "nor", "neither", "without")
OPENERS = ("Hey.", "Hi.", "Quick favour.", "Real quick.")
THANKS = ("Thanks!", "Thanks.", "Cheers.")
# [Frozen when the paraphrase run is approved] The paraphrase prompt, in rewrites.py's form.
PARAPHRASE = ("Rewrite this instruction so it means exactly the same thing. Keep every colour, object, number "
              "and person (me, you) and add nothing.")


def unit_seed(seed: int, *parts: str) -> int:
    return int(hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).hexdigest()[:8], 16)


def plural(word: str) -> str:
    if word.endswith("fe"):
        return word[:-2] + "ves"
    if word.endswith("f"):
        return word[:-1] + "ves"
    if word.endswith(("s", "x", "ch", "sh", "o")):
        return word + "es"
    return word + "s"


def vh_object_words(clips_root: Path) -> set[str]:
    found = set()
    for path in sorted((clips_root / "virtual-home" / "dataset" / "problems").glob("*.pddl")):
        text = path.read_text(encoding="utf-8")
        block = re.search(r"\(:objects(.*?)\)", text, re.S)
        if not block:
            continue
        for token in re.findall(r"[A-Za-z][A-Za-z_-]*\d*", block.group(1)):
            name = re.sub(r"\d+$", "", token.lower())
            if name and name not in ("itype", "item", "fixture", "agent", "actor", "helper"):
                found |= {name, plural(name)}
                if name in COMPOUND_HEADS:
                    found |= {COMPOUND_HEADS[name], plural(COMPOUND_HEADS[name])}
    return found


def protected_set(clips_root: Path) -> frozenset[str]:
    words = set(COLOURS) | set(MDKG_OBJECTS) | set(NUMBER_WORDS) | set(PRONOUNS) | set(NEGATIONS)
    return frozenset(words | vh_object_words(clips_root))


def protected_tokens(text: str, protected: frozenset[str]) -> Counter:
    lowered = text.lower().replace("’", "'").replace("‘", "'")
    lowered = re.sub(r"n't\b", " not", lowered)
    lowered = re.sub(r"'(ll|m|re|ve|d|s)\b", " ", lowered)
    tokens = re.findall(r"[a-z]+|\d+", lowered)
    return Counter(token for token in tokens if token in protected or token.isdigit())


def valid(original: str, rewrite: str | None, protected: frozenset[str]) -> bool:
    return bool(rewrite) and rewrite.strip() != original.strip() and \
        protected_tokens(rewrite, protected) == protected_tokens(original, protected)


def utterance_key(row: dict) -> str:
    return f"{row['domain']}:{row['plan_id']}:{row['index']}"


def perturb_instruction(row: dict, protected: frozenset[str], seed: int, lexicon: frozenset[str]) -> list[dict]:
    text, key = row["text"], utterance_key(row)
    units = []

    def add(kind: str, rewrite: str, **extra) -> None:
        units.append({"unit_id": f"{key}:{kind}", "utterance": key, "domain": row["domain"], "plan_id": row["plan_id"],
                      "index": row["index"], "timestep": row["timestep"], "kind": kind, "original": text,
                      "rewrite": rewrite, "valid": valid(text, rewrite, protected), **extra})

    candidates = mw.typo_candidates(text, None, mw.protected_words(protected))
    order = candidates[:]
    random.Random(unit_seed(seed, key, "typo")).shuffle(order)
    for dose in (1, 2):
        if len(order) >= dose:
            rewrite, edits = mw.edit_words(text, order[:dose], random.Random(unit_seed(seed, key, "ops")), lexicon)
            add(f"typo{dose}", rewrite, edits=edits, available=len(candidates))
        else:
            add(f"typo{dose}", None, edits=[], available=len(candidates))
    rng = random.Random(unit_seed(seed, key, "fill"))
    opener, thanks = rng.choice(OPENERS), rng.choice(THANKS)
    add("fill1", f"{text.strip()} {thanks}", added=[thanks])
    add("fill2", f"{opener} {text.strip()} {thanks}", added=[opener, thanks])
    add("lower", text.lower())
    add("nopunct", mw.nopunct(text))
    return units


def build_units(utterances: list[dict], protected: frozenset[str], seed: int, lexicon: frozenset[str]) -> list[dict]:
    rows = []
    for row in sorted(utterances, key=utterance_key):
        rows += perturb_instruction(row, protected, seed, lexicon)
    return rows


def counts(units: list[dict]) -> dict:
    found = {}
    for domain in ("mdkg", "vh"):
        mine = [unit for unit in units if unit["domain"] == domain]
        found[domain] = {kind: {"units": sum(1 for unit in mine if unit["kind"] == kind),
                                "valid": sum(1 for unit in mine if unit["kind"] == kind and unit["valid"])}
                         for kind in KINDS if any(unit["kind"] == kind for unit in mine)}
    return found


def utterance_prompts(particles: list[dict], utterances: list[dict]) -> dict[str, list[str]]:
    """Each instruction's distinct recorded prompts: those of every particle at its time step."""
    by_time: dict[tuple, set] = defaultdict(set)
    for row in particles:
        by_time[(row["plan_id"], row["t"], row["output"].strip())] |= set(row["prompts"])
    found = {}
    for row in utterances:
        prompts = by_time.get((row["plan_id"], row["timestep"], row["text"].strip()))
        if not prompts:
            raise ValueError(f"no particle prompts for {utterance_key(row)}")
        found[utterance_key(row)] = sorted(prompts)
    return found


def batches(units: list[dict], utterances: list[dict], prompts_by_utterance: dict[str, list[str]]) -> list[dict]:
    """One row per (instruction, prompt): the original output first, then every valid perturbed output."""
    outputs: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        if unit["valid"]:
            output = " " + unit["rewrite"].strip()
            if output not in outputs[unit["utterance"]]:
                outputs[unit["utterance"]].append(output)
    rows = []
    for row in sorted(utterances, key=utterance_key):
        key = utterance_key(row)
        original = " " + row["text"].strip()
        found = [original] + [output for output in outputs.get(key, []) if output != original]
        if len(found) == 1:
            continue
        for prompt in prompts_by_utterance[key]:
            rows.append({"utterance": key, "prompt_id": prompt_id(prompt), "outputs": found})
    return rows
