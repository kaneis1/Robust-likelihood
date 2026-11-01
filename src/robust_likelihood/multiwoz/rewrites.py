"""Claims plan 2b: meaning-preserving rewrites of utterances and hypothesis texts.

Utterance side, for each clean item: `typo` and `pad` are programmatic and seeded;
`para1`, `para2`, `verbose`, and `zh` come from the `claude` provider with the frozen
prompts below. Hypothesis side, for every distinct hypothesis text of those items:
`hpara1`-`hpara3` and `hzh`. A Chinese rewrite is back-translated by the same model.

A rewrite is accepted only if [Frozen]:
1. the regex slot tuple is unchanged (utterances against the 2.2 annotation,
   hypotheses against the slots the text was built from; Chinese through the
   back-translation), and
2. NLI entailment is at least 0.8 in both directions between the original and the
   rewrite (Chinese through the back-translation).
A rejected rewrite is regenerated up to 2 times, then dropped.

`claude-fable-5-1` rejects a temperature, so the plan's temperature 1 is the model's
default sampling; every call records what was sent.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.jev_run import choose_items, git_state
from robust_likelihood.multiwoz.semantic import Extractor, annotated_tuple, hypothesis_tuple, load_vocab, normalize_text
from robust_likelihood.storage import append_jsonl, read_jsonl, write_json, write_jsonl
from robust_likelihood.verbal import api_client, api_verbal

SEED = 20261027
ATTEMPTS = 3
NLI_THRESHOLD = 0.8
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
UTTERANCE_KINDS = ("typo", "pad", "para1", "para2", "verbose", "zh")
HYPOTHESIS_KINDS = ("hpara1", "hpara2", "hpara3", "hzh")
LLM_KINDS = {"para1", "para2", "verbose", "zh", "hpara1", "hpara2", "hpara3", "hzh"}
CHINESE_KINDS = {"zh", "hzh"}
# `pad` keeps the message untouched between a fixed opener and closer, and the message never entails a
# greeting, so NLI rejects pads by design; pads are checked by regex only (docs/claims_decisions.md).
REGEX_ONLY_KINDS = {"pad"}
# Accepted rewrites the human audit judged not meaning-preserving (docs/claim1_results/rewrite_audit.csv).
# They stay in the run's accepted.jsonl but are left out of derived/rewrites.jsonl.
AUDIT_REJECTED = {
    "u:SNG0606:typo": "price became rice, so the moderate price range is no longer stated",
}

# [Frozen] prompts.
PARAPHRASE = (
    "Rewrite this message so it means exactly the same thing. Keep every detail (cuisine, area, price, people, "
    "day, time) and add nothing."
)
VERBOSE = "Rewrite this message to be about twice as long without adding any new information, preference, or detail."
HYPOTHESIS_PARAPHRASE = (
    "Rewrite this description of a user of a restaurant booking assistant so it means exactly the same thing. "
    "Keep every detail and add nothing."
)
KEEP_NO_PREFERENCE = " Keep the statement about what the user has no preference about."
TO_CHINESE = "Translate this text into Chinese. Keep every detail and add nothing."
TO_ENGLISH = "Translate this text into English. Keep every detail and add nothing."
REPLY_ONLY = "Reply with the rewritten text only."
TRANSLATION_ONLY = "Reply with the translation only."
OPENERS = (
    "Hi there! Hope your day is going well.",
    "Hello! Thanks for taking the time to help me.",
    "Good afternoon, I hope you're doing well.",
    "Hi! I hope this message finds you well.",
    "Hey there, thanks in advance for your help.",
)
CLOSERS = (
    "Thanks so much!",
    "I really appreciate your help.",
    "Thank you in advance!",
    "Looking forward to your reply.",
    "Many thanks, take care!",
)


def unit_seed(label: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{label}".encode("utf-8")).hexdigest()[:16], 16)


def protected_spans(text: str, extractor: Extractor) -> list[tuple[int, int]]:
    """Character ranges of every slot word the extractor reads, so a typo never touches them."""
    lowered = text.lower().replace("’", "'")
    patterns = [extractor.food, extractor.area, extractor.price] + [
        pattern for patterns in extractor.dontcare.values() for pattern in patterns
    ]
    return [(match.start(), match.end()) for pattern in patterns for match in pattern.finditer(lowered)]


def typo(text: str, rng: random.Random, protected: list[tuple[int, int]]) -> str:
    """About one edit per 12 characters: swap two adjacent letters or drop one, never inside a protected span."""
    def free(index: int) -> bool:
        return not any(start <= index < end for start, end in protected)

    letters = [index for index, char in enumerate(text) if char.isalpha() and free(index)]
    count = min(len(letters), max(1, round(len(text) / 12)))
    chosen = sorted(rng.sample(letters, count), reverse=True)
    chars = list(text)
    used: set[int] = set()
    for index in chosen:
        if index in used:
            continue
        nxt = index + 1
        if rng.random() < 0.5 and nxt < len(chars) and chars[nxt].isalpha() and free(nxt) and nxt not in used \
                and chars[nxt] != chars[index]:
            chars[index], chars[nxt] = chars[nxt], chars[index]
            used.update({index, nxt})
        else:
            chars[index] = ""
            used.add(index)
    return "".join(chars)


def pad(text: str, rng: random.Random) -> str:
    return f"{rng.choice(OPENERS)} {text.strip()} {rng.choice(CLOSERS)}"


def llm_messages(kind: str, text: str, explicit: bool = False) -> list[dict]:
    if kind in ("para1", "para2"):
        instruction, tail = PARAPHRASE, REPLY_ONLY
    elif kind == "verbose":
        instruction, tail = VERBOSE, REPLY_ONLY
    elif kind in CHINESE_KINDS:
        instruction, tail = TO_CHINESE, TRANSLATION_ONLY
    elif kind.startswith("hpara"):
        instruction, tail = HYPOTHESIS_PARAPHRASE + (KEEP_NO_PREFERENCE if explicit else ""), REPLY_ONLY
    elif kind == "back":
        instruction, tail = TO_ENGLISH, TRANSLATION_ONLY
    else:
        raise ValueError(kind)
    return [{"role": "user", "content": f"{instruction} {tail}\n\n{text}"}]


def build_units(items: list[dict], probe: list[dict], vocab: dict) -> list[dict]:
    """One unit per (original text, kind), with the slot tuple a valid rewrite must keep."""
    units = []
    for item in items:
        truth = annotated_tuple(item, vocab)
        for kind in UTTERANCE_KINDS:
            units.append({"unit_id": f"u:{item['item_id']}:{kind}", "side": "utterance", "kind": kind,
                          "item_id": item["item_id"], "original": item["utterance"], "expected": truth,
                          "explicit": False})
    texts = {}
    chosen = {item["item_id"] for item in items}
    for row in probe:
        if row["item_id"] in chosen:
            texts.setdefault(row["text"], row)
    for text, row in sorted(texts.items()):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        expected = {slot: [value] for slot, value in hypothesis_tuple(row).items()}
        for kind in HYPOTHESIS_KINDS:
            units.append({"unit_id": f"h:{digest}:{kind}", "side": "hypothesis", "kind": kind, "text_id": digest,
                          "original": text, "expected": expected, "explicit": row["regime"] == "expl"})
    return units


def slots_kept(extractor: Extractor, text: str | None, expected: dict[str, list[str]]) -> bool:
    if not text:
        return False
    found = extractor.extract(text)
    return all(found[slot] in expected[slot] for slot in SLOTS)


class NLI:
    """Entailment probability from the pinned DeBERTa NLI model; needs the lm extra plus sentencepiece."""

    def __init__(self, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from robust_likelihood.lmcore import pick_device

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
        self.device = pick_device(device)
        self.model.to(self.device).eval()
        labels = {name.lower(): int(index) for index, name in self.model.config.id2label.items()}
        self.entailment = labels["entailment"]
        self.batch_size = batch_size

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        found = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            encoded = self.tokenizer([premise for premise, _ in chunk], [claim for _, claim in chunk],
                                     return_tensors="pt", padding=True, truncation=True, max_length=256)
            with self.torch.no_grad():
                logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits
            found.extend(self.torch.softmax(logits.float(), dim=-1)[:, self.entailment].tolist())
        return found


def run_rewrites(
    root: Path,
    *,
    pilot: int | None = None,
    workers: int = 4,
    nli_device: str | None = None,
    label: str | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    transport=None,
    environ=None,
    sleeper=time.sleep,
    nli=None,
    now: datetime | None = None,
) -> dict:
    derived = derived or root / DERIVED_DIR
    vocab = load_vocab(derived)
    extractor = Extractor(vocab)
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    chosen = set(choose_items(items, "clean", pilot, unit_seed("rewrites:pilot") if pilot else None))
    items = [item for item in items if item["item_id"] in chosen]
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    units = build_units(items, probe, vocab)
    client, settings = api_client(root, "claude", transport=transport, environ=environ)
    started = now or datetime.now(timezone.utc)
    run_dir = (results_root or root / RESULTS_DIR / "rewrites") / (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label or ('pilot' if pilot else 'full')}")
    manifest = {
        "kind": "rewrites", "git": git_state(root), "pilot": pilot, "items": len(items), "units": len(units),
        "provider": "claude", "requested_model_id": settings["model_id"], "requested_temperature": 1.0,
        "prompts": {"para": PARAPHRASE, "verbose": VERBOSE, "hpara": HYPOTHESIS_PARAPHRASE,
                    "keep_no_preference": KEEP_NO_PREFERENCE, "zh": TO_CHINESE, "back": TO_ENGLISH,
                    "reply_only": REPLY_ONLY, "translation_only": TRANSLATION_ONLY},
        "openers": list(OPENERS), "closers": list(CLOSERS),
        "nli": {"model": NLI_MODEL, "revision": NLI_REVISION, "threshold": NLI_THRESHOLD,
                "regex_only_kinds": sorted(REGEX_ONLY_KINDS)},
        "attempts": ATTEMPTS, "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(run_dir / "manifest.json", manifest)
    temperature = {"value": 1.0}
    calls: list[dict] = []

    def ask(messages: list[dict]) -> dict:
        found = api_verbal("claude", messages, client=client, settings=settings, temperature=temperature["value"],
                           sleeper=sleeper)
        if found["temperature_rejected"]:
            temperature["value"] = None
        calls.append({key: found[key] for key in ("status", "usage", "returned_model_id", "temperature_sent",
                                                  "attempts", "latency_ms")})
        return found

    def generate(unit: dict, attempt: int) -> dict:
        rng = random.Random(unit_seed(f"{unit['unit_id']}:{attempt}"))
        record = {"unit_id": unit["unit_id"], "attempt": attempt, "rewrite": None, "back": None, "error": None}
        if unit["kind"] == "typo":
            record["rewrite"] = typo(unit["original"], rng, protected_spans(unit["original"], extractor))
        elif unit["kind"] == "pad":
            record["rewrite"] = pad(unit["original"], rng)
        else:
            answer = ask(llm_messages(unit["kind"], unit["original"], unit["explicit"]))
            record["rewrite"], record["error"] = answer["text"], answer["error"]
            record["returned_model_id"], record["temperature_sent"] = answer["returned_model_id"], answer["temperature_sent"]
            if unit["kind"] in CHINESE_KINDS and answer["text"]:
                back = ask(llm_messages("back", answer["text"]))
                record["back"], record["error"] = back["text"], back["error"]
        record["rewrite"] = record["rewrite"].strip() if record["rewrite"] else record["rewrite"]
        record["back"] = record["back"].strip() if record["back"] else record["back"]
        return record

    if nli is None:
        nli = NLI(nli_device)
    accepted: dict[str, dict] = {}
    pending = list(units)
    by_id = {unit["unit_id"]: unit for unit in units}
    if pending and any(unit["kind"] in LLM_KINDS for unit in pending):
        ask(llm_messages("para1", "I would like a cheap restaurant."))  # one call first settles the temperature
    for attempt in range(ATTEMPTS):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            records = list(pool.map(lambda unit: generate(unit, attempt), pending))
        checked = []
        for record in records:
            unit = by_id[record["unit_id"]]
            english = record["back"] if unit["kind"] in CHINESE_KINDS else record["rewrite"]
            record["english"] = english
            record["slots_kept"] = slots_kept(extractor, english, unit["expected"])
            record["changed"] = bool(english) and normalize_text(english) != normalize_text(unit["original"])
            if record["slots_kept"] and english and unit["kind"] not in REGEX_ONLY_KINDS:
                checked.append(record)
        scores = nli.entailment_probs(
            [(by_id[record["unit_id"]]["original"], record["english"]) for record in checked]
            + [(record["english"], by_id[record["unit_id"]]["original"]) for record in checked]
        )
        for index, record in enumerate(checked):
            record["nli_forward"], record["nli_backward"] = scores[index], scores[len(checked) + index]
        for record in records:
            if by_id[record["unit_id"]]["kind"] in REGEX_ONLY_KINDS:
                record.update(nli_forward=None, nli_backward=None, accepted=bool(record["slots_kept"]))
            else:
                record["accepted"] = bool(record["slots_kept"] and record.get("nli_forward") is not None
                                          and record["nli_forward"] >= NLI_THRESHOLD
                                          and record["nli_backward"] >= NLI_THRESHOLD)
            if record["accepted"]:
                accepted[record["unit_id"]] = record
        append_jsonl(run_dir / "candidates.jsonl", records)
        pending = [unit for unit in pending if unit["unit_id"] not in accepted]
    rates = {}
    for kind in UTTERANCE_KINDS + HYPOTHESIS_KINDS:
        mine = [unit for unit in units if unit["kind"] == kind]
        if mine:
            rates[kind] = {"units": len(mine), "accepted": sum(1 for unit in mine if unit["unit_id"] in accepted),
                           "rate": sum(1 for unit in mine if unit["unit_id"] in accepted) / len(mine)}
    usage_in = sum(int((call["usage"] or {}).get("input_tokens") or 0) for call in calls)
    usage_out = sum(int((call["usage"] or {}).get("output_tokens") or 0) for call in calls)
    manifest.update({
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature_sent": temperature["value"],
        "returned_model_ids": sorted({call["returned_model_id"] for call in calls if call["returned_model_id"]}),
        "api_calls": len(calls),
        "api_failures": sum(1 for call in calls if call["status"] != "success"),
        "input_tokens": usage_in,
        "output_tokens": usage_out,
        "acceptance": rates,
        "accepted": len(accepted),
    })
    write_json(run_dir / "manifest.json", manifest)
    write_jsonl(run_dir / "accepted.jsonl", [
        {**{key: by_id[unit_id][key] for key in ("unit_id", "side", "kind", "original")},
         **{key: by_id[unit_id].get(key) for key in ("item_id", "text_id")},
         "rewrite": record["rewrite"], "back": record["back"], "attempt": record["attempt"],
         "nli_forward": record.get("nli_forward"), "nli_backward": record.get("nli_backward")}
        for unit_id, record in sorted(accepted.items())
    ])
    return {"run_dir": str(run_dir), "items": len(items), "units": len(units),
            "llm_units": sum(1 for unit in units if unit["kind"] in LLM_KINDS), "accepted": len(accepted),
            "unchanged": sum(1 for record in accepted.values() if not record["changed"]),
            "api_calls": len(calls), "input_tokens": usage_in, "output_tokens": usage_out, "acceptance": rates,
            "temperature_sent": temperature["value"]}


def full_llm_units(derived: Path) -> int:
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    texts = {row["text"] for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"}
    return len(items) * len(LLM_KINDS & set(UTTERANCE_KINDS)) + len(texts) * len(HYPOTHESIS_KINDS)


def cost_estimate(summary: dict, full_units: int, price: dict) -> dict:
    """Scale the pilot's calls and tokens per LLM-generated unit to the full run."""
    scale = full_units / summary["llm_units"] if summary["llm_units"] else 0
    found = {"pilot_calls": summary["api_calls"], "full_llm_units": full_units,
             "full_calls_estimate": round(summary["api_calls"] * scale),
             "full_input_tokens": round(summary["input_tokens"] * scale),
             "full_output_tokens": round(summary["output_tokens"] * scale)}
    found["full_usd"] = round((found["full_input_tokens"] * price["input"] + found["full_output_tokens"] * price["output"])
                              / 1e6, 2)
    return found


def audit_rows(accepted: list[dict], seed: int = SEED, per_side: int = 50) -> list[dict]:
    rng = random.Random(unit_seed("rewrites:audit") ^ seed)
    rows = []
    for side in ("utterance", "hypothesis"):
        mine = sorted((row for row in accepted if row["side"] == side), key=lambda row: row["unit_id"])
        for row in sorted(rng.sample(mine, min(per_side, len(mine))), key=lambda row: row["unit_id"]):
            rows.append({"unit_id": row["unit_id"], "side": side, "kind": row["kind"], "original": row["original"],
                         "rewrite": row["rewrite"], "back_translation": row.get("back") or "",
                         "meaning_preserved (y/n)": "", "note": ""})
    return rows


def acceptance_markdown(manifest: dict, accepted: list[dict]) -> str:
    same = [row for row in accepted if row["kind"] not in CHINESE_KINDS
            and normalize_text(row["rewrite"]) == normalize_text(row["original"])]
    lines = ["# Rewrite acceptance", "",
             f"Run started {manifest['started_at']}: {manifest['items']} clean items, {manifest['units']} rewrite units, "
             f"{manifest['api_calls']} calls to `{', '.join(manifest['returned_model_ids'])}` "
             f"(temperature sent: {manifest['temperature_sent']}), {manifest['input_tokens']:,} input and "
             f"{manifest['output_tokens']:,} output tokens.", "",
             "A rewrite is kept if its regex slot tuple is unchanged and NLI entailment is at least "
             f"{manifest['nli']['threshold']} both ways (`{manifest['nli']['model']}`); "
             f"{', '.join(manifest['nli']['regex_only_kinds'])} is checked by regex only. Up to "
             f"{manifest['attempts']} attempts per unit.", "",
             "| Kind | Accepted | Units | Rate |", "|---|---|---|---|"]
    lines += [f"| {kind} | {found['accepted']} | {found['units']} | {100 * found['rate']:.1f}% |"
              for kind, found in manifest["acceptance"].items()]
    lines += ["", f"Accepted rewrites identical to their original (not counting Chinese, whose back-translation may "
              f"match): {len(same)}" + (f" ({', '.join(row['unit_id'] for row in same)})." if same else "."), ""]
    return "\n".join(lines)


def robustness_subset(items: list[dict], size: int = 60) -> list[str]:
    return choose_items(items, "clean", size, unit_seed("rewrites:robustness"))


def write_derived(run_dir: Path, derived: Path, items: list[dict]) -> dict:
    accepted = read_jsonl(run_dir / "accepted.jsonl")
    kept = [row for row in accepted if row["unit_id"] not in AUDIT_REJECTED]
    write_jsonl(derived / "rewrites.jsonl", kept)
    subset = robustness_subset(items)
    (derived / "robustness_subset.json").write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8",
                                                    newline="\n")
    return {"rewrites": len(kept), "audit_rejected": len(accepted) - len(kept), "robustness_subset": len(subset)}
