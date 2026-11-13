"""Simulator study, Phase 3 (docs/simulator_study_plan.md): metrics for every (model, format) run.

Inputs are the finished sim-score runs, plus Phase 8 and claims job A as `qwen25_inst|P0`. Every
metric is computed per run on the 235 clean items; intervals come from the item bootstrap of
claim1.py (2,000 replicates, seed 20261103, configs/simulator.json). The machinery is the earlier
code's: claim1.py for Q3, metrics.py for the raw Occam gap, typv0.py for Q4.

Q1 coverage. A pair's human percentile is the share of the hypothesis's own samples with S >= S(U|H).
A sample cut off at max_new_tokens counts as >= when its lower bound S_lower already is, and is
left out otherwise. Semantic coverage asks whether any sample under the gold hypothesis carries U's
annotated slot tuple (the regex extractor); style compares each human turn with one sample under
its gold hypothesis (the first that ended). The typo proxy is a word outside the MultiWOZ 2.2 train
vocabulary, counting words that occur in at least DICTIONARY_DIALOGUES dialogues.
Q2 fidelity. A sample names a slot when the extractor finds a value for it (not none, not
dontcare). a_model(X): samples under the gold hypothesis naming constrained X; b_model(X): samples
under minus:X naming X. ln(a/b) from pooled counts, b smoothed as (n + 0.5) / (N + 1).
Q5 concentration over the clean hypothesis texts, and paired item-bootstrap differences.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, load_config
from robust_likelihood.multiwoz.claim1 import ItemBoot, Scorer, _np, item_meta, posterior, rewording, token_log, utterance
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.metrics import analyze_scorer, goal_key
from robust_likelihood.multiwoz.semantic import DONTCARE, NONE, annotated_tuple, load_vocab
from robust_likelihood.multiwoz.sim import load_sim_config
from robust_likelihood.multiwoz.typv0 import (
    Inputs,
    bin_agreement,
    corpus_bootstrap,
    load_typv0_config,
    matches,
    moments,
    occupancy,
    p_sem,
    pool_weights,
    score_map,
    size_and_information,
    variant_scores,
)
from robust_likelihood.storage import read_json, read_jsonl, write_json

DICTIONARY_DIALOGUES = 5
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
GREETING = re.compile(r"^\W*(?:hi|hello|hey|hiya|howdy|greetings|good (?:morning|afternoon|evening|day))\b",
                      re.IGNORECASE)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
PAIRED = (("osim_mid|S", "osim_final|S"), ("qwen3_base|F", "osim_mid|S"), ("qwen3_inst|S", "osim_final|S"))
INSTRUCT = ("qwen25_inst|S", "qwen3_inst|S")
P0_LABEL = "qwen25_inst|P0"


# ---------------------------------------------------------------- pure helpers


def human_percentile(surprisal: float, samples: list[dict]) -> float | None:
    """Share of samples at least as surprising as the human turn; cut-off samples by their lower bound."""
    above = counted = 0
    for row in samples:
        if row.get("S") is not None:
            counted += 1
            above += row["S"] >= surprisal
        elif row.get("S_lower") is not None and row["S_lower"] >= surprisal:
            counted += 1
            above += 1
    return above / counted if counted else None


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {f"q{int(q * 100):02d}": ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
            for q in QUANTILES}


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def distinct_2(texts: list[str]) -> float | None:
    bigrams = [pair for text in texts for pair in zip(words(text), words(text)[1:])]
    return len(set(bigrams)) / len(bigrams) if bigrams else None


def style(texts: list[str], dictionary: set[str]) -> dict:
    if not texts:
        return {"n": 0}
    return {"n": len(texts), "median_words": statistics.median(len(text.split()) for text in texts),
            "greeting": sum(1 for text in texts if GREETING.search(text)) / len(texts),
            "question": sum(1 for text in texts if "?" in text) / len(texts),
            "non_dictionary": sum(1 for text in texts if any(word not in dictionary for word in words(text))) / len(texts),
            "distinct_2": distinct_2(texts)}


def corpus_dictionary(raw: Path, minimum: int = DICTIONARY_DIALOGUES) -> set[str]:
    """Words in at least `minimum` MultiWOZ 2.2 train dialogues: a typo seldom recurs that often."""
    counts: Counter = Counter()
    for path in sorted((raw / "MultiWOZ_2.2" / "train").glob("dialogues_*.json")):
        for dialogue in json.loads(path.read_text(encoding="utf-8")):
            counts.update({word for turn in dialogue["turns"] for word in words(turn["utterance"])})
    return {word for word, count in counts.items() if count >= minimum}


def named(value: str) -> bool:
    return value not in (NONE, DONTCARE)


def weighted_replicates(boot: ItemBoot, item_ids: list[str], values: list[float], kind: str):
    """Replicate statistics (mean or median) with the bootstrap's own draws, as ItemBoot computes them."""
    np = _np()
    values = np.asarray(values, dtype=float)
    index = np.asarray([boot.index[item] for item in item_ids], dtype=np.intp)
    weights = boot.counts[:, index].astype(float)
    if kind == "mean":
        totals = weights.sum(axis=1)
        return np.where(totals > 0, (weights @ values) / np.maximum(totals, 1), np.nan)
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[:, order], axis=1)
    total = cumulative[:, -1:]
    position = (cumulative >= total / 2).argmax(axis=1)
    return np.where(total[:, 0] > 0, values[order][position], np.nan)


def paired_difference(boot: ItemBoot, first: tuple[list, list], second: tuple[list, list], kind: str) -> dict:
    """second - first, both resampled with the same item draws."""
    np = _np()
    if not first[1] or not second[1]:
        return {"difference": None, "ci95": None}
    point = (statistics.fmean if kind == "mean" else statistics.median)
    replicates = weighted_replicates(boot, *second, kind) - weighted_replicates(boot, *first, kind)
    return {"difference": point(second[1]) - point(first[1]),
            "ci95": boot.interval([value for value in replicates.tolist() if not np.isnan(value)])}


# ---------------------------------------------------------------- one run


class Run:
    """One (model, format): its human scores, its samples, and typv0's Inputs on them."""

    def __init__(self, root: Path, derived: Path, label: str, scores_dir: Path, lm_dirs: list[Path], model_key: str,
                 primary: bool, sources: list[str]):
        self.label, self.primary, self.sources = label, primary, sources
        self.inputs = Inputs(root, scores_dir, lm_dirs, derived, model_key=model_key)
        self.scores = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(scores_dir / "lm_scores.jsonl")
                       if row["set"] == "clean"}
        pool = set(self.inputs.pool)
        self.samples: dict[str, list[dict]] = {}
        for row in read_jsonl(scores_dir / "lm_samples.jsonl"):
            if row["text"] in pool:
                self.samples.setdefault(row["text"], []).append(row)
        self.has_rewrites = bool(self.inputs.lm.get("urewrite") or self.inputs.lm.get("hrewrite"))

    def log(self) -> dict:
        return token_log({key: -value for key, value in self.scores.items()}, self.inputs.lm, self.inputs.clean)


def coverage(run: Run, metas: dict, boot: ItemBoot, vocab: dict, dictionary: set[str]) -> tuple[dict, dict]:
    """Q1 metrics, and the per-pair rows the paired differences reuse."""
    rows = {"all": [], "gold": []}
    per_word_human, human_items, per_word_sample, sample_items = [], [], [], []
    semantic_items, covered, p_values = [], [], []
    human_texts, sample_texts = [], []
    human_gold, sample_gold = [], []
    for item, meta in sorted(metas.items()):
        gold = goal_key(len(meta["constrained"]), "impl")
        for key in meta["keys"]:
            value, text = run.scores.get((item, key)), meta["texts"][key]
            found = run.samples.get(text)
            if value is None or not found:
                continue
            ended = [row["S"] for row in found if row.get("S") is not None]
            moment = moments(ended)
            row = {"item": item, "key": key, "percentile": human_percentile(value, found),
                   "z": (value - moment[0]) / moment[1] if moment and moment[1] > 0 else None,
                   "mu": moment[0] if moment else None}
            rows["all"].append(row)
            if key == gold:
                rows["gold"].append(row)
                utterance_text = meta["item"]["utterance"]
                per_word_human.append(value / max(1, len(utterance_text.split())))
                human_items.append(item)
                human_gold.append(value)
                for sample in found:
                    if sample.get("S") is not None:
                        per_word_sample.append(sample["S"] / max(1, len(sample["reply"].split())))
                        sample_items.append(item)
                        sample_gold.append(sample["S"])
                tuples = run.inputs.samples.get(text, {}).get("tuples", [])
                if tuples:
                    wanted = annotated_tuple(meta["item"], vocab)
                    semantic_items.append(item)
                    covered.append(float(any(matches(found_tuple, wanted) for found_tuple in tuples)))
                    p_values.append(p_sem(tuples, wanted))
                first = next((sample for sample in sorted(found, key=lambda s: s["sample"]) if sample.get("S") is not None),
                             None)
                if first is not None:
                    human_texts.append(utterance_text)
                    sample_texts.append(first["reply"])
    result = {}
    for scope, found in rows.items():
        usable = [row for row in found if row["percentile"] is not None]
        ids = [row["item"] for row in usable]
        z_rows = [row for row in found if row["z"] is not None]
        zs = sorted(row["z"] for row in z_rows)
        result[scope] = {
            "pairs": len(found),
            "median_percentile": boot.median(ids, [row["percentile"] for row in usable]),
            "share_below_005": boot.mean(ids, [float(row["percentile"] < 0.05) for row in usable]),
            "share_beyond_all": boot.mean(ids, [float(row["percentile"] == 0) for row in usable]),
            "z_median": boot.median([row["item"] for row in z_rows], [row["z"] for row in z_rows]),
            "z_p90": zs[int(0.9 * (len(zs) - 1))] if zs else None,
        }
    result["per_word"] = {"human": boot.median(human_items, per_word_human),
                          "samples": boot.median(sample_items, per_word_sample)}
    result["semantic"] = {"items": len(semantic_items), "covered": boot.mean(semantic_items, covered),
                          "median_p_sem": boot.median(semantic_items, p_values)}
    result["style"] = {"human": style(human_texts, dictionary), "samples": style(sample_texts, dictionary)}
    result["gold_surprisal"] = {"human": quantiles(human_gold), "samples": quantiles(sample_gold)}
    return result, rows


def naming(run: Run, metas: dict, boot: ItemBoot, reference: dict) -> dict:
    """Q2: slot-naming rates of the model's samples next to the corpus a and b."""
    position = {slot: index for index, slot in enumerate(SLOTS)}
    a_rows: dict[str, list] = {slot: [] for slot in SLOTS}
    b_rows: dict[tuple[str, str], list] = {(slot, regime): [] for slot in SLOTS for regime in ("impl", "expl")}
    for item, meta in sorted(metas.items()):
        gold = run.inputs.samples.get(meta["texts"].get(goal_key(len(meta["constrained"]), "impl"), ""), {}).get("tuples")
        for slot in meta["constrained"]:
            if gold:
                a_rows[slot].append((item, sum(named(found[position[slot]]) for found in gold), len(gold)))
            for regime in ("impl", "expl"):
                text = meta["texts"].get(f"minus:{slot}|{regime}")
                tuples = run.inputs.samples.get(text, {}).get("tuples") if text else None
                if tuples:
                    b_rows[(slot, regime)].append((item, sum(named(found[position[slot]]) for found in tuples), len(tuples)))

    def rate(rows: list) -> dict:
        return boot.mean([row[0] for row in rows], [row[1] / row[2] for row in rows]) if rows else {"n": 0, "mean": None}

    result = {"slots": {}}
    for slot in SLOTS:
        a_named, a_total = sum(row[1] for row in a_rows[slot]), sum(row[2] for row in a_rows[slot])
        corpus = reference["slots"][slot]
        block = {"a_model": rate(a_rows[slot]), "a_corpus": corpus["a"], "b_corpus": corpus["b"],
                 "ln_a_over_b_corpus": corpus["named"]}
        for regime in ("impl", "expl"):
            rows = b_rows[(slot, regime)]
            b_named, b_total = sum(row[1] for row in rows), sum(row[2] for row in rows)
            block[f"b_model_{regime}"] = rate(rows)
            block[f"ln_a_over_b_{regime}"] = (math.log((a_named / a_total) / ((b_named + 0.5) / (b_total + 1)))
                                              if a_total and b_total and a_named else None)
        result["slots"][slot] = block
    every_a = [row for slot in SLOTS for row in a_rows[slot]]
    result["slots_mean"] = {"a_model": rate(every_a),
                            "a_corpus": statistics.fmean(reference["slots"][slot]["a"] for slot in SLOTS),
                            "b_corpus": statistics.fmean(reference["slots"][slot]["b"] for slot in SLOTS)}
    for regime in ("impl", "expl"):
        result["slots_mean"][f"b_model_{regime}"] = rate([row for slot in SLOTS for row in b_rows[(slot, regime)]])
    return result


def occam_summary(items: list[dict], log: dict, changed: dict, corpus_boot) -> dict:
    """Q2: the raw log-likelihood Occam gap with scores -S, from metrics.py."""
    block = analyze_scorer(items, score_map(log), changed, corpus_boot)
    keep = {key: value for key, value in block["occam"].items() if "|slots_mean|" in key or "|bin|" in key}
    return {"occam": keep, "wording_gap": block["wording_gap"].get("all"),
            "contra": {name: block["controls"]["ctrl:contra"].get(name) for name in ("n", "auroc", "mean_gap",
                                                                                    "mean_gap_ci95")}}


def typicality(run: Run, typ: dict, changed: dict, corpus_boot, boot: ItemBoot, clean_ids: list[str]) -> dict:
    """Q4: typv0's variants with this model as the reference; occupancy under both weightings, the
    size principle and contradiction gap under uniform weighting, bin agreement where rewrites exist."""
    weights = {scheme: pool_weights(run.inputs.probe, scheme) for scheme in typ["weightings"]}
    specs = dict(typ["variants"])
    specs["cent1B:6bins"] = dict(typ["variants"]["cent1B"], edges=typ["sensitivity"]["edges"]["6bins"])
    result = {}
    for name, spec in specs.items():
        schemes = ["uniform"] if spec["normalization"] == "per" else list(typ["weightings"])
        block = {"spec": spec}
        for scheme in schemes:
            log, bins, pool = variant_scores(run.inputs, spec, weights[scheme])
            found = occupancy(bins, len(spec["edges"]) + 1)
            block[scheme] = {"pool": pool, "scores": len(log), "occupancy": found, "extreme_share": found[-1]}
            if scheme == "uniform":
                block[scheme]["size"] = size_and_information(run.inputs.items, score_map(log), changed, corpus_boot)
                if run.has_rewrites:
                    block[scheme]["bin_agreement"] = bin_agreement(bins, clean_ids, boot)
        result[name] = block
    return result


def concentration(run: Run) -> dict:
    means, distinct, top = [], [], []
    for rows in run.samples.values():
        ended = [row["S"] for row in rows if row.get("S") is not None]
        if ended:
            means.append(statistics.fmean(ended))
        replies = Counter(" ".join(row["reply"].split()) for row in rows)
        distinct.append(len(replies))
        top.append(max(replies.values()) / len(rows))
    return {"texts": len(run.samples), "mean_sample_surprisal": statistics.fmean(means) if means else None,
            "distinct_per_text": statistics.fmean(distinct) if distinct else None,
            "top_share": statistics.fmean(top) if top else None,
            "samples_per_text": statistics.median(len(rows) for rows in run.samples.values()) if run.samples else None}


# ---------------------------------------------------------------- predictions


def _get(block, *path):
    for key in path:
        if block is None:
            return None
        block = block.get(key)
    return block


def predictions(runs: dict) -> list[dict]:
    """S1-S6 of the plan, on the primary runs."""
    primary = {label: block for label, block in runs.items() if block["primary"]}

    def pct(label):
        return _get(primary.get(label), "q1", "all", "median_percentile", "median")

    def beyond(label):
        return _get(primary.get(label), "q1", "all", "share_beyond_all", "mean")

    found = []
    ranked = sorted((label for label in primary if pct(label) is not None), key=pct, reverse=True)
    lowest_beyond = min((label for label in primary if beyond(label) is not None), key=beyond, default=None)
    worst_two = ranked[-2:]
    found.append({"id": "S1", "statement": "osim_mid has the best coverage; the instruct models the worst",
                  "value": {"ranking_by_median_percentile": [(label, pct(label)) for label in ranked],
                            "lowest_share_beyond_all": lowest_beyond},
                  "pass": bool(ranked) and ranked[0] == "osim_mid|S" and lowest_beyond == "osim_mid|S"
                  and set(worst_two) == set(INSTRUCT)})
    mid, final = primary.get("osim_mid|S"), primary.get("osim_final|S")
    if mid and final:
        mid_s, final_s = _get(mid, "q5", "mean_sample_surprisal"), _get(final, "q5", "mean_sample_surprisal")
        drop = None if not mid_s or final_s is None else 1 - final_s / mid_s
        found.append({"id": "S2", "statement": "osim_final's mean sample surprisal >= 30% below osim_mid's, "
                                               "and its coverage worse",
                      "value": {"relative_drop": drop, "median_percentile": [pct("osim_mid|S"), pct("osim_final|S")]},
                      "pass": drop is not None and drop >= 0.30 and (pct("osim_final|S") or 0) < (pct("osim_mid|S") or 0)})
    if mid:
        a = _get(mid, "q2", "naming", "slots_mean", "a_model", "mean")
        b = _get(mid, "q2", "naming", "slots_mean", "b_model_impl", "mean")
        a_corpus = _get(mid, "q2", "naming", "slots_mean", "a_corpus")
        b_corpus = _get(mid, "q2", "naming", "slots_mean", "b_corpus")
        instruct_a = {label: _get(primary.get(label), "q2", "naming", "slots_mean", "a_model", "mean") for label in INSTRUCT}
        ok = (a is not None and b is not None and abs(a - a_corpus) <= 0.10 and b_corpus / 3 <= b <= b_corpus * 3
              and all(value is not None and value >= 0.9 for value in instruct_a.values()))
        found.append({"id": "S3", "statement": "osim_mid names slots at corpus rates (a within 0.10, b within x3); "
                                               "instruct models a >= 0.9",
                      "value": {"osim_mid_a": a, "osim_mid_b_impl": b, "corpus_a": a_corpus, "corpus_b": b_corpus,
                                "instruct_a": instruct_a}, "pass": ok})
    shifts = {label: {group: _get(block, "q3", "utterance", group, "abs_delta", "median") for group in ("para", "zh")}
              for label, block in primary.items() if _get(block, "q3")}
    found.append({"id": "S4", "statement": "claim 1 holds for every model: median |delta log BF| under para and zh > 2 nats",
                  "value": shifts, "pass": bool(shifts) and all(value is not None and value > 2
                                                                for found_shift in shifts.values()
                                                                for value in found_shift.values())})
    if mid:
        share = _get(mid, "q4", "glob1A", "uniform", "extreme_share")
        found.append({"id": "S5", "statement": "with osim_mid as reference, global one-sided typicality (glob1A) puts "
                                               "< 50% of human pairs in its extreme bin",
                      "value": share, "pass": share is not None and share < 0.5})
    found.append({"id": "S6", "statement": "osim_mid beats userlm on coverage; qwen25_broad beats qwen25_inst but "
                                           "stays below osim_mid",
                  "value": {label: (pct(label), beyond(label)) for label in ("osim_mid|S", "userlm|S", "qwen25_broad|B",
                                                                             "qwen25_inst|S")},
                  "pass": None not in (pct("osim_mid|S"), pct("userlm|S"), pct("qwen25_broad|B"), pct("qwen25_inst|S"))
                  and pct("osim_mid|S") > pct("userlm|S") and beyond("osim_mid|S") < beyond("userlm|S")
                  and pct("qwen25_inst|S") < pct("qwen25_broad|B") < pct("osim_mid|S")})
    return found


# ---------------------------------------------------------------- run


def discover(run_dirs: list[Path]) -> dict[str, Path]:
    """The newest finished sim-score run per (model, format)."""
    found: dict[str, Path] = {}
    for run_dir in sorted(run_dirs):
        path = run_dir / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        if manifest.get("kind") != "sim-score" or "finished_at" not in manifest:
            continue
        found[f"{manifest['model_key']}|{manifest['format']}"] = run_dir
    return found


def run_sim_analyze(root: Path, *, runs: list[Path], phase8: Path | None, p0_lm: list[Path], raw: Path, out: Path,
                    derived: Path | None = None, labels: list[str] | None = None) -> dict:
    clock = time.monotonic()
    derived = derived or root / DERIVED_DIR
    config, sim, typ = load_config(root), load_sim_config(root), load_typv0_config(root)
    seed, replicates = int(sim["bootstrap"]["seed"]), int(sim["bootstrap"]["replicates"])
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    clean_ids = sorted(item["item_id"] for item in items)
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    metas = item_meta(items, probe)
    vocab = load_vocab(derived)
    changed = {(row["item_id"], row["hyp_key"]): row["changed"]["slot"]
               for row in read_jsonl(derived / "probe.jsonl") if "changed" in row}
    reference = read_json(derived / "reference_table.json")["variants"]["primary"]
    corpus_boot = corpus_bootstrap(derived, config, items)
    boot = ItemBoot(clean_ids, replicates, seed)
    dictionary = corpus_dictionary(raw)

    chosen = discover(runs)
    sources: dict[str, tuple] = {}
    for label, run_dir in sorted(chosen.items()):
        manifest = read_json(run_dir / "manifest.json")
        sources[label] = (run_dir, [run_dir], manifest["model_key"], bool(manifest.get("primary")), [run_dir.as_posix()])
    if phase8 is not None:
        sources[P0_LABEL] = (phase8, p0_lm, "qwen", False, [phase8.as_posix()] + [path.as_posix() for path in p0_lm])
    if labels:
        sources = {label: found for label, found in sources.items() if label in labels}

    metrics: dict = {"runs": {}, "paired": {}}
    rows_of: dict[str, dict] = {}
    examples: dict[str, list[str]] = {}
    example_item = sorted(metas)[0]
    for label, (scores_dir, lm_dirs, model_key, primary, paths) in sources.items():
        run = Run(root, derived, label, scores_dir, lm_dirs, model_key, primary, paths)
        q1, rows_of[label] = coverage(run, metas, boot, vocab, dictionary)
        log = run.log()
        block = {"primary": primary, "sources": paths, "pairs_scored": len(run.scores),
                 "q1": q1,
                 "q2": {"naming": naming(run, metas, boot, reference), "raw_occam": occam_summary(items, log, changed,
                                                                                                  corpus_boot)},
                 "q4": typicality(run, typ, changed, corpus_boot, boot, clean_ids),
                 "q5": concentration(run)}
        if run.has_rewrites:
            scorer = Scorer(label, "loglik", log, items=clean_ids)
            block["q3"] = {"rewording": rewording(scorer, metas, boot), "utterance": utterance(scorer, metas, boot),
                           "posterior": posterior(scorer, metas, boot, seed)}
        gold_text = metas[example_item]["texts"][goal_key(len(metas[example_item]["constrained"]), "impl")]
        examples[label] = [row["reply"] for row in sorted(run.samples.get(gold_text, []), key=lambda r: r["sample"])[:10]]
        metrics["runs"][label] = block
    for first, second in PAIRED:
        if first in rows_of and second in rows_of:
            pair = {}
            for name, scope, field, kind in (("median_percentile", "all", "percentile", "median"),
                                             ("share_beyond_all", "all", "percentile", "beyond"),
                                             ("mean_mu", "all", "mu", "mean"),
                                             ("gold_z_median", "gold", "z", "median")):
                def column(label):
                    rows = [row for row in rows_of[label][scope] if row[field] is not None]
                    values = [float(row[field] == 0) if kind == "beyond" else row[field] for row in rows]
                    return [row["item"] for row in rows], values

                pair[name] = paired_difference(boot, column(first), column(second), "mean" if kind != "median" else "median")
            metrics["paired"][f"{second} - {first}"] = pair
    metrics["examples"] = {"item_id": example_item, "samples": examples}
    metrics["predictions"] = predictions(metrics["runs"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    manifest = {"kind": "sim-analyze", "git": git_state(root),
                "inputs": {label: found[4] for label, found in sources.items()},
                "bootstrap": {"replicates": replicates, "seed": seed},
                "dictionary": {"rule": f"words in >= {DICTIONARY_DIALOGUES} MultiWOZ 2.2 train dialogues",
                               "size": len(dictionary)},
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_seconds": round(time.monotonic() - clock, 1)}
    write_json(out / "manifest.json", manifest)
    return {"out": str(out), "runs": sorted(metrics["runs"]), "predictions": metrics["predictions"],
            "elapsed_seconds": manifest["elapsed_seconds"]}
