"""Claims plan 2e: claim 1 (token log-probs are correct but fragile) and claim 2 (verbalized
probabilities disagree with generation probabilities), from the outputs of jobs A and B.

Every score is read on a log scale: language-model log-likelihood l = -S in nats, its clipped
form max(l_H - max_H' l_H', -3) and its centered form l_H - l_broad|impl; the natural log of a
verbalized probability (an answer of 0 has no log: it is counted and left out of log-scale
metrics, and enters posteriors as probability 0); Jev's yes-probability on the logit scale, as
metrics.py reads it. Robustness is measured on log Bayes factors and posteriors, never on raw
log p (the plan's frozen assumption); raw shifts are reported only for context.

A score is keyed (item, hyp_key, hypothesis variant, utterance variant). A variant is
`original` or a rewrite kind; a rewrite changes one side only. A semantic hypothesis is a
hyp_key (type and regime) of an item; its textual variants are the original text and its
accepted rewrites. Bootstrap intervals resample items (2,000 replicates, seed 20261027); one
scorer's metrics share one resample of its items.

numpy (with the figures extra) is imported only to analyze.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR
from robust_likelihood.multiwoz.corpus import SLOTS
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.metrics import REGIMES, analyze, goal_key, logit_score
from robust_likelihood.multiwoz.rewrites import slots_kept
from robust_likelihood.multiwoz.semantic import Extractor, annotated_tuple, load_vocab
from robust_likelihood.storage import read_json, read_jsonl, write_json

ORIGINAL = "original"
H_KINDS = ("hpara1", "hpara2", "hpara3", "hzh")
U_KINDS = ("typo", "pad", "para1", "para2", "verbose", "zh")
U_GROUPS = {"typo": ("typo",), "pad": ("pad",), "para": ("para1", "para2"), "verbose": ("verbose",), "zh": ("zh",)}
LN10 = math.log(10)
POSTERIOR_DRAWS = 50

SCORER_LABELS = {
    "qwen_tok": "Qwen token log-prob",
    "qwen_tok_subset": "Qwen token log-prob, 60-item subset",
    "qwen_clip": "Qwen clipped log-prob",
    "qwen_cent": "Qwen centered log-prob",
    "olmo_tok": "OLMo token log-prob",
    "qwen_verbal_exact": "Qwen verbalized, exact string",
    "qwen_verbal_meaning": "Qwen verbalized, meaning",
    "claude_verbal": "Claude verbalized, exact string",
    "gpt_verbal": "GPT verbalized, exact string",
    "jev_likely": "Jev, likely wording",
    "jev_typical": "Jev, typical wording",
}
SCORER_ORDER = tuple(SCORER_LABELS)


def _np():
    import numpy

    return numpy


# ---------------------------------------------------------------- scores


class Scorer:
    """Log-scale scores, and the likelihood weights a posterior uses: exp of the log for language
    models (taken as a softmax), the answered probability for verbalized scorers and Jev."""

    def __init__(self, name: str, kind: str, log: dict, like: dict | None = None, items=None, notes=None):
        self.name = name
        self.label = SCORER_LABELS.get(name, name)
        self.kind = kind  # "loglik" or "prob"
        self.log = log
        self.like = like if like is not None else {}
        self.items = sorted(items if items is not None else {key[0] for key in (self.like or self.log)})
        self.notes = notes or {}

    def restricted(self, name: str, items) -> "Scorer":
        chosen = set(items)
        return Scorer(name, self.kind, {key: value for key, value in self.log.items() if key[0] in chosen},
                      {key: value for key, value in self.like.items() if key[0] in chosen},
                      sorted(chosen & set(self.items)), dict(self.notes))


def read_lm_runs(dirs: list[Path]) -> dict[str, dict[str, list[dict]]]:
    """model key -> part -> rows, over every claim1-lm run folder given."""
    found: dict[str, dict[str, list[dict]]] = {}
    for run_dir in dirs:
        manifest = read_json(run_dir / "manifest.json")
        if "finished_at" not in manifest:
            raise ValueError(f"{run_dir} did not finish")
        mine = found.setdefault(manifest["model_key"], {})
        for part in manifest["parts"]:
            path = run_dir / f"{part}.jsonl"
            if path.exists():
                mine.setdefault(part, []).extend(read_jsonl(path))
            if part == "hsample":
                mine.setdefault("hsample_hypotheses", []).extend(read_jsonl(run_dir / "hsample_hypotheses.jsonl"))
    return found


def token_log(originals: dict[tuple[str, str], float], parts: dict[str, list[dict]], clean: set[str]) -> dict:
    found = {(item, hyp, ORIGINAL, ORIGINAL): value for (item, hyp), value in originals.items() if item in clean}
    for row in parts.get("original", []):
        if row["item_id"] in clean:
            found[(row["item_id"], row["hyp_key"], ORIGINAL, ORIGINAL)] = -row["S"]
    for row in parts.get("hrewrite", []):
        if row["item_id"] in clean:
            found[(row["item_id"], row["hyp_key"], row["kind"], ORIGINAL)] = -row["S"]
    for row in parts.get("urewrite", []):
        if row["item_id"] in clean:
            found[(row["item_id"], row["hyp_key"], ORIGINAL, row["kind"])] = -row["S"]
    return found


def clipped_log(log: dict, clip: float) -> dict:
    """max(l_H - m, -clip), m the best original hypothesis of the item under the same utterance variant."""
    best: dict[tuple[str, str], float] = {}
    for (item, _, hvar, uvar), value in log.items():
        if hvar == ORIGINAL:
            best[(item, uvar)] = max(value, best.get((item, uvar), -math.inf))
    return {key: max(value - best[(key[0], key[3])], -clip) for key, value in log.items() if (key[0], key[3]) in best}


def centered_log(log: dict) -> dict:
    """l_H - l_broad|impl, the broad implicit hypothesis in its original text under the same utterance."""
    return {key: value - log[(key[0], "broad|impl", ORIGINAL, key[3])] for key, value in log.items()
            if (key[0], "broad|impl", ORIGINAL, key[3]) in log}


def _variant(source: str, side: str) -> tuple[str, str]:
    if source == ORIGINAL or side == "none":
        return ORIGINAL, ORIGINAL
    return (source, ORIGINAL) if side == "hypothesis" else (ORIGINAL, source)


def verbal_scorers(rows: list[dict]) -> tuple[list[Scorer], list[dict]]:
    """One scorer per (model, framing); normalization rows are returned apart."""
    logs: dict[str, dict] = {}
    likes: dict[str, dict] = {}
    notes: dict[str, dict] = {}
    norm = []
    for row in rows:
        name = f"qwen_verbal_{row['framing']}" if row["model"] == "qwen" else f"{row['model']}_verbal"
        if row["source"] == "norm":
            norm.append(dict(row, scorer=name))
            continue
        note = notes.setdefault(name, {"queries": 0, "failed_calls": 0, "unparsed": 0, "zero": 0})
        note["queries"] += 1
        if row.get("status", "success") != "success":
            note["failed_calls"] += 1
            continue
        if row["p"] is None:
            note["unparsed"] += 1
            continue
        hvar, uvar = _variant(row["source"], row["side"])
        key = (row["item_id"], row["hyp_key"], hvar, uvar)
        likes.setdefault(name, {})[key] = row["p"]
        if row["p"] > 0:
            logs.setdefault(name, {})[key] = math.log(row["p"])
        else:
            note["zero"] += 1
    return [Scorer(name, "prob", logs.get(name, {}), likes[name], notes=notes[name]) for name in sorted(likes)], norm


def jev_scorers(full_rows: list[dict], rewrite_rows: list[dict], clean: set[str], epsilon: float) -> list[Scorer]:
    found = []
    for version, name in (("multiwoz_likely", "jev_likely"), ("multiwoz_typical", "jev_typical")):
        like = {}
        for row in full_rows:
            if row["prompt_version"] == version and row.get("repeat_id") is None and row["item_id"] in clean:
                like[(row["item_id"], row["hyp_key"], ORIGINAL, ORIGINAL)] = row["noul"]
        rewritten = set()
        for row in rewrite_rows:
            if row["prompt_version"] == version and row.get("repeat_id") is None:
                hyp_key, kind = row["hyp_key"].rsplit("@", 1)
                hvar, uvar = (kind, ORIGINAL) if kind in H_KINDS else (ORIGINAL, kind)
                like[(row["item_id"], hyp_key, hvar, uvar)] = row["noul"]
                rewritten.add(row["item_id"])
        if like:
            log = {key: logit_score(value, epsilon) for key, value in like.items()}
            found.append(Scorer(name, "prob", log, like, items=rewritten or {key[0] for key in like}))
    return found


# ---------------------------------------------------------------- bootstrap


def rankdata(values):
    """Average ranks, ties shared (as scipy.stats.rankdata)."""
    np = _np()
    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    inverse = np.empty(sorter.size, dtype=np.intp)
    inverse[sorter] = np.arange(sorter.size)
    ordered = values[sorter]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    dense = first.cumsum()[inverse]
    count = np.r_[np.nonzero(first)[0], len(first)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y) -> float | None:
    np = _np()
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


class ItemBoot:
    """Item-level bootstrap: replicate r draws each item counts[r, i] times."""

    def __init__(self, items: list[str], replicates: int, seed: int, chunk: int = 250):
        np = _np()
        self.items = sorted(items)
        self.index = {item: position for position, item in enumerate(self.items)}
        size = len(self.items)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(size, np.full(size, 1.0 / size), size=replicates) if size else np.zeros((0, 0))
        self.chunk = chunk

    @staticmethod
    def interval(values) -> list[float] | None:
        np = _np()
        values = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
        if values.size == 0:
            return None
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    def _idx(self, item_ids):
        np = _np()
        return np.asarray([self.index[item] for item in item_ids], dtype=np.intp)

    def median(self, item_ids: list[str], values: list[float]) -> dict:
        np = _np()
        if not values:
            return {"n": 0, "items": 0, "median": None, "ci95": None}
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ordered, index = values[order], self._idx(item_ids)[order]
        replicates = []
        for start in range(0, self.counts.shape[0], self.chunk):
            weights = self.counts[start:start + self.chunk][:, index].astype(np.float32)
            cumulative = np.cumsum(weights, axis=1)
            total = cumulative[:, -1:]
            position = (cumulative >= total / 2).argmax(axis=1)
            replicates.extend(np.where(total[:, 0] > 0, ordered[position], np.nan).tolist())
        return {"n": int(values.size), "items": len(set(item_ids)), "median": float(np.median(values)),
                "ci95": self.interval(replicates)}

    def mean(self, item_ids: list[str], values: list[float]) -> dict:
        """The mean over observations, observations weighted by their item's draw count."""
        np = _np()
        if not values:
            return {"n": 0, "items": 0, "mean": None, "ci95": None}
        values = np.asarray(values, dtype=float)
        weights = self.counts[:, self._idx(item_ids)].astype(float)
        totals = weights.sum(axis=1)
        replicates = np.where(totals > 0, (weights @ values) / np.maximum(totals, 1), np.nan)
        return {"n": int(values.size), "items": len(set(item_ids)), "mean": float(values.mean()),
                "ci95": self.interval(replicates.tolist())}

    def statistic(self, rows_by_item: dict[str, list], function) -> list:
        """function(rows) on each replicate's resampled rows (an item drawn twice appears twice)."""
        replicates = []
        present = [item for item in self.items if rows_by_item.get(item)]
        for counts in self.counts:
            rows = []
            for item in present:
                rows.extend(rows_by_item[item] * int(counts[self.index[item]]))
            replicates.append(function(rows) if rows else None)
        return replicates


# ---------------------------------------------------------------- items


def item_meta(items: list[dict], probe: list[dict]) -> dict[str, dict]:
    keys: dict[str, list[str]] = {}
    texts: dict[tuple[str, str], str] = {}
    for row in probe:
        keys.setdefault(row["item_id"], []).append(row["hyp_key"])
        texts[(row["item_id"], row["hyp_key"])] = row["text"]
    return {
        item["item_id"]: {"constrained": [slot for slot in SLOTS if slot in item["goal"]], "keys": keys[item["item_id"]],
                          "texts": {key: texts[(item["item_id"], key)] for key in keys[item["item_id"]]}, "item": item}
        for item in items if item["item_id"] in keys
    }


def bf_pairs(meta: dict) -> list[tuple[str, str, str]]:
    """(G, H, family) for the size-principle pairs and the contradiction control."""
    constrained, keys = meta["constrained"], set(meta["keys"])
    found = []
    for regime in REGIMES:
        goal = goal_key(len(constrained), regime)
        if goal not in keys:
            continue
        found += [(goal, f"minus:{slot}|{regime}", "minus") for slot in constrained if f"minus:{slot}|{regime}" in keys]
        if f"broad|{regime}" in keys:
            found.append((goal, f"broad|{regime}", "broad"))
    if "ctrl:contra" in keys and "G|impl" in keys:
        found.append(("G|impl", "ctrl:contra", "contra"))
    return found


# ---------------------------------------------------------------- metric 2: hypothesis rewording


def rewording(scorer: Scorer, metas: dict, boot: ItemBoot) -> dict:
    """ICC(1) of item-centered scores across each semantic hypothesis's textual variants, the
    within-to-between SD ratio for the same utterance, and the shift each rewrite kind causes."""
    np = _np()
    log = scorer.log
    aggregates, ratios, ratio_items = {}, [], []
    shifts: dict[str, tuple[list, list, list]] = {kind: ([], [], []) for kind in H_KINDS}
    for item in scorer.items:
        keys = metas[item]["keys"]
        originals = {key: log[(item, key, ORIGINAL, ORIGINAL)] for key in keys if (item, key, ORIGINAL, ORIGINAL) in log}
        if len(originals) < 2:
            continue
        center = statistics.fmean(originals.values())
        total = groups = within = sums = between = squares = 0.0
        pooled, dof = 0.0, 0
        for key in originals:
            values = [log[(item, key, variant, ORIGINAL)] - center for variant in (ORIGINAL,) + H_KINDS
                      if (item, key, variant, ORIGINAL) in log]
            for kind in H_KINDS:
                if (item, key, kind, ORIGINAL) in log:
                    delta = log[(item, key, kind, ORIGINAL)] - originals[key]
                    shifts[kind][0].append(item)
                    shifts[kind][1].append(abs(delta))
                    shifts[kind][2].append(delta)
            if len(values) < 2:
                continue
            size, total_value = len(values), sum(values)
            spread_sq = sum((value - total_value / size) ** 2 for value in values)  # two-pass: no cancellation
            total += size
            groups += 1
            within += spread_sq
            sums += total_value
            between += total_value * total_value / size
            squares += size * size
            pooled += spread_sq
            dof += size - 1
        if groups:
            aggregates[item] = [total, groups, within, sums, between, squares]
        spread = statistics.stdev(originals.values())
        if dof and spread > 0:
            ratios.append(math.sqrt(pooled / dof) / spread)
            ratio_items.append(item)
    result = {"icc1": None, "icc1_ci95": None, "groups": 0}
    if len(aggregates) >= 2:
        matrix = np.asarray([aggregates.get(item, [0.0] * 6) for item in boot.items], dtype=float)

        def icc(weights):
            n, g, ssw, s, sb, n2 = weights @ matrix
            if g < 2 or n <= g:
                return None
            grand = s / n
            msb = (sb - n * grand * grand) / (g - 1)
            msw = ssw / (n - g)
            k0 = (n - n2 / n) / (g - 1)
            denominator = msb + (k0 - 1) * msw
            return float((msb - msw) / denominator) if denominator > 0 else None

        result = {"icc1": icc(np.ones(len(boot.items))), "icc1_ci95": boot.interval([icc(c) for c in boot.counts]),
                  "groups": int(matrix[:, 1].sum())}
    result["sd_ratio"] = boot.median(ratio_items, ratios)
    result["shift_abs"] = {kind: boot.median(found[0], found[1]) for kind, found in shifts.items()}
    result["shift_signed"] = {kind: boot.median(found[0], found[2]) for kind, found in shifts.items()}
    return result


# ---------------------------------------------------------------- metric 3: utterance rewrites


def utterance_rows(scorer: Scorer, metas: dict) -> list[dict]:
    log = scorer.log
    rows = []
    for item in scorer.items:
        for goal, other, family in bf_pairs(metas[item]):
            before = [log.get((item, goal, ORIGINAL, ORIGINAL)), log.get((item, other, ORIGINAL, ORIGINAL))]
            if None in before:
                continue
            bf0 = before[0] - before[1]
            for kind in U_KINDS:
                after = [log.get((item, goal, ORIGINAL, kind)), log.get((item, other, ORIGINAL, kind))]
                if None in after:
                    continue
                bf1 = after[0] - after[1]
                rows.append({"item_id": item, "family": family, "kind": kind, "bf0": bf0, "bf1": bf1,
                             "delta": bf1 - bf0, "flip": bf0 * bf1 < 0, "raw": after[0] - before[0]})
    return rows


def utterance(scorer: Scorer, metas: dict, boot: ItemBoot) -> dict:
    rows = utterance_rows(scorer, metas)
    found = {}
    for group, kinds in U_GROUPS.items():
        mine = [row for row in rows if row["kind"] in kinds]
        relative = [row for row in mine if row["bf0"] != 0]
        found[group] = {
            "pairs": len(mine),
            "abs_delta": boot.median([row["item_id"] for row in mine], [abs(row["delta"]) for row in mine]),
            "relative_delta": boot.median([row["item_id"] for row in relative],
                                          [abs(row["delta"]) / abs(row["bf0"]) for row in relative]),
            "flip_rate": boot.mean([row["item_id"] for row in mine], [float(row["flip"]) for row in mine]),
            "raw_shift_abs": boot.median([row["item_id"] for row in mine], [abs(row["raw"]) for row in mine]),
            "by_family": {family: boot.median([row["item_id"] for row in mine if row["family"] == family],
                                              [abs(row["delta"]) for row in mine if row["family"] == family])
                          for family in ("minus", "broad", "contra")},
        }
    return found


# ---------------------------------------------------------------- metric 4: posterior instability


def posterior_classes(meta: dict) -> list[str]:
    keys = set(meta["keys"])
    classes = ["G|impl"] + [f"minus:{slot}|impl" for slot in meta["constrained"]] + ["broad|impl"]
    if "ctrl:contra" in keys:
        classes.append("ctrl:contra")
    return [key for key in classes if key in keys]


def _posterior(scorer: Scorer, item: str, chosen: list[tuple[str, str]]) -> list[float] | None:
    if scorer.kind == "loglik":
        values = [scorer.log[(item, key, variant, ORIGINAL)] for key, variant in chosen]
        top = max(values)
        weights = [math.exp(value - top) for value in values]
    else:
        weights = [scorer.like[(item, key, variant, ORIGINAL)] for key, variant in chosen]
    total = sum(weights)
    return None if total <= 0 else [weight / total for weight in weights]


def posterior(scorer: Scorer, metas: dict, boot: ItemBoot, seed: int, draws: int = POSTERIOR_DRAWS) -> dict:
    store = scorer.log if scorer.kind == "loglik" else scorer.like
    sd_items, sds, flip_items, flips, skipped = [], [], [], [], 0
    for item in scorer.items:
        classes = posterior_classes(metas[item])
        variants = {key: [variant for variant in (ORIGINAL,) + H_KINDS if (item, key, variant, ORIGINAL) in store]
                    for key in classes}
        if len(classes) < 2 or any(ORIGINAL not in found for found in variants.values()):
            continue
        base = _posterior(scorer, item, [(key, ORIGINAL) for key in classes])
        if base is None:
            continue
        top = max(range(len(classes)), key=base.__getitem__)
        rng = random.Random(f"{seed}:{item}")
        p_goal, changed = [], 0
        for _ in range(draws):
            found = _posterior(scorer, item, [(key, rng.choice(variants[key])) for key in classes])
            if found is None:
                skipped += 1
                continue
            p_goal.append(found[0])
            changed += max(range(len(classes)), key=found.__getitem__) != top
        if len(p_goal) >= 2:
            sd_items.append(item)
            sds.append(statistics.pstdev(p_goal))
            flip_items.append(item)
            flips.append(changed / len(p_goal))
    return {"items": len(sd_items), "draws": draws, "skipped_draws": skipped,
            "sd_p_goal": boot.mean(sd_items, sds), "argmax_flip_rate": boot.mean(flip_items, flips)}


# ---------------------------------------------------------------- metric 5: claim 2


def semantic_truth(extractor: Extractor, vocab: dict, metas: dict, samples_by_text: dict[str, list[str]],
                   samples_by_unit: dict[str, list[str]]):
    """P_sem(U|H) = (samples under H with U's annotated slot tuple + 0.5) / (samples + 1)."""
    cache: dict[tuple[str, str], float | None] = {}

    def truth(item: str, hyp_key: str, hvar: str, unit_id: str | None) -> float | None:
        source = unit_id if hvar != ORIGINAL else metas[item]["texts"][hyp_key]
        if (source, item) not in cache:
            replies = samples_by_unit.get(unit_id) if hvar != ORIGINAL else samples_by_text.get(source)
            if not replies:
                cache[(source, item)] = None
            else:
                wanted = annotated_tuple(metas[item]["item"], vocab)
                hits = sum(1 for reply in replies if slots_kept(extractor, reply, wanted))
                cache[(source, item)] = (hits + 0.5) / (len(replies) + 1)
        return cache[(source, item)]

    return truth


def claim2(verbal_rows: list[dict], norm_rows: list[dict], qwen_log: dict, sample_s: dict, truth_sem,
           boots: dict, seed: int, replicates: int) -> dict:
    found = {}
    rows_by_scorer: dict[str, list[dict]] = {}
    for row in verbal_rows:
        name = f"qwen_verbal_{row['framing']}" if row["model"] == "qwen" else f"{row['model']}_verbal"
        rows_by_scorer.setdefault(name, []).append(row)
    for name, rows in sorted(rows_by_scorer.items()):
        points = []
        for row in rows:
            if row["source"] == "norm" or row.get("status", "success") != "success" or not row["p"]:
                continue
            hvar, uvar = _variant(row["source"], row["side"])
            key = (row["item_id"], row["hyp_key"], hvar, uvar)
            exact = qwen_log.get(key)
            semantic = truth_sem(row["item_id"], row["hyp_key"], hvar, row.get("unit_id"))
            points.append({"item_id": row["item_id"], "original": hvar == uvar == ORIGINAL, "y": math.log10(row["p"]),
                           "exact": None if exact is None else exact / LN10,
                           "semantic": None if semantic is None else math.log10(semantic)})
        items = sorted({point["item_id"] for point in points})
        boot = boots.setdefault(tuple(items), ItemBoot(items, replicates, seed))
        block = {"answered_nonzero": len(points),
                 "zero_answers": sum(1 for row in rows if row["source"] != "norm" and row["p"] == 0),
                 "queries": sum(1 for row in rows if row["source"] != "norm")}
        for truth in ("exact", "semantic"):
            for scope in ("all", "original"):
                mine = [point for point in points if point[truth] is not None and (scope == "all" or point["original"])]
                by_item: dict[str, list] = {}
                for point in mine:
                    by_item.setdefault(point["item_id"], []).append((point["y"], point[truth]))

                def rho(pairs):
                    return spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])

                block[f"{truth}_{scope}"] = {
                    "n": len(mine),
                    "spearman": rho([(point["y"], point[truth]) for point in mine]),
                    "spearman_ci95": boot.interval(boot.statistic(by_item, rho)),
                    "gap_orders": boot.median([point["item_id"] for point in mine],
                                              [point["y"] - point[truth] for point in mine]),
                }
        found[name] = block
    norm: dict[str, dict[str, list]] = {}
    for row in norm_rows:
        if row.get("status", "success") != "success" or row["p"] is None:
            continue
        mine = norm.setdefault(row["scorer"], {})
        mine.setdefault(row["item_id"], []).append(row)
    for name, by_item in norm.items():
        items = sorted(by_item)
        boot = boots.setdefault(tuple(items), ItemBoot(items, replicates, seed))
        sums = [sum(row["p"] for row in by_item[item]) for item in items]
        true_sums = [sum(math.exp(-sample_s[(row["hyp_text"], row["sample"])]) for row in by_item[item]
                         if (row["hyp_text"], row["sample"]) in sample_s) for item in items]
        found.setdefault(name, {})["normalization"] = {
            "items": len(items), "samples_per_item": statistics.median(len(by_item[item]) for item in items),
            "sum": boot.median(items, sums), "mean_sum": boot.mean(items, sums),
            "share_above_1": boot.mean(items, [float(value > 1) for value in sums]),
            "true_sum_median": statistics.median(true_sums) if true_sums else None,
            "true_sum_max": max(true_sums) if true_sums else None,
        }
    return found


# ---------------------------------------------------------------- predictions


def _value(block: dict | None, key: str):
    return None if block is None else block.get(key)


def predictions(metrics: dict) -> list[dict]:
    found = []
    occam = metrics.get("r1_occam") or {}
    regimes = [occam.get(regime) for regime in ("expl", "impl")]
    if all(regimes):
        passed = all(block["mean_L_J"] > 0 and block["mean_L_J"] >= 0.5 * block["L_E"] for block in regimes)
        found.append({"id": "R1", "statement": "Qwen raw log-likelihood, named single-slot pairs: L_J > 0 in both regimes "
                      "and L_J ≥ 0.5·L_E (mean over slots)", "pass": passed,
                      "value": {regime: {"L_J": block["mean_L_J"], "L_E": block["L_E"], "OG": block["OG"]}
                                for regime, block in zip(("expl", "impl"), regimes)}})
    else:
        found.append({"id": "R1", "statement": "Qwen raw log-likelihood Occam gap", "pass": None, "status": "missing"})
    tok = metrics["scorers"].get("qwen_tok")
    if tok:
        icc = tok["rewording"]["icc1"]
        shift = tok["rewording"]["shift_abs"]["hzh"]["median"]
        found.append({"id": "R2", "statement": "Qwen token log-prob: rewording ICC(1) < 0.5, and a Chinese hypothesis "
                      "moves the log Bayes factor by a median > 2 nats", "pass": icc is not None and shift is not None
                      and icc < 0.5 and shift > 2, "value": {"icc1": icc, "hzh_median_abs_shift": shift}})
        groups = tok["utterance"]
        values = {group: groups[group]["abs_delta"]["median"] for group in ("pad", "typo", "para", "zh")}
        passed = None not in values.values() and values["pad"] < 0.5 and values["typo"] < 0.5 \
            and values["para"] > 1 and values["zh"] > 1
        found.append({"id": "R3", "statement": "Qwen token log-prob: median |Δ log BF| < 0.5 nats under pad and typo, "
                      "> 1 nat under para and zh", "pass": passed, "value": values,
                      "raw_shift": {group: groups[group]["raw_shift_abs"]["median"] for group in values}})
    claim2_block = metrics.get("claim2", {}).get("qwen_verbal_exact")
    if claim2_block and "normalization" in claim2_block:
        gap = claim2_block["exact_all"]["gap_orders"]["median"]
        total = claim2_block["normalization"]["sum"]["median"]
        found.append({"id": "R4", "statement": "Qwen verbalized (exact string) vs exp(−S): median gap > 6 orders of "
                      "magnitude, and the 20-sample normalization sum > 1 (median over items)",
                      "pass": gap is not None and total is not None and gap > 6 and total > 1,
                      "value": {"median_gap_orders": gap, "median_norm_sum": total}})
    meaning = metrics.get("claim2", {}).get("qwen_verbal_meaning")
    if meaning:
        rho = meaning["semantic_all"]["spearman"]
        found.append({"id": "R5", "statement": "Qwen verbalized (meaning) vs P_sem: Spearman < 0.5",
                      "pass": rho is not None and rho < 0.5, "value": rho,
                      "ci95": meaning["semantic_all"]["spearman_ci95"]})
    return found


# ---------------------------------------------------------------- run


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_claim1(
    root: Path,
    *,
    phase8: Path,
    lm_runs: list[Path],
    verbal: list[Path],
    jev_full: Path,
    jev: Path | None,
    out: Path,
    derived: Path | None = None,
    figures: bool = True,
) -> dict:
    from robust_likelihood.multiwoz import load_config

    clock = time.monotonic()
    derived = derived or root / DERIVED_DIR
    config = load_config(root)
    claims = read_json(root / "configs" / "claims.json")
    seed, replicates = int(claims["bootstrap"]["seed"]), int(claims["bootstrap"]["replicates"])
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    clean = {item["item_id"] for item in items}
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    metas = item_meta(items, probe)
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    vocab = load_vocab(derived)
    extractor = Extractor(vocab)

    phase8_rows = read_jsonl(phase8 / "lm_scores.jsonl")
    lm = read_lm_runs(lm_runs)
    qwen_originals = {(row["item_id"], row["hyp_key"]): -row["S"] for row in phase8_rows}
    scorers = []
    qwen = Scorer("qwen_tok", "loglik", token_log(qwen_originals, lm.get("qwen", {}), clean), items=clean)
    scorers += [qwen, qwen.restricted("qwen_tok_subset", subset),
                Scorer("qwen_clip", "loglik", clipped_log(qwen.log, float(claims["clip_nats"])), items=clean),
                Scorer("qwen_cent", "loglik", centered_log(qwen.log), items=clean)]
    pending = {}
    olmo_parts = lm.get("olmo", {})
    if olmo_parts.get("original"):
        scorers.append(Scorer("olmo_tok", "loglik", token_log({}, olmo_parts, clean), items=clean))
    elif olmo_parts:
        pending["olmo_tok"] = ("OLMo scored the rewrites only; its original pairs are needed to compare with them "
                               "(claim1-lm --model olmo --parts original).")
    verbal_rows = [row for run_dir in verbal for row in read_jsonl(run_dir / "scores.jsonl")]
    verbal_found, norm_rows = verbal_scorers(verbal_rows)
    scorers += verbal_found
    epsilon = float(config["score_mapping"]["clip_epsilon"])
    scorers += jev_scorers(read_jsonl(jev_full / "scores.jsonl"), read_jsonl(jev / "scores.jsonl") if jev else [],
                           clean, epsilon)
    scorers.sort(key=lambda scorer: SCORER_ORDER.index(scorer.name) if scorer.name in SCORER_ORDER else 99)

    # Noise floor: job A's centering part scores S(U | broad|impl), the same prompt and utterance Phase 8
    # scored; batches of other shapes take other bfloat16 kernels, so the two can differ slightly.
    center_rows = lm.get("qwen", {}).get("center", [])
    gaps = sorted(abs(row["S"] + qwen.log[(row["item_id"], "broad|impl", ORIGINAL, ORIGINAL)])
                  for row in center_rows if row.get("source") == "utterance"
                  and (row["item_id"], "broad|impl", ORIGINAL, ORIGINAL) in qwen.log)
    center_gap = None if not gaps else {
        "pairs": len(gaps), "identical": sum(1 for gap in gaps if gap == 0), "median": statistics.median(gaps),
        "p90": gaps[int(0.9 * (len(gaps) - 1))], "max": gaps[-1]}

    boots: dict[tuple, ItemBoot] = {}
    metrics: dict = {"scorers": {}, "pending": pending}
    for scorer in scorers:
        boot = boots.setdefault(tuple(scorer.items), ItemBoot(scorer.items, replicates, seed))
        metrics["scorers"][scorer.name] = {
            "label": scorer.label, "kind": scorer.kind, "items": len(scorer.items), "scores": len(scorer.log),
            "notes": scorer.notes, "rewording": rewording(scorer, metas, boot),
            "utterance": utterance(scorer, metas, boot), "posterior": posterior(scorer, metas, boot, seed),
        }

    samples_by_text: dict[str, list[str]] = {}
    sample_s: dict[tuple[str, int], float] = {}
    for row in read_jsonl(phase8 / "lm_samples.jsonl"):
        samples_by_text.setdefault(row["text"], []).append(row["reply"])
        sample_s[(row["text"], row["sample"])] = row["S"]
    samples_by_unit: dict[str, list[str]] = {}
    for row in lm.get("qwen", {}).get("hsample", []):
        samples_by_unit.setdefault(row["unit_id"], []).append(row["reply"])
    query_text = {}
    for run_dir in verbal:
        for query in read_jsonl(run_dir / "queries.jsonl"):
            query_text[query["query_id"]] = query["hypothesis"]
    norm_rows = [dict(row, hyp_text=query_text.get(row["query_id"])) for row in norm_rows]
    truth = semantic_truth(extractor, vocab, metas, samples_by_text, samples_by_unit)
    metrics["claim2"] = claim2(verbal_rows, norm_rows, qwen.log, sample_s, truth, boots, seed, replicates)

    occam_metrics = analyze(jev_full, derived, config, lm_dir=phase8)
    occam = occam_metrics["sets"]["clean"]["lm"]["lm_loglik"]["occam"]
    metrics["r1_occam"] = {regime: occam.get(f"{regime}|slots_mean|named") for regime in ("expl", "impl")}
    metrics["r1_occam_by_slot"] = {key: value for key, value in occam.items() if key.endswith("|named")}
    metrics["checks"] = {"rescoring_noise_nats": center_gap}
    metrics["predictions"] = predictions(metrics)

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    paths = []
    if figures:
        from robust_likelihood.multiwoz.claim1_report import write_figures

        paths = write_figures(out, metrics, verbal_rows, qwen.log, truth)
    from robust_likelihood.multiwoz.claim1_report import summary_markdown

    inputs = {"phase8": phase8.as_posix(), "lm_runs": [path.as_posix() for path in lm_runs],
              "verbal": [path.as_posix() for path in verbal], "jev_full": jev_full.as_posix(),
              "jev": None if jev is None else jev.as_posix()}
    manifest = {
        "kind": "claim1-analysis", "git": git_state(root), "inputs": inputs,
        "input_sha256": {name: _sha256(derived / name) for name in
                         ("probe.jsonl", "rewrites.jsonl", "items.jsonl", "robustness_subset.json")},
        "models": {name: read_json(path / "manifest.json").get("model") or read_json(path / "manifest.json").get(
            "requested_model_id") for name, path in [(path.name, path) for path in lm_runs + verbal + [jev_full]]},
        "api_models": {provider: found.get("returned_model_ids") for run_dir in verbal
                       for provider, found in (read_json(run_dir / "manifest.json").get("providers") or {}).items()},
        "bootstrap": {"replicates": replicates, "seed": seed}, "posterior_draws": POSTERIOR_DRAWS,
        "clip_nats": float(claims["clip_nats"]),
        "counts": {name: {"items": block["items"], "scores": block["scores"]} for name, block in metrics["scorers"].items()},
        "pending": pending,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.monotonic() - clock, 1),
    }
    write_json(out / "manifest.json", manifest)
    (out / "summary.md").write_text(summary_markdown(metrics, manifest), encoding="utf-8", newline="\n")
    return {"out": str(out), "predictions": metrics["predictions"], "figures": [str(path) for path in paths],
            "pending": pending, "elapsed_seconds": manifest["elapsed_seconds"]}
