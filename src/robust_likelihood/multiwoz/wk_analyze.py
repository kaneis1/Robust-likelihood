"""Week of 24 Nov, Parts A-C (docs/week_nov24_plan.md): metrics from the uperturb, hsample and Phase 2 runs.

Part A reads each model's uperturb.jsonl. Every delta is taken against the original turn scored in the
same batch (S_orig), never against Phase 2's human score. Per unit, "Delta log p" is the median over the
item's hypothesis texts of S_orig - S (negative when the edit costs probability); log BF is
l(U | G) - l(U | H) over claim1.bf_pairs, l = -S. Posteriors are claim1's softmax over
posterior_classes. Typicality takes each model's own Phase 2 samples as the reference and puts the
co-batched original and the unit's text in bins with the same pool statistics.

Part B reads the Phase 2 hrewrite part (and, for the models with one, the hsample part) through
typv0.Inputs. Part C's variants are wk_quantile.py's. Intervals: item bootstrap, 2,000 replicates,
seed 20261124.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, load_config
from robust_likelihood.multiwoz.claim1 import (
    H_KINDS,
    ORIGINAL,
    ItemBoot,
    Scorer,
    bf_pairs,
    item_meta,
    posterior,
    posterior_classes,
    token_log,
)
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.semantic import Extractor, load_vocab
from robust_likelihood.multiwoz.sim import load_sim_config
from robust_likelihood.multiwoz.sim_analyze import discover
from robust_likelihood.multiwoz.typv0 import (
    Inputs,
    corpus_bootstrap,
    load_typv0_config,
    pool_weights,
    score_map,
    size_and_information,
)
from robust_likelihood.multiwoz.wk_quantile import evaluate, scores_of, specs
from robust_likelihood.multiwoz.wk_units import load_units
from robust_likelihood.storage import read_json, read_jsonl, write_json

SEED = 20261124
REPLICATES = 2000
HUMAN_LIKE = ("qwen3_base|F", "osim_mid|S", "osim_final|S", "userlm|S")
INSTRUCT = ("qwen25_inst|S", "qwen3_inst|S")
TYPO = ("typo1", "typo2", "typo4")
FILL = ("fill1", "fill2", "fill4")
STYLE = ("lower", "nopunct", "noapos")
KINDS = TYPO + ("slottypo1",) + STYLE + FILL + ("terse", "fixed")
TEACHER = "qwen3_base|F"
TYPICALITY = ("glob1A", "glob1B", "cent1B", "cent1B:6bins", "glob1Q", "cent1Q", "glob1Q:3q", "cent1Q:3q")


def orig(kind: str) -> str:
    return f"orig:{kind}"


# ---------------------------------------------------------------- inputs


def discover_week(run_dirs: list[Path]) -> dict[str, dict[str, Path]]:
    """label -> part -> the newest finished sim-week run carrying that part (smoke runs left out)."""
    found: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(run_dirs):
        path = run_dir / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        if manifest.get("kind") != "sim-week" or "finished_at" not in manifest or manifest.get("items"):
            continue
        if run_dir.name.endswith("-smoke"):
            continue
        for part in manifest["parts"]:
            found.setdefault(f"{manifest['model_key']}|{manifest['format']}", {})[part] = run_dir
    return found


def perturb_log(rows: list[dict]) -> dict:
    """claim1-style log: (item, hyp, original, kind) = -S, (item, hyp, original, orig:kind) = -S_orig."""
    log = {}
    for row in rows:
        log[(row["item_id"], row["hyp_key"], ORIGINAL, row["kind"])] = -row["S"]
        log[(row["item_id"], row["hyp_key"], ORIGINAL, orig(row["kind"]))] = -row["S_orig"]
    return log


def perturb_cells(rows: list[dict]) -> list[dict]:
    """typv0 cells for the co-batched originals and the units, cent against the same batch's broad|impl."""
    broad = {(row["item_id"], row["kind"]): (row["S_orig"], row["S"]) for row in rows if row["hyp_key"] == "broad|impl"}
    cells = []
    for row in rows:
        found = broad.get((row["item_id"], row["kind"]))
        if found is None:
            continue
        for uvar, value, centre in ((orig(row["kind"]), row["S_orig"], found[0]), (row["kind"], row["S"], found[1])):
            cells.append({"key": (row["item_id"], row["hyp_key"], ORIGINAL, uvar), "source": row["text"], "S": value,
                          "cent": value - centre, "sem": None, "semf": None})
    return cells


# ---------------------------------------------------------------- Part A: raw log-likelihood


def slope(points: list[tuple[float, float]]) -> float | None:
    if len({x for x, _ in points}) < 2:
        return None
    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / sxx


def slope_block(rows_by_item: dict[str, list[tuple[float, float]]], boot: ItemBoot) -> dict:
    points = [point for rows in rows_by_item.values() for point in rows]
    return {"slope": slope(points), "ci95": boot.interval(boot.statistic(rows_by_item, slope)),
            "units": len(points)}


def unit_rows(log: dict, metas: dict, units: dict[str, dict], extractor: Extractor) -> list[dict]:
    """One row per unit: its Delta log p, its Bayes-factor pairs and its posterior change."""
    found = []
    for unit_id, unit in sorted(units.items()):
        item, kind = unit["item_id"], unit["kind"]
        meta = metas.get(item)
        if meta is None:
            continue
        deltas = []
        for key in meta["keys"]:
            after, before = log.get((item, key, ORIGINAL, kind)), log.get((item, key, ORIGINAL, orig(kind)))
            if after is not None and before is not None:
                deltas.append(after - before)
        if not deltas:
            continue
        pairs = []
        for goal, other, family in bf_pairs(meta):
            keys = [(item, goal, ORIGINAL, orig(kind)), (item, other, ORIGINAL, orig(kind)),
                    (item, goal, ORIGINAL, kind), (item, other, ORIGINAL, kind)]
            if all(key in log for key in keys):
                bf0, bf1 = log[keys[0]] - log[keys[1]], log[keys[2]] - log[keys[3]]
                pairs.append({"goal": goal, "other": other, "family": family, "bf0": bf0, "bf1": bf1,
                              "delta": bf1 - bf0, "flip": bf0 * bf1 < 0})
        classes = posterior_classes(meta)
        post = None
        if len(classes) >= 2 and all((item, key, ORIGINAL, uvar) in log for key in classes for uvar in (kind, orig(kind))):
            p0, p1 = (softmax([log[(item, key, ORIGINAL, uvar)] for key in classes]) for uvar in (orig(kind), kind))
            top0, top1 = max(range(len(classes)), key=p0.__getitem__), max(range(len(classes)), key=p1.__getitem__)
            post = {"abs_delta_p_goal": abs(p1[0] - p0[0]), "flip": top0 != top1}
        found.append({"unit_id": unit_id, "item_id": item, "kind": kind, "dose": unit.get("dose"),
                      "op": unit["edits"][0]["op"] if unit.get("edits") else None,
                      "delta_log_p": statistics.median(deltas), "pairs": pairs, "posterior": post,
                      "slot": slot_hit(unit, extractor) if kind == "slottypo1" else None,
                      "added_sentences": len(unit.get("added") or [])})
    return found


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def slot_hit(unit: dict, extractor: Extractor) -> str | None:
    """The slot whose value word the slot typo changed."""
    edit = unit["edits"][0]
    lowered = unit["original"].lower().replace("’", "'")
    for slot, pattern in (("food", extractor.food), ("area", extractor.area), ("pricerange", extractor.price)):
        for match in pattern.finditer(lowered):
            if match.start() <= edit["start"] < match.end():
                return slot
    return None


def token_costs(rows: list[dict]) -> dict[str, float]:
    """unit_id -> median over the item's texts of Delta log p per added token (fillers, terse)."""
    by_unit: dict[str, list[float]] = {}
    for row in rows:
        added = row["tokens"] - row["tokens_orig"]
        if added:
            by_unit.setdefault(row["unit_id"], []).append((row["S_orig"] - row["S"]) / added)
    return {unit: statistics.median(values) for unit, values in by_unit.items()}


def raw_metrics(found: list[dict], per_token: dict[str, float], boot: ItemBoot) -> dict:
    result = {}
    for kind in KINDS:
        mine = [row for row in found if row["kind"] == kind]
        if not mine:
            continue
        pairs = [(row["item_id"], pair) for row in mine for pair in row["pairs"]]
        relative = [(item, pair) for item, pair in pairs if pair["bf0"] != 0]
        posts = [(row["item_id"], row["posterior"]) for row in mine if row["posterior"]]
        block = {
            "units": len(mine),
            "delta_log_p": boot.median([row["item_id"] for row in mine], [row["delta_log_p"] for row in mine]),
            "abs_delta_log_bf": boot.median([item for item, _ in pairs], [abs(pair["delta"]) for _, pair in pairs]),
            "relative_delta_log_bf": boot.median([item for item, _ in relative],
                                                 [abs(pair["delta"]) / abs(pair["bf0"]) for _, pair in relative]),
            "sign_change": boot.mean([item for item, _ in pairs], [float(pair["flip"]) for _, pair in pairs]),
            "abs_delta_p_goal": boot.median([item for item, _ in posts], [post["abs_delta_p_goal"] for _, post in posts]),
            "argmax_flip": boot.mean([item for item, _ in posts], [float(post["flip"]) for _, post in posts]),
            "by_family": {family: boot.median([item for item, pair in pairs if pair["family"] == family],
                                              [abs(pair["delta"]) for _, pair in pairs if pair["family"] == family])
                          for family in ("minus", "broad", "contra")},
        }
        doses = [row for row in mine if row["dose"]]
        if kind in TYPO + FILL and doses:
            block["delta_log_p_per_unit_dose"] = boot.median([row["item_id"] for row in doses],
                                                            [row["delta_log_p"] / row["dose"] for row in doses])
        if kind == "typo1":
            block["by_op"] = {op: boot.median([row["item_id"] for row in mine if row["op"] == op],
                                              [row["delta_log_p"] for row in mine if row["op"] == op])
                              for op in sorted({row["op"] for row in mine if row["op"]})}
        tokens = [(row["item_id"], per_token[row["unit_id"]]) for row in mine if row["unit_id"] in per_token]
        if kind in FILL + ("terse",) and tokens:
            block["delta_log_p_per_token"] = boot.median([item for item, _ in tokens], [value for _, value in tokens])
        if kind == "slottypo1":
            kept = []
            for row in mine:
                for pair in row["pairs"]:
                    if (row["slot"] and pair["goal"] == "G|impl" and pair["other"] == f"minus:{row['slot']}|impl"
                            and pair["bf0"] > 1):
                        kept.append((row["item_id"], pair["bf1"] / pair["bf0"]))
            block["named_slot_bf_kept"] = boot.median([item for item, _ in kept], [value for _, value in kept])
        result[kind] = block
    result["dose"] = {}
    for family, kinds in (("typo", TYPO), ("fill", FILL)):
        bf_points: dict[str, list] = {}
        p_points: dict[str, list] = {}
        for row in found:
            if row["kind"] in kinds and row["pairs"]:
                dose = float(row["dose"])
                bf_points.setdefault(row["item_id"], []).append(
                    (dose, statistics.fmean(abs(pair["delta"]) for pair in row["pairs"])))
                p_points.setdefault(row["item_id"], []).append((dose, row["delta_log_p"]))
        result["dose"][family] = {"abs_delta_log_bf": slope_block(bf_points, boot),
                                  "delta_log_p": slope_block(p_points, boot)}
    return result


# ---------------------------------------------------------------- Part A: typicality


def typicality_changes(inputs: Inputs, cells: list[dict], variants: dict[str, dict], weights: dict, boot: ItemBoot) -> dict:
    """Per variant and kind: share of human pairs whose bin changes, share moved into the extreme bin, |Delta score|."""
    result = {}
    for name, spec in variants.items():
        log, bins, _ = scores_of(inputs, spec, weights, cells)
        top = (len(spec["quantiles"]) if spec.get("score") == "Q" else len(spec["edges"]))
        block = {}
        for kind in KINDS:
            ids, changed, into, delta = [], [], [], []
            for key, value in bins.items():
                item, hyp, _, uvar = key
                if uvar != kind:
                    continue
                before = (item, hyp, ORIGINAL, orig(kind))
                if before not in bins:
                    continue
                ids.append(item)
                changed.append(float(value != bins[before]))
                into.append(float(value == top and bins[before] != top))
                delta.append(abs(log[key] - log[before]))
            if ids:
                block[kind] = {"bin_change": boot.mean(ids, changed), "into_extreme": boot.mean(ids, into),
                               "abs_delta_score": boot.median(ids, delta)}
        result[name] = block
    return result


def fixed_bins(inputs: Inputs, cells: list[dict], variants: dict[str, dict], weights: dict) -> list[dict]:
    """A3: the teacher's bins of the six misspelled turns before and after the correction, gold hypothesis."""
    rows = []
    for name in ("cent1B:6bins", "cent1Q"):
        log, bins, _ = scores_of(inputs, variants[name], weights, cells)
        for key, value in sorted(bins.items()):
            item, hyp, _, uvar = key
            if uvar == "fixed" and hyp == "G|impl":
                before = (item, hyp, ORIGINAL, orig("fixed"))
                rows.append({"variant": name, "item_id": item, "bin_typo": bins.get(before), "bin_fixed": value,
                             "score_typo": log.get(before), "score_fixed": log[key]})
    return rows


# ---------------------------------------------------------------- Part B


def hypothesis_rewording(inputs: Inputs, log: dict, label: str, metas: dict, boot: ItemBoot, changed: dict,
                         corpus_boot) -> dict:
    """Argmax flips and SD of P(G) over hypothesis rewordings, and the size principle under each rewording."""
    scorer = Scorer(label, "loglik", log, items=sorted(inputs.clean))
    found = {"posterior": posterior(scorer, metas, boot, SEED), "size_kept": {}}
    for kind in H_KINDS:
        reworded: dict[str, dict[str, float]] = {}
        for (item, key, hvar, uvar), value in log.items():
            if hvar == kind and uvar == ORIGINAL:
                reworded.setdefault(item, {})[key] = value
        if not reworded:
            continue
        size = size_and_information(inputs.items, reworded, changed, corpus_boot)
        group = size["occam"].get("impl|slots_mean|named")
        found["size_kept"][kind] = None if not group or not group.get("L_E") else group["mean_L_J"] / group["L_E"]
    return found


# ---------------------------------------------------------------- run


def predictions(models: dict) -> list[dict]:
    def get(label, *path):
        block = models.get(label, {})
        for key in path:
            if block is None:
                return None
            block = block.get(key) if isinstance(block, dict) else None
        return block

    found = []
    t1 = {label: get(label, "raw", "typo1", "delta_log_p", "median") for label in models}
    found.append({"id": "T1", "statement": "median Delta log p <= -3 nats per typo at typo1, every model", "value": t1,
                  "pass": bool(t1) and all(value is not None and value <= -3 for value in t1.values())})
    t2 = {label: get(label, "raw", "typo1", "abs_delta_log_bf", "median") for label in models}
    human = [t2[label] for label in HUMAN_LIKE if t2.get(label) is not None]
    inst = [t2[label] for label in INSTRUCT if t2.get(label) is not None]
    found.append({"id": "T2", "statement": "median |Delta log BF| <= 0.5 at typo1 for the human-like models, larger "
                                           "for qwen25_inst and qwen3_inst", "value": t2,
                  "pass": len(human) == len(HUMAN_LIKE) and len(inst) == len(INSTRUCT) and max(human) <= 0.5
                  and min(inst) > max(human)})
    glob = get(TEACHER, "typicality", "glob1B", "typo2", "bin_change", "mean")
    cent = get(TEACHER, "typicality", "cent1B:6bins", "typo2", "bin_change", "mean")
    found.append({"id": "T3", "statement": "typo2, qwen3_base reference: glob1B changes bin for >= 20% of human pairs, "
                                           "cent1B:6bins for <= 10%", "value": {"glob1B": glob, "cent1B:6bins": cent},
                  "pass": glob is not None and cent is not None and glob >= 0.2 and cent <= 0.1})
    t4 = {label: get(label, "raw", "slottypo1", "named_slot_bf_kept", "median") for label in HUMAN_LIKE}
    found.append({"id": "T4", "statement": "human-like models keep >= 50% of the named slot's log BF under slottypo1",
                  "value": t4, "pass": all(value is not None and value >= 0.5 for value in t4.values())})
    cost = {label: {kind: get(label, "raw", kind, "delta_log_p_per_unit_dose", "median") for kind in FILL}
            for label in models}
    fill4 = {label: get(label, "raw", "fill4", "abs_delta_log_bf", "median") for label in HUMAN_LIKE}
    found.append({"id": "T5", "statement": "each added sentence costs >= 5 nats of log p; |Delta log BF| at fill4 "
                                           "<= 1 nat median for the human-like models",
                  "value": {"per_sentence": cost, "fill4_abs_delta_log_bf": fill4},
                  "pass": all(value is not None and value <= -5 for block in cost.values() for value in block.values())
                  and all(value is not None and value <= 1 for value in fill4.values())})
    t6 = {label: {kind: get(label, "raw", kind, "abs_delta_log_bf", "median") for kind in STYLE} for label in HUMAN_LIKE}
    found.append({"id": "T6", "statement": "lower, nopunct, noapos move |Delta log BF| by <= 0.5 nats median for the "
                                           "human-like models", "value": t6,
                  "pass": all(value is not None and value <= 0.5 for block in t6.values() for value in block.values())})
    return found


def teacher_choice(models: dict) -> dict:
    """The frozen rule of plan 5.1, from qwen3_base's Part A and C metrics."""
    block = models.get(TEACHER, {})
    part_c = block.get("part_c", {})
    typ = block.get("typicality", {})

    def agree_typo1(name):
        found = typ.get(name, {}).get("typo1", {}).get("bin_change", {}).get("mean")
        return None if found is None else 1 - found

    q = {"para1": part_c.get("cent1Q", {}).get("agreement", {}).get("para1", {}).get("mean"),
         "typo1": agree_typo1("cent1Q"), "size_kept": part_c.get("cent1Q", {}).get("size_kept", {}).get("impl")}
    b = {"para1": part_c.get("cent1B:6bins", {}).get("agreement", {}).get("para1", {}).get("mean"),
         "typo1": agree_typo1("cent1B:6bins")}
    ok = (None not in (q["para1"], q["typo1"], b["para1"], b["typo1"], q["size_kept"])
          and q["para1"] >= b["para1"] - 0.02 and q["typo1"] >= b["typo1"] - 0.02 and 0.5 <= q["size_kept"] <= 2)
    return {"rule": "cent1Q if its bin agreement under para1 and typo1 is within 2 points of cent1B:6bins or better "
                    "and its size principle kept (impl, slot named) is in [0.5, 2]; otherwise cent1B:6bins",
            "cent1Q": q, "cent1B:6bins": b, "choice": "cent1Q" if ok else "cent1B:6bins"}


def run_wk_analyze(root: Path, *, runs: list[Path], out: Path, derived: Path | None = None,
                   parts: tuple[str, ...] = ("a", "b", "c")) -> dict:
    """parts: any of a, b, c; typ-quantile runs c alone (Part C on the claims rewrites, no GPU results)."""
    clock = time.monotonic()
    derived = derived or root / DERIVED_DIR
    config, sim, typ = load_config(root), load_sim_config(root), load_typv0_config(root)
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    clean_ids = sorted(item["item_id"] for item in items)
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    metas = item_meta(items, probe)
    changed = {(row["item_id"], row["hyp_key"]): row["changed"]["slot"]
               for row in read_jsonl(derived / "probe.jsonl") if "changed" in row}
    corpus_boot = corpus_bootstrap(derived, config, items)
    boot = ItemBoot(clean_ids, REPLICATES, SEED)
    extractor = Extractor(load_vocab(derived))
    units = {row["unit_id"]: row for row in load_units(derived)}
    variants = specs(typ)
    phase2 = discover(runs)
    week = discover_week(runs)
    labels = [label.replace(":", "|") for label in sim["week_nov24"]["uperturb_models"]]
    primary = [label for label, run_dir in sorted(phase2.items()) if read_json(run_dir / "manifest.json").get("primary")]
    models: dict = {}
    for label in sorted(set(labels) | set(primary)):
        if label not in phase2:
            continue
        key = label.split("|")[0]
        lm_dirs = [phase2[label]] + ([week[label]["hsample"]] if "hsample" in week.get(label, {}) else [])
        inputs = Inputs(root, phase2[label], lm_dirs, derived, model_key=key)
        weights = pool_weights(inputs.probe, "uniform")
        block: dict = {"sources": [path.as_posix() for path in lm_dirs], "part_c": {}}
        for name in TYPICALITY:
            block["part_c"][name] = evaluate(inputs, variants[name], weights, changed, corpus_boot, boot)
        if "b" not in parts and "a" not in parts:
            models[label] = block
            continue
        phase2_scores = {(row["item_id"], row["hyp_key"]): row["S"] for row in read_jsonl(phase2[label] / "lm_scores.jsonl")
                         if row["set"] == "clean"}
        raw_log = token_log({pair: -value for pair, value in phase2_scores.items()}, inputs.lm, inputs.clean)
        block["part_b"] = {"raw": hypothesis_rewording(inputs, raw_log, f"{label}:raw", metas, boot, changed, corpus_boot),
                           "hsample": "hsample" in week.get(label, {}) or bool(inputs.lm.get("hsample"))}
        for name in TYPICALITY:
            log, _, _ = scores_of(inputs, variants[name], weights)
            block["part_b"][name] = hypothesis_rewording(inputs, log, f"{label}:{name}", metas, boot, changed,
                                                         corpus_boot)
        if "a" in parts and "uperturb" in week.get(label, {}):
            rows = read_jsonl(week[label]["uperturb"] / "uperturb.jsonl")
            block["uperturb_run"] = week[label]["uperturb"].as_posix()
            found = unit_rows(perturb_log(rows), metas, units, extractor)
            block["raw"] = raw_metrics(found, token_costs(rows), boot)
            cells = perturb_cells(rows)
            block["typicality"] = typicality_changes(inputs, cells, {name: variants[name] for name in TYPICALITY},
                                                     weights, boot)
            if label == TEACHER:
                block["fixed"] = fixed_bins(inputs, cells, variants, weights)
            spread = {}
            for row in rows:
                spread.setdefault((row["item_id"], row["hyp_key"]), []).append(row["S_orig"])
            gaps = [max(values) - min(values) for values in spread.values() if len(values) > 1]
            block["layout_spread"] = {"originals": len(gaps), "median": statistics.median(gaps) if gaps else None,
                                      "p90": sorted(gaps)[int(0.9 * (len(gaps) - 1))] if gaps else None}
        models[label] = block
    metrics = {"models": models, "predictions": predictions({label: block for label, block in models.items()
                                                             if label in labels and "raw" in block}),
               "teacher_choice": teacher_choice(models)}
    out.mkdir(parents=True, exist_ok=True)
    name = "metrics_abc.json" if "a" in parts else "metrics_c.json"
    write_json(out / name, metrics)
    manifest = {"kind": "wk-analyze", "git": git_state(root),
                "inputs": {"phase2": {label: path.as_posix() for label, path in phase2.items()},
                           "week": {label: {part: path.as_posix() for part, path in parts.items()}
                                    for label, parts in week.items()}},
                "bootstrap": {"replicates": REPLICATES, "seed": SEED},
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_seconds": round(time.monotonic() - clock, 1)}
    manifest["parts"] = list(parts)
    write_json(out / name.replace("metrics", "manifest"), manifest)
    return {"out": str(out), "models": sorted(models), "predictions": metrics["predictions"],
            "teacher_choice": metrics["teacher_choice"], "elapsed_seconds": manifest["elapsed_seconds"]}
