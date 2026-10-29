"""Dice metrics: slopes in k, accuracy at k = 1, Bayes factors, and normalization sums.

A score is a natural-log likelihood: -S for token log-probs (S1), ln p for a parsed
verbalized answer (S2, S3). Unparsed answers and answers of exactly 0 have no log
score; they are counted and left out of the fits, never imputed.

Slopes regress log10(score) on k over sequences scored under their own generating die,
by ordinary least squares, with a bootstrap that resamples sequences within each k.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from robust_likelihood.dice.score import load_queries
from robust_likelihood.multiwoz.reference import bootstrap_rng, interval
from robust_likelihood.storage import read_json, read_jsonl
from robust_likelihood.verbal import parse_probability

LN10 = math.log(10)
BF_PAIRS = (("even", "fair"), ("loaded", "fair"), ("even", "loaded"))
TRUE_SLOPES = {"fair": -math.log10(6), "even": -math.log10(3)}
D2_LIMIT = 0.15
D3_LIMIT = 0.10
D4_LIMIT = 2.0
LOCAL_PARSE_LIMIT = 0.02


def ols(xs: list[float], ys: list[float]) -> dict | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    return {"slope": slope, "intercept": intercept, "r2": 1 - residual / total if total > 0 else None, "n": n}


def _strata_replicates(rows: list[dict], key: str, replicates: int, seed: int, label: str) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    rng = bootstrap_rng(seed, label)
    found = []
    for _ in range(replicates):
        sample = []
        for name in sorted(groups):
            members = groups[name]
            sample.extend(members[index] for index in rng.choices(range(len(members)), k=len(members)))
        found.append(sample)
    return found


def load_scores(run_dirs: list[Path]) -> tuple[dict[tuple[str, str], dict[str, dict]], list[dict]]:
    """Rows by (model, scorer) and query id, and a summary of every run read. A later run overrides an earlier one."""
    sources: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    runs = []
    for run_dir in run_dirs:
        manifest = read_json(run_dir / "manifest.json")
        rows = read_jsonl(run_dir / ("calls.jsonl" if manifest["kind"] == "api" else "scores.jsonl"))
        for row in rows:
            if row["scorer"] != "S1":
                # The raw answer is the record; parsing is redone so every run is read by the same parser.
                row = dict(row, p=parse_probability(row.get("raw")) if row.get("status", "success") == "success" else None)
            sources[(row["model"], row["scorer"])][row["query_id"]] = row
        runs.append({"run_dir": run_dir.as_posix(), "kind": manifest["kind"], "rows": len(rows),
                     "pilot": manifest.get("pilot")})
    return dict(sources), runs


def log_score(row: dict | None) -> float | None:
    if row is None:
        return None
    if row["scorer"] == "S1":
        return row["logp"]
    p = row.get("p")
    return math.log(p) if p is not None and p > 0 else None


def score_probability(row: dict | None) -> float | None:
    if row is None:
        return None
    if row["scorer"] == "S1":
        return math.exp(row["logp"])
    return row.get("p")


def slope_block(rows: list[dict], values: dict[str, float | None], bootstrap: dict, label: str) -> dict:
    usable = [row for row in rows if values.get(row["query_id"]) is not None]
    fit = ols([row["k"] for row in usable], [values[row["query_id"]] / LN10 for row in usable])
    truth = ols([row["k"] for row in usable], [row["true_log10"] for row in usable])
    block = {"n": len(rows), "n_used": len(usable), "fit": fit, "truth": truth}
    if fit is None or truth is None:
        return block
    slopes, ratios = [], []
    for sample in _strata_replicates(usable, "k", int(bootstrap["replicates"]), int(bootstrap["seed"]), label):
        found = ols([row["k"] for row in sample], [values[row["query_id"]] / LN10 for row in sample])
        exact = ols([row["k"] for row in sample], [row["true_log10"] for row in sample])
        if found is not None:
            slopes.append(found["slope"])
            if exact is not None and exact["slope"]:
                ratios.append(found["slope"] / exact["slope"])
    block.update({
        "slope_ci95": interval(slopes),
        "ratio_to_truth": fit["slope"] / truth["slope"] if truth["slope"] else None,
        "ratio_ci95": interval(ratios),
    })
    per_k = defaultdict(lambda: {"score": [], "truth": []})
    for row in usable:
        per_k[row["k"]]["score"].append(values[row["query_id"]] / LN10)
        per_k[row["k"]]["truth"].append(row["true_log10"])
    block["per_k"] = [
        {"k": k, "n": len(found["score"]), "mean_log10": sum(found["score"]) / len(found["score"]),
         "min_log10": min(found["score"]), "max_log10": max(found["score"]),
         "mean_true_log10": sum(found["truth"]) / len(found["truth"])}
        for k, found in sorted(per_k.items())
    ]
    return block


def bayes_block(pairs: list[dict], values: dict[str, float | None], bootstrap: dict, label: str) -> dict:
    by_sequence = defaultdict(dict)
    for row in pairs:
        by_sequence[row["seq_id"]][row["hypothesis"]] = row
    points = []
    impossible = []
    for seq_id, rows in sorted(by_sequence.items()):
        for top, bottom in BF_PAIRS:
            if top not in rows or bottom not in rows:
                continue
            first, second = values.get(rows[top]["query_id"]), values.get(rows[bottom]["query_id"])
            predicted = None if first is None or second is None else first - second
            truth_top, truth_bottom = rows[top]["true_logp"], rows[bottom]["true_logp"]
            faces = rows[top]["faces"]
            if truth_top is None or truth_bottom is None:
                if (top, bottom) == ("even", "fair") and predicted is not None:
                    impossible.append(predicted)
                continue
            points.append({"seq_id": seq_id, "k": rows[top]["k"], "pair": f"{top}/{bottom}",
                           "all_even": all(face % 2 == 0 for face in faces),
                           "true": truth_top - truth_bottom, "predicted": predicted})
    usable = [point for point in points if point["predicted"] is not None]
    fit = ols([point["true"] for point in usable], [point["predicted"] for point in usable])
    even = [point for point in usable if point["pair"] == "even/fair" and point["all_even"]]
    even_fit = ols([point["true"] for point in even], [point["predicted"] for point in even])
    block = {"n_pairs": len(points), "n_used": len(usable), "fit": fit, "even_vs_fair_all_even": even_fit,
             "even_vs_fair_impossible": {
                 "n": len(impossible),
                 "median_predicted": sorted(impossible)[len(impossible) // 2] if impossible else None,
                 "share_favouring_fair": sum(1 for value in impossible if value < 0) / len(impossible) if impossible else None,
             }}
    if fit is not None:
        slopes, even_slopes = [], []
        for sample in _strata_replicates(usable, "k", int(bootstrap["replicates"]), int(bootstrap["seed"]), label):
            found = ols([point["true"] for point in sample], [point["predicted"] for point in sample])
            if found is not None:
                slopes.append(found["slope"])
            evens = [point for point in sample if point["pair"] == "even/fair" and point["all_even"]]
            found = ols([point["true"] for point in evens], [point["predicted"] for point in evens])
            if found is not None:
                even_slopes.append(found["slope"])
        block["slope_ci95"] = interval(slopes)
        block["even_vs_fair_slope_ci95"] = interval(even_slopes)
    block["points"] = points
    return block


def k1_accuracy(norm: list[dict], rows: dict[str, dict]) -> dict:
    found = []
    for query in norm:
        if not query["norm_set"].startswith("k1_"):
            continue
        p = score_probability(rows.get(query["query_id"]))
        found.append({"hypothesis": query["hypothesis"], "face": query["faces"][0], "true": query["true_p"], "score": p,
                      "abs_error": None if p is None else abs(p - query["true_p"])})
    errors = [row["abs_error"] for row in found if row["abs_error"] is not None]
    return {
        "n": len(found),
        "missing": len(found) - len(errors),
        "mean_abs_error": sum(errors) / len(errors) if errors else None,
        "max_abs_error": max(errors) if errors else None,
        "share_within_0_10": sum(1 for value in errors if value <= D3_LIMIT) / len(errors) if errors else None,
        "rows": found,
    }


def normalization(norm: list[dict], rows: dict[str, dict]) -> dict:
    sets = defaultdict(list)
    for query in norm:
        sets[query["norm_set"]].append(score_probability(rows.get(query["query_id"])))
    return {
        name: {"sum": sum(value for value in found if value is not None), "n": len(found),
               "missing": sum(1 for value in found if value is None), "true_sum": 1.0}
        for name, found in sorted(sets.items())
    }


def source_block(model: str, scorer: str, rows: dict[str, dict], pairs: list[dict], norm: list[dict],
                 bootstrap: dict) -> dict:
    scored_pairs = [row for row in pairs if row["query_id"] in rows]
    values = {query_id: log_score(row) for query_id, row in rows.items()}
    verbal = scorer != "S1"
    failed = [row for row in rows.values() if verbal and (row.get("status", "success") != "success" or row.get("p") is None)]
    block = {
        "model": model,
        "scorer": scorer,
        "queries": len(rows),
        "pairs": len(scored_pairs),
        "parse_failures": len(failed),
        "parse_failure_rate": len(failed) / len(rows) if rows else None,
        "zeros": sum(1 for row in rows.values() if verbal and row.get("p") == 0),
        "cut_off": sum(1 for row in rows.values() if row.get("cut_off")),
        "slopes": {},
    }
    for hypothesis in ("fair", "even", "loaded"):
        own = [row for row in scored_pairs if row["own"] and row["hypothesis"] == hypothesis]
        if own:
            block["slopes"][hypothesis] = slope_block(own, values, bootstrap, f"dice:{model}:{scorer}:{hypothesis}")
    impossible = [rows[row["query_id"]] for row in scored_pairs if row["true_logp"] is None]
    if verbal:
        block["impossible"] = {"n": len(impossible),
                               "share_answered_zero": sum(1 for row in impossible if row.get("p") == 0) / len(impossible)
                               if impossible else None}
    else:
        found = sorted(row["logp"] for row in impossible)
        block["impossible"] = {"n": len(found), "median_logp": found[len(found) // 2] if found else None}
    block["k1_accuracy"] = k1_accuracy(norm, rows)
    block["bayes"] = bayes_block(scored_pairs, values, bootstrap, f"dice:{model}:{scorer}:bayes")
    block["normalization"] = normalization(norm, rows)
    return block


def _fair(block: dict | None) -> dict:
    return ((block or {}).get("slopes") or {}).get("fair") or {}


def predictions(blocks: dict[str, dict]) -> list[dict]:
    rows = []
    for key, block in sorted(blocks.items()):
        model, scorer = block["model"], block["scorer"]
        fair = _fair(block)
        fit = fair.get("fit") or {}
        if scorer == "S1":
            ratio, r2 = fair.get("ratio_to_truth"), fit.get("r2")
            passed = None if ratio is None or r2 is None else (0.5 <= ratio <= 2.0 and r2 > 0.9)
            rows.append({"id": "D1", "model": model, "pass": passed, "value": {"ratio": ratio, "r2": r2},
                         "claim": "S1 fair slope within a factor of 2 of the truth, R² > 0.9"})
        if scorer == "S2":
            slope = fit.get("slope")
            rows.append({"id": "D2", "model": model, "pass": None if slope is None else abs(slope) < D2_LIMIT,
                         "value": slope, "claim": f"S2 |fair slope| < {D2_LIMIT} per roll"})
            accuracy = block["k1_accuracy"]["mean_abs_error"]
            rows.append({"id": "D3", "model": model, "pass": None if accuracy is None else accuracy <= D3_LIMIT,
                         "value": accuracy, "claim": f"S2 mean |error| at k = 1 ≤ {D3_LIMIT}"})
            total = block["normalization"].get("k2_fair", {})
            rows.append({"id": "D4", "model": model,
                         "pass": None if not total or total["n"] == total["missing"] else total["sum"] > D4_LIMIT,
                         "value": total.get("sum"), "missing": total.get("missing"),
                         "claim": f"S2 sum over the 36 sequences at k = 2 > {D4_LIMIT}"})
        if scorer == "S3":
            rows.append({"id": "D5", "model": model, "pass": None, "status": "no prediction",
                         "value": {"ratio": fair.get("ratio_to_truth"), "r2": fit.get("r2")},
                         "claim": "S3 reported without a prediction"})
    order = {"D1": 0, "D2": 1, "D3": 2, "D4": 3, "D5": 4}
    return sorted(rows, key=lambda row: (order[row["id"]], row["model"]))


def acceptance(blocks: dict[str, dict], local_models: set[str]) -> dict:
    local = {key: block["parse_failure_rate"] for key, block in blocks.items()
             if block["model"] in local_models and block["scorer"] != "S1"}
    return {"local_parse_failure_rates": local,
            "local_parse_failures_under_2pct": all(rate < LOCAL_PARSE_LIMIT for rate in local.values()) if local else None}


def analyze(root: Path, config: dict, run_dirs: list[Path], data_dir: Path) -> dict:
    pairs, norm = load_queries(data_dir)
    sources, runs = load_scores(run_dirs)
    blocks = {
        f"{model}:{scorer}": source_block(model, scorer, rows, pairs, norm, config["bootstrap"])
        for (model, scorer), rows in sorted(sources.items())
    }
    return {
        "quantity": "Exact dice likelihoods against token log-probs (S1) and verbalized probabilities (S2, S3)",
        "runs": runs,
        "counts": {"pairs": len(pairs), "normalization_queries": len(norm),
                   "sequences": len({row["seq_id"] for row in pairs})},
        "true_slopes_log10_per_roll": TRUE_SLOPES,
        "sources": blocks,
        "predictions": predictions(blocks),
        "acceptance": acceptance(blocks, set(config["local_models"])),
    }
