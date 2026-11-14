"""Simulator study, Phase 5: coverage of later user turns, from the sim-pilot runs.

Each pilot run gives turns 2 and 3; turn 1 comes from the same model's Phase 2 run on the same
dialogues, with each hypothesis's first 32 of its 64 samples (draws are independent, so these are 32
samples), so that every turn's percentile and share beyond all samples rest on the same sample
count. Coverage is Phase 3's: a pair's human percentile is the share of its own context's samples at
least as surprising as the human turn (cut-off samples by their lower bound); intervals come from the
item bootstrap over the pilot dialogues; turn t against turn 1 is a paired difference with the same
draws. The ranking orders the models as Phase 3 does, by median percentile and then share beyond all
samples, over all hypotheses.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR
from robust_likelihood.multiwoz.claim1 import ItemBoot
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.multiwoz.metrics import goal_key
from robust_likelihood.multiwoz.report import INK, _pyplot, write_csv
from robust_likelihood.multiwoz.sim import load_sim_config
from robust_likelihood.multiwoz.sim_analyze import discover, human_percentile, paired_difference
from robust_likelihood.multiwoz.sim_report import LABELS, _dots, _get, _save, num, pct, table, with_ci
from robust_likelihood.storage import read_json, read_jsonl, write_json

HEALTH = ("cut_off", "empty", "assistant_opening", "restaurant_voice", "think_block", "other_flag")
EXAMPLE_SAMPLES = 3


def discover_pilot(run_dirs: list[Path]) -> dict[str, Path]:
    """The newest finished sim-pilot run per (model, format)."""
    found: dict[str, Path] = {}
    for run_dir in sorted(run_dirs):
        path = run_dir / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        if manifest.get("kind") == "sim-pilot" and "finished_at" in manifest:
            found[f"{manifest['model_key']}|{manifest['format']}"] = run_dir
    return found


def health(samples: list[dict]) -> dict:
    """Shares of cut-off, empty and flagged samples."""
    if not samples:
        return {"samples": 0}
    known = {"empty", "truncated", "assistant_opening", "restaurant_voice", "think_block"}
    counts = {name: 0 for name in HEALTH}
    for row in samples:
        counts["cut_off"] += not row["ended"]
        for name in ("empty", "assistant_opening", "restaurant_voice", "think_block"):
            counts[name] += name in row["flags"]
        counts["other_flag"] += any(flag not in known for flag in row["flags"])
    return {"samples": len(samples), **{name: value / len(samples) for name, value in counts.items()}}


def turn_one_pairs(scores: list[dict], samples: list[dict], texts: dict, utterances: dict, dialogues: set[str],
                   count: int) -> tuple[list[dict], list[dict]]:
    """Phase 2 pairs on the pilot dialogues, each with the first count samples of its hypothesis text."""
    by_text: dict[str, list[dict]] = {}
    wanted = {texts[(row["item_id"], row["hyp_key"])] for row in scores
              if row["set"] == "clean" and row["item_id"] in dialogues}
    for row in samples:
        if row["text"] in wanted and row["sample"] < count:
            by_text.setdefault(row["text"], []).append(row)
    pairs = []
    for row in scores:
        if row["set"] != "clean" or row["item_id"] not in dialogues:
            continue
        text = texts[(row["item_id"], row["hyp_key"])]
        pairs.append({"item": row["item_id"], "key": row["hyp_key"], "S": row["S"], "samples": by_text.get(text, []),
                      "utterance": utterances[(row["item_id"], 1)]})
    used = [row for text in sorted(by_text) for row in by_text[text]]
    return pairs, used


def later_pairs(scores: list[dict], samples: list[dict], utterances: dict, turn: int) -> tuple[list[dict], list[dict]]:
    by_context: dict[tuple, list[dict]] = {}
    used = [row for row in samples if row["turn"] == turn]
    for row in used:
        by_context.setdefault((row["item_id"], row["text"]), []).append(row)
    pairs = [{"item": row["item_id"], "key": row["hyp_key"], "S": row["S"],
              "samples": by_context.get((row["item_id"], row["text"]), []),
              "utterance": utterances[(row["item_id"], turn)]}
             for row in scores if row["turn"] == turn]
    return pairs, used


def turn_coverage(pairs: list[dict], used: list[dict], gold: dict[str, str], boot: ItemBoot) -> tuple[dict, dict]:
    """Coverage over all hypotheses and the gold one, per-word surprisal under gold, and sample health."""
    found, columns = {}, {}
    for scope in ("all", "gold"):
        rows = [pair for pair in pairs if scope == "all" or pair["key"] == gold[pair["item"]]]
        usable = [(pair["item"], human_percentile(pair["S"], pair["samples"])) for pair in rows]
        usable = [(item, value) for item, value in usable if value is not None]
        ids, values = [item for item, _ in usable], [value for _, value in usable]
        columns[scope] = (ids, values)
        found[scope] = {"pairs": len(rows),
                        "median_percentile": boot.median(ids, values),
                        "share_beyond_all": boot.mean(ids, [float(value == 0) for value in values]),
                        "share_below_005": boot.mean(ids, [float(value < 0.05) for value in values])}
    gold_pairs = [pair for pair in pairs if pair["key"] == gold[pair["item"]]]
    human = [(pair["item"], pair["S"] / max(1, len(pair["utterance"].split()))) for pair in gold_pairs]
    sample = [(pair["item"], row["S"] / max(1, len(row["reply"].split())))
              for pair in gold_pairs for row in pair["samples"] if row.get("S") is not None]
    found["per_word"] = {"human": boot.median([item for item, _ in human], [value for _, value in human]),
                         "samples": boot.median([item for item, _ in sample], [value for _, value in sample])}
    found["health"] = health(used)
    return found, columns


def ranking(runs: dict, turn: str) -> list[str]:
    def place(label):
        block = runs[label]["turns"].get(turn, {}).get("all", {})
        median = _get(block, "median_percentile", "median")
        beyond = _get(block, "share_beyond_all", "mean")
        return (-(median if median is not None else -1.0), beyond if beyond is not None else 2.0, label)

    return sorted(runs, key=place)


def discordant(first: list[str], second: list[str]) -> list[list[str]]:
    """Pairs of models that the two orders place differently."""
    where = {label: index for index, label in enumerate(second)}
    return [[a, b] for index, a in enumerate(first) for b in first[index + 1:]
            if a in where and b in where and where[a] > where[b]]


def examples(seed: int, dialogues: list[str], plan: list[dict], gold: dict[str, str], texts: dict,
             runs_samples: dict, runs_scores: dict) -> list[dict]:
    """One seeded dialogue: at each later turn, its history, the human turn and its percentile under the
    gold hypothesis, and each model's first ended samples there."""
    item = random.Random(seed).choice(dialogues)
    found = []
    for row in plan:
        if row["item_id"] != item:
            continue
        text = texts[(item, gold[item])]
        models = {}
        for label in runs_samples:
            mine = [sample for sample in runs_samples[label]
                    if sample["item_id"] == item and sample["turn"] == row["turn"] and sample["text"] == text]
            score = runs_scores[label].get((item, row["turn"], gold[item]))
            models[label] = {"human_S": score,
                             "percentile": None if score is None else human_percentile(score, mine),
                             "samples": [sample["reply"] for sample in sorted(mine, key=lambda s: s["sample"])
                                         if sample["ended"]][:EXAMPLE_SAMPLES]}
        found.append({"item_id": item, "turn": row["turn"], "hypothesis": text, "history": row["history"],
                      "utterance": row["utterance"], "models": models})
    return found


def run_pilot_analyze(root: Path, *, runs: list[Path], out: Path, derived: Path | None = None,
                      figures: bool = True, now: datetime | None = None) -> dict:
    config = load_sim_config(root)
    phase5, boot_config = config["phase5"], config["bootstrap"]
    derived = derived or root / DERIVED_DIR
    pilots, phase2 = discover_pilot(runs), discover(runs)
    if not pilots:
        raise ValueError("no finished sim-pilot run among --runs")
    plan = read_jsonl(root / phase5["file"])
    dialogues = sorted({row["item_id"] for row in plan})
    items = {row["item_id"]: row for row in read_jsonl(derived / "items.jsonl")}
    probe = [row for row in read_jsonl(derived / "probe.jsonl") if row["set"] == "clean"]
    texts = {(row["item_id"], row["hyp_key"]): row["text"] for row in probe}
    gold = {item: goal_key(items[item]["n_constrained"], "impl") for item in dialogues}
    utterances = {(item, 1): items[item]["utterance"] for item in dialogues}
    utterances.update({(row["item_id"], row["turn"]): row["utterance"] for row in plan})
    boot = ItemBoot(dialogues, int(boot_config["replicates"]), int(boot_config["seed"]))
    count = int(phase5["samples"])
    metrics_runs, sources, all_samples, all_scores = {}, {}, {}, {}
    for label, run_dir in sorted(pilots.items()):
        manifest = read_json(run_dir / "manifest.json")
        scores, samples = read_jsonl(run_dir / "pilot_scores.jsonl"), read_jsonl(run_dir / "pilot_samples.jsonl")
        all_samples[label] = samples
        all_scores[label] = {(row["item_id"], row["turn"], row["hyp_key"]): row["S"] for row in scores}
        block, columns = {"turns": {}, "paired": {}}, {}
        sources[label] = {"pilot": run_dir.name, "commit": (manifest.get("git") or {}).get("commit"),
                          "checks": manifest.get("checks"), "elapsed_seconds": manifest.get("elapsed_seconds")}
        if label in phase2:
            first_scores = read_jsonl(phase2[label] / "lm_scores.jsonl")
            first_samples = read_jsonl(phase2[label] / "lm_samples.jsonl")
            pairs, used = turn_one_pairs(first_scores, first_samples, texts, utterances, set(dialogues), count)
            block["turns"]["1"], columns["1"] = turn_coverage(pairs, used, gold, boot)
            sources[label]["turn_1"] = phase2[label].name
        for turn in sorted({row["turn"] for row in scores}):
            pairs, used = later_pairs(scores, samples, utterances, turn)
            block["turns"][str(turn)], columns[str(turn)] = turn_coverage(pairs, used, gold, boot)
        if "1" in columns:
            for turn in (key for key in columns if key != "1"):
                block["paired"][turn] = {
                    "median_percentile": paired_difference(boot, columns["1"]["all"], columns[turn]["all"], "median"),
                    "share_beyond_all": paired_difference(
                        boot, (columns["1"]["all"][0], [float(value == 0) for value in columns["1"]["all"][1]]),
                        (columns[turn]["all"][0], [float(value == 0) for value in columns[turn]["all"][1]]), "mean")}
        metrics_runs[label] = block
    turns = sorted({turn for block in metrics_runs.values() for turn in block["turns"]}, key=int)
    orders = {turn: ranking(metrics_runs, turn) for turn in turns}
    holds = {turn: discordant(orders["1"], orders[turn]) for turn in turns if turn != "1" and "1" in orders}
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics = {"generated_at": stamp, "git": git_state(root), "dialogues": dialogues, "samples_per_context": count,
               "turns": turns, "runs": metrics_runs, "sources": sources, "ranking": orders,
               "discordant_with_turn_1": holds,
               "examples": examples(int(config["seed"]), dialogues, plan, gold, texts, all_samples, all_scores)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "pilot_metrics.json", metrics)
    paths = write_figure(out, metrics) if figures else []
    discussion = out / "discussion.md"
    (out / "pilot.md").write_text(pilot_markdown(metrics, discussion.read_text(encoding="utf-8")
                                                 if discussion.exists() else None), encoding="utf-8", newline="\n")
    return {"out": str(out), "runs": sorted(metrics_runs), "turns": turns, "ranking": orders,
            "discordant_with_turn_1": holds, "figures": [str(path) for path in paths]}


# ---------------------------------------------------------------- figure and markdown


def ordered_labels(runs: dict) -> list[str]:
    order = [label for label in LABELS if label in runs]
    return order + sorted(label for label in runs if label not in order)


def write_figure(out: Path, metrics: dict) -> list[Path]:
    plt = _pyplot()
    runs, turns = metrics["runs"], metrics["turns"]
    labels = ordered_labels(runs)
    figure, axes = plt.subplots(2, len(turns), figsize=(3.2 * len(turns) + 2.2, 0.45 * len(labels) * 2 + 2.2),
                                sharey=True, squeeze=False)
    rows = []
    for column, turn in enumerate(turns):
        for row_index, (field, stat, title) in enumerate((("median_percentile", "median", "median human percentile"),
                                                          ("share_beyond_all", "mean", "share beyond all samples"))):
            ax = axes[row_index][column]
            points, intervals = [], []
            for label in labels:
                block = _get(runs[label], "turns", turn, "all", field)
                points.append(_get(block, stat))
                intervals.append(_get(block, "ci95"))
                rows.append({"run": label, "turn": turn, "metric": field, "value": _get(block, stat),
                             "ci_low": (_get(block, "ci95") or [None, None])[0],
                             "ci_high": (_get(block, "ci95") or [None, None])[1]})
            _dots(ax, [LABELS.get(label, label) for label in labels], points, intervals)
            ax.set_xlim(-0.03, 1.03)
            if row_index == 0:
                ax.set_title(f"user turn {turn}" + (" (Phase 2, 32 samples)" if turn == "1" else ""), fontsize=9,
                             color=INK["primary"])
            ax.set_xlabel(title, fontsize=8)
    figure.suptitle("Coverage of human user turns by turn, 60 dialogues, all hypotheses", fontsize=10,
                    color=INK["primary"])
    figure.tight_layout()
    write_csv(out / "fig7_multiturn.csv", rows)
    paths = _save(figure, out / "fig7_multiturn")
    plt.close(figure)
    return paths


def pilot_markdown(metrics: dict, discussion: str | None) -> str:
    runs, turns = metrics["runs"], metrics["turns"]
    labels = ordered_labels(runs)
    name = {label: LABELS.get(label, label) for label in labels}
    commits = sorted({source.get("commit") or "?" for source in metrics["sources"].values()})
    lines = ["# Simulator study, Phase 5: multi-turn pilot", "",
             f"User turns 2 and 3 of {len(metrics['dialogues'])} clean dialogues (seeded draw from the 235), scored "
             "and sampled by the S-format chat models given the hypothesis and the real history: format S's system "
             "prompt and opener, then every earlier turn in its role. Every hypothesis of the dialogue's first-turn "
             f"set, {metrics['samples_per_context']} samples per (dialogue, turn, hypothesis), pure sampling. "
             "Turn 1 is the Phase 2 run on the same dialogues with each hypothesis's first 32 samples, so every turn "
             "has the same sample count. Brackets are 95% item-bootstrap intervals over the dialogues. "
             f"Pilot runs at commit {', '.join(f'`{commit[:7]}`' for commit in commits)}; generated "
             f"{metrics['generated_at']}.", ""]
    if discussion:
        lines += [discussion.rstrip("\n"), ""]
    for scope, heading in (("all", "all hypotheses"), ("gold", "gold hypothesis")):
        lines += [f"## Coverage by turn, {heading}", "",
                  "Median human percentile (the share of the context's own samples at least as surprising as the "
                  "human turn; 0.5 is a typical sample), then the share of pairs beyond all samples.", ""]
        header = ["Model"] + [f"Median pct, turn {turn}" for turn in turns] + [f"Beyond all, turn {turn}" for turn in turns]
        body = []
        for label in labels:
            blocks = [_get(runs[label], "turns", turn, scope) for turn in turns]
            body.append([name[label]] + [with_ci(_get(block, "median_percentile"), "median") for block in blocks]
                        + [with_ci(_get(block, "share_beyond_all"), "mean", percent=True) for block in blocks])
        lines += [table(header, body), ""]
    lines += ["## Ranking by turn", "",
              "Ordered by median percentile, then share beyond all samples (all hypotheses), as in Phase 3.", ""]
    lines += [table(["Turn", "Order (best first)", "Pairs ordered differently from turn 1"],
                    [[turn, " > ".join(name.get(label, label) for label in metrics["ranking"][turn]),
                      "–" if turn == "1" else (", ".join(f"{name.get(a, a)} / {name.get(b, b)}"
                                                         for a, b in metrics["discordant_with_turn_1"].get(turn, []))
                                               or "none")]
                     for turn in turns]), ""]
    lines += ["## Change from turn 1", "", "Paired item-bootstrap differences, later turn minus turn 1, all hypotheses.",
              ""]
    later = [turn for turn in turns if turn != "1"]
    body = []
    for label in labels:
        paired = runs[label].get("paired", {})
        body.append([name[label]]
                    + [with_ci(_get(paired, turn, "median_percentile"), "difference") for turn in later]
                    + [with_ci(_get(paired, turn, "share_beyond_all"), "difference", percent=True) for turn in later])
    lines += [table(["Model"] + [f"Δ median pct, turn {turn}" for turn in later]
                    + [f"Δ beyond all, turn {turn}" for turn in later], body), ""]
    lines += ["## Surprisal per word under the gold hypothesis", "", "Median nats per word.", ""]
    body = [[name[label]] + [num(_get(runs[label], "turns", turn, "per_word", who, "median"))
                             for turn in turns for who in ("human", "samples")] for label in labels]
    lines += [table(["Model"] + [f"{who}, turn {turn}" for turn in turns for who in ("human", "samples")], body), ""]
    lines += ["## Sample health by turn", "",
              "Cut off: no end-of-turn token within 96 new tokens. The flags are Phase 1's heuristics.", ""]
    body = []
    for label in labels:
        for turn in turns:
            found = _get(runs[label], "turns", turn, "health") or {}
            body.append([name[label], turn, str(found.get("samples", 0))]
                        + [pct(found.get(field)) for field in HEALTH])
    lines += [table(["Model", "Turn", "Samples", "Cut off", "Empty", "Assistant opening", "Restaurant voice",
                     "Think block", "Other flag token"], body), ""]
    for example in metrics["examples"]:
        lines += [f"## Example: {example['item_id']}, user turn {example['turn']}", "",
                  f"Gold hypothesis: {example['hypothesis']}", ""]
        lines += [f"- *{turn['speaker'].lower()}*: {turn['utterance']}" for turn in example["history"]]
        lines += [f"- **human user turn {example['turn']}**: {example['utterance']}", ""]
        for label in labels:
            found = example["models"].get(label)
            if not found:
                continue
            lines.append(f"{name[label]} (human S {num(found['human_S'], 1)}, percentile "
                         f"{num(found['percentile'])}):")
            lines += [f"- {' '.join(sample.split()) or '∅'}" for sample in found["samples"]] + [""]
    return "\n".join(lines).rstrip("\n") + "\n"
