"""Figures and summary.md for a run folder of the Gaussian test.

Everything is read back from config.json, manifest.json, metrics.jsonl, cells.jsonl,
and e3_heatmaps.json, so a stored run can be reported again without re-running it.
summary.md opens with the checklist of pre-registered predictions; its tables are
the table view of every figure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from robust_likelihood.storage import read_json, read_jsonl

MISSING = "n/a"
ROUNDING = 1e-12


@dataclass(frozen=True, eq=False)
class RunData:
    """The stored rows of one run folder, indexed for lookup."""

    run_dir: Path
    config: dict
    manifest: dict
    metrics: list[dict]
    cells: list[dict]
    heatmaps: list[dict]

    @classmethod
    def load(cls, run_dir: Path) -> RunData:
        heatmap_path = run_dir / "e3_heatmaps.json"
        data = cls(
            run_dir=run_dir,
            config=read_json(run_dir / "config.json"),
            manifest=read_json(run_dir / "manifest.json"),
            metrics=read_jsonl(run_dir / "metrics.jsonl"),
            cells=read_jsonl(run_dir / "cells.jsonl"),
            heatmaps=read_json(heatmap_path)["heatmaps"] if heatmap_path.is_file() else [],
        )
        index = {}
        for row in data.metrics:
            key = (row["experiment"], row["cell_key"], row["contamination"], row["subset"], row["scorer"], row["n"])
            index[key] = row
        object.__setattr__(data, "_index", index)
        return data

    @property
    def preset(self) -> dict:
        return self.config["preset"]

    def ran(self, experiment: str) -> bool:
        return experiment in self.config["experiments"]

    def row(
        self,
        experiment: str,
        cell_key: str,
        scorer: str,
        n: int,
        *,
        contamination: str = "C0",
        subset: str = "all",
    ) -> dict | None:
        return self._index.get((experiment, cell_key, contamination, subset, scorer, n))

    def value(
        self,
        experiment: str,
        cell_key: str,
        scorer: str,
        n: int,
        metric: str = "p_true",
        *,
        contamination: str = "C0",
        subset: str = "all",
    ) -> float | None:
        """Trial mean of a metric, or None when the row or the metric is absent."""
        row = self.row(experiment, cell_key, scorer, n, contamination=contamination, subset=subset)
        if row is None or row.get(metric) is None:
            return None
        return row[metric]["mean"]

    def cell_rows(self, experiment: str, kind: str) -> list[dict]:
        return [row for row in self.cells if row["experiment"] == experiment and row["kind"] == kind]

    def n90(
        self,
        experiment: str,
        cell_key: str,
        scorer: str,
        *,
        contamination: str = "C0",
        subset: str = "all",
    ) -> tuple[bool, int | None]:
        """(found, n90) for one cell and scorer; n90 is None when mean p_true never reached 0.9."""
        for row in self.cell_rows(experiment, "n90"):
            if (
                row["cell_key"] == cell_key
                and row["scorer"] == scorer
                and row["contamination"] == contamination
                and row["subset"] == subset
            ):
                return True, row["n90"]
        return False, None


def fmt(value: float | None, digits: int = 3) -> str:
    """Fixed-point text. Magnitudes below 1e-12 are rounding and print as 0; other small values use an exponent."""
    if value is None:
        return MISSING
    if abs(value) < ROUNDING:
        return f"{0.0:.{digits}f}"
    if abs(value) < 10.0 ** (-digits):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def r_key(d: int, r: float) -> str:
    return f"d={d}|r={r:g}"


def _header(data: RunData) -> list[str]:
    manifest = data.manifest
    git = manifest.get("git", {})
    commit = (git.get("commit") or "unknown")[:12]
    dirty = git.get("dirty")
    dirty_note = ""
    if dirty:
        dirty_note = f" The working tree had {len(git.get('dirty_paths', []))} uncommitted path(s): " + ", ".join(
            f"`{path}`" for path in git.get("dirty_paths", [])
        ) + "."
    times = manifest.get("wall_clock_seconds", {})
    per_experiment = ", ".join(f"{name} {times[name]:.0f}s" for name in data.config["experiments"] if name in times)
    total = times.get("total")
    lines = [
        "# Gaussian typicality test",
        "",
        f"Run `{data.run_dir.name}`. Preset `{data.preset['name']}`, base seed {manifest.get('base_seed')}, "
        f"commit `{commit}`.{dirty_note}",
        f"Started {manifest.get('started_at')}, finished {manifest.get('finished_at')}. "
        f"Wall-clock {fmt(total, 0) if total is not None else MISSING} s ({per_experiment}).",
        "",
        "These are exact Gaussian computations on simulated observations. They are not language-model scores.",
        "",
    ]
    if data.preset["name"] != "full":
        lines += [f"Preset `{data.preset['name']}` is a smoke run with reduced grids and trials. It is not reportable.", ""]
    return lines


def _e0_section(data: RunData) -> list[str]:
    entropy = data.cell_rows("e0", "entropy")
    if not entropy:
        return []
    lines = ["## E0. Closed forms against Monte Carlo", ""]
    lines.append(
        "H-hat and V-hat are the mean and variance of S surprisals drawn from the hypothesis. "
        "z is the error in standard errors: sqrt(V/S) for H-hat, and the exact SE of a sample variance for V-hat. "
        "The plug-in shift is (H - H-hat) / sqrt(V), how far an estimated entropy moves the center of T."
    )
    lines.append("")
    shifts = {row["S"]: row for row in data.cell_rows("e0", "plugin_shift")}
    rows = []
    for size in sorted({row["S"] for row in entropy}):
        chosen = [row for row in entropy if row["S"] == size]
        shift = shifts.get(size, {})
        rows.append(
            [
                str(size),
                str(len(chosen)),
                fmt(max(abs(row["z_H"]) for row in chosen), 2),
                fmt(max(abs(row["z_V"]) for row in chosen), 2),
                fmt(shift.get("rms_shift"), 4),
                fmt(shift.get("predicted_sd"), 4),
            ]
        )
    lines += table(["S", "cells", "max abs z, H-hat", "max abs z, V-hat", "RMS plug-in shift of T", "1/sqrt(S)"], rows)
    lines.append("")
    moments = data.cell_rows("e0", "moments")
    rows = [
        [
            str(row["d"]),
            f"{row['sigma2']:g}",
            fmt(row["mean"], 4),
            fmt(row["var"], 4),
            fmt(row["skew"], 4),
            fmt(row["target_skew"], 4),
            fmt(max(abs(row["z_mean"]), abs(row["z_var"]), abs(row["z_skew"])), 2),
        ]
        for row in moments
    ]
    if rows:
        lines.append(f"Moments of T from {moments[0]['draws']:,} draws per cell, against (0, 1, sqrt(8/d)).")
        lines.append("")
        lines += table(["d", "sigma2", "mean", "variance", "skewness", "sqrt(8/d)", "max abs z"], rows)
        lines.append("")
    masses = data.cell_rows("e0", "bin_mass")
    if masses:
        dims = sorted({row["d"] for row in masses})
        schemes = list(dict.fromkeys(row["scheme"] for row in masses))
        lookup = {(row["scheme"], row["d"]): row for row in masses}
        lines.append(
            f"Chi-square goodness-of-fit p-values of the exact bin masses against {masses[0]['draws']:,} draws of T "
            "per d, pooled over the variances."
        )
        lines.append("")
        lines += table(
            ["scheme"] + [f"d={d}" for d in dims],
            [[name] + [fmt(lookup[(name, d)]["p_value"], 3) if (name, d) in lookup else MISSING for d in dims]
             for name in schemes],
        )
        dropped = [f"{row['scheme']} at d={row['d']}: bins {row['dropped_bins']}" for row in masses if row["dropped_bins"]]
        if dropped:
            lines += ["", "Bins dropped as empty (mass below 1e-300): " + "; ".join(dropped) + "."]
        lines.append("")
    return lines


def _e1_section(data: RunData) -> list[str]:
    if not data.ran("e1"):
        return []
    config = data.preset["e1"]
    scorers = config["scorers"]
    cells = [(d, r) for d in config["d_grid"] for r in config["r_grid"]]
    n_max = config["n_max"]
    lines = ["## E1. Information loss and convergence, mean-shift family", ""]
    lines.append(
        f"M = 20, delta = r (2d)^(1/4), {config['trials']} trials per cell, n up to {n_max}. "
        "n90 is the smallest n with mean p_true >= 0.9."
    )
    lines.append("")

    def grid(label: str, value) -> list[str]:
        rows = []
        for d, r in cells:
            delta = (2.0 * d) ** 0.25 * r
            rows.append([str(d), f"{r:g}", fmt(delta, 2)] + [value(r_key(d, r), scorer) for scorer in scorers])
        return [f"**{label}**", ""] + table(["d", "r", "delta"] + scorers, rows) + [""]

    def n90_text(cell_key: str, scorer: str) -> str:
        found, value = data.n90("e1", cell_key, scorer)
        if not found:
            return MISSING
        return "not reached" if value is None else str(value)

    lines += grid("n90", n90_text)
    lines += grid("bits(1): single-observation information about the truth, in bits",
                  lambda key, scorer: fmt(data.value("e1", key, scorer, 1, "bits")))
    for n in sorted({min(20, n_max), n_max}):
        lines += grid(f"p_true({n})", lambda key, scorer, n=n: fmt(data.value("e1", key, scorer, n)))
    lines += grid(f"kl_to_exact({n_max}), nats",
                  lambda key, scorer: fmt(data.value("e1", key, scorer, n_max, "kl_to_exact")))
    return lines


def _e2_section(data: RunData) -> list[str]:
    limits = data.cell_rows("e2", "limit")
    if not limits:
        return []
    config = data.preset["e2"]
    scorers = config["scorers"]
    truths = sorted({row["truth_sigma2"] for row in limits})
    dims = config["d_grid"]
    lines = ["## E2. Sharp versus broad hypotheses, scale family", ""]
    lines.append(
        f"sigma2 in (0.25, 0.5, 1, 2, 4), {config['trials']} trials per d with the truth cycling, n up to "
        f"{config['n_max']}. Each entry is the predicted limit, the argmax of E[l_z(x)] under the truth, then the share "
        f"of trials whose posterior mode at n = {config['n_max']} equals it. A tie at the top is not a mode; its share "
        "is shown when it occurs. The margin is the gap, in nats per observation, between the best and second-best "
        "E[l_z(x)]; a posterior cannot separate two hypotheses whose margin times n is small."
    )
    lines.append("")
    lookup = {(row["scorer"], row["cell"]["d"], row["truth_sigma2"]): row for row in limits}
    for scorer in scorers:
        rows = []
        for d in dims:
            entries = []
            for truth in truths:
                row = lookup.get((scorer, d, truth))
                if row is None:
                    entries.append(MISSING)
                    continue
                predicted = row["predicted_sigma2"]
                if predicted is None:
                    entries.append("tied objective")
                    continue
                text = f"{predicted:g}: {100 * row['mode_match']:.0f}%"
                ties = row["mode_counts"].get("tie", 0)
                if ties:
                    text += f", ties {100 * ties / row['trials']:.0f}%"
                entries.append(text + f"; margin {fmt(row['objective_margin'], 3)}")
            rows.append([str(d)] + entries)
        lines += [f"**{scorer}**", ""] + table(["d"] + [f"truth {t:g}" for t in truths], rows) + [""]
    rows = [
        [str(d)] + [fmt(data.value("e2", f"d={d}", scorer, config["n_max"])) for scorer in scorers] for d in dims
    ]
    lines += [f"**p_true({config['n_max']})**, all truths", ""] + table(["d"] + scorers, rows) + [""]
    return lines


def _e3_section(data: RunData) -> list[str]:
    if not data.ran("e3"):
        return []
    config = data.preset["e3"]
    scorers = config["scorers"]
    n = config["n_max"]
    cells = [(s, d) for s in config["spacing_grid"] for d in config["d_grid"]]
    lines = ["## E3. Localization with nuisance dimensions", ""]
    lines.append(
        f"441 hypotheses on a 21 x 21 grid in coordinates 1-2, {config['trials']} trials per cell, truth at least two "
        "cells from the border. Content-oracle scorers use the two in-plane coordinates with d_eff = 2."
    )
    lines.append("")
    for metric, label in (("p_true", "p_true"), ("loc_error", "loc_error, grid units"), ("post_entropy", "post_entropy, bits")):
        rows = [
            [f"{s:g}", str(d)] + [fmt(data.value("e3", f"s={s:g}|d={d}", scorer, n, metric)) for scorer in scorers]
            for s, d in cells
        ]
        lines += [f"**{label} at n = {n}**", ""] + table(["spacing", "d"] + scorers, rows) + [""]
    checks = data.cell_rows("e3", "check")
    if checks:
        rows = [
            [row["name"], row["cell_key"], f"{row['max_abs_diff']:.2e}", f"{row['threshold']:.0e}",
             "yes" if row["passed"] else "no"]
            for row in checks
        ]
        lines += ["**Identity checks**", ""] + table(["check", "cell", "max abs difference", "threshold", "passed"], rows)
        lines.append("")
    return lines


def _e4_section(data: RunData) -> list[str]:
    if not data.ran("e4"):
        return []
    config = data.preset["e4"]
    scorers = config["scorers"]
    keys = config["contamination_keys"]
    n_max = config["n_max"]
    lines = ["## E4. Robustness, mean-shift family", ""]
    lines.append(
        f"r = {config['r']:g}, M = 20, {config['trials']} trials, n up to {n_max}. A tie counts as incorrect."
    )
    lines.append("")
    for d in config["d_grid"]:
        cell_key = r_key(d, config["r"])
        for n in sorted({min(20, n_max), n_max}):
            rows = [
                [key] + [fmt(data.value("e4", cell_key, scorer, n, "acc", contamination=key), 2) for scorer in scorers]
                for key in keys
            ]
            lines += [f"**acc({n}), d = {d}**", ""] + table(["contamination"] + scorers, rows) + [""]
        if "C5" in keys:
            row = data.row("e4", cell_key, "bins_abs3", min(20, n_max), contamination="C5")
            if row is not None:
                lines.append(
                    f"C5 at d = {d}, bins_abs3, n = {min(20, n_max)}: mean p_true {fmt(row['p_true']['mean'])}; the "
                    f"largest trial-mean posterior on a wrong hypothesis is {fmt(row['max_wrong_mean_posterior'])} "
                    f"(hypothesis {row['max_wrong_hypothesis']})."
                )
                lines.append("")
    return lines


def _e5_section(data: RunData) -> list[str]:
    choices = data.cell_rows("e5", "choice")
    if not choices:
        return []
    config = data.preset["e5"]
    lines = ["## E5. Bin design", ""]
    lines.append(
        f"Calibration: d = {config['calibration_d']}, r = {config['calibration_r']:g}, "
        f"{config['calibration_trials']} trials with their own seed. Objective: mean bits({config['objective_n']}) on "
        f"clean data, subject to acc({config['constraint_n']}) >= {config['min_accuracy']:g} under "
        f"{choices[0]['contamination']}. {len(data.cell_rows('e5', 'candidate'))} candidate absolute schemes."
    )
    lines.append("")
    rows = []
    for row in choices + data.cell_rows("e5", "named_scheme"):
        rows.append(
            [
                row["scheme"],
                ", ".join(f"{edge:g}" for edge in row["edges"]),
                fmt(row["clean_bits"]["mean"]),
                fmt(row["contaminated_acc"]["mean"], 2),
                "yes" if row["feasible"] else "no",
            ]
        )
    lines += table(["scheme", "|T| edges", f"clean bits({config['objective_n']})",
                    f"contaminated acc({config['constraint_n']})", "meets constraint"], rows)
    flags = [row["flag"] for row in choices if row.get("flag")]
    if flags:
        lines += [""] + [f"Flag: {flag}." for flag in flags]
    lines.append("")
    scorers = config["heldout_e1"]["scorers"]
    shown = [scorer for scorer in scorers if scorer != "exact"]
    objective_n = config["objective_n"]
    rows = []
    for d in config["heldout_e1"]["d_grid"]:
        for r in config["heldout_e1"]["r_grid"]:
            rows.append(
                [f"E1 d={d}, r={r:g}", f"bits({objective_n})"]
                + [fmt(data.value("e5_heldout", r_key(d, r), scorer, objective_n, "bits")) for scorer in shown]
            )
    e4 = config["heldout_e4"]
    for d in e4["d_grid"]:
        for key in e4["contamination_keys"]:
            rows.append(
                [f"E4 d={d}, {key}", f"acc({e4['n_max']})"]
                + [fmt(data.value("e5_heldout", r_key(d, e4["r"]), scorer, e4["n_max"], "acc", contamination=key), 2)
                   for scorer in shown]
            )
    lines += ["**Held-out cells**", ""] + table(["cell", "metric"] + shown, rows) + [""]
    return lines


PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Verdict:
    """One pre-registered prediction, its status, and the numbers behind it."""

    prediction: str
    claim: str
    status: str
    numbers: str


def _combine(statuses: Iterable[str]) -> str:
    statuses = list(statuses)
    if FAIL in statuses:
        return FAIL
    if INCONCLUSIVE in statuses or not statuses:
        return INCONCLUSIVE
    return PASS


def _contamination_rows(data: RunData, experiment: str, code: str, radius: float | None = None) -> list[tuple[float, str]]:
    """(rho, key) of the C3 or C4 contaminations of an experiment's config, sorted by rho."""
    config = data.preset[experiment] if experiment == "e4" else data.preset["e5"]["heldout_e4"]
    found = []
    for contamination in config["contaminations"]:
        if contamination["code"] != code or (radius is not None and contamination["radius"] != radius):
            continue
        key = f"{code}:rho={contamination['rho']:g}:R={contamination['radius']:g}"
        found.append((contamination["rho"], key))
    return sorted(found)


def _p1(data: RunData) -> Verdict:
    claim = "Closed forms and Monte Carlo agree."
    entropy = data.cell_rows("e0", "entropy")
    moments = data.cell_rows("e0", "moments")
    masses = data.cell_rows("e0", "bin_mass")
    if not (entropy and moments and masses):
        return Verdict("P1", claim, INCONCLUSIVE, "E0 was not run.")
    worst_h = max(entropy, key=lambda row: abs(row["z_H"]))

    def moment_z(row: dict) -> tuple[float, str]:
        return max((abs(row["z_mean"]), "mean"), (abs(row["z_var"]), "variance"), (abs(row["z_skew"]), "skewness"))

    worst_m = max(moments, key=lambda row: moment_z(row)[0])
    worst_p = min(masses, key=lambda row: row["p_value"])
    status = PASS
    if abs(worst_h["z_H"]) > 4.0 or moment_z(worst_m)[0] > 4.0 or worst_p["p_value"] < 0.001:
        status = FAIL
    numbers = (
        f"Largest abs z of H-hat {abs(worst_h['z_H']):.2f} over {len(entropy)} cells "
        f"(d={worst_h['d']}, sigma2={worst_h['sigma2']:g}, S={worst_h['S']}). "
        f"Largest abs z of a T moment {moment_z(worst_m)[0]:.2f} ({moment_z(worst_m)[1]}, d={worst_m['d']}, "
        f"sigma2={worst_m['sigma2']:g}). Smallest bin-mass p {worst_p['p_value']:.3g} "
        f"({worst_p['scheme']}, d={worst_p['d']}) over {len(masses)} tests."
    )
    return Verdict("P1", claim, status, numbers)


def _p2(data: RunData) -> Verdict:
    claim = "The exact likelihood is dimension-free; typicality-based scorers have a threshold at (2d)^(1/4)."
    if not data.ran("e1"):
        return Verdict("P2", claim, INCONCLUSIVE, "E1 was not run.")
    config = data.preset["e1"]
    if config["n_max"] < 20:
        return Verdict("P2", claim, INCONCLUSIVE, "E1 stops before n = 20.")
    parts = []
    cells = [(d, r) for d in config["d_grid"] for r in config["r_grid"] if r * (2.0 * d) ** 0.25 >= 2.0]
    values = {(d, r): data.value("e1", r_key(d, r), "exact", 20) for d, r in cells}
    worst = min(values.items(), key=lambda item: item[1])
    statuses = [PASS if worst[1] >= 0.99 else FAIL]
    parts.append(
        f"exact: smallest p_true(20) {fmt(worst[1], 4)} over the {len(cells)} cells with delta >= 2 "
        f"(d={worst[0][0]}, r={worst[0][1]:g})."
    )
    dims = config["d_grid"]
    for scorer in ("typ_normal", "bins_abs3"):
        if len(dims) < 3:
            statuses.append(INCONCLUSIVE)
            parts.append(f"{scorer}: E1 has {len(dims)} values of d, not three.")
            continue
        ratios = {}
        for r in config["r_grid"]:
            n90 = [data.n90("e1", r_key(d, r), scorer)[1] for d in dims]
            if all(value is not None for value in n90):
                ratios[r] = (max(n90) / min(n90), n90)
        if len(ratios) < 3:
            statuses.append(INCONCLUSIVE)
            parts.append(f"{scorer}: n90 reached at every d for only {len(ratios)} values of r.")
            continue
        bad = {r: item for r, item in ratios.items() if item[0] > 1.5}
        statuses.append(FAIL if bad else PASS)
        detail = "; ".join(
            f"r={r:g}: n90 {'/'.join(str(v) for v in item[1])}, ratio {item[0]:.2f}" for r, item in ratios.items()
        )
        parts.append(f"{scorer} ({'/'.join(str(d) for d in dims)}): {detail}.")
    return Verdict("P2", claim, _combine(statuses), " ".join(parts))


def _p3(data: RunData) -> Verdict:
    claim = "Most of the information loss comes from reducing x to T, not from binning."
    if not data.ran("e1"):
        return Verdict("P3", claim, INCONCLUSIVE, "E1 was not run.")
    config = data.preset["e1"]
    cells = [(d, r) for d in config["d_grid"] for r in config["r_grid"] if r <= 2.0]
    holds = []
    for d, r in cells:
        key = r_key(d, r)
        exact, typ, bins = (data.value("e1", key, scorer, 1, "bits") for scorer in ("exact", "typ_normal", "bins_abs3"))
        holds.append(((typ - bins) < (exact - typ), d, r, typ - bins, exact - typ))
    count = sum(1 for item in holds if item[0])
    status = PASS if count >= 0.8 * len(holds) else FAIL
    misses = ", ".join(
        f"d={d} r={r:g} (binning gap {binning:.3f} vs reduction gap {reduction:.3f} bits)"
        for ok, d, r, binning, reduction in holds
        if not ok
    )
    numbers = f"The binning gap is the smaller one in {count} of {len(holds)} cells with r <= 2 (80% needs {math.ceil(0.8 * len(holds))})."
    if misses:
        numbers += f" Exceptions: {misses}."
    return Verdict("P3", claim, status, numbers)


def _p4(data: RunData) -> Verdict:
    claim = "With hypotheses of different spread at small d, typ_normal converges to a hypothesis broader than the truth."
    limits = data.cell_rows("e2", "limit")
    if not limits:
        return Verdict("P4", claim, INCONCLUSIVE, "E2 was not run.")
    n_max = data.preset["e2"]["n_max"]
    typ = [row for row in limits if row["scorer"] == "typ_normal"]
    exact = [row for row in limits if row["scorer"] == "exact"]
    low = [row for row in typ if row["mode_match"] is None or row["mode_match"] < 0.9]
    worst = min(typ, key=lambda row: -1.0 if row["mode_match"] is None else row["mode_match"])
    exact_bad = [row for row in exact if row["predicted_sigma2"] != row["truth_sigma2"] or row["truth_match"] < 0.9]
    worst_exact = min(exact, key=lambda row: row["truth_match"])
    numbers = (
        f"typ_normal: posterior mode at n = {n_max} equals the predicted limit in at least 90% of trials in "
        f"{len(typ) - len(low)} of {len(typ)} (d, truth) cells; lowest {100 * (worst['mode_match'] or 0.0):.0f}% "
        f"(d={worst['cell']['d']}, truth {worst['truth_sigma2']:g}, predicted {worst['predicted_sigma2']})."
    )
    if low:
        numbers += " Below 90%: " + ", ".join(
            f"d={row['cell']['d']} truth {row['truth_sigma2']:g} -> {row['predicted_sigma2']} "
            f"({100 * (row['mode_match'] or 0.0):.0f}%)"
            for row in low
        ) + "."
    numbers += (
        f" exact: the predicted limit is the truth in every cell, and the mode is the truth in at least 90% of "
        f"trials in {len(exact) - len(exact_bad)} of {len(exact)} cells; lowest {100 * worst_exact['truth_match']:.0f}%."
    )
    if n_max != 1000:
        return Verdict("P4", claim, INCONCLUSIVE, f"The criterion is at n = 1000; this run stops at n = {n_max}. " + numbers)
    return Verdict("P4", claim, FAIL if (low or exact_bad) else PASS, numbers)


def _p5(data: RunData) -> Verdict:
    claim = "In localization, the exact posterior does not depend on nuisance dimensions; typicality degrades as they grow."
    if not data.ran("e3"):
        return Verdict("P5", claim, INCONCLUSIVE, "E3 was not run.")
    config = data.preset["e3"]
    checks = data.cell_rows("e3", "check")
    exact_checks = [row for row in checks if row["name"] == "exact_identical_across_d"]
    oracle_checks = [row for row in checks if row["name"] == "content_oracle_matches_typ_normal_at_d2"]
    dims = sorted(config["d_grid"])
    if len(dims) < 2 or not exact_checks or not oracle_checks:
        return Verdict("P5", claim, INCONCLUSIVE, "E3 needs d = 2 and at least one larger d.")
    statuses = [PASS if all(row["passed"] for row in exact_checks) else FAIL]
    parts = [f"exact: largest difference from d = 2 is {max(row['max_abs_diff'] for row in exact_checks):.1e} (threshold 1e-9)."]
    n = config["n_max"]
    monotone = []
    for spacing in config["spacing_grid"]:
        values = [data.value("e3", f"s={spacing:g}|d={d}", "typ_normal", n) for d in dims]
        decreasing = all(right < left for left, right in zip(values, values[1:]))
        monotone.append(decreasing)
        parts.append(f"typ_normal p_true({n}) at s={spacing:g}: " + ", ".join(fmt(v, 4) for v in values) + ".")
    statuses.append(PASS if all(monotone) else FAIL)
    statuses.append(PASS if all(row["passed"] for row in oracle_checks) else FAIL)
    parts.append(
        f"typ_content_oracle against typ_normal at d = 2: largest difference "
        f"{max(row['max_abs_diff'] for row in oracle_checks):.1e} over {len(oracle_checks)} cells."
    )
    return Verdict("P5", claim, _combine(statuses), " ".join(parts))


def _e4_acc(data: RunData, d: int, scorer: str, key: str) -> float | None:
    config = data.preset["e4"]
    return data.value("e4", r_key(d, config["r"]), scorer, config["n_max"], "acc", contamination=key)


def _p6(data: RunData) -> Verdict:
    claim = "Bins bound each observation's evidence, so gross outliers cannot dominate; tempering cannot bound it."
    if not data.ran("e4") or data.preset["e4"]["n_max"] < 40:
        return Verdict("P6", claim, INCONCLUSIVE, "E4 with n = 40 was not run.")
    rows = [(rho, key) for rho, key in _contamination_rows(data, "e4", "C3", 50.0) if rho <= 0.3]
    heavy = [(rho, key) for rho, key in rows if rho >= 0.2]
    if not heavy:
        return Verdict("P6", claim, INCONCLUSIVE, "No C3 run with R = 50 and rho >= 0.2.")
    statuses, parts = [], []
    for d in data.preset["e4"]["d_grid"]:
        bins = [(_e4_acc(data, d, "bins_abs3", key), rho) for rho, key in rows]
        others = [
            (_e4_acc(data, d, scorer, key), scorer, rho)
            for rho, key in heavy
            for scorer in ("exact", "tempered_0.5", "tempered_0.1", "typ_normal")
        ]
        low_bins = min(bins)
        high_other = max(others)
        statuses.append(PASS if low_bins[0] >= 0.9 and high_other[0] < 0.5 else FAIL)
        parts.append(
            f"d={d}: bins_abs3 acc(40) at least {fmt(low_bins[0], 3)} for rho in "
            f"{{{', '.join(f'{rho:g}' for rho, _ in rows)}}}; the highest acc(40) of exact, tempered_*, and "
            f"typ_normal at rho >= 0.2 is {fmt(high_other[0], 3)} ({high_other[1]}, rho={high_other[2]:g})."
        )
    return Verdict("P6", claim, _combine(statuses), " ".join(parts))


def _p6b(data: RunData) -> Verdict:
    claim = "The matched-cap clipped likelihood is as robust as bins on C3."
    if not data.ran("e4") or data.preset["e4"]["n_max"] < 40:
        return Verdict("P6b", claim, INCONCLUSIVE, "E4 with n = 40 was not run.")
    rows = [(rho, key) for rho, key in _contamination_rows(data, "e4", "C3", 50.0) if rho <= 0.3]
    if not rows:
        return Verdict("P6b", claim, INCONCLUSIVE, "No C3 run with R = 50.")
    statuses, parts = [], []
    for d in data.preset["e4"]["d_grid"]:
        low = min((_e4_acc(data, d, "clipped", key), rho) for rho, key in rows)
        statuses.append(PASS if low[0] >= 0.9 else FAIL)
        text = f"d={d}: clipped acc(40) at least {fmt(low[0], 3)} under C3 with R = 50 (lowest at rho={low[1]:g})"
        other = [(rho, key) for rho, key in _contamination_rows(data, "e4", "C3", 20.0) if rho <= 0.3]
        if other:
            low20 = min((_e4_acc(data, d, "clipped", key), rho) for rho, key in other)
            text += f"; with R = 20, at least {fmt(low20[0], 3)}"
        parts.append(text + ".")
    return Verdict("P6b", claim, _combine(statuses), " ".join(parts) + " Evaluated at R = 50, the setting of P6.")


def _p6c(data: RunData) -> Verdict:
    claim = "Under shared misspecification (C1, C2) exact loses nothing; under corruption in every observation (C5) bins fail safe."
    if not data.ran("e4"):
        return Verdict("P6c", claim, INCONCLUSIVE, "E4 was not run.")
    config = data.preset["e4"]
    keys = set(config["contamination_keys"])
    if not {"C1", "C2", "C5"} <= keys or config["n_max"] < 40:
        return Verdict("P6c", claim, INCONCLUSIVE, "E4 did not run C1, C2, and C5 to n = 40.")
    statuses, parts = [], []
    for d in config["d_grid"]:
        shared = []
        for key in ("C1", "C2"):
            exact, bins = _e4_acc(data, d, "exact", key), _e4_acc(data, d, "bins_abs3", key)
            statuses.append(PASS if exact >= bins else FAIL)
            shared.append(f"{key} exact {fmt(exact, 3)} vs bins_abs3 {fmt(bins, 3)}")
        row = data.row("e4", r_key(d, config["r"]), "bins_abs3", 20, contamination="C5")
        p_true = row["p_true"]["mean"]
        wrong = row["max_wrong_mean_posterior"]
        statuses.append(PASS if p_true <= 0.2 and wrong <= 0.5 else FAIL)
        parts.append(
            f"d={d}: acc(40) {'; '.join(shared)}. C5: bins_abs3 mean p_true(20) {fmt(p_true, 3)}, largest mean "
            f"posterior on one wrong hypothesis {fmt(wrong, 3)}."
        )
    return Verdict("P6c", claim, _combine(statuses), " ".join(parts))


def _p7(data: RunData) -> Verdict:
    claim = "Equal-mass bins carry no information; optimized edges help."
    checks = [row for row in data.cells if row["kind"] == "check" and row["name"] == "equal_mass_posterior_is_prior"]
    statuses, parts = [], []
    if checks:
        worst = max(checks, key=lambda row: row["max_abs_diff"])
        statuses.append(PASS if all(row["passed"] for row in checks) else FAIL)
        parts.append(
            f"Equal-mass posterior minus prior: largest {worst['max_abs_diff']:.1e} over {len(checks)} "
            f"(cell, scheme) checks (threshold 1e-12)."
        )
    else:
        statuses.append(INCONCLUSIVE)
        parts.append("No equal-mass scorer was run.")
    if not data.cell_rows("e5", "choice"):
        statuses.append(INCONCLUSIVE)
        parts.append("E5 was not run.")
        return Verdict("P7", claim, _combine(statuses), " ".join(parts))
    config = data.preset["e5"]
    heldout = config["heldout_e1"]
    n = config["objective_n"]
    cells = [(d, r) for d in heldout["d_grid"] for r in heldout["r_grid"]]
    better = []
    for d, r in cells:
        opt3 = data.value("e5_heldout", r_key(d, r), "bins_opt3", n, "bits")
        abs3 = data.value("e5_heldout", r_key(d, r), "bins_abs3", n, "bits")
        better.append((opt3 >= abs3, d, r, opt3, abs3))
    wins = sum(1 for item in better if item[0])
    detail = "; ".join(f"d={d} r={r:g}: {fmt(opt3)} vs {fmt(abs3)}" for _, d, r, opt3, abs3 in better)
    parts.append(f"bins_opt3 bits({n}) >= bins_abs3 in {wins} of {len(cells)} held-out E1 cells ({detail}).")
    if len(cells) != 6:
        statuses.append(INCONCLUSIVE)
        parts.append("The criterion is stated for 6 held-out E1 cells.")
    else:
        statuses.append(PASS if wins >= 4 else FAIL)
    return Verdict("P7", claim, _combine(statuses), " ".join(parts))


def predictions(data: RunData) -> list[Verdict]:
    """The checklist of Section 7, evaluated from the stored rows."""
    return [rule(data) for rule in (_p1, _p2, _p3, _p4, _p5, _p6, _p6b, _p6c, _p7)]


def _checklist_section(verdicts: list[Verdict], data: RunData) -> list[str]:
    lines = ["## Prediction checklist", ""]
    if data.preset["name"] != "full":
        lines += [
            f"Evaluated mechanically on the `{data.preset['name']}` grids. A reduced preset cannot confirm or refute "
            "the predictions.",
            "",
        ]
    lines += table(
        ["ID", "Prediction", "Result", "Numbers"],
        [[v.prediction, v.claim, f"**{v.status}**", v.numbers] for v in verdicts],
    )
    lines.append("")
    failed = [v for v in verdicts if v.status == FAIL]
    if failed:
        lines.append(
            "Failed predictions are results, not bugs. The method, bins, and grids were not changed after seeing them: "
            + ", ".join(v.prediction for v in failed)
            + "."
        )
        lines.append("")
    if any(v.prediction == "P6b" and v.status == PASS for v in verdicts):
        lines.append(
            "In this test the robustness advantage comes from bounded influence, not from typicality specifically. "
            "The clipped likelihood caps each observation's evidence at the bins_abs3 log-mass ratio and stays as "
            "accurate as bins_abs3 under C3."
        )
        lines.append("")
    return lines


def summary_lines(data: RunData, figures: list[Path] | None = None) -> list[str]:
    lines = _header(data)
    lines += _checklist_section(predictions(data), data)
    for section in (_e0_section, _e1_section, _e2_section, _e3_section, _e4_section, _e5_section):
        lines += section(data)
    if figures:
        lines += ["## Figures", ""]
        for path in figures:
            relative = path.relative_to(data.run_dir).as_posix()
            lines += [f"![{path.stem}]({relative})", ""]
    return lines


def write_report(run_dir: Path, *, figures: bool = True) -> Path:
    """Write figures/*.png and summary.md into the run folder from its stored rows; return the summary path."""
    data = RunData.load(Path(run_dir))
    written = write_figures(data) if figures else []
    path = data.run_dir / "summary.md"
    path.write_text("\n".join(summary_lines(data, written)).rstrip() + "\n", encoding="utf-8")
    return path


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
SCORER_PRIORITY = (
    "exact",
    "typ_normal",
    "bins_abs3",
    "typ_density",
    "bins_signed5",
    "clipped",
    "bins_abs2",
    "tempered_0.1",
)
NEUTRAL_SCORERS = {"tempered_0.5": "--", "bins_equalmass3": "-"}
MARKERS = {
    "exact": "o",
    "typ_normal": "s",
    "bins_abs3": "^",
    "typ_density": "D",
    "bins_signed5": "v",
    "clipped": "P",
    "bins_abs2": "X",
    "tempered_0.1": "<",
    "tempered_0.5": ">",
    "bins_equalmass3": None,
}
LINE_WIDTH = 1.4
MARKER_SIZE = 4.5


@dataclass(frozen=True)
class SeriesStyle:
    color: str
    linestyle: str
    marker: str | None
    hollow: bool = False


def series_styles(scorers: Iterable[str]) -> dict[str, SeriesStyle]:
    """Line styles for the scorers of one figure.

    Colored scorers take categorical slots 1, 2, ... in SCORER_PRIORITY order, never
    skipping a slot, so exact, typ_normal, and bins_abs3 keep the same hue in every
    figure. tempered_0.5 and the equal-mass control are gray. A content oracle takes
    its base scorer's hue, dashed, with hollow markers.
    """
    from robust_likelihood.gaussian.scorers import content_oracle_base

    scorers = list(scorers)
    bases: list[str] = []
    for scorer in scorers:
        base = content_oracle_base(scorer) or scorer
        if base not in NEUTRAL_SCORERS and base not in bases:
            bases.append(base)
    ordered = [name for name in SCORER_PRIORITY if name in bases]
    ordered += [name for name in bases if name not in SCORER_PRIORITY]
    if len(ordered) > len(CATEGORICAL):
        raise ValueError("a figure can color at most eight scorers")
    slot = {name: CATEGORICAL[index] for index, name in enumerate(ordered)}
    styles = {}
    for scorer in scorers:
        base = content_oracle_base(scorer)
        if scorer in NEUTRAL_SCORERS:
            styles[scorer] = SeriesStyle(INK_MUTED, NEUTRAL_SCORERS[scorer], MARKERS.get(scorer))
        elif base is not None:
            styles[scorer] = SeriesStyle(slot[base], "--", MARKERS.get(base, "o"), hollow=True)
        else:
            styles[scorer] = SeriesStyle(slot[scorer], "-", MARKERS.get(scorer, "o"))
    return styles


def _rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "text.color": INK,
    }


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRIDLINE, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def _values(values: Iterable[float | None]):
    import numpy as np

    return np.array([np.nan if value is None else value for value in values], dtype=float)


def _line(ax, x, y, style: SeriesStyle, label: str, *, markevery=None, zorder: float = 2.0) -> None:
    ax.plot(
        x,
        _values(y),
        color=style.color,
        linestyle=style.linestyle,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        dash_capstyle="round",
        marker=style.marker,
        markersize=MARKER_SIZE,
        markevery=markevery,
        markerfacecolor=SURFACE if style.hollow else style.color,
        markeredgecolor=style.color if style.hollow else SURFACE,
        markeredgewidth=1.0 if style.hollow else 0.9,
        label=label,
        zorder=zorder,
    )


def _reference(ax, y: float, label: str, *, side: str = "right") -> None:
    """A recessive horizontal hairline with a small direct label."""
    ax.axhline(y, color=BASELINE, linewidth=0.8, zorder=1)
    x = 0.99 if side == "right" else 0.01
    ax.text(x, y, label, transform=ax.get_yaxis_transform(), ha=side, va="bottom", color=INK_MUTED, fontsize=7.0)


def _markevery(count: int, index: int) -> tuple[int, int] | None:
    if count <= 12:
        return None
    step = max(2, count // 6)
    return (index % step, step)


def _finish(fig, title: str, handles: list | None = None, labels: list[str] | None = None) -> None:
    """Wrap the title to the figure width and put the legend, if any, below the panels in as many columns as fit."""
    import textwrap

    width = fig.get_size_inches()[0]
    fig.suptitle(textwrap.fill(title, width=max(40, int(width * 12.5))), color=INK, fontsize=10)
    if not handles:
        return
    em = 8.0 / 72.0
    entry = max((5.0 + 0.52 * len(label)) * em for label in labels)
    ncol = max(1, min(len(labels), int(0.95 * width // entry)))
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, handlelength=2.6, columnspacing=1.6)


def _plain_log_ticks(axis, ticks: Iterable[float]) -> None:
    ticks = list(ticks)
    axis.set_ticks(ticks)
    axis.set_ticklabels([f"{tick:g}" for tick in ticks])


def _save(fig, path: Path) -> Path:
    import time

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.25
    for attempt in range(12):
        try:
            fig.savefig(path, dpi=150)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    plt.close(fig)
    return path


def _ordinal_colors(count: int) -> list[str]:
    """count steps of the blue ramp between steps 250 and 650, light to dark."""
    if count == 1:
        return [SEQUENTIAL[7]]
    return [SEQUENTIAL[round(3 + 8 * k / (count - 1))] for k in range(count)]


def _figure_e1_convergence(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not data.ran("e1"):
        return None
    config = data.preset["e1"]
    dims, rs, n_max, scorers = config["d_grid"], config["r_grid"], config["n_max"], config["scorers"]
    styles = series_styles(scorers)
    n = np.arange(1, n_max + 1)
    fig, axes = plt.subplots(
        len(dims),
        len(rs),
        figsize=(1.9 * len(rs) + 1.0, 1.7 * len(dims) + 1.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for i, d in enumerate(dims):
        for j, r in enumerate(rs):
            ax = axes[i][j]
            _grid(ax)
            for index, scorer in enumerate(scorers):
                values = [data.value("e1", r_key(d, r), scorer, int(k)) for k in n]
                _line(ax, n, values, styles[scorer], scorer, markevery=_markevery(n_max, index))
            ax.set_xlim(1, n_max)
            ax.set_ylim(-0.03, 1.03)
            ax.text(
                0.97,
                0.14,
                f"δ = {r * (2.0 * d) ** 0.25:.2f}",
                transform=ax.transAxes,
                ha="right",
                color=INK_MUTED,
                fontsize=7.0,
            )
            if i == 0:
                ax.set_title(f"r = {r:g}")
            if j == 0:
                ax.set_ylabel(f"d = {d}\np_true(n)")
            if i == len(dims) - 1:
                ax.set_xlabel("observations n")
    _finish(
        fig,
        "E1. Posterior probability of the true hypothesis after n observations, mean over trials "
        "(M = 20, δ = r (2d)^1/4).",
        *axes[0][0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e1_convergence.png")


def _figure_e1_collapse(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not data.ran("e1"):
        return None
    config = data.preset["e1"]
    dims, rs, n_max = config["d_grid"], config["r_grid"], config["n_max"]
    colors = _ordinal_colors(len(dims))
    shapes = ("o", "s", "^", "D", "v")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), sharey=True, layout="constrained")
    for ax, scorer in zip(axes, ("typ_normal", "bins_abs3")):
        _grid(ax, "both")
        for k, d in enumerate(dims):
            values = [data.n90("e1", r_key(d, r), scorer)[1] for r in rs]
            ax.plot(
                rs,
                _values(values),
                color=colors[k],
                linewidth=LINE_WIDTH,
                marker=shapes[k % len(shapes)],
                markersize=5.5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.9,
                label=f"d = {d}",
            )
        ax.set_yscale("log", base=2)
        ticks = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= n_max]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0.8, n_max * 1.15)
        ax.set_xticks(rs)
        ax.set_xticklabels([f"{r:g}" for r in rs])
        ax.set_xlabel("r = δ / (2d)^1/4")
        ax.set_title(scorer)
    axes[0].set_ylabel("n90, log scale")
    _finish(
        fig,
        f"E1. Observations needed for mean p_true to reach 0.9. Lines that overlap collapse across d (P2); "
        f"a missing point was not reached by n = {n_max}.",
        *axes[0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e1_collapse.png")


def _figure_e1_bits(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not data.ran("e1"):
        return None
    config = data.preset["e1"]
    dims, rs, scorers = config["d_grid"], config["r_grid"], config["scorers"]
    styles = series_styles(scorers)
    fig, axes = plt.subplots(
        1, len(dims), figsize=(3.4 * len(dims) + 0.6, 3.9), sharey=True, squeeze=False, layout="constrained"
    )
    ceiling = math.log2(20.0)
    for ax, d in zip(axes[0], dims):
        _grid(ax)
        for scorer in scorers:
            values = [data.value("e1", r_key(d, r), scorer, 1, "bits") for r in rs]
            _line(ax, rs, values, styles[scorer], scorer)
        _reference(ax, ceiling, "log2 20: certainty", side="left")
        ax.set_xticks(rs)
        ax.set_xticklabels([f"{r:g}" for r in rs])
        ax.set_xlabel("r = δ / (2d)^1/4")
        ax.set_title(f"d = {d}")
        ax.set_ylim(-0.15, ceiling + 0.45)
    axes[0][0].set_ylabel("bits(1), information from one observation")
    _finish(
        fig,
        "E1. Information about the truth from a single observation, log2 π1(z*) − log2 π0(z*), mean over trials.",
        *axes[0][0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e1_single_observation_bits.png")


def _figure_e2(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    limits = data.cell_rows("e2", "limit")
    if not limits:
        return None
    config = data.preset["e2"]
    dims, n_grid, scorers = config["d_grid"], config["n_grid"], config["scorers"]
    styles = series_styles(scorers)
    sigma2 = sorted({row["truth_sigma2"] for row in limits})
    fig, axes = plt.subplots(2, len(dims), figsize=(2.45 * len(dims) + 1.0, 6.4), squeeze=False, layout="constrained")
    for j, d in enumerate(dims):
        ax = axes[0][j]
        _grid(ax)
        for scorer in scorers:
            _line(ax, n_grid, [data.value("e2", f"d={d}", scorer, n) for n in n_grid], styles[scorer], scorer)
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis, [t for t in (1, 10, 100, 1000) if t <= n_grid[-1]])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"d = {d}")
        ax.set_xlabel("observations n, log scale")
        if j == 0:
            ax.set_ylabel("p_true(n), all truths")
        ax = axes[1][j]
        _grid(ax, "both")
        style = styles["typ_normal"]
        low, high = sigma2[0] / 1.6, sigma2[-1] * 1.6
        ax.plot([low, high], [low, high], color=BASELINE, linewidth=0.9, zorder=1)
        factor = 1.0 + 2.0 / d
        ax.plot(
            [low, high],
            [low * factor, high * factor],
            color=style.color,
            linewidth=LINE_WIDTH,
            zorder=2,
            label="typ_normal limit (1 + 2/d) σ*²",
        )
        for row in limits:
            if row["scorer"] != "typ_normal" or row["cell"]["d"] != d:
                continue
            for mode, count in row["mode_counts"].items():
                if mode == "tie" or count == 0:
                    continue
                share = count / row["trials"]
                ax.scatter(
                    [row["truth_sigma2"]],
                    [float(mode)],
                    s=12 + 110 * share,
                    color=style.color,
                    edgecolors=SURFACE,
                    linewidths=0.9,
                    zorder=3,
                )
            predicted = row["predicted_sigma2"]
            if predicted is not None:
                ax.annotate(
                    f"{100 * row['mode_match']:.0f}%",
                    (row["truth_sigma2"], predicted),
                    xytext=(5, -9),
                    textcoords="offset points",
                    color=INK_SECONDARY,
                    fontsize=7.0,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(sigma2)
        ax.set_xticklabels([f"{s:g}" for s in sigma2])
        ax.set_yticks(sigma2)
        ax.set_yticklabels([f"{s:g}" for s in sigma2])
        ax.minorticks_off()
        ax.set_xlim(low, high)
        ax.set_ylim(low, high * 1.4)
        ax.set_xlabel("true σ²")
        if j == 0:
            ax.set_ylabel(f"typ_normal posterior mode at n = {config['n_max']}")
            ax.text(0.03, 0.97, "gray: mode = truth", transform=ax.transAxes, va="top", color=INK_MUTED, fontsize=7.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1][0].get_legend_handles_labels()
    _finish(
        fig,
        "E2. Scale family. Top: convergence of each scorer. Bottom: typ_normal's posterior mode against the truth; "
        "dot area is the share of trials, the label is the share at the predicted limit.",
        handles + extra_handles,
        labels + extra_labels,
    )
    return _save(fig, folder / "e2_consistency.png")


def _figures_e3_heatmaps(data: RunData, folder: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    if not data.heatmaps:
        return []
    config = data.preset["e3"]
    cmap = LinearSegmentedColormap.from_list("posterior", [SURFACE, *SEQUENTIAL])
    norm = LogNorm(vmin=1e-4, vmax=1.0)
    paths = []
    for d in sorted({row["d"] for row in data.heatmaps}):
        chosen = [row for row in data.heatmaps if row["d"] == d]
        scorers = [s for s in config["heatmap_scorers"] if any(row["scorer"] == s for row in chosen)]
        ns = sorted({row["n"] for row in chosen})
        lookup = {(row["scorer"], row["n"]): row for row in chosen}
        fig, axes = plt.subplots(
            len(scorers),
            len(ns),
            figsize=(1.7 * len(ns) + 1.6, 1.75 * len(scorers) + 1.0),
            squeeze=False,
            layout="constrained",
        )
        image = None
        for i, scorer in enumerate(scorers):
            for j, n in enumerate(ns):
                ax = axes[i][j]
                row = lookup[(scorer, n)]
                posterior = np.clip(np.array(row["posterior"]), 1e-4, 1.0)
                image = ax.imshow(posterior, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
                true_row, true_column = row["true_cell"]
                ax.scatter([true_column], [true_row], s=70, facecolors="none", edgecolors=SURFACE, linewidths=2.6)
                ax.scatter([true_column], [true_row], s=70, facecolors="none", edgecolors=INK, linewidths=1.1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(GRIDLINE)
                if i == 0:
                    ax.set_title(f"n = {n}")
                if j == 0:
                    ax.set_ylabel(scorer)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
        colorbar.set_label("posterior probability, log scale", color=INK_SECONDARY)
        colorbar.outline.set_visible(False)
        first = chosen[0]
        _finish(
            fig,
            f"E3. Posterior over the 21 × 21 grid at d = {d}, spacing {first['spacing']:g}, trial {first['trial']}. "
            "The ring marks the true cell.",
        )
        paths.append(_save(fig, folder / f"e3_heatmap_d{d}.png"))
    return paths


def _figure_e3_nuisance(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not data.ran("e3"):
        return None
    config = data.preset["e3"]
    spacings, dims, scorers, n = config["spacing_grid"], sorted(config["d_grid"]), config["scorers"], config["n_max"]
    styles = series_styles(scorers)
    fig, axes = plt.subplots(
        1, len(spacings), figsize=(3.5 * len(spacings) + 0.6, 3.9), sharey=True, squeeze=False, layout="constrained"
    )
    for ax, spacing in zip(axes[0], spacings):
        _grid(ax, "both")
        for scorer in scorers:
            values = [data.value("e3", f"s={spacing:g}|d={d}", scorer, n) for d in dims]
            _line(ax, dims, values, styles[scorer], scorer)
        _reference(ax, 1.0 / 441.0, "prior 1/441")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log_ticks(ax.xaxis, dims)
        _plain_log_ticks(ax.yaxis, (0.003, 0.01, 0.03, 0.1, 0.3, 1.0))
        ax.minorticks_off()
        ax.set_ylim(1.0 / 441.0 / 1.6, 1.25)
        ax.set_xlabel("dimension d, log scale")
        ax.set_title(f"spacing {spacing:g}")
    axes[0][0].set_ylabel(f"p_true({n}), log scale")
    _finish(
        fig,
        "E3. Localization with nuisance dimensions. Dashed lines are content oracles, which see only the two "
        "in-plane coordinates.",
        *axes[0][0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e3_nuisance_dimensions.png")


def _figure_e4(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not data.ran("e4"):
        return None
    config = data.preset["e4"]
    dims, scorers, n_max = config["d_grid"], list(config["scorers"]), config["n_max"]
    panels = []
    for code in ("C3", "C4"):
        for radius in sorted({c["radius"] for c in config["contaminations"] if c["code"] == code}):
            rows = _contamination_rows(data, "e4", code, radius)
            if rows:
                panels.append((code, radius, rows))
    if not panels:
        return None
    merged = []
    for tempered in ("tempered_0.5", "tempered_0.1"):
        if (
            tempered in scorers
            and "exact" in scorers
            and all(
                _e4_acc(data, d, tempered, key) == _e4_acc(data, d, "exact", key)
                for d in dims
                for _, _, rows in panels
                for _, key in rows
            )
        ):
            merged.append(tempered)
    shown = [scorer for scorer in scorers if scorer not in merged]
    styles = series_styles(shown)
    labels = {scorer: scorer for scorer in shown}
    if merged:
        labels["exact"] = f"exact (= {', '.join(merged)}: same argmax)"
    fig, axes = plt.subplots(
        len(panels),
        len(dims),
        figsize=(3.6 * len(dims) + 0.8, 2.25 * len(panels) + 1.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for i, (code, radius, rows) in enumerate(panels):
        rhos = [rho for rho, _ in rows]
        for j, d in enumerate(dims):
            ax = axes[i][j]
            _grid(ax)
            for scorer in shown:
                values = [_e4_acc(data, d, scorer, key) for _, key in rows]
                _line(ax, rhos, values, styles[scorer], labels[scorer])
            _reference(ax, 0.9, "0.9")
            _reference(ax, 0.5, "0.5")
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(rhos)
            ax.set_xticklabels([f"{rho:g}" for rho in rhos])
            if i == 0:
                ax.set_title(f"d = {d}")
            if j == 0:
                ax.set_ylabel(f"{code}, R = {radius:g}\nacc({n_max})")
            if i == len(panels) - 1:
                ax.set_xlabel("outlier probability ρ")
    _finish(
        fig,
        f"E4. Accuracy after {n_max} observations under gross outliers of radius R. C3 pushes toward hypothesis 1, "
        "C4 in random directions; a tie counts as incorrect.",
        *axes[0][0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e4_robustness.png")


def _figure_e5(data: RunData, folder: Path) -> Path | None:
    import matplotlib.pyplot as plt

    candidates = data.cell_rows("e5", "candidate")
    if not candidates:
        return None
    import matplotlib.patches as patches

    config = data.preset["e5"]
    named = data.cell_rows("e5", "named_scheme")
    choices = data.cell_rows("e5", "choice")

    def point(row: dict) -> tuple[float, float]:
        return row["clean_bits"]["mean"], row["contaminated_acc"]["mean"]

    groups: dict[tuple[float, float], list[dict]] = {}
    for row in choices + named:
        x, y = point(row)
        groups.setdefault((round(x, 4), round(y, 4)), []).append(row)
    marked = [(point(rows[0]), "/".join(row["scheme"] for row in rows)) for rows in groups.values()]
    feasible = [xy for xy, _ in marked if xy[1] >= config["min_accuracy"]]
    window = None
    if feasible:
        xs = [x for x, _ in feasible]
        pad = 0.25 * (max(xs) - min(xs)) + 0.03
        window = (min(xs) - pad, max(xs) + pad, min(config["min_accuracy"], min(y for _, y in feasible)) - 0.03, 1.07)
    fig, axes = plt.subplots(
        1, 2 if window else 1, figsize=(11.0 if window else 7.2, 4.9), squeeze=False, layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0]} if window else None,
    )
    ks = sorted({row["K"] for row in candidates})
    color_of = {k: CATEGORICAL[index] for index, k in enumerate(ks)}
    for panel, ax in enumerate(axes[0]):
        _grid(ax, "both")
        for k in ks:
            chosen = [row for row in candidates if row["K"] == k]
            ax.scatter(
                [point(row)[0] for row in chosen],
                [point(row)[1] for row in chosen],
                s=18,
                color=color_of[k],
                edgecolors=SURFACE,
                linewidths=0.6,
                zorder=2,
                label=f"K = {k} candidates ({len(chosen)})",
            )
        ax.scatter(
            [point(row)[0] for row in named],
            [point(row)[1] for row in named],
            s=46,
            marker="D",
            facecolors=SURFACE,
            edgecolors=INK,
            linewidths=1.1,
            zorder=3,
            label="named schemes",
        )
        ax.scatter(
            [point(row)[0] for row in choices],
            [point(row)[1] for row in choices],
            s=150,
            marker="*",
            color=[color_of.get(row["K"], INK) for row in choices],
            edgecolors=INK,
            linewidths=0.8,
            zorder=4,
            label="chosen opt<K>",
        )
        _reference(ax, config["min_accuracy"], f"constraint {config['min_accuracy']:g}", side="left")
        inside = panel == 1
        shown = [
            (xy, text)
            for xy, text in marked
            if window is None or inside == (window[0] <= xy[0] <= window[1] and window[2] <= xy[1] <= window[3])
        ]
        offsets = [(0, 16), (0, -20), (0, 30), (0, -34), (0, 44), (0, -48)]
        for index, ((x, y), text) in enumerate(sorted(shown)):
            dx, dy = offsets[index % len(offsets)] if inside else (-6, 7)
            ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center" if inside else "right",
                va="center",
                color=INK_SECONDARY,
                fontsize=7.5,
                zorder=5,
                arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.6) if inside else None,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9),
            )
        ax.set_xlabel(f"clean bits({config['objective_n']}): information about the truth")
        if inside:
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])
            ax.set_title("Zoom on the schemes that meet the constraint", loc="left")
        else:
            ax.set_ylim(-0.05, 1.08)
            ax.set_ylabel(f"acc({config['constraint_n']}) under {choices[0]['contamination']}")
            ax.set_title(f"All {len(candidates)} candidate absolute schemes", loc="left")
            if window:
                ax.add_patch(
                    patches.Rectangle(
                        (window[0], window[2]), window[1] - window[0], window[3] - window[2],
                        fill=False, edgecolor=INK_MUTED, linewidth=0.8, zorder=1,
                    )
                )
    _finish(
        fig,
        f"E5. Bin design on the calibration setting d = {config['calibration_d']}, r = {config['calibration_r']:g}: "
        f"clean information against contaminated accuracy. Stars mark the chosen edges for each K.",
        *axes[0][0].get_legend_handles_labels(),
    )
    return _save(fig, folder / "e5_tradeoff.png")


def write_figures(data: RunData) -> list[Path]:
    """Render every figure the run's rows support into <run>/figures/ and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = data.run_dir / "figures"
    paths: list[Path] = []
    with plt.rc_context(_rc()):
        for builder in (_figure_e1_convergence, _figure_e1_collapse, _figure_e1_bits, _figure_e2):
            path = builder(data, folder)
            if path is not None:
                paths.append(path)
        paths.extend(_figures_e3_heatmaps(data, folder))
        for builder in (_figure_e3_nuisance, _figure_e4, _figure_e5):
            path = builder(data, folder)
            if path is not None:
                paths.append(path)
    return paths
