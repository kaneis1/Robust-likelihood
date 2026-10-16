"""summary.md for a run folder of the Gaussian test.

Everything is read back from config.json, manifest.json, metrics.jsonl, cells.jsonl,
and e3_heatmaps.json, so a stored run can be reported again without re-running it.
"""

from __future__ import annotations

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


def summary_lines(data: RunData) -> list[str]:
    lines = _header(data)
    for section in (_e0_section, _e1_section, _e2_section, _e3_section, _e4_section, _e5_section):
        lines += section(data)
    return lines


def write_report(run_dir: Path) -> Path:
    """Write summary.md into the run folder from its stored rows, and return its path."""
    data = RunData.load(Path(run_dir))
    path = data.run_dir / "summary.md"
    path.write_text("\n".join(summary_lines(data)).rstrip() + "\n", encoding="utf-8")
    return path
