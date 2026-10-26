from pathlib import Path

import pytest

from robust_likelihood.multiwoz import load_config
from robust_likelihood.multiwoz.__main__ import prepare
from robust_likelihood.multiwoz.metrics import analyze
from robust_likelihood.multiwoz.report import (
    bin_label,
    headline_rows,
    metrics_markdown,
    summary_markdown,
    write_figures,
)
from robust_likelihood.storage import read_json, read_jsonl, repo_root, write_json, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "multiwoz_mini"
SECTIONS = (
    "## 1. Setup and counts",
    "## 2. Reference table",
    "## 3. Headline: Occam gap",
    "## 4. Wording gap",
    "## 5. Monotonicity",
    "## 6. Controls and tie offset",
    "## 7. Reverse direction",
    "## 8. Typicality and the language model",
    "## 9. Prediction scorecard",
    "## 10. Notes for Xuan",
)


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("report")
    derived = folder / "derived"
    prepare(repo_root(), load_config(repo_root()), raw=FIXTURE, derived=derived)
    rows = []
    for version, lift in (("multiwoz_likely", 0.0), ("multiwoz_typical", 0.04)):
        for row in read_jsonl(derived / "probe.jsonl"):
            breadth = 0.05 * (3 - len(row["constrained"]))
            noul = min(0.95, 0.55 + lift - breadth + (0.1 if row["regime"] == "impl" else 0.0))
            rows.append({"item_id": row["item_id"], "hyp_key": row["hyp_key"], "prompt_version": version,
                         "repeat_id": None, "noul": noul})
    run_dir = folder / "run"
    write_jsonl(run_dir / "scores.jsonl", rows)
    manifest = {
        "selection": {"item_ids": [item["item_id"] for item in read_jsonl(derived / "items.jsonl")]},
        "requested_model_id": "jev-1.13.0",
        "returned_model_ids": {"jev-1.13.0": len(rows)},
        "http_requests": len(rows),
        "failures": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    config = load_config(repo_root())
    config["bootstrap"] = {"replicates": 40, "seed": 5}
    metrics = analyze(run_dir, derived, config)
    return run_dir, derived, metrics, manifest


def test_bin_labels_read_as_words():
    assert bin_label("C2k0-1") == "2 slots, 0–1 named"
    assert bin_label("C3k2") == "3 slots, 2 named"


def test_headline_rows_cover_single_slots_and_bins(analyzed):
    _, _, metrics, _ = analyzed
    rows = headline_rows(metrics["sets"]["clean"]["mappings"]["logit"]["multiwoz_likely"], "multiwoz_likely")
    kinds = {(row["kind"], row["regime"]) for row in rows}
    assert ("single_named", "explicit") in kinds and ("full", "implicit") in kinds
    assert all(row["OG"] == pytest.approx(row["mean_L_J"] - row["L_E"]) for row in rows)
    assert {row["group"] for row in rows if row["kind"] == "full"} <= {"2 slots, 0–2 named", "2 slots, 2 named",
                                                                      "3 slots, 0–1 named", "3 slots, 1 named",
                                                                      "3 slots, 0–2 named"}


def test_summary_has_every_section_and_marks_the_language_model_pending(analyzed):
    run_dir, derived, metrics, manifest = analyzed
    text = summary_markdown(metrics, read_json(derived / "reference_table.json"), manifest, "f" * 40)
    positions = [text.index(section) for section in SECTIONS]
    assert positions == sorted(positions)
    assert "has not run yet" in text
    assert "fig_occam_both.png" in text
    tables = metrics_markdown(metrics)
    assert "## Prediction scorecard" in tables and "| P6 |" in tables


def test_figures_are_written_with_their_data(analyzed):
    pytest.importorskip("matplotlib")
    run_dir, derived, metrics, _ = analyzed
    written = {path.name for path in write_figures(run_dir, derived, metrics)}
    for stem in ("fig_occam_likely", "fig_occam_typical", "fig_occam_both"):
        assert {f"{stem}.png", f"{stem}.pdf", f"{stem}.csv"} <= written
    assert {"fig_items_hexbin_likely.png", "fig_scores_by_type_typical.png"} <= written
    header = (run_dir / "fig_occam_likely.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("scorer,regime,kind,group,n,L_E")
