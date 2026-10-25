"""The probe set: one row per (item, hypothesis), with the item's first user turn."""

from __future__ import annotations

from collections import Counter

from robust_likelihood.multiwoz.hypotheses import item_hypotheses

APPENDIX_A4 = {"rows": 3727, "by_set": {"clean": 2355, "fail_info": 1372}}


def build_probe(items: list[dict], pools: dict[str, list[str]]) -> list[dict]:
    rows = []
    for item in items:
        for hypothesis in item_hypotheses(item, pools):
            rows.append({"item_id": item["item_id"], "set": item["set"], "utterance": item["utterance"], **hypothesis})
    return rows


def is_control(row: dict) -> bool:
    return row["hyp_type"].startswith("ctrl:")


def probe_summary(rows: list[dict]) -> dict:
    by_set = Counter(row["set"] for row in rows)
    return {
        "rows": len(rows),
        "by_set": {name: by_set[name] for name in ("clean", "fail_info")},
        "by_hyp_key": dict(sorted(Counter(row["hyp_key"] for row in rows).items())),
        "distinct_texts": len({row["text"] for row in rows}),
        "distinct_non_control_texts": len({row["text"] for row in rows if not is_control(row)}),
        "distinct_control_texts": len({row["text"] for row in rows if is_control(row)}),
    }
