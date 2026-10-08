import pytest

from robust_likelihood.constants import SCHEMA_TRANSFORMATIONS
from robust_likelihood.drafts import AUTHORED_IDS, DRAFTS
from robust_likelihood.pilot import load_train_rows, prepare, rows_for_intent, select_originals
from robust_likelihood.storage import read_json, read_jsonl, repo_root


def test_sampler_uses_numeric_train_ids_only():
    rows = [
        {"id": "10", "intent": "alarm_set", "partition": "train", "utt": "a"},
        {"id": "2", "intent": "alarm_set", "partition": "train", "utt": "b"},
        {"id": "1", "intent": "alarm_set", "partition": "test", "utt": "held out"},
    ]
    assert [row["id"] for row in rows_for_intent(rows, "alarm_set")] == ["2", "10"]


def test_seeded_sample_matches_authored_drafts():
    train = repo_root() / "data" / "massive" / "en-US" / "train.jsonl"
    if not train.exists() or train.stat().st_size == 0:
        pytest.skip("MASSIVE train split is not on disk")
    selected = select_originals(load_train_rows(train), ("alarm_set", "alarm_query", "alarm_remove"), 5, 20260927)
    assert [row["id"] for row in selected] == AUTHORED_IDS


def test_prepare_writes_sixty_drafts_and_an_empty_review(tmp_path):
    root = repo_root()
    train = root / "data" / "massive" / "en-US" / "train.jsonl"
    if not train.exists() or train.stat().st_size == 0:
        pytest.skip("MASSIVE train split is not on disk")
    summary = prepare(root, output_dir=tmp_path)
    assert summary["draft_inputs"] == 60
    assert summary["meaning_change_controls"] == 15
    assert summary["review_items"] == 75
    drafts = read_jsonl(tmp_path / "draft_inputs.jsonl")
    controls = read_jsonl(tmp_path / "meaning_change_controls.jsonl")
    review = read_json(tmp_path / "review_manifest.json")
    assert "polite_restatement" in SCHEMA_TRANSFORMATIONS
    assert all(row["transformation"] != "polite_restatement" for row in drafts)
    assert {row["transformation"] for row in drafts} == {"original", "paraphrase", "minor_typo", "repetition"}
    assert all(row["approved"] is False and row["reviewer"] is None for row in review["items"])
    for row in drafts:
        draft = DRAFTS[row["original_example_id"]]
        if row["transformation"] == "repetition":
            assert row["utt"] == f"{row['source_utt']} {row['source_utt']}"
        elif row["transformation"] == "original":
            assert row["utt"] == draft["utt"]
        elif row["transformation"] == "paraphrase":
            assert row["utt"] == draft["paraphrase"]
        elif row["transformation"] == "minor_typo":
            assert row["utt"] == draft["minor_typo"]
            assert row["utt"] != draft["utt"]
        if row["transformation"] in {"original", "paraphrase", "minor_typo"}:
            for token in draft["preserve"]:
                assert token in row["utt"]
        if " not " not in row["source_utt"] and "n't" not in row["source_utt"]:
            assert " not " not in row["utt"]
            assert "n't" not in row["utt"]
        assert row["label_changed"] is False
        assert row["intent"] == row["original_intent"]
    for row in controls:
        assert row["label_changed"] is True
        assert row["intent"] != row["original_intent"]
        assert row["transformation"] == "meaning_change"
    sampling = read_json(tmp_path / "sampling_manifest.json")
    assert sampling["seed"] == 20260927
    assert sampling["split"] == "train"
    assert sampling["official_test_split"] == "untouched"
    assert sampling["example_ids"] == AUTHORED_IDS
