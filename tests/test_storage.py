from pathlib import Path

from robust_likelihood.storage import read_jsonl, write_jsonl


def test_jsonl_round_trips_text_with_unicode_line_separators(tmp_path: Path):
    rows = [{"reply": "one two"}, {"reply": "three four\x85five"}, {"reply": "plain"}]
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_jsonl_reads_crlf_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a":1}\r\n{"a":2}\r\n')
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
