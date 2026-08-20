"""Tests for utils/resumable.py: 0-byte header recovery, duplicate-id
suppression, and two processes sharing --out. No model download."""
from __future__ import annotations

import csv
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils.resumable import load_done_ids, open_for_append, shard_rows

FIELDS = ["id", "value"]


def _write_rows(path: str, rows: list[dict]) -> None:
    f, w = open_for_append(Path(path), FIELDS)
    try:
        for row in rows:
            w.writerow(row)
    finally:
        f.close()


def test_zero_byte_existing_file_still_gets_a_header(tmp_path):
    path = tmp_path / "out.csv"
    path.write_bytes(b"")
    f, w = open_for_append(path, FIELDS)
    try:
        w.writerow({"id": "1", "value": "a"})
    finally:
        f.close()
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == FIELDS
    assert rows == [{"id": "1", "value": "a"}]


def test_same_id_written_twice_in_one_process_is_kept_once(tmp_path):
    path = tmp_path / "out.csv"
    _write_rows(str(path), [
        {"id": "1", "value": "first"},
        {"id": "1", "value": "second"},
        {"id": "2", "value": "ok"},
    ])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [r["id"] for r in rows] == ["1", "2"]
    assert rows[0]["value"] == "first"


def test_resume_skips_ids_already_on_disk(tmp_path):
    path = tmp_path / "out.csv"
    _write_rows(str(path), [{"id": "1", "value": "a"}])
    _write_rows(str(path), [{"id": "1", "value": "dup"}, {"id": "2", "value": "b"}])
    assert load_done_ids(path) == {"1", "2"}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [r["id"] for r in rows] == ["1", "2"]
    assert rows[0]["value"] == "a"


def test_two_processes_sharing_out_do_not_duplicate_ids(tmp_path):
    path = tmp_path / "out.csv"
    rows = [{"id": str(i), "value": "x"} for i in range(8)]
    ctx = multiprocessing.get_context("spawn")
    a = ctx.Process(target=_write_rows, args=(str(path), rows))
    b = ctx.Process(target=_write_rows, args=(str(path), rows))
    a.start()
    b.start()
    a.join(timeout=15)
    b.join(timeout=15)
    assert a.exitcode == 0 and b.exitcode == 0
    got = [r["id"] for r in csv.DictReader(path.open(newline="", encoding="utf-8"))]
    assert got == [str(i) for i in range(8)]
    assert len(got) == len(set(got))


def test_shard_rows_splits_by_position():
    rows = [{"id": str(i)} for i in range(6)]
    assert [r["id"] for r in shard_rows(rows, 0, 2)] == ["0", "2", "4"]
    assert [r["id"] for r in shard_rows(rows, 1, 2)] == ["1", "3", "5"]
    assert shard_rows(rows, 0, 1) == rows
