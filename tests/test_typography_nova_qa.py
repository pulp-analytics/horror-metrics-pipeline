"""Unit tests for 24_typography_nova_qa.py's pure logic (bin_register/
load_rows/pick_sample) -- no network calls, no Bedrock access needed."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
qa = importlib.import_module("24_typography_nova_qa")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_bin_register_highest_axis_is_ornate_lowest_is_minimal():
    axes = [float(i) for i in range(20)]  # 0..19, evenly spread
    regs = qa.bin_register(axes)
    assert regs[-1] == "ornate"  # highest axis value
    assert regs[0] == "minimal"  # lowest axis value


def test_bin_register_returns_one_of_five_registers():
    axes = [0.1, 0.5, -0.3, 0.9, -0.9, 0.0, 0.2, 0.4, -0.1, -0.5]
    regs = qa.bin_register(axes)
    assert len(regs) == len(axes)
    assert set(regs) <= set(qa.REGISTERS)


def test_load_rows_joins_on_id_and_computes_register_from_axis(tmp_path):
    in_path = tmp_path / "in.csv"
    typo_path = tmp_path / "typography.csv"
    _write_csv(in_path, [
        {"id": str(i), "title": f"T{i}", "year": "2000", "poster_path": f"/{i}.jpg"}
        for i in range(10)
    ], ["id", "title", "year", "poster_path"])
    _write_csv(typo_path, [
        {"id": str(i), "axis": str(i - 5)} for i in range(10)
    ], ["id", "axis"])

    rows = qa.load_rows(in_path, typo_path)
    assert len(rows) == 10
    assert all("clip_register" in r and r["clip_register"] in qa.REGISTERS for r in rows)
    # highest axis (id=9, axis=4) should land in the ornate end
    by_id = {r["id"]: r for r in rows}
    assert by_id["9"]["clip_register"] == "ornate"
    assert by_id["0"]["clip_register"] == "minimal"


def test_pick_sample_returns_at_most_n_rows():
    rows = [{"id": str(i), "clip_register": qa.REGISTERS[i % 5], "clip_axis": 0.0} for i in range(20)]
    sample = qa.pick_sample(rows, n=6, seed=1)
    assert len(sample) == 6


def test_fields_declares_expected_columns():
    for col in ("id", "clip_register", "clip_axis", "model", "status",
                "nova_register", "agree", "agree_adjacent", "reason", "latency_s", "error"):
        assert col in qa.FIELDS
