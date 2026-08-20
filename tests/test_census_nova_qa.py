"""Unit tests for 23_census_nova_qa.py's pure logic (load_rows/pick_sample)
-- no network calls, no Bedrock access needed for these."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
qa = importlib.import_module("23_census_nova_qa")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_rows_joins_on_id_and_drops_ids_missing_poster_path(tmp_path):
    in_path = tmp_path / "in.csv"
    census_path = tmp_path / "census.csv"
    _write_csv(in_path, [
        {"id": "1", "title": "A", "year": "2000", "poster_path": "/a.jpg"},
        {"id": "2", "title": "B", "year": "2001", "poster_path": ""},
    ], ["id", "title", "year", "poster_path"])
    _write_csv(census_path, [
        {"id": "1", "label": "vampire", "score": "0.8"},
        {"id": "2", "label": "zombie", "score": "0.5"},
        {"id": "3", "label": "ghost", "score": "0.6"},
    ], ["id", "label", "score"])

    rows = qa.load_rows(in_path, census_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["clip_label"] == "vampire"
    assert rows[0]["poster_path"] == "/a.jpg"


def test_pick_sample_returns_at_most_n_rows():
    rows = [{"id": str(i), "clip_label": "vampire" if i % 2 else "uncertain",
             "clip_score": str(0.1 * i)} for i in range(20)]
    sample = qa.pick_sample(rows, n=5, seed=0)
    assert len(sample) == 5


def test_pick_sample_deterministic_for_same_seed():
    rows = [{"id": str(i), "clip_label": "vampire" if i % 2 else "uncertain",
             "clip_score": str(0.1 * i)} for i in range(20)]
    a = qa.pick_sample(rows, n=8, seed=7)
    b = qa.pick_sample(rows, n=8, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_fields_declares_expected_columns():
    for col in ("id", "clip_label", "clip_score", "model", "status",
                "nova_label", "agree", "reason", "latency_s", "error"):
        assert col in qa.FIELDS


def test_categories_match_06_clip_census_taxonomy():
    census = importlib.import_module("06_clip_census")
    assert set(qa.CATEGORIES) == set(census.TAXONOMY.keys())


def test_clip_label_for_agree_maps_uncertain_to_none():
    assert qa.clip_label_for_agree("uncertain") == "none"
    assert qa.clip_label_for_agree("  uncertain  ") == "none"
    assert qa.clip_label_for_agree("vampire") == "vampire"
    assert qa.clip_label_for_agree("none") == "none"
    assert qa.clip_label_for_agree("") == ""


def test_labels_agree_treats_clip_uncertain_as_nova_none():
    assert qa.labels_agree("uncertain", "none")
    assert not qa.labels_agree("uncertain", "vampire")
    assert qa.labels_agree("vampire", "vampire")
    assert qa.labels_agree("none", "none")
    assert not qa.labels_agree("vampire", "none")
    assert not qa.labels_agree("none", "vampire")
