"""Unit tests for 22_creature_weapon_nova_qa.py's pure logic (load_detections/
pick_sample) -- no network calls, no Bedrock access needed for these."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
qa = importlib.import_module("22_creature_weapon_nova_qa")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_load_detections_flattens_creature_and_weapon_boxes(tmp_path):
    in_path = tmp_path / "in.csv"
    boxes_path = tmp_path / "boxes.csv"
    _write_csv(in_path, [
        {"id": "1", "title": "A", "year": "2000", "poster_path": "/a.jpg"},
    ], ["id", "title", "year", "poster_path"])
    creature_boxes = json.dumps([{"label": "vampire", "score": 0.4, "box": [0, 0, 0.1, 0.1]}])
    weapon_boxes = json.dumps([{"label": "knife", "score": 0.6, "box": [0.2, 0.2, 0.1, 0.1]}])
    _write_csv(boxes_path, [
        {"id": "1", "title": "A", "year": "2000",
         "creature_n": "1", "creature_top_label": "vampire", "creature_top_score": "0.4", "creature_boxes": creature_boxes,
         "weapon_n": "1", "weapon_top_label": "knife", "weapon_top_score": "0.6", "weapon_boxes": weapon_boxes,
         "error": ""},
    ], ["id", "title", "year", "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
        "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error"])

    rows = qa.load_detections(in_path, boxes_path, "owlv2")
    assert len(rows) == 2
    kinds = {r["kind"] for r in rows}
    assert kinds == {"creature", "weapon"}
    assert all(r["poster_path"] == "/a.jpg" for r in rows)


def test_load_detections_skips_ids_missing_from_metrics_input(tmp_path):
    in_path = tmp_path / "in.csv"
    boxes_path = tmp_path / "boxes.csv"
    _write_csv(in_path, [], ["id", "title", "year", "poster_path"])
    _write_csv(boxes_path, [
        {"id": "1", "title": "A", "year": "2000",
         "creature_n": "0", "creature_top_label": "", "creature_top_score": "", "creature_boxes": "[]",
         "weapon_n": "0", "weapon_top_label": "", "weapon_top_score": "", "weapon_boxes": "[]", "error": ""},
    ], ["id", "title", "year", "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
        "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error"])

    rows = qa.load_detections(in_path, boxes_path, "owlv2")
    assert rows == []


def test_pick_sample_skews_toward_low_confidence():
    rows = (
        [{"id": str(i), "kind": "creature", "label": "vampire", "score": 0.1, "box": [0, 0, 0.1, 0.1]} for i in range(50)]
        + [{"id": str(i), "kind": "creature", "label": "vampire", "score": 0.9, "box": [0, 0, 0.1, 0.1]} for i in range(50, 60)]
    )
    sample = qa.pick_sample(rows, n=20, seed=0)
    assert len(sample) == 20
    low_conf = sum(1 for r in sample if r["score"] < 0.3)
    assert low_conf >= 10  # 60% of 20 = 12, allow some slack from bucket rounding


def test_fields_declares_expected_columns():
    for col in ("id", "source", "kind", "label", "score", "box",
                "model", "status", "verdict", "actual", "reason", "latency_s", "error"):
        assert col in qa.FIELDS


def test_source_names_cover_both_detectors():
    assert set(qa.SOURCE_NAMES) == {"owlv2", "dino"}
