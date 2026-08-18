"""Unit tests for 20_creature_weapon_owlv2.py's pure-math filter_boxes()/
_summarize() -- no network calls, no model download needed for these."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
owlv2 = importlib.import_module("20_creature_weapon_owlv2")


def test_filter_boxes_drops_scores_below_min_score():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    scores = [0.5, 0.1]
    labels = ["zombie", "zombie"]
    rows = owlv2.filter_boxes(boxes, scores, labels, w=100, h=100, min_score=0.2)
    assert len(rows) == 1
    assert rows[0]["score"] == 0.5


def test_filter_boxes_drops_tiny_and_huge_areas():
    # tiny box (area << 0.002 of 100x100) and a near-full-image box (area > 0.95)
    boxes = [[0, 0, 1, 1], [0, 0, 99, 99]]
    scores = [0.9, 0.9]
    labels = ["knife", "knife"]
    rows = owlv2.filter_boxes(boxes, scores, labels, w=100, h=100)
    assert rows == []


def test_filter_boxes_sorts_by_score_descending_and_caps_at_max_boxes():
    boxes = [[0, 0, 20, 20]] * 5
    scores = [0.3, 0.9, 0.5, 0.7, 0.4]
    labels = ["ghost"] * 5
    rows = owlv2.filter_boxes(boxes, scores, labels, w=100, h=100, max_boxes=3)
    assert len(rows) == 3
    assert [r["score"] for r in rows] == [0.9, 0.7, 0.5]


def test_filter_boxes_normalizes_and_sorts_reversed_coordinates():
    # x0 > x1 and y0 > y1 -- real OWLv2 output shouldn't do this, but the
    # sort-then-subtract logic must not silently produce a negative xywh
    boxes = [[30, 30, 10, 10]]
    scores = [0.9]
    labels = ["axe"]
    rows = owlv2.filter_boxes(boxes, scores, labels, w=100, h=100)
    assert len(rows) == 1
    x, y, bw, bh = rows[0]["box"]
    assert x >= 0 and y >= 0 and bw > 0 and bh > 0


def test_summarize_empty_rows():
    s = owlv2._summarize([])
    assert s == {"n": 0, "top_label": "", "top_score": "", "boxes": "[]"}


def test_summarize_picks_highest_score_as_top():
    rows = [
        {"label": "wolf", "score": 0.4, "box": [0, 0, 0.1, 0.1]},
        {"label": "spider", "score": 0.8, "box": [0, 0, 0.1, 0.1]},
    ]
    s = owlv2._summarize(rows)
    assert s["n"] == 2
    assert s["top_label"] == "spider"
    assert s["top_score"] == 0.8
    assert json.loads(s["boxes"]) == rows


def test_fields_declares_expected_columns():
    for col in ("id", "title", "year",
                "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
                "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error"):
        assert col in owlv2.FIELDS


def test_creature_and_weapon_query_maps_dont_overlap():
    assert set(owlv2.CREATURE_QUERIES) & set(owlv2.WEAPON_QUERIES) == set()
