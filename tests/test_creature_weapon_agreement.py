"""Unit tests for 25_creature_weapon_agreement.py -- IoU math, greedy
same-label matching, and per-poster summarization. No model download."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
agree = importlib.import_module("25_creature_weapon_agreement")


def test_iou_identical_boxes_is_one():
    box = [0.1, 0.2, 0.3, 0.4]
    assert agree.iou_xywh(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert agree.iou_xywh([0, 0, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]) == 0.0


def test_iou_partial_overlap_is_between_zero_and_one():
    iou = agree.iou_xywh([0, 0, 1, 1], [0.5, 0.5, 1, 1])
    assert 0.1 < iou < 0.5


def test_match_requires_same_label_even_if_boxes_overlap():
    owl = [{"label": "vampire", "score": 0.9, "box": [0, 0, 0.5, 0.5]}]
    dino = [{"label": "zombie", "score": 0.9, "box": [0, 0, 0.5, 0.5]}]
    assert agree.match_boxes(owl, dino, min_iou=0.3) == []


def test_match_pairs_same_label_when_iou_clears_threshold():
    owl = [{"label": "knife", "score": 0.8, "box": [0.1, 0.1, 0.2, 0.2]}]
    dino = [{"label": "knife", "score": 0.6, "box": [0.12, 0.12, 0.2, 0.2]}]
    pairs = agree.match_boxes(owl, dino, min_iou=0.3)
    assert len(pairs) == 1
    assert pairs[0]["label"] == "knife"
    assert pairs[0]["owlv2_score"] == 0.8
    assert pairs[0]["dino_score"] == 0.6
    assert pairs[0]["iou"] >= 0.3


def test_match_does_not_reuse_a_dino_box():
    owl = [
        {"label": "vampire", "score": 0.9, "box": [0, 0, 0.5, 0.5]},
        {"label": "vampire", "score": 0.8, "box": [0.05, 0.05, 0.5, 0.5]},
    ]
    dino = [{"label": "vampire", "score": 0.7, "box": [0, 0, 0.5, 0.5]}]
    pairs = agree.match_boxes(owl, dino, min_iou=0.3)
    assert len(pairs) == 1


def test_summarize_pairs_empty():
    assert agree.summarize_pairs([]) == {
        "n": 0, "top_label": "", "top_score": "", "boxes": "[]"}


def test_summarize_pairs_top_score_is_min_of_the_two():
    pairs = [
        {"label": "axe", "iou": 0.5, "owlv2_score": 0.9, "dino_score": 0.4,
         "owlv2_box": [0, 0, 0.1, 0.1], "dino_box": [0, 0, 0.1, 0.1]},
    ]
    s = agree.summarize_pairs(pairs)
    assert s["n"] == 1
    assert s["top_label"] == "axe"
    assert s["top_score"] == 0.4
    assert json.loads(s["boxes"]) == pairs


def test_label_agree_requires_nonempty_equal_strings():
    assert agree.label_agree("vampire", "vampire") == 1
    assert agree.label_agree("vampire", "zombie") == 0
    assert agree.label_agree("", "") == 0
    assert agree.label_agree("vampire", "") == 0


def test_agree_row_on_a_poster_with_one_matching_creature():
    meta = {"id": "1", "title": "X", "year": "1970"}
    owl = {
        "creature_top_label": "vampire",
        "creature_boxes": json.dumps([{"label": "vampire", "score": 0.5, "box": [0, 0, 0.4, 0.4]}]),
        "weapon_top_label": "",
        "weapon_boxes": "[]",
    }
    dino = {
        "creature_top_label": "vampire",
        "creature_boxes": json.dumps([{"label": "vampire", "score": 0.4, "box": [0.05, 0.05, 0.4, 0.4]}]),
        "weapon_top_label": "knife",
        "weapon_boxes": json.dumps([{"label": "knife", "score": 0.3, "box": [0.8, 0.8, 0.1, 0.1]}]),
    }
    row = agree.agree_row(meta, owl, dino, min_iou=0.3)
    assert row["creature_n"] == 1
    assert row["creature_top_label"] == "vampire"
    assert row["creature_label_agree"] == 1
    assert row["weapon_n"] == 0
    assert row["weapon_label_agree"] == 0
    assert list(row.keys()) == list(agree.FIELDS)


def test_fields_declares_expected_columns():
    for col in ("id", "title", "year",
                "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
                "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes",
                "creature_label_agree", "weapon_label_agree"):
        assert col in agree.FIELDS
