"""Unit tests for 21_creature_weapon_dino.py's pure-math build_prompt()/
_summarize() -- no network calls, no model download needed for these."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
dino = importlib.import_module("21_creature_weapon_dino")


def test_build_prompt_joins_phrases_with_period_space():
    caption, lookup = dino.build_prompt({"a": "Alpha", "b": "beta thing"})
    assert caption == "alpha. beta thing."
    assert lookup == {"alpha": "a", "beta thing": "b"}


def test_build_prompt_lowercases_and_strips():
    caption, lookup = dino.build_prompt({"x": "  Giant Monster  "})
    assert caption == "giant monster."
    assert lookup == {"giant monster": "x"}


def test_summarize_empty_rows():
    s = dino._summarize([])
    assert s == {"n": 0, "top_label": "", "top_score": "", "boxes": "[]"}


def test_summarize_picks_highest_score_as_top():
    rows = [
        {"label": "knife", "score": 0.3, "box": [0, 0, 0.1, 0.1]},
        {"label": "axe", "score": 0.6, "box": [0, 0, 0.1, 0.1]},
    ]
    s = dino._summarize(rows)
    assert s["n"] == 2
    assert s["top_label"] == "axe"
    assert s["top_score"] == 0.6
    assert json.loads(s["boxes"]) == rows


def test_fields_declares_expected_columns():
    for col in ("id", "title", "year",
                "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
                "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error"):
        assert col in dino.FIELDS


def test_creature_and_weapon_vocab_matches_owlv2_port():
    """The module docstring promises these stay in sync with
    20_creature_weapon_owlv2.py -- a cross-check is only valid if both
    detectors are actually run on the same vocabulary."""
    owlv2 = importlib.import_module("20_creature_weapon_owlv2")
    assert dino.CREATURE_QUERIES == owlv2.CREATURE_QUERIES
    assert dino.WEAPON_QUERIES == owlv2.WEAPON_QUERIES


def test_creature_and_weapon_query_maps_dont_overlap():
    assert set(dino.CREATURE_QUERIES) & set(dino.WEAPON_QUERIES) == set()
