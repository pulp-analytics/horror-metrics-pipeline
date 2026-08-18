"""Unit tests for 16_geometric_composition.py's pure OpenCV metric
functions -- no network calls, no AWS. Synthetic images only, chosen so
each metric's expected direction is unambiguous (a real poster's "correct"
answer isn't obvious by eye the way a solid-color or checkerboard image's
is)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
geo = importlib.import_module("16_geometric_composition")


def _bgr_gray(h=300, w=200, fill=128):
    bgr = np.full((h, w, 3), fill, dtype=np.uint8)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def test_composition_solid_image_is_maximally_symmetric_and_flat():
    bgr, gray = _bgr_gray()
    m = geo.composition(bgr, gray)
    assert m["symmetry"] == pytest.approx(1.0, abs=0.01)
    assert m["neg_space"] == pytest.approx(1.0, abs=0.01)
    assert m["complexity"] == pytest.approx(0.0, abs=0.01)


def test_composition_vertical_split_breaks_symmetry():
    bgr, gray = _bgr_gray()
    gray[:, : gray.shape[1] // 2] = 0  # left half black, right half stays 128
    m = geo.composition(bgr, gray)
    assert m["symmetry"] < 0.9


def test_composition_centered_bright_square_centers_mass():
    bgr, gray = _bgr_gray(fill=0)
    H, W = gray.shape
    gray[H // 2 - 10 : H // 2 + 10, W // 2 - 10 : W // 2 + 10] = 255
    m = geo.composition(bgr, gray)
    assert m["mass_x"] == pytest.approx(0.5, abs=0.1)
    assert m["mass_y"] == pytest.approx(0.5, abs=0.1)


def test_typography_blank_image_has_no_text_regions():
    bgr, gray = _bgr_gray()
    m = geo.typography(bgr, gray)
    assert m["text_regions"] == 0
    assert m["text_area"] == pytest.approx(0.0)
    assert m["text_y"] == -1.0  # sentinel: no text found


def test_grid_alignment_needs_at_least_two_blocks():
    bgr, gray = _bgr_gray()
    m = geo.grid_alignment(bgr, gray)
    # a blank frame has no text blocks and no strong-enough gradient
    # contour to count as a "main visual mass" -- both sentinels apply
    assert m["align_score"] == -1.0
    assert m["thirds_dist"] == -1.0


def test_aesthetic_solid_image_has_no_dominant_hue_scheme():
    bgr, gray = _bgr_gray()
    m = geo.aesthetic(bgr, gray)
    # a single flat color has one hue peak, not >=2 -- harmony sentinel
    assert m["harmony"] == -1.0


def test_diagonal_pyramid_horizontal_lines_score_near_zero():
    bgr, gray = _bgr_gray(fill=255)
    for y in range(20, gray.shape[0], 40):
        cv2.line(gray, (10, y), (gray.shape[1] - 10, y), 0, 2)
    m = geo.diagonal_pyramid(bgr, gray)
    assert m["diagonal_score"] < 0.1


def test_diagonal_pyramid_45deg_lines_score_high():
    bgr, gray = _bgr_gray(fill=255)
    cv2.line(gray, (10, 10), (gray.shape[1] - 10, gray.shape[0] - 10), 0, 2)
    m = geo.diagonal_pyramid(bgr, gray)
    assert m["diagonal_score"] > 0.5


def test_analyze_poster_end_to_end_on_a_real_file(tmp_path):
    # exercises the full read -> resize -> five-metric-group pipeline
    # without needing a downloaded poster
    img = np.random.default_rng(0).integers(0, 255, (600, 400, 3), dtype=np.uint8)
    p = tmp_path / "synthetic.jpg"
    cv2.imwrite(str(p), img)

    m = geo.analyze_poster(p)

    for col in geo.FIELDS:
        if col in ("id", "title", "year"):
            continue
        assert col in m, f"missing column {col}"
