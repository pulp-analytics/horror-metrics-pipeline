"""Unit tests for 01_color_metrics.py's pure color-math functions -- no
network calls, no AWS."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
color_metrics = importlib.import_module("01_color_metrics")


def test_srgb_to_lab_black_is_zero_lightness():
    lab = color_metrics.srgb_to_lab(np.array([[0.0, 0.0, 0.0]]))
    assert lab[0, 0] == pytest.approx(0.0, abs=0.5)


def test_srgb_to_lab_white_is_full_lightness():
    lab = color_metrics.srgb_to_lab(np.array([[1.0, 1.0, 1.0]]))
    assert lab[0, 0] == pytest.approx(100.0, abs=0.5)


def test_lab_and_srgb_roundtrip_hex():
    # pure red should round-trip through CIELAB back to (close to) #ff0000
    lab = color_metrics.srgb_to_lab(np.array([1.0, 0.0, 0.0]))
    hexcode = color_metrics.lab_to_hex(lab)
    r, g, b = int(hexcode[1:3], 16), int(hexcode[3:5], 16), int(hexcode[5:7], 16)
    assert r > 200 and g < 40 and b < 40


def test_rgb_to_hsv_pure_colors():
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    h, s, v = color_metrics.rgb_to_hsv(rgb)
    assert h[0] == pytest.approx(0.0, abs=1.0)      # red
    assert h[1] == pytest.approx(120.0, abs=1.0)    # green
    assert h[2] == pytest.approx(240.0, abs=1.0)    # blue
    np.testing.assert_allclose(s, 1.0)
    np.testing.assert_allclose(v, 1.0)


def test_rgb_to_hsv_grey_has_zero_saturation():
    h, s, v = color_metrics.rgb_to_hsv(np.array([[0.5, 0.5, 0.5]]))
    assert s[0] == pytest.approx(0.0)


def test_analyze_poster_on_synthetic_solid_red_image():
    # a solid near-pure-red image should score high on red_share and low
    # on dark_share/saturation-of-neutrals -- exercises the full
    # analyze_poster() pipeline (resize, CIELAB, HSV, banding, k-means
    # palette) without needing a real downloaded poster
    import io
    from PIL import Image

    img = Image.new("RGB", (200, 300), color=(200, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    rng = np.random.default_rng(0)

    result = color_metrics.analyze_poster(buf.getvalue(), rng)

    assert result["red_share"] > 0.8
    assert result["dark_share"] < 0.1
    assert len(result["palette"]) == color_metrics.K
    assert len(result["palette_share"]) == color_metrics.K
    assert sum(result["palette_share"]) == pytest.approx(1.0, abs=0.01)
    assert all(p.startswith("#") and len(p) == 7 for p in result["palette"])
