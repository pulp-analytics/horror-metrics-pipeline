"""Unit tests for 18_saliency_prediction.py -- exercises the normalization
math directly (no model download/network needed for these), plus one
slow end-to-end test gated behind a marker since it needs MSI-Net's
real weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
sal_mod = importlib.import_module("18_saliency_prediction")


def test_fields_declares_all_four_metrics_plus_id_columns():
    for col in ("id", "title", "year", "peak_x", "peak_y",
                "top10pct_mass", "mean_saliency", "error"):
        assert col in sal_mod.FIELDS


def test_summarize_saliency_finds_peak_at_hottest_pixel():
    sal = np.zeros((10, 10))
    sal[3, 7] = 1.0  # row 3 (y), col 7 (x)
    m = sal_mod.summarize_saliency(sal)
    assert m["peak_x"] == pytest.approx(0.7)
    assert m["peak_y"] == pytest.approx(0.3)


def test_summarize_saliency_top10pct_mass_is_higher_for_concentrated_map():
    concentrated = np.zeros((10, 10))
    concentrated[0, 0] = 1.0
    diffuse = np.ones((10, 10))

    m_concentrated = sal_mod.summarize_saliency(concentrated)
    m_diffuse = sal_mod.summarize_saliency(diffuse)
    assert m_concentrated["top10pct_mass"] > m_diffuse["top10pct_mass"]
    # a perfectly uniform map has 10% of pixels holding ~10% of the mass
    assert m_diffuse["top10pct_mass"] == pytest.approx(0.10, abs=0.01)


def test_summarize_saliency_all_zero_map_does_not_divide_by_zero():
    sal = np.zeros((10, 10))
    m = sal_mod.summarize_saliency(sal)
    assert m["top10pct_mass"] == 0.0
    assert m["mean_saliency"] == 0.0


@pytest.mark.slow
def test_predict_saliency_on_a_real_poster(tmp_path):
    # Requires network access the first time (downloads MSI-Net's real
    # weights via huggingface_hub, cached after) -- marked slow so the
    # default `pytest tests/` run (no network assumed) can skip it with
    # `-m "not slow"`.
    import cv2

    img = np.random.default_rng(0).integers(0, 255, (600, 400, 3), dtype=np.uint8)
    p = tmp_path / "synthetic.jpg"
    cv2.imwrite(str(p), img)

    model = sal_mod.load_msinet()
    m = sal_mod.predict_saliency(model, p)

    assert 0.0 <= m["peak_x"] <= 1.0
    assert 0.0 <= m["peak_y"] <= 1.0
    assert 0.0 <= m["top10pct_mass"] <= 1.0
    assert m["mean_saliency"] >= 0.0
