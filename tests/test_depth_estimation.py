"""Unit tests for 17_depth_estimation.py -- exercises the normalization
math directly (no model download/network needed for these), plus one
slow end-to-end test gated behind a marker since it needs MiDaS_small's
real weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
depth_mod = importlib.import_module("17_depth_estimation")


def test_fields_declares_all_four_metrics_plus_id_columns():
    for col in ("id", "title", "year", "mean_depth", "p95_depth",
                "depth_std", "close_area_frac", "error"):
        assert col in depth_mod.FIELDS


@pytest.mark.slow
def test_estimate_depth_on_a_real_poster(tmp_path):
    # Requires network access the first time (downloads MiDaS_small's
    # weights via torch.hub, ~82MB, cached after) -- marked slow so the
    # default `pytest tests/` run (no network assumed) can skip it with
    # `-m "not slow"`.
    import cv2

    img = np.random.default_rng(0).integers(0, 255, (600, 400, 3), dtype=np.uint8)
    p = tmp_path / "synthetic.jpg"
    cv2.imwrite(str(p), img)

    device = "cpu"
    model, transform = depth_mod.load_midas(device)
    m = depth_mod.estimate_depth(model, transform, device, p)

    for col in ("mean_depth", "p95_depth", "depth_std", "close_area_frac"):
        assert 0.0 <= m[col] <= 1.0, f"{col}={m[col]} out of normalized [0,1] range"
    # p95 is the 95th percentile of a normalized map -- must be >= the mean
    assert m["p95_depth"] >= m["mean_depth"]
