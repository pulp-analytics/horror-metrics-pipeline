"""Unit tests for 19_pose_dynamism.py's pure-math compute_metrics() --
no network calls, no model download needed for these."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
pose = importlib.import_module("19_pose_dynamism")


def _standing_keypoints():
    """A roughly symmetric standing pose in a 100x200 box: shoulders level,
    hips level, wrists at the sides -- low limb_asymmetry expected."""
    kpts = np.zeros((17, 2))
    scores = np.zeros(17)
    box = [0.0, 0.0, 100.0, 200.0]

    kpts[pose.L_SHOULDER] = [30, 60]; kpts[pose.R_SHOULDER] = [70, 60]
    kpts[pose.L_HIP] = [35, 120]; kpts[pose.R_HIP] = [65, 120]
    kpts[pose.L_WRIST] = [20, 110]; kpts[pose.R_WRIST] = [80, 110]
    kpts[pose.L_ELBOW] = [25, 90]; kpts[pose.R_ELBOW] = [75, 90]
    kpts[pose.L_KNEE] = [35, 160]; kpts[pose.R_KNEE] = [65, 160]
    kpts[pose.L_ANKLE] = [35, 195]; kpts[pose.R_ANKLE] = [65, 195]
    for i in (pose.L_SHOULDER, pose.R_SHOULDER, pose.L_HIP, pose.R_HIP,
              pose.L_WRIST, pose.R_WRIST, pose.L_ELBOW, pose.R_ELBOW,
              pose.L_KNEE, pose.R_KNEE, pose.L_ANKLE, pose.R_ANKLE):
        scores[i] = 0.9
    return kpts, scores, box


def test_symmetric_standing_pose_scores_low_asymmetry():
    kpts, scores, box = _standing_keypoints()
    m = pose.compute_metrics(kpts, scores, box)
    assert m["limb_asymmetry"] < 0.05


def test_one_arm_raised_increases_asymmetry():
    kpts, scores, box = _standing_keypoints()
    kpts[pose.R_WRIST] = [70, 20]  # raise the right wrist far above the shoulder
    m_before = pose.compute_metrics(*_standing_keypoints())
    m_after = pose.compute_metrics(kpts, scores, box)
    assert m_after["limb_asymmetry"] > m_before["limb_asymmetry"]


def test_too_few_confident_keypoints_returns_none_metrics():
    kpts = np.zeros((17, 2))
    scores = np.full(17, 0.1)  # all below KPT_CONF_THRESHOLD
    m = pose.compute_metrics(kpts, scores, [0.0, 0.0, 100.0, 200.0])
    assert m["kpt_bbox_area_frac"] is None
    assert m["limb_asymmetry"] is None
    assert m["mean_kpt_confidence"] == pytest.approx(0.1, abs=1e-6)


def test_kpt_bbox_area_frac_is_between_zero_and_one_for_compact_pose():
    kpts, scores, box = _standing_keypoints()
    m = pose.compute_metrics(kpts, scores, box)
    assert 0.0 < m["kpt_bbox_area_frac"] < 1.0


def test_fields_declares_expected_columns():
    for col in ("id", "title", "year", "n_persons", "kpt_bbox_area_frac",
                "limb_asymmetry", "mean_kpt_confidence", "box", "keypoints", "error"):
        assert col in pose.FIELDS
