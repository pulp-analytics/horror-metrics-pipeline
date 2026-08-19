"""Guards that every Hub/GitHub/file load this repo owns is pinned -- the
same check MODELS.md exists for. Importing the numbered scripts pulls in
torch at module level for 17/19/20/21; that's fine for the fast suite
(no weights are downloaded). Does not import 04 (open_clip) or call
load_* so CI can skip tensorflow/pyiqa/ultralytics/open_clip."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_siglip_revision_is_a_full_commit():
    siglip = importlib.import_module("utils.siglip_backbone")
    assert _SHA1.match(siglip.MODEL_REVISION)


def test_owlv2_revision_is_a_full_commit():
    owlv2 = importlib.import_module("20_creature_weapon_owlv2")
    assert _SHA1.match(owlv2.MODEL_REVISION)


def test_dino_revision_is_a_full_commit():
    dino = importlib.import_module("21_creature_weapon_dino")
    assert _SHA1.match(dino.MODEL_REVISION)


def test_msinet_revision_is_a_full_commit():
    sal = importlib.import_module("18_saliency_prediction")
    assert _SHA1.match(sal.MODEL_REVISION)


def test_vitpose_revision_is_a_full_commit():
    pose = importlib.import_module("19_pose_dynamism")
    assert _SHA1.match(pose.VITPOSE_REVISION)


def test_midas_revision_is_a_full_commit():
    depth = importlib.import_module("17_depth_estimation")
    assert _SHA1.match(depth.MIDAS_REVISION)
    assert depth.MIDAS_GITHUB == "intel-isl/MiDaS"


def test_yolo_and_yunet_are_pinned_by_sha256():
    pose = importlib.import_module("19_pose_dynamism")
    yunet = importlib.import_module("14_face_detect")
    assert _SHA256.match(pose.YOLO_SHA256)
    assert _SHA256.match(yunet.MODEL_SHA256)
    assert pose.YOLO_URL.endswith("/v8.3.0/yolov8n.pt")


def test_ensure_yolo_rejects_a_wrong_hash(tmp_path):
    pose = importlib.import_module("19_pose_dynamism")
    bogus = tmp_path / "yolov8n.pt"
    bogus.write_bytes(b"not the real weights")
    with pytest.raises(RuntimeError, match="doesn't match the pinned"):
        pose.ensure_yolo(bogus)
