#!/usr/bin/env python3
"""Body pose dynamism over movie posters -- is the figure static
(portrait pose, arms at sides) or dynamic (running, falling, reaching,
fighting)? Two-stage pipeline: YOLOv8n detects person bounding boxes,
ViTPose (COCO 17-keypoint) estimates the skeleton within the largest
detected box (the primary/foreground figure).

Per-poster metrics:
  n_persons            how many people YOLOv8n found (0 is a legitimate
                        result, not an error -- plenty of posters have no
                        legible human figure at all)
  kpt_bbox_area_frac    bounding box area of all confident keypoints,
                        normalized by the person's own detection box
                        area. Low = compact/static pose (arms at sides);
                        high = limbs spread out (running, falling,
                        reaching -- classic action pose)
  limb_asymmetry        mean absolute left/right limb-position
                        difference relative to torso center, normalized
                        by torso size. Symmetric standing poses score
                        low; mid-stride/off-balance/struggling poses
                        score high
  mean_kpt_confidence   average keypoint detection confidence (sanity
                        check -- low values mean the pose is unreliable,
                        e.g. a heavily stylized/painted figure the model
                        wasn't trained on)

Also persists the raw detection (for drawing the skeleton later instead
of just the summary numbers):
  box          [x0, y0, x1, y1] of the YOLOv8n person box used, JSON-encoded
  keypoints    the 17 COCO keypoints as [[x, y, score], ...] in a fixed
               order (NOSE..R_ANKLE, see below), JSON-encoded, already in
               the original poster's pixel space

  python3 19_pose_dynamism.py --in data/sample_input/sample_100_posters.csv

Unlike the real project's own version of this script, this one runs
YOLOv8n on every poster in --in rather than pre-filtering to ids with a
detected face (faces_v2*.csv) -- that pre-filter was a full-corpus
compute-cost optimization (skip the ~40% of the corpus with no legible
body, at 145k-poster scale), not a correctness requirement. This repo's
per-script independence convention (see README) means every script
computes from raw poster bytes on its own; a poster with n_persons=0 is
exactly as valid an answer as any other.

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("pose_dynamism")

# COCO 17-keypoint order used by ViTPose
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16
LEFT_RIGHT_PAIRS = [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
                    (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)]

KPT_CONF_THRESHOLD = 0.3
FIELDS = ["id", "title", "year", "n_persons", "kpt_bbox_area_frac", "limb_asymmetry",
          "mean_kpt_confidence", "box", "keypoints", "error"]

VITPOSE_ID = "usyd-community/vitpose-base-simple"
# Pinned to the HF Hub repo's current commit, verified 2026-08-19 via
# curl https://huggingface.co/api/models/usyd-community/vitpose-base-simple
VITPOSE_REVISION = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"

# Ultralytics YOLO("yolov8n.pt") otherwise fetches whatever that filename
# currently is on the assets repo's latest release. Pin the v8.3.0 asset
# by URL + sha256, same pattern as 14_face_detect.py's YuNet.
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
DEFAULT_YOLO_PATH = "data/models/yolov8n.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_yolo(path: Path) -> Path:
    """Download yolov8n.pt if missing and refuse to proceed if the sha256
    doesn't match YOLO_SHA256 -- on first download or on a pre-existing file."""
    if path.exists():
        actual = _sha256(path)
        if actual != YOLO_SHA256:
            raise RuntimeError(
                f"{path} exists but its sha256 ({actual}) doesn't match the pinned "
                f"YOLO_SHA256 ({YOLO_SHA256}) -- delete it and re-run to re-download "
                f"a verified copy, don't just ignore this")
        return path
    log.info(f"downloading YOLOv8n weights to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YOLO_URL, path)
    actual = _sha256(path)
    if actual != YOLO_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded YOLOv8n weights' sha256 ({actual}) doesn't match the pinned "
            f"YOLO_SHA256 ({YOLO_SHA256}) -- refusing to use them. Either the upstream "
            f"file at YOLO_URL changed (update YOLO_SHA256 after checking why) or the "
            f"download was corrupted/tampered with")
    log.info("YOLOv8n downloaded and verified (sha256 matches)")
    return path


def load_models(yolo_path: Path | str = DEFAULT_YOLO_PATH):
    from ultralytics import YOLO
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    yolo = YOLO(str(ensure_yolo(Path(yolo_path))))
    processor = AutoProcessor.from_pretrained(VITPOSE_ID, revision=VITPOSE_REVISION)
    model = VitPoseForPoseEstimation.from_pretrained(
        VITPOSE_ID, revision=VITPOSE_REVISION).eval()
    return yolo, processor, model


def compute_metrics(keypoints: np.ndarray, scores: np.ndarray, box: list[float]) -> dict:
    x0, y0, x1, y1 = box
    box_w, box_h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    box_area = box_w * box_h

    conf_mask = scores >= KPT_CONF_THRESHOLD
    mean_conf = float(scores.mean())
    if conf_mask.sum() < 3:
        return {"kpt_bbox_area_frac": None, "limb_asymmetry": None, "mean_kpt_confidence": round(mean_conf, 3)}

    pts = keypoints[conf_mask]
    kpt_x0, kpt_y0 = pts[:, 0].min(), pts[:, 1].min()
    kpt_x1, kpt_y1 = pts[:, 0].max(), pts[:, 1].max()
    kpt_bbox_area_frac = ((kpt_x1 - kpt_x0) * (kpt_y1 - kpt_y0)) / box_area

    torso_size = max(box_w, box_h)
    if scores[L_SHOULDER] >= KPT_CONF_THRESHOLD and scores[R_SHOULDER] >= KPT_CONF_THRESHOLD and \
       scores[L_HIP] >= KPT_CONF_THRESHOLD and scores[R_HIP] >= KPT_CONF_THRESHOLD:
        center = (keypoints[L_SHOULDER] + keypoints[R_SHOULDER] + keypoints[L_HIP] + keypoints[R_HIP]) / 4
    else:
        center = pts.mean(axis=0)

    diffs = []
    for li, ri in LEFT_RIGHT_PAIRS:
        if scores[li] >= KPT_CONF_THRESHOLD and scores[ri] >= KPT_CONF_THRESHOLD:
            dl = np.linalg.norm(keypoints[li] - center)
            dr = np.linalg.norm(keypoints[ri] - center)
            diffs.append(abs(dl - dr) / torso_size)
    limb_asymmetry = float(np.mean(diffs)) if diffs else None

    return {
        "kpt_bbox_area_frac": round(float(kpt_bbox_area_frac), 4),
        "limb_asymmetry": round(limb_asymmetry, 4) if limb_asymmetry is not None else None,
        "mean_kpt_confidence": round(mean_conf, 3),
    }


def analyze_pose(yolo, processor, model, img_path: Path) -> dict:
    img = Image.open(img_path).convert("RGB")
    yres = yolo(str(img_path), classes=[0], verbose=False)
    boxes = yres[0].boxes.xyxy.cpu().numpy().tolist() if yres[0].boxes is not None else []

    row = {"n_persons": len(boxes), "kpt_bbox_area_frac": "", "limb_asymmetry": "",
           "mean_kpt_confidence": "", "box": "", "keypoints": ""}
    if not boxes:
        return row

    # largest box = primary/foreground figure
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    inputs = processor(img, boxes=[boxes[:1]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_pose_estimation(outputs, boxes=[boxes[:1]])
    person = results[0][0]
    kpts = person["keypoints"].numpy()
    scores = person["scores"].numpy()

    metrics = compute_metrics(kpts, scores, boxes[0])
    row.update({k: (v if v is not None else "") for k, v in metrics.items()})
    row["box"] = json.dumps([round(float(c), 1) for c in boxes[0]])
    row["keypoints"] = json.dumps([[round(float(x), 1), round(float(y), 1), round(float(s), 3)]
                                    for (x, y), s in zip(kpts.tolist(), scores.tolist())])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/pose_dynamism.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    log.info("loading YOLOv8n (person detector)...")
    log.info("loading ViTPose...")
    yolo, processor, model = load_models()
    log.info("models loaded")

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = shard_rows(rows, args.shard_index, args.shard_count)

    out_path = Path(args.out)
    done = load_done_ids(out_path)
    todo = [row for row in rows if row["id"] not in done and row.get("poster_path")]
    if done:
        log.info(f"resuming: {len(done)} already done, {len(todo)} remaining")

    posters_dir = Path(args.posters_dir)
    session = requests.Session()
    t0 = time.time()
    n_ok = n_err = 0

    f, w = open_for_append(out_path, FIELDS)
    try:
        for i, row in enumerate(todo, 1):
            out = {"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""), "error": ""}
            poster_file = posters_dir / f"{row['id']}.jpg"
            if not fetch_poster_file(session, row["poster_path"], poster_file,
                                      args.posters_s3_bucket, args.posters_s3_prefix):
                out["error"] = "download_failed"
                n_err += 1
            else:
                try:
                    out.update(analyze_pose(yolo, processor, model, poster_file))
                    n_ok += 1
                except Exception as e:
                    out["error"] = str(e)[:200]
                    n_err += 1
            w.writerow(out)
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                log.info(f"{i}/{len(todo)} rate={rate:.2f}/s ok={n_ok} err={n_err}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} scored, {n_err} failed (this run)")


if __name__ == "__main__":
    main()
