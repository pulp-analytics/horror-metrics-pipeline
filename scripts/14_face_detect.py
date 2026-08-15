#!/usr/bin/env python3
"""Face detection via YuNet (cv2.FaceDetectorYN) -- a compact (230KB) ONNX
face detector, run at a fixed 320px detection width.

Real methodology note: this replaced an earlier Haar cascade approach,
which undercounted badly on stylized poster artwork -- the real project's
own worked example: *Resident Evil: Welcome to Raccoon City* (2021)
scored 0 of 6 faces with Haar. YuNet's own hand-verified validation set
(below) is unchanged from the real script.

Fully local, CPU-only, no AWS involved -- the model is auto-downloaded
once from its real public source (opencv_zoo) if not already present at
--model-path.

  python3 14_face_detect.py --in data/sample_input/sample_100_posters.csv
  python3 14_face_detect.py --validate

Resumable: re-running with the same --out skips ids already processed.

Not something this script does: decade-level face-share aggregation
(the real faces_v2.py's finalize() step) -- same Scope reasoning as
every other script here (see the README's Scope note).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("face_detect")

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
# Verified 2026-08-14 against the file MODEL_URL served at that time:
# shasum -a 256 face_detection_yunet_2023mar.onnx
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEFAULT_MODEL_PATH = "data/models/face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 320
CONF_THRESHOLD = 0.6

FIELDS = ["id", "title", "year", "n_faces", "face_area", "max_conf", "face_boxes", "error"]

# hand-verified by actually counting faces in the artwork -- real
# validation set, unchanged from the real faces_v2.py. (title, year,
# expected face count, tolerance)
VALIDATION = [
    ("Resident Evil: Welcome to Raccoon City", 2021, 4, 2),
    ("Scream", 1996, 6, 1),
    ("Get Out", 2017, 1, 0),
    ("Psycho", 1960, 3, 1),
    ("The Exorcist", 1973, 1, 1),
    ("Us", 2019, 1, 1),
    ("Halloween", 1978, 0, 0),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: Path) -> None:
    if path.exists():
        actual = _sha256(path)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                f"{path} exists but its sha256 ({actual}) doesn't match the pinned "
                f"MODEL_SHA256 ({MODEL_SHA256}) -- delete it and re-run to re-download "
                f"a verified copy, don't just ignore this")
        return
    log.info(f"downloading YuNet model to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    actual = _sha256(path)
    if actual != MODEL_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded YuNet model's sha256 ({actual}) doesn't match the pinned "
            f"MODEL_SHA256 ({MODEL_SHA256}) -- refusing to use it. Either the upstream "
            f"file at MODEL_URL changed (update MODEL_SHA256 after checking why) or the "
            f"download was corrupted/tampered with")
    log.info("model downloaded and verified (sha256 matches)")


def make_detector(model_path: Path):
    return cv2.FaceDetectorYN.create(str(model_path), "", (DETECT_WIDTH, DETECT_WIDTH), CONF_THRESHOLD, 0.3, 5000)


def detect(det, path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"n_faces": 0, "face_area": 0.0, "max_conf": 0.0, "face_boxes": ""}
    h, w = img.shape[:2]
    s = DETECT_WIDTH / w
    img = cv2.resize(img, (DETECT_WIDTH, int(h * s)))
    ih, iw = img.shape[:2]
    det.setInputSize((iw, ih))
    _, faces = det.detect(img)
    if faces is None:
        return {"n_faces": 0, "face_area": 0.0, "max_conf": 0.0, "face_boxes": ""}
    area = float(sum(f[2] * f[3] for f in faces) / (ih * iw))
    boxes = []
    for f in faces:
        boxes.append([
            round(float(f[0]) / iw, 4), round(float(f[1]) / ih, 4),
            round(float(f[2]) / iw, 4), round(float(f[3]) / ih, 4),
        ])
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes_s = "|".join(f"{x},{y},{bw},{bh}" for x, y, bw, bh in boxes)
    return {
        "n_faces": len(faces),
        "face_area": round(area, 4),
        "max_conf": round(float(faces[:, -1].max()), 3),
        "face_boxes": boxes_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/face_detect.csv")
    add_poster_source_args(ap)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    ensure_model(model_path)
    det = make_detector(model_path)
    log.info("YuNet loaded")

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.validate:
        posters_dir = Path(args.posters_dir)
        session = __import__("requests").Session()
        print(f'{"title":45}{"expected":>10}{"detected":>10}   boxes')
        ok = found = 0
        for title, year, expected, tol in VALIDATION:
            match = [r for r in rows if r["title"] == title and int(float(r["year"])) == year]
            if not match:
                print(f"{title:45} NOT FOUND")
                continue
            row = match[0]
            poster_file = posters_dir / f"{row['id']}.jpg"
            if not fetch_poster_file(session, row["poster_path"], poster_file,
                                      args.posters_s3_bucket, args.posters_s3_prefix):
                print(f"{title:45} POSTER FETCH FAILED")
                continue
            found += 1
            r = detect(det, poster_file)
            flag = "OK" if abs(r["n_faces"] - expected) <= tol else "FAIL"
            ok += flag == "OK"
            print(f'{title:45}{expected:10}{r["n_faces"]:10} {flag}  {r["face_boxes"][:40]}')
        print(f"\nVALIDATION: {ok}/{found} within tolerance (of {len(VALIDATION)} total in the validation set)")
        return

    rows = shard_rows(rows, args.shard_index, args.shard_count)
    out_path = Path(args.out)
    done = load_done_ids(out_path)
    todo = [row for row in rows if row["id"] not in done and row.get("poster_path")]
    if done:
        log.info(f"resuming: {len(done)} already done, {len(todo)} remaining")

    posters_dir = Path(args.posters_dir)
    import requests
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
                    out.update(detect(det, poster_file))
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
