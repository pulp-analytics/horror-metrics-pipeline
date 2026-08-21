#!/usr/bin/env python3
"""AWS Rekognition per-poster enrichment: general scene/object labels,
image-quality properties (brightness/contrast/sharpness/dominant colors),
and face demographics (age range/gender/dominant emotion of the largest
detected face).

Ported from the real project's rekognition_enrich.py, which called
DetectLabels + DetectModerationLabels + DetectFaces together (~3 API
calls/poster). This script deliberately drops the moderation call --
gore/violence/nudity flagging is a corpus-*validation* decision, already
faithfully ported as poster-corpus-validation's gate 15
(15_content_moderation.py, Nova + Rekognition DetectModerationLabels
cross-check) -- duplicating it here would just be a second, unlinked copy
of the same signal in the wrong repo. What's left (labels, image
properties, faces) are genuinely per-poster *metrics*, the same category
as this repo's other CLIP/SigLIP/composition scripts, just from a
different vendor model -- hence living here, not in the validation repo.

Complements, doesn't replace, this repo's own local signals: `rek_labels`/
`rek_top` is a second, independently-trained object/scene read next to
06_clip_census.py's zero-shot CLIP taxonomy; `rek_bright`/`rek_contrast`/
`rek_sharp`/`rek_colors` is a second color/quality read next to
01_color_metrics.py's own CIELAB analysis; `rek_age_lo`/`rek_age_hi`/
`rek_gender`/`rek_emotion` has no local equivalent at all -- 14_face_detect.py
only counts/locates faces, it doesn't attempt demographics or expression
(15_face_expression.py's CLIP zero-shot read is a different kind of
signal, not age/gender).

Unlike scripts 01-21, this ONE needs real AWS access (Rekognition, via the
standard boto3 credential chain -- an AWS_PROFILE env var or otherwise
configured credentials), same exception already carved out for the Nova
QA tools (22-24). NOT YET LIVE-VERIFIED on this repo's own corpus and NOT
YET wired into statemachine/compute_metrics.asl.json in
poster-analysis-infrastructure -- both pending real AWS access.

  export AWS_PROFILE=sandbox_bedrock
  python3 26_rekognition_enrich.py --in data/sample_input/sample_100_posters.csv --workers 8

Resumable: re-running with the same --out skips ids already processed.
Shares this repo's poster cache with every other script -- see
utils/posters.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("rekognition_enrich")

REGION = "us-east-1"
MAX_BYTES = 5_000_000

FIELDS = [
    "id", "title", "year",
    "rek_labels", "rek_top", "rek_top_conf",
    "rek_weapon", "rek_animal", "rek_person", "rek_water", "rek_fire", "rek_silhouette",
    "rek_n_boxes", "rek_bright", "rek_sharp", "rek_contrast", "rek_colors",
    "rek_n_faces", "rek_emotion", "rek_gender", "rek_age_lo", "rek_age_hi",
    "error",
]

# Ported as-is from the real rekognition_enrich.py's label-presence vocabularies.
WEAPON = {
    "weapon", "blade", "knife", "dagger", "sword", "gun", "handgun", "rifle",
    "axe", "hatchet", "bow", "arrow", "spear", "mace", "chainsaw",
}
ANIMAL = {
    "animal", "shark", "fish", "sea life", "insect", "bird", "dog", "cat",
    "wolf", "bear", "snake", "spider", "bat", "crow", "raven", "great white shark",
}
PERSON = {"person", "human", "man", "woman", "boy", "girl", "adult", "child", "face", "baby"}
WATER = {"water", "ocean", "sea", "lake", "beach", "wave", "underwater"}
FIRE = {"fire", "flame", "smoke", "explosion"}
SILHOUETTE = {"silhouette"}


def _flag(labels: list[tuple[str, float]], vocab: set[str]) -> float:
    """Ported as-is: highest confidence among labels whose name is in vocab, else 0."""
    best = 0.0
    for name, conf in labels:
        if name.lower() in vocab:
            best = max(best, conf)
    return round(best, 4)


def analyze(client, img_bytes: bytes) -> dict:
    lab = client.detect_labels(
        Image={"Bytes": img_bytes},
        MaxLabels=20,
        MinConfidence=50,
        Features=["GENERAL_LABELS", "IMAGE_PROPERTIES"],
        Settings={"ImageProperties": {"MaxDominantColors": 5}},
    )
    labels = [(l["Name"], float(l["Confidence"]) / 100.0) for l in lab.get("Labels", [])]
    n_boxes = sum(len(l.get("Instances") or []) for l in lab.get("Labels", []))
    ip = lab.get("ImageProperties") or {}
    q = ip.get("Quality") or {}
    colors = ip.get("DominantColors") or []
    color_s = "|".join(f"{c.get('HexCode', '')}:{round(float(c.get('PixelPercent', 0)), 1)}" for c in colors[:5])
    label_s = "|".join(f"{n}:{c:.2f}" for n, c in labels[:10])
    top_n, top_c = (labels[0][0], labels[0][1]) if labels else ("", 0.0)

    faces = client.detect_faces(Image={"Bytes": img_bytes}, Attributes=["ALL"])
    details = faces.get("FaceDetails") or []
    emotion = gender = ""
    age_lo = age_hi = -1
    if details:
        # largest face by bounding box area, same tiebreak as the real script
        details = sorted(details, key=lambda f: f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"], reverse=True)
        f0 = details[0]
        emos = sorted(f0.get("Emotions") or [], key=lambda e: -e["Confidence"])
        if emos:
            emotion = f"{emos[0]['Type']}:{emos[0]['Confidence']/100:.2f}"
        gender = (f0.get("Gender") or {}).get("Value") or ""
        ar = f0.get("AgeRange") or {}
        age_lo = int(ar.get("Low", -1))
        age_hi = int(ar.get("High", -1))

    return {
        "rek_labels": label_s, "rek_top": top_n, "rek_top_conf": round(top_c, 4),
        "rek_weapon": _flag(labels, WEAPON), "rek_animal": _flag(labels, ANIMAL),
        "rek_person": _flag(labels, PERSON), "rek_water": _flag(labels, WATER),
        "rek_fire": _flag(labels, FIRE), "rek_silhouette": _flag(labels, SILHOUETTE),
        "rek_n_boxes": int(n_boxes),
        "rek_bright": round(float(q.get("Brightness") or 0), 2),
        "rek_sharp": round(float(q.get("Sharpness") or 0), 2),
        "rek_contrast": round(float(q.get("Contrast") or 0), 2),
        "rek_colors": color_s,
        "rek_n_faces": len(details), "rek_emotion": emotion, "rek_gender": gender,
        "rek_age_lo": age_lo, "rek_age_hi": age_hi,
    }


def process_one(row: dict, client, session, posters_dir: Path, s3_bucket: str, s3_prefix: str) -> dict:
    out = {"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""), "error": ""}
    poster_file = posters_dir / f"{row['id']}.jpg"
    if not fetch_poster_file(session, row.get("poster_path", ""), poster_file, s3_bucket, s3_prefix):
        out["error"] = "download_failed"
        return out
    try:
        img_bytes = poster_file.read_bytes()
        if len(img_bytes) > MAX_BYTES:
            out["error"] = "image_too_large"
            return out
        out.update(analyze(client, img_bytes))
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/rekognition_enrich.csv")
    add_poster_source_args(ap)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    import boto3
    client = boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).client("rekognition", region_name=args.region)

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = shard_rows(rows, args.shard_index, args.shard_count)

    out_path = Path(args.out)
    done = load_done_ids(out_path)
    todo = [row for row in rows if row["id"] not in done and row.get("poster_path")]
    if done:
        log.info(f"resuming: {len(done)} already done, {len(todo)} remaining")

    posters_dir = Path(args.posters_dir)
    import requests
    t0 = time.time()
    n_ok = n_err = 0

    f, w = open_for_append(out_path, FIELDS)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_one, row, client, requests.Session(), posters_dir,
                          args.posters_s3_bucket, args.posters_s3_prefix): row
                for row in todo
            }
            n_done = 0
            for fut in as_completed(futs):
                out = fut.result()
                w.writerow(out)
                n_done += 1
                if out.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    rate = n_done / max(time.time() - t0, 1e-9)
                    log.info(f"{n_done}/{len(todo)} rate={rate:.2f}/s ok={n_ok} err={n_err}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} scored, {n_err} failed (this run)")


if __name__ == "__main__":
    main()
