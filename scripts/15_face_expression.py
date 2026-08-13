#!/usr/bin/env python3
"""Zero-shot CLIP expression classifier over the face crops 14_face_detect.py
found. For every detected face box, crops the region (with 25% padding),
encodes it fresh with CLIP ViT-B/32 (face crops were never part of
05_clip_embed.py's whole-poster cache, so there's nothing to reuse here),
and classifies against 8 fear-oriented expression prototypes -- same
prompt-ensemble + cosine-softmax method as 06_clip_census.py, applied to
face crops instead of whole posters.

  python3 15_face_expression.py --faces data/sample_output/face_detect.csv --in data/sample_input/sample_100_posters.csv

One output row per detected face, not per poster (a poster with 3 faces
gets 3 rows). Resumable at the poster level: re-running with the same
--out skips ids that already have every one of their faces scored.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.clip_backbone import get_tokenizer, load_clip, text_prototype
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file

log = get_logger("face_expression")

FIELDS = ["id", "face_idx", "box", "label", "score"]

# fear-oriented, not a general emotion taxonomy -- this project cares
# whether a poster's faces read as dread/threat, not neutral affect
# science. Real prompts, unchanged from the private pipeline.
PROTOTYPES = {
    "terrified": ["a close-up of a terrified human face, eyes wide with fear",
                  "a person's face frozen in fear and dread"],
    "screaming": ["a person screaming in horror, mouth wide open",
                  "a human face mid-scream, terror"],
    "shocked": ["a shocked, startled human face, wide eyes, surprise",
                "a face reacting in sudden shock"],
    "menacing": ["a menacing, evil human face, threatening stare",
                 "a villain's face with a sinister, predatory expression"],
    "angry": ["an angry, enraged human face, gritted teeth"],
    "sad": ["a sad, sorrowful, or crying human face"],
    "in_pain": ["a human face in physical pain or agony"],
    "calm": ["a calm, neutral human face, no strong emotion",
              "an ordinary relaxed human face, plain expression"],
}


def parse_boxes(face_boxes: str) -> list[tuple[float, float, float, float]]:
    boxes = []
    for chunk in (face_boxes or "").split("|"):
        parts = chunk.split(",")
        if len(parts) != 4:
            continue
        boxes.append(tuple(float(p) for p in parts))
    return boxes


def crop_face(img: Image.Image, box: tuple[float, float, float, float], pad: float = 0.25) -> Image.Image:
    W, H = img.size
    x, y, w, h = box
    px, py = w * pad, h * pad
    x0 = max(0.0, x - px) * W
    y0 = max(0.0, y - py) * H
    x1 = min(1.0, x + w + px) * W
    y1 = min(1.0, y + h + py) * H
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--faces", default="data/sample_output/face_detect.csv",
                     help="output of 14_face_detect.py -- id,...,face_boxes")
    ap.add_argument("--out", default="data/sample_output/face_expression.csv")
    add_poster_source_args(ap)
    ap.add_argument("--min-score", type=float, default=0.35,
                     help="below this, label becomes 'uncertain' instead of a low-confidence guess")
    ap.add_argument("--temp", type=float, default=100.0, help="softmax temperature")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    model, preprocess = load_clip(device)
    tok = get_tokenizer()
    log.info("computing expression prototypes...")
    protos = {label: text_prototype(model, tok, prompts, device) for label, prompts in PROTOTYPES.items()}
    labels = list(protos)
    P = np.stack([protos[l] for l in labels])

    faces_df = pd.read_csv(args.faces, usecols=["id", "n_faces", "face_boxes"])
    faces: dict[int, list[tuple[float, float, float, float]]] = {}
    for _, r in faces_df.iterrows():
        if int(r["n_faces"]) <= 0 or not isinstance(r["face_boxes"], str):
            continue
        boxes = parse_boxes(r["face_boxes"])
        if boxes:
            faces[int(r["id"])] = boxes
    log.info(f"posters with faces: {len(faces):,}")

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = {int(row["id"]): row for row in csv.DictReader(f)}
    want_ids = [i for i in faces if i in rows]
    log.info(f"todo: {len(want_ids):,} posters")

    out_path = Path(args.out)
    done_ids: set[int] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        done_ids = set(pd.read_csv(out_path)["id"])
        want_ids = [i for i in want_ids if i not in done_ids]
        log.info(f"resuming: {len(done_ids):,} posters already done, {len(want_ids):,} remaining")

    posters_dir = Path(args.posters_dir)
    import requests
    session = requests.Session()

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0 = time.time()
    n_done_posters = n_faces_done = n_err = 0
    try:
        for pid in want_ids:
            row = rows[pid]
            poster_file = posters_dir / f"{pid}.jpg"
            if not fetch_poster_file(session, row["poster_path"], poster_file,
                                      args.posters_s3_bucket, args.posters_s3_prefix):
                n_err += 1
                continue
            try:
                img = Image.open(poster_file).convert("RGB")
                boxes = faces[pid]
                crops = [preprocess(crop_face(img, b)) for b in boxes]
                batch = torch.stack(crops).to(device)
                with torch.no_grad():
                    emb = model.encode_image(batch)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = emb.cpu().numpy() @ P.T
                probs = torch.softmax(torch.tensor(sims * args.temp), dim=1).numpy()
                top = probs.argmax(1)
                for face_idx, (b, ti, sc) in enumerate(zip(boxes, top, probs.max(1))):
                    label = labels[ti] if sc >= args.min_score else "uncertain"
                    writer.writerow({
                        "id": pid, "face_idx": face_idx,
                        "box": ",".join(f"{v:.4f}" for v in b),
                        "label": label, "score": round(float(sc), 3),
                    })
                    n_faces_done += 1
            except Exception as e:
                n_err += 1
                log.warning(f"FAIL {pid}: {e}")
                continue
            n_done_posters += 1
            if n_done_posters % 25 == 0 or n_done_posters == len(want_ids):
                f_out.flush()
                rate = n_done_posters / max(time.time() - t0, 1e-9)
                log.info(f"{n_done_posters}/{len(want_ids)} posters ({n_faces_done} faces) rate={rate:.2f}/s err={n_err}")
    finally:
        f_out.close()

    log.info(f"wrote {out_path}: posters={len(done_ids) + n_done_posters:,} faces={n_faces_done:,} err={n_err}")


if __name__ == "__main__":
    main()
