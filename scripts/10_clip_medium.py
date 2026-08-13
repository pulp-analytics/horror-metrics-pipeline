#!/usr/bin/env python3
"""Painted/illustrated vs. photographic poster classification, via CLIP
zero-shot. Dates the death of the illustrated horror poster.

Unlike 06/07/08/09, this does NOT read 05_clip_embed.py's cache -- it
embeds posters fresh with its own preprocessing pipeline. That's how the
real script was written (not a gap introduced by this port); porting it
faithfully means keeping that choice rather than "fixing" it to match the
other scripts' pattern.

  python3 10_clip_medium.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already scored.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.clip_backbone import get_tokenizer, load_clip
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("clip_medium")

# prompt ensembles -- averaged for robustness
PAINTED_PROMPTS = [
    "a hand-painted illustrated movie poster, drawn artwork",
    "a vintage movie poster with painted illustration art",
    "an illustrated poster, painting, brush strokes, drawn characters",
]
PHOTO_PROMPTS = [
    "a movie poster made from a photograph of real actors",
    "a photographic movie poster, photo of a person or scene",
    "a poster with photographic imagery, camera photograph",
]

# sanity checks from the real run -- should come out painted: Creepshow,
# The Evil Dead; photographic: Scream, Hereditary
SANITY_CHECK_TITLES = ["Creepshow", "The Evil Dead", "Scream", "Hereditary"]

FIELDS = ["id", "title", "year", "p_painted", "painted", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/medium.csv")
    add_poster_source_args(ap)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    model, preprocess = load_clip(device)
    tok = get_tokenizer()

    with torch.no_grad():
        t = tok(PAINTED_PROMPTS + PHOTO_PROMPTS).to(device)
        tf = model.encode_text(t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        t_painted = tf[:len(PAINTED_PROMPTS)].mean(0)
        t_photo = tf[len(PAINTED_PROMPTS):].mean(0)
        t_painted = t_painted / t_painted.norm()
        t_photo = t_photo / t_photo.norm()

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

    f_out, w = open_for_append(out_path, FIELDS)
    try:
        batch_imgs, batch_rows = [], []

        def flush():
            nonlocal batch_imgs, batch_rows, n_ok
            if not batch_imgs:
                return
            with torch.no_grad():
                im = torch.stack(batch_imgs).to(device)
                feat = model.encode_image(im)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logits = torch.stack([feat @ t_painted, feat @ t_photo], dim=1) * 100
                p_painted = logits.softmax(dim=1)[:, 0].cpu().numpy()
            for row, pp in zip(batch_rows, p_painted):
                w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                            "p_painted": round(float(pp), 4), "painted": int(pp > 0.5), "error": ""})
                n_ok += 1
            batch_imgs, batch_rows = [], []

        for i, row in enumerate(todo, 1):
            poster_file = posters_dir / f"{row['id']}.jpg"
            if not fetch_poster_file(session, row["poster_path"], poster_file,
                                      args.posters_s3_bucket, args.posters_s3_prefix):
                w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                            "p_painted": "", "painted": "", "error": "download_failed"})
                n_err += 1
                continue
            try:
                batch_imgs.append(preprocess(Image.open(poster_file).convert("RGB")))
                batch_rows.append(row)
            except Exception as e:
                w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                            "p_painted": "", "painted": "", "error": str(e)[:200]})
                n_err += 1
                continue
            if len(batch_imgs) >= args.batch or i == len(todo):
                flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                log.info(f"{i}/{len(todo)} rate={rate:.2f}/s ok={n_ok} err={n_err}")
    finally:
        f_out.close()

    log.info(f"wrote {out_path}: {n_ok} scored, {n_err} failed (this run)")


if __name__ == "__main__":
    main()
