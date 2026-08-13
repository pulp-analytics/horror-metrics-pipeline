#!/usr/bin/env python3
"""One-time CLIP (ViT-B/32) image embedding per poster, cached to a .npz.

Embed once, then every zero-shot semantic question (creature census, fear
axis, typography axis, genre agreement -- 06/07/08/09) reads this cache
and runs in seconds, no re-embedding of images. Only 10_clip_medium.py
doesn't use this cache (see its own docstring for why).

  python3 05_clip_embed.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out only embeds ids not already in
the cache. Output: --out is a .npz with `ids` (int array) and `vecs`
(float16, L2-normalized, 512-d per poster) -- float16 to keep the cache
file small; embeddings are re-normalized to float32 after loading by
every downstream script anyway, so the precision loss doesn't compound.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.clip_backbone import load_clip
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import shard_rows

log = get_logger("clip_embed")
BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/clip_embeddings.npz")
    add_poster_source_args(ap)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    model, preprocess = load_clip(device)
    log.info("CLIP ViT-B/32 loaded")

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = shard_rows(rows, args.shard_index, args.shard_count)

    out_path = Path(args.out)
    ids_done: list[int] = []
    vecs: list[np.ndarray] = []
    if out_path.exists():
        z = np.load(out_path)
        ids_done, vecs = list(z["ids"]), list(z["vecs"])
        log.info(f"resuming: {len(ids_done):,} already embedded")
    done = set(int(x) for x in ids_done)
    todo = [row for row in rows if int(row["id"]) not in done and row.get("poster_path")]
    log.info(f"pending: {len(todo):,}")

    posters_dir = Path(args.posters_dir)
    session = requests.Session()
    t0 = time.time()
    imgs, batch_ids = [], []

    def flush():
        nonlocal imgs, batch_ids
        if not imgs:
            return
        with torch.no_grad():
            f = model.encode_image(torch.stack(imgs).to(device))
            f = f / f.norm(dim=-1, keepdim=True)
        vecs.extend(f.cpu().numpy().astype(np.float16))
        ids_done.extend(batch_ids)
        imgs, batch_ids = [], []

    for i, row in enumerate(todo, 1):
        poster_file = posters_dir / f"{row['id']}.jpg"
        if not fetch_poster_file(session, row["poster_path"], poster_file,
                                  args.posters_s3_bucket, args.posters_s3_prefix):
            continue
        try:
            imgs.append(preprocess(Image.open(poster_file).convert("RGB")))
            batch_ids.append(int(row["id"]))
        except Exception:
            continue
        if len(imgs) >= args.batch:
            flush()
        if i % 25 == 0 or i == len(todo):
            rate = (len(ids_done) - len(done)) / max(time.time() - t0, 1e-9)
            log.info(f"{i}/{len(todo)} rate={rate:.1f}/s")
    flush()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=np.array(ids_done), vecs=np.array(vecs))
    log.info(f"wrote {out_path}: {len(ids_done):,} embeddings total")


if __name__ == "__main__":
    main()
