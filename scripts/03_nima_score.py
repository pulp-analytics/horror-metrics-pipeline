#!/usr/bin/env python3
"""NIMA (Neural Image Assessment, InceptionV2 backbone, trained on the AVA
aesthetic-ratings dataset) per poster, via the pyiqa toolbox.

Second, independently-trained aesthetic model -- cross-checks against
04_laion_aesthetic_score.py's CLIP-ViT-L14-based model to see whether
"which posters look good" trends agree across two models trained on
different data with different architectures, rather than trusting a
single aesthetic scorer's biases.

  python3 03_nima_score.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
THREADS (default 8) caps OMP/MKL/torch thread count -- unlike
02_iqa_multi_score.py, this one benefits from capping when several
scoring processes share one machine (matches the real project's
deployment). Set THREADS to your instance's physical core count.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)

import pyiqa
import requests
import torch
from PIL import Image

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", _THREADS)))

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("nima_score")

FIELDS = ["id", "title", "year", "nima_score", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/nima_score.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    metric = pyiqa.create_metric("nima", device=device)
    log.info("NIMA model loaded")

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
                    img = Image.open(poster_file).convert("RGB")
                    with torch.inference_mode():
                        out["nima_score"] = round(float(metric(img).item()), 4)
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
