#!/usr/bin/env python3
"""Three no-reference image-quality metrics per poster, via the pyiqa
toolbox -- each measuring a genuinely different kind of signal, run
together so they can be sanity-checked against each other:

  clipiqa  CLIP-based quality/naturalness score (0-1, higher = better)
  musiq    multi-scale transformer quality score, trained on KonIQ
           (general photographic quality, not aesthetic preference)
  brisque  classic statistical no-reference IQA, no deep learning at all
           (lower = better) -- a non-learned baseline, useful as a check
           on whether the two learned metrics agree for a real reason or
           are both just picking up on the same training-data bias

This is quality/naturalness, not liking -- a technically clean, sharp,
well-exposed poster scores well here even if nobody finds it appealing;
that's what 03_nima_score.py and 04_laion_aesthetic_score.py are for.

  python3 02_iqa_multi_score.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
No thread limiting on purpose (unlike 03/04) -- this matches a real
finding from running all three at the real project's scale: capping
OMP/MKL threads only helped when several of these processes shared one
machine; for one process per machine (this script's actual deployment
pattern), capping was a measured 10x+ slowdown, not a speedup.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pyiqa
import requests
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("iqa_multi_score")

METRIC_NAMES = ["clipiqa", "musiq", "brisque"]
FIELDS = ["id", "title", "year", *METRIC_NAMES, "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/iqa_multi_score.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    metrics = {}
    for name in METRIC_NAMES:
        metrics[name] = pyiqa.create_metric(name, device=device)
        log.info(f"{name} loaded")

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
                    with torch.inference_mode():
                        for name, m in metrics.items():
                            out[name] = round(float(m(str(poster_file)).item()), 4)
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
