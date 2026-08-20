#!/usr/bin/env python3
"""Monocular depth estimation (MiDaS) over movie posters -- how close/
"in your face" is the foreground threat, vs. a distant/landscape
composition.

MiDaS depth is relative and scale-ambiguous per image (no absolute
units), so every metric below is computed after per-image min-max
normalization to [0,1]:
  mean_depth       average closeness across the whole frame
  p95_depth        closeness of the nearest major foreground mass
                   (robust proxy for "how close is the biggest threat" --
                   avoids the single-pixel noise a plain max() would pick up)
  depth_std        compositional depth contrast (flat graphic-design
                   poster vs. photographic foreground/background separation)
  close_area_frac  fraction of pixels above 0.7 normalized closeness
                   ("how much of the frame is close-up foreground")

  python3 17_depth_estimation.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
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
from utils.device import add_device_arg, pick_device
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("depth_estimation")

# intel-isl/MiDaS redirects to isl-org/MiDaS. Pin the GitHub ref torch.hub
# clones -- without :REF this tracks that repo's default branch. Tag v3_1
# is commit 1645b7e1675301fdfac03640738fe5a6531e17d6, verified 2026-08-19.
MIDAS_GITHUB = "intel-isl/MiDaS"
MIDAS_REVISION = "1645b7e1675301fdfac03640738fe5a6531e17d6"
FIELDS = ["id", "title", "year", "mean_depth", "p95_depth", "depth_std", "close_area_frac", "error"]


def load_midas(device: str):
    import torch.hub as hub
    # MiDaS's hubconf makes nested torch.hub.load() calls (e.g. to
    # rwightman/gen-efficientnet-pytorch) that don't forward trust_repo=True,
    # so those prompt via input() and crash with EOFError on a non-interactive
    # run. Neutralize the trust check outright rather than chase every nested
    # repo through trust_repo plumbing.
    hub._check_repo_is_trusted = lambda *a, **k: None
    repo = f"{MIDAS_GITHUB}:{MIDAS_REVISION}"
    model = hub.load(repo, "MiDaS_small", trust_repo=True)
    model.to(device).eval()
    transforms = hub.load(repo, "transforms", trust_repo=True)
    return model, transforms.small_transform


def estimate_depth(model, transform, device: str, img_path: Path) -> dict:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    inp = transform(arr).to(device)
    with torch.no_grad():
        pred = model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=arr.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = pred.cpu().numpy()
    dmin, dmax = depth.min(), depth.max()
    norm = np.zeros_like(depth) if dmax - dmin < 1e-6 else (depth - dmin) / (dmax - dmin)
    return {
        "mean_depth": round(float(norm.mean()), 4),
        "p95_depth": round(float(np.percentile(norm, 95)), 4),
        "depth_std": round(float(norm.std()), 4),
        "close_area_frac": round(float((norm > 0.7).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/depth_estimation.csv")
    add_poster_source_args(ap)
    add_device_arg(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = pick_device(args.device)
    log.info(f"device={device}")
    model, transform = load_midas(device)
    log.info("MiDaS_small loaded")

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
                    out.update(estimate_depth(model, transform, device, poster_file))
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
