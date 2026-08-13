"""Shared poster fetch-and-cache helper, used by every per-poster metric
script in this repo (01_color_metrics.py, 02_iqa_multi_score.py,
03_nima_score.py, 04_laion_aesthetic_score.py, and future CLIP/SigLIP/
geometric-composition scripts). One shared cache, not one per script,
because these are independent metric categories analyzing the SAME
image -- sharing the downloaded file is a pure I/O optimization (avoids
fetching the same poster N times for N categories), not a dependency
between categories: each script still computes its own metrics from the
image entirely on its own.

Three tiers, checked in order, matching the real project's own storage
pattern (see download_posters_by_id.py in the private pipeline for the
real S3 key convention this mirrors):

  1. local disk cache (--posters-dir)      fastest, this run's scratch copy
  2. S3, if --posters-s3-bucket is set     durable, shared across every run
                                            and every repo -- once a poster is
                                            in S3 it never needs to hit TMDB
                                            again, from any script, ever
  3. TMDB's public image CDN               the fallback / how S3 gets
                                            populated in the first place

S3 is optional and off by default (no bucket = skip straight to TMDB) so
this repo's sample data still runs with zero AWS involvement -- matching
fear_pipeline.py's own --posters-s3-bucket/POSTER_BUCKET convention.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

IMG_BASE = "https://image.tmdb.org/t/p/w500"


def fetch_poster_file(session: requests.Session, poster_path: str, dest: Path,
                       s3_bucket: str = "", s3_prefix: str = "") -> bool:
    """Ensures `dest` exists locally, trying disk cache -> S3 -> TMDB CDN in
    that order. Returns whether `dest` is usable afterward."""
    if dest.exists():
        return True

    if s3_bucket:
        try:
            import boto3
            dest.parent.mkdir(parents=True, exist_ok=True)
            boto3.client("s3").download_file(s3_bucket, f"{s3_prefix}/posters/{dest.stem}.jpg", str(dest))
            if dest.exists() and dest.stat().st_size > 0:
                return True
        except Exception:
            dest.unlink(missing_ok=True)

    try:
        resp = session.get(f"{IMG_BASE}{poster_path}", timeout=30)
        if resp.status_code == 200 and resp.content:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


def add_poster_source_args(ap) -> None:
    """Shared --posters-dir/--posters-s3-bucket/--posters-s3-prefix flags,
    so every script exposes them identically."""
    ap.add_argument("--posters-dir", default="data/posters_cache",
                     help="local cache dir, shared across every metric script -- a poster "
                          "downloaded by one script is reused by the others, not re-fetched")
    ap.add_argument("--posters-s3-bucket", default=os.environ.get("POSTER_BUCKET", ""),
                     help="if set, check s3://BUCKET/PREFIX/posters/{id}.jpg before falling "
                          "back to TMDB's CDN -- optional, matches the real project's storage "
                          "pattern; leave unset to use TMDB only (no AWS involved)")
    ap.add_argument("--posters-s3-prefix", default=os.environ.get("POSTER_PREFIX", ""))
