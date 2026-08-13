#!/usr/bin/env python3
"""Per-poster color analysis: dominant palette, brightness, saturation, and
"blood red" share, computed the same way as the real 145k-poster corpus run
(see docs/METHODOLOGY.md for the ACM "Colour of Horror" 2022 method this is
based on).

For each poster: convert to CIELAB (D65), then run a saturation-weighted
k-means (k=5) to get a dominant palette that isn't dominated by large flat
backgrounds -- a plain unweighted k-means tends to just return shades of
whatever color fills the most pixels (often a dark background), not the
colors a human would actually call "the palette." Also computes six
hue-family shares (red/warm/green/blue/purple/dark-or-grey) that a
downstream aggregation step (out of scope for this repo -- see the
README) turns into the "Color River" chart.

  python3 01_color_metrics.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
Downloads run concurrently (--workers, default 12) since this is a CDN
fetch + local CPU computation, not a rate-limited API call -- unlike
horror-corpus-validation's AWS-service gates, there's no per-request quota
to pace against here.

Shares its poster cache (--posters-dir, optionally backed by S3 -- see
utils/posters.py) with 02_iqa_multi_score.py, 03_nima_score.py, and
04_laion_aesthetic_score.py: whichever of these scripts runs first
downloads a given poster, every other one reuses that same file instead
of fetching it again. Pure I/O sharing, not a dependency between
categories -- each script still computes its own metrics independently.

Shardable: --shard-index/--shard-count split --in's rows by position, for
running N copies of this script in parallel (e.g. an AWS Batch array job,
same convention as the sibling horror-corpus-validation repo).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("color_metrics")

ANALYSIS_SIZE = (96, 144)                       # downsample before clustering
K = 5                                            # palette size
FIELDS = ["id", "title", "year", "brightness", "dark_share", "saturation", "red_share",
          "palette", "palette_share", "band_red", "band_warm", "band_green",
          "band_blue", "band_purple", "band_dark"]


# ---------------------------- color math (numpy) -----------------------------
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb float array (...,3) in [0,1] -> CIELAB (D65). Vectorized."""
    r = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = r @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    idx = m & (mx == r); h[idx] = (60 * ((g - b)[idx] / d[idx])) % 360
    idx = m & (mx == g); h[idx] = 60 * ((b - r)[idx] / d[idx]) + 120
    idx = m & (mx == b); h[idx] = 60 * ((r - g)[idx] / d[idx]) + 240
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0)
    return h, s, mx


def lab_to_hex(lab: np.ndarray) -> str:
    """Approximate LAB -> sRGB hex for palette output."""
    L, a, b = lab
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200

    def finv(t):
        return np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)

    xyz = np.array([finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883])
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = M @ xyz
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    rgb = np.clip(rgb, 0, 1)
    return "#{:02x}{:02x}{:02x}".format(*(rgb * 255).astype(int))


# ---------------------------- per-poster metrics ------------------------------
def analyze_poster(img_bytes: bytes, rng: np.random.Generator) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(ANALYSIS_SIZE)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    px = rgb.reshape(-1, 3)
    lab = srgb_to_lab(px)
    h, s, v = rgb_to_hsv(px)

    brightness = float(lab[:, 0].mean())                       # mean L*, 0-100
    dark_share = float((lab[:, 0] < 20).mean())                # near-black px
    saturation = float(s.mean())
    red = ((h >= 345) | (h <= 15)) & (s > 0.4) & (v > 0.15)     # blood red
    red_share = float(red.mean())

    # hue-family shares (feeds the Color River chart)
    dark_or_grey = (v < 0.12) | (lab[:, 0] < 15) | (s < 0.15)
    chrom = ~dark_or_grey

    def band(lo, hi):
        m = ((h >= lo) | (h < hi)) if lo > hi else ((h >= lo) & (h < hi))
        return round(float((m & chrom).mean()), 4)

    bands = dict(band_red=band(345, 15), band_warm=band(15, 70),
                 band_green=band(70, 170), band_blue=band(170, 260),
                 band_purple=band(260, 345),
                 band_dark=round(float(dark_or_grey.mean()), 4))

    # saturation-weighted k-means palette (ACM "Colour of Horror" method --
    # weighting by saturation keeps a large flat dark background from
    # dominating all 5 clusters, while `0.25 +` keeps some weight on
    # neutrals so a genuinely monochrome poster doesn't get a fabricated
    # colorful palette)
    w = 0.25 + s
    idx = rng.choice(len(px), size=min(4000, len(px)), p=w / w.sum())
    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(lab[idx])
    counts = np.bincount(km.labels_, minlength=K)
    order = np.argsort(-counts)
    palette = [lab_to_hex(km.cluster_centers_[i]) for i in order]
    pal_share = [round(float(counts[i]) / counts.sum(), 3) for i in order]

    return dict(brightness=round(brightness, 2), dark_share=round(dark_share, 4),
                saturation=round(saturation, 4), red_share=round(red_share, 4),
                palette=palette, palette_share=pal_share, **bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/color_metrics.csv")
    add_poster_source_args(ap)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = shard_rows(rows, args.shard_index, args.shard_count)

    out_path = Path(args.out)
    done = load_done_ids(out_path)
    todo = [row for row in rows if row["id"] not in done and row.get("poster_path")]
    if done:
        log.info(f"resuming: {len(done)} already done, {len(todo)} remaining")

    posters_dir = Path(args.posters_dir)
    rng = np.random.default_rng(42)
    session = requests.Session()
    n_ok = n_failed = 0

    f, w = open_for_append(out_path, FIELDS)
    try:
        # concurrent fetch-to-cache first (this is the CDN-bound phase);
        # analysis itself is CPU-bound and runs sequentially below
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            poster_files = {row["id"]: posters_dir / f"{row['id']}.jpg" for row in todo}
            futs = {ex.submit(fetch_poster_file, session, row["poster_path"], poster_files[row["id"]],
                               args.posters_s3_bucket, args.posters_s3_prefix): row for row in todo}
            fetched: dict[str, bool] = {}
            for i, fut in enumerate(as_completed(futs), 1):
                row = futs[fut]
                fetched[row["id"]] = fut.result()
                if i % 25 == 0 or i == len(todo):
                    log.info(f"fetch {i}/{len(todo)}")

        for i, row in enumerate(todo, 1):
            if not fetched.get(row["id"]):
                n_failed += 1
                continue
            try:
                m = analyze_poster(poster_files[row["id"]].read_bytes(), rng)
            except Exception as e:
                log.info(f"  {row['id']}: analysis failed ({e})")
                n_failed += 1
                continue
            m.update(id=row["id"], title=row.get("title", ""), year=row.get("year", ""),
                     palette=json.dumps(m["palette"]), palette_share=json.dumps(m["palette_share"]))
            w.writerow(m)
            n_ok += 1
            if i % 25 == 0 or i == len(todo):
                log.info(f"analyze {i}/{len(todo)}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} analyzed, {n_failed} failed/unreachable (this run)")


if __name__ == "__main__":
    main()
