#!/usr/bin/env python3
"""Per-poster geometric composition: symmetry, negative space, visual
complexity/mass, MSER text coverage, grid/thirds alignment, visual
balance, color-scheme harmony, and diagonal/pyramid weight-shift.

Pure OpenCV pixel/edge/contour heuristics (Sobel gradients, Canny edges,
MSER glyph candidates, Hough line segments, saliency, hue-histogram
peaks) -- no CLIP/SigLIP model involved, despite this category's
`clip_attributes_*` column prefix in the real project's own
master_dataset.csv (a historical naming artifact from when it lived
alongside the CLIP-based categories, not a dependency on one). See
docs/METHODOLOGY.md for why each heuristic was chosen over a heavier
layout model (LayoutParser/Detectron2 are trained on document layouts,
not illustrated poster art, and hit an unresolved Detectron2 checkpoint
bug on the path we tried).

  python3 16_geometric_composition.py --in data/sample_input/sample_100_posters.csv

Five independent metric groups, all computed from the same downsampled
grayscale/BGR frame per poster (matching the real project's own
multi_analyze.py, which lets any one group be re-run without re-decoding
the image for the others):
  - composition: symmetry, neg_space, complexity, mass_x/y
  - typography:  text_area, text_y, text_regions (MSER glyph coverage --
                 NOT per-poster title *boxes*, which come from OCR
                 elsewhere in this project; re-running this must not be
                 confused with that separate signal)
  - grid:        align_score, thirds_dist, n_blocks (Lee et al., "Neural
                 Design Network", ECCV 2020 -- how tightly detected
                 layout-block edges share alignment lines, plus rule-of-
                 thirds distance)
  - aesthetic:   balance (saliency centroid vs. geometric center),
                 harmony (dominant hues vs. classic color-wheel schemes)
  - diagonal:    diagonal_score (share of Hough line length that's
                 diagonal), pyramid_shift (bottom-third vs. top-third
                 horizontal spread)

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with 01_color_metrics.py and the other
per-poster scripts -- see utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("geometric_composition")

ANALYSIS_WIDTH = 180
FIELDS = ["id", "title", "year",
          "symmetry", "neg_space", "complexity", "mass_y", "mass_x",
          "text_area", "text_y", "text_regions",
          "align_score", "thirds_dist", "n_blocks",
          "balance", "harmony",
          "diagonal_score", "pyramid_shift"]


# ---------------------------- shared helpers ------------------------------
def _mser_text_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Filtered MSER glyph candidates: small-to-medium, wider than tall or
    roughly square. Heuristic, not OCR -- trend-comparable across posters,
    not absolute truth. Shared by typography() and grid_alignment()."""
    mser = cv2.MSER_create(delta=5, min_area=15, max_area=2000)
    out = mser.detectRegions(gray)
    regions = out[0] if isinstance(out, (tuple, list)) else out
    if regions is None:
        regions = []
    H, W = gray.shape
    boxes = []
    for pts in regions:
        pts = np.asarray(pts)
        if pts.ndim == 1:
            if pts.size < 2:
                continue
            pts = pts.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        ar = w / max(h, 1)
        if 0.1 < ar < 12 and 4 <= h <= H * 0.12 and w < W * 0.9:
            boxes.append((x, y, w, h))
    return boxes


# ---------------------------- metric groups -------------------------------
def composition(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Symmetry, negative space, visual complexity, center of visual mass."""
    small = cv2.resize(gray, (64, 96))
    sym = 1.0 - float(np.abs(small.astype(int) - small[:, ::-1]).mean()) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    flat = float((mag < 8).mean())
    edges = float(cv2.Canny(gray, 60, 140).mean()) / 255
    tot = mag.sum() + 1e-9
    ys, xs = np.indices(mag.shape)
    cy = float((ys * mag).sum() / tot) / mag.shape[0]
    cx = float((xs * mag).sum() / tot) / mag.shape[1]
    return dict(symmetry=round(sym, 4), neg_space=round(flat, 4),
                complexity=round(edges, 4), mass_y=round(cy, 4), mass_x=round(cx, 4))


def typography(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Text coverage via MSER -- corpus-trend signal, not per-poster OCR."""
    H, W = gray.shape
    boxes = _mser_text_boxes(gray)
    mask = np.zeros((H, W), np.uint8)
    for x, y, w, h in boxes:
        mask[y:y + h, x:x + w] = 1
    area = float(mask.mean())
    ys = np.where(mask.any(axis=1))[0]
    ty = float(ys.mean() / H) if len(ys) else -1.0
    return dict(text_area=round(area, 4), text_y=round(ty, 4), text_regions=len(boxes))


def grid_alignment(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Layout blocks (clustered text regions + dominant visual mass) and
    how tightly their edges share alignment lines, plus rule-of-thirds
    distance for the main visual mass."""
    H, W = gray.shape

    txt_mask = np.zeros((H, W), np.uint8)
    for x, y, w, h in _mser_text_boxes(gray):
        txt_mask[y:y + h, x:x + w] = 1
    txt_mask = cv2.dilate(txt_mask, np.ones((5, 15), np.uint8))
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(txt_mask)
    boxes = [(x, y, x + w, y + h) for x, y, w, h, area in stats[1:] if area >= 25]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    busy = cv2.dilate((mag > 25).astype(np.uint8), np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(busy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_box = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) >= 0.01 * H * W:
            x, y, w, h = cv2.boundingRect(c)
            main_box = (x, y, x + w, y + h)
            boxes.append(main_box)

    n_blocks = len(boxes)

    align_score = -1.0
    if n_blocks >= 2:
        # bounds of an axis-aligned box (x0,y0,x1,y1) are just itself --
        # no geometry library needed for this.
        lx = [(b[0], (b[0] + b[2]) / 2, b[2]) for b in boxes]
        ly = [(b[1], (b[1] + b[3]) / 2, b[3]) for b in boxes]
        dists = []
        for axis_vals, norm in ((lx, W), (ly, H)):
            for k in range(3):
                vals = [v[k] for v in axis_vals]
                for i, vi in enumerate(vals):
                    others = vals[:i] + vals[i + 1:]
                    dists.append(min(abs(vi - vj) for vj in others) / norm)
        align_score = round(float(np.mean(dists)), 4)

    thirds_dist = -1.0
    if main_box:
        cx = (main_box[0] + main_box[2]) / 2 / W
        cy = (main_box[1] + main_box[3]) / 2 / H
        pts = [(px, py) for px in (1 / 3, 2 / 3) for py in (1 / 3, 2 / 3)]
        thirds_dist = round(min(np.hypot(cx - px, cy - py) for px, py in pts), 4)

    return dict(align_score=align_score, thirds_dist=thirds_dist, n_blocks=n_blocks)


def aesthetic(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Visual balance (saliency centroid vs. geometric center) and color
    harmony (dominant hues vs. classic color-wheel schemes)."""
    H, W = gray.shape

    balance = -1.0
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    sout = sal.computeSaliency(gray)
    if isinstance(sout, (tuple, list)):
        ok, smap = (bool(sout[0]), sout[1]) if len(sout) >= 2 else (True, sout[0])
    else:
        ok, smap = sout is not None, sout
    if ok and smap is not None:
        smap = np.asarray(smap)
        tot = smap.sum() + 1e-9
        ys, xs = np.indices(smap.shape)
        cy = float((ys * smap).sum() / tot) / H
        cx = float((xs * smap).sum() / tot) / W
        balance = round(float(np.hypot(cx - 0.5, cy - 0.5)), 4)

    harmony = -1.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    peaks = np.argsort(hist)[-3:]
    peaks = peaks[hist[peaks] > hist.sum() * 0.03]
    hues = sorted(peaks * 10.0)
    if len(hues) >= 2:
        scheme_angles = (30, 90, 120, 180)
        devs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                d = abs(hues[i] - hues[j])
                d = min(d, 360 - d)
                nearest = min(scheme_angles, key=lambda a: abs(d - a))
                devs.append(abs(d - nearest) / 180)
        harmony = round(1.0 - float(np.mean(devs)), 4)

    return dict(balance=balance, harmony=harmony)


def diagonal_pyramid(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Diagonal linework share (Hough line segments 25-65deg off
    horizontal) and pyramid/funnel weight-shift (bottom-third vs.
    top-third horizontal spread of gradient energy)."""
    H, W = gray.shape

    edges = cv2.Canny(gray, 60, 140)
    min_len = int(min(H, W) * 0.12)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                             minLineLength=min_len, maxLineGap=6)
    diag_len = total_len = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            ang = min(ang, 180 - ang)
            total_len += length
            if 25 <= ang <= 65:
                diag_len += length
    diagonal_score = round(diag_len / total_len, 4) if total_len > 0 else 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.clip(np.hypot(gx, gy) - 15, 0, None)
    xs = np.arange(W)

    def band_spread(y0, y1):
        band = mag[y0:y1].sum(axis=0)
        tot = band.sum()
        if tot < 1e-6:
            return 0.0
        cx = (xs * band).sum() / tot
        var = ((xs - cx) ** 2 * band).sum() / tot
        return 2 * np.sqrt(var) / W

    pyramid_shift = round(band_spread(2 * H // 3, H) - band_spread(0, H // 3), 4)
    return dict(diagonal_score=diagonal_score, pyramid_shift=pyramid_shift)


def analyze_poster(path: Path) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError("cv2.imread returned None")
    h, w = bgr.shape[:2]
    s = ANALYSIS_WIDTH / w
    bgr = cv2.resize(bgr, (ANALYSIS_WIDTH, int(h * s)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {}
    out.update(composition(bgr, gray))
    out.update(typography(bgr, gray))
    out.update(grid_alignment(bgr, gray))
    out.update(aesthetic(bgr, gray))
    out.update(diagonal_pyramid(bgr, gray))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/geometric_composition.csv")
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
    n_ok = n_failed = 0

    f, w = open_for_append(out_path, FIELDS)
    try:
        import requests
        session = requests.Session()
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
                m = analyze_poster(poster_files[row["id"]])
            except Exception as e:
                log.info(f"  {row['id']}: analysis failed ({e})")
                n_failed += 1
                continue
            m.update(id=row["id"], title=row.get("title", ""), year=row.get("year", ""))
            w.writerow(m)
            n_ok += 1
            if i % 25 == 0 or i == len(todo):
                log.info(f"analyze {i}/{len(todo)}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} analyzed, {n_failed} failed/unreachable (this run)")


if __name__ == "__main__":
    main()
