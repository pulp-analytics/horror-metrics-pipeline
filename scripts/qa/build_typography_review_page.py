#!/usr/bin/env python3
"""Blind HTML review of title-lettering register (08) -- layer 3 of
Validation methodology. Does not show CLIP's axis or Nova's register.

  python3 scripts/qa/build_typography_review_page.py
  open data/ground_truth/typography_review.html
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).parent))

from review_page import fetch_to_cache, poster_data_uri, write_review_html
from utils.posters import add_poster_source_args

REGISTERS = ["ornate", "decorative", "standard", "clean", "minimal", "no_seguro"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/ground_truth/typography_review.html")
    add_poster_source_args(ap)
    args = ap.parse_args()

    with Path(args.in_path).open(newline="", encoding="utf-8") as f:
        posters = list(csv.DictReader(f))

    session = requests.Session()
    posters_dir = Path(args.posters_dir)
    rows = []
    for p in posters:
        dest = posters_dir / f"{p['id']}.jpg"
        img = ""
        if fetch_to_cache(session, p["poster_path"], dest, args.posters_s3_bucket, args.posters_s3_prefix):
            img = poster_data_uri(dest)
        rows.append({
            "id": p["id"], "key": p["id"], "title": p["title"], "year": p.get("year", ""),
            "question": "Title lettering style (ignore tagline/credits): ornate → decorative → standard → clean → minimal",
            "img": img,
        })

    write_review_html(
        Path(args.out),
        title="Typography review",
        blurb="Blind human ground truth for 08_clip_typography_axis.py. CLIP axis and Nova register are not on this page.",
        storage_key="typography_review_v1",
        export_name="typography_human_labels.csv",
        verdicts=[{"v": r, "label": r} for r in REGISTERS],
        rows=rows,
    )
    print(f"wrote {args.out} ({len(rows)} posters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
