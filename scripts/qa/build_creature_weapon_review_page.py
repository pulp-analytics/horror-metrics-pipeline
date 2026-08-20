#!/usr/bin/env python3
"""Blind HTML review of creature/weapon boxes (20/21) -- layer 3 of
Validation methodology. Draws the detector box in red. Asks whether that
rectangle contains the claimed label. Does not show detector score or
Nova's verdict.

  python3 scripts/qa/build_creature_weapon_review_page.py
  open data/ground_truth/creature_weapon_review.html
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import requests

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).parent))

from review_page import fetch_to_cache, write_review_html
from utils.posters import add_poster_source_args

qa22 = importlib.import_module("22_creature_weapon_nova_qa")


def jpeg_data_uri(raw: bytes) -> str:
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--boxes", default="data/sample_output/creature_weapon_owlv2.csv")
    ap.add_argument("--source", choices=["owlv2", "dino"], default="owlv2")
    ap.add_argument("--n", type=int, default=0, help="0 = all detections")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ground_truth/creature_weapon_review.html")
    add_poster_source_args(ap)
    args = ap.parse_args()

    with Path(args.in_path).open(newline="", encoding="utf-8") as f:
        meta = {r["id"]: r for r in csv.DictReader(f)}

    detections = qa22.load_detections(Path(args.in_path), Path(args.boxes), args.source)
    if args.n:
        detections = qa22.pick_sample(detections, args.n, args.seed)

    session = requests.Session()
    posters_dir = Path(args.posters_dir)
    rows = []
    for r in detections:
        dest = posters_dir / f"{r['id']}.jpg"
        img = ""
        if fetch_to_cache(session, r["poster_path"], dest, args.posters_s3_bucket, args.posters_s3_prefix):
            img = jpeg_data_uri(qa22.draw_box(dest, r["box"]))
        box = r["box"]
        m = meta.get(r["id"], {})
        key = f"{r['id']}|{r['kind']}|{r['label']}|{json.dumps(box)}"
        rows.append({
            "id": r["id"], "key": key, "title": m.get("title", ""), "year": m.get("year", ""),
            "kind": r["kind"], "label": r["label"], "box": json.dumps(box),
            "question": f"Does the red rectangle contain a {r['label']}?",
            "img": img,
        })

    write_review_html(
        Path(args.out),
        title="Creature/weapon box review",
        blurb="Blind human ground truth for 20/21 boxes (the claim is the label on the question; detector score and Nova verdict are not on this page).",
        storage_key="creature_weapon_review_v1",
        export_name="creature_weapon_human_labels.csv",
        verdicts=[
            {"v": "si", "label": "Yes"},
            {"v": "no", "label": "No"},
            {"v": "no_seguro", "label": "Not sure"},
        ],
        rows=rows,
    )
    print(f"wrote {args.out} ({len(rows)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
