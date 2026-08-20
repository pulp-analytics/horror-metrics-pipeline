#!/usr/bin/env python3
"""OWLv2 ∩ Grounding DINO agreement -- the citable creature/weapon signal.

20 and 21 are independent zero-shot detectors over the same vocabulary.
Neither is trustworthy alone (OWLv2's creature boxes were ~62.5% false
positives under Nova QA; see docs/RESULTS.md). This script does not run
a model: it reads both CSVs and keeps a detection only when the two
agree on label AND the boxes overlap (IoU >= --min-iou).

Per-poster output, same shape as 20/21 so assemble_master_dataset.py
can left-join it:
  creature_n, creature_top_label, creature_top_score, creature_boxes
  weapon_n, weapon_top_label, weapon_top_score, weapon_boxes

Each agreed box in the JSON is a pair:
  {label, iou, owlv2_score, dino_score, owlv2_box, dino_box}

creature_label_agree / weapon_label_agree are a looser poster-level
check: 1 iff both detectors reported a non-empty top_label and those
strings match, even if the boxes do not overlap. Box agreement (the
n/top_*/boxes columns) is the stricter, citable signal.

  python3 scripts/25_creature_weapon_agreement.py
  python3 scripts/25_creature_weapon_agreement.py --min-iou 0.5

No AWS, no model download. Cheap enough to rewrite --out in full each
run (not resumable/shardable -- it's a join of two already-computed
files, not a per-poster inference loop).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.resumable import write_csv_rows

log = get_logger("creature_weapon_agreement")

MIN_IOU = 0.3
FIELDS = [
    "id", "title", "year",
    "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
    "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes",
    "creature_label_agree", "weapon_label_agree",
]


def iou_xywh(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two normalized [x, y, w, h] boxes."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return min(1.0, inter / union)


def parse_boxes(raw) -> list[dict]:
    if raw is None or raw == "":
        return []
    try:
        rows = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        box = row.get("box")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        out.append({
            "label": str(row.get("label", "")),
            "score": float(row.get("score") or 0.0),
            "box": [float(v) for v in box],
        })
    return out


def match_boxes(owl: list[dict], dino: list[dict], min_iou: float) -> list[dict]:
    """Greedy one-to-one matching: walk OWLv2 boxes by score descending,
    pair each with the highest-IoU unused DINO box of the same label
    that clears min_iou."""
    remaining = list(enumerate(dino))
    pairs = []
    for o in sorted(owl, key=lambda r: r["score"], reverse=True):
        best_i = None
        best_iou = min_iou
        best_d = None
        for i, d in remaining:
            if d["label"] != o["label"]:
                continue
            iou = iou_xywh(o["box"], d["box"])
            if iou >= best_iou:
                best_iou = iou
                best_i = i
                best_d = d
        if best_d is None:
            continue
        remaining = [(i, d) for i, d in remaining if i != best_i]
        pairs.append({
            "label": o["label"],
            "iou": round(best_iou, 4),
            "owlv2_score": round(o["score"], 3),
            "dino_score": round(best_d["score"], 3),
            "owlv2_box": [round(v, 4) for v in o["box"]],
            "dino_box": [round(v, 4) for v in best_d["box"]],
        })
    pairs.sort(key=lambda r: min(r["owlv2_score"], r["dino_score"]), reverse=True)
    return pairs


def summarize_pairs(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0, "top_label": "", "top_score": "", "boxes": "[]"}
    top = pairs[0]
    score = round(min(top["owlv2_score"], top["dino_score"]), 3)
    return {
        "n": len(pairs),
        "top_label": top["label"],
        "top_score": score,
        "boxes": json.dumps(pairs),
    }


def label_agree(owl_top: str, dino_top: str) -> int:
    o, d = (owl_top or "").strip(), (dino_top or "").strip()
    return int(bool(o) and o == d)


def load_by_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = (row.get("id") or "").strip()
            if i:
                out[i] = row
    return out


def agree_row(meta: dict, owl: dict | None, dino: dict | None, min_iou: float) -> dict:
    owl = owl or {}
    dino = dino or {}
    out = {"id": meta["id"], "title": meta.get("title", ""), "year": meta.get("year", "")}
    for kind in ("creature", "weapon"):
        pairs = match_boxes(
            parse_boxes(owl.get(f"{kind}_boxes")),
            parse_boxes(dino.get(f"{kind}_boxes")),
            min_iou,
        )
        s = summarize_pairs(pairs)
        out[f"{kind}_n"] = s["n"]
        out[f"{kind}_top_label"] = s["top_label"]
        out[f"{kind}_top_score"] = s["top_score"]
        out[f"{kind}_boxes"] = s["boxes"]
        out[f"{kind}_label_agree"] = label_agree(
            owl.get(f"{kind}_top_label", ""), dino.get(f"{kind}_top_label", ""))
    return {k: out[k] for k in FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--owlv2", default="data/sample_output/creature_weapon_owlv2.csv")
    ap.add_argument("--dino", default="data/sample_output/creature_weapon_dino.csv")
    ap.add_argument("--out", default="data/sample_output/creature_weapon_agreement.csv")
    ap.add_argument("--min-iou", type=float, default=MIN_IOU,
                     help="minimum IoU for a same-label box pair to count as agreement "
                          f"(default {MIN_IOU}; illustrated-poster boxes are looser than COCO)")
    args = ap.parse_args()

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    owl_by_id = load_by_id(Path(args.owlv2))
    dino_by_id = load_by_id(Path(args.dino))
    log.info(f"in={len(rows)} owlv2={len(owl_by_id)} dino={len(dino_by_id)} min_iou={args.min_iou}")

    out_rows = [
        agree_row(row, owl_by_id.get(row["id"]), dino_by_id.get(row["id"]), args.min_iou)
        for row in rows if row.get("id")
    ]
    write_csv_rows(args.out, out_rows)
    n_c = sum(1 for r in out_rows if r["creature_n"] > 0)
    n_w = sum(1 for r in out_rows if r["weapon_n"] > 0)
    n_cl = sum(r["creature_label_agree"] for r in out_rows)
    n_wl = sum(r["weapon_label_agree"] for r in out_rows)
    log.info(f"wrote {args.out}: {len(out_rows)} posters, "
             f"creature box-agree {n_c}, weapon box-agree {n_w}, "
             f"creature label-agree {n_cl}, weapon label-agree {n_wl}")


if __name__ == "__main__":
    main()
