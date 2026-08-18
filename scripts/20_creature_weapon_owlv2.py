#!/usr/bin/env python3
"""OWLv2 open-vocabulary creature + weapon detection over movie posters.

One zero-shot detector, one forward pass per poster, two label vocabularies
(18 creature phrases -- vampire, zombie, giant monster, evil clown, etc. --
and 12 weapon phrases -- knife, axe, chainsaw, etc.), split by which
vocabulary each detected box's label came from.

Per-poster metrics (top-3 boxes by score, kept in full as JSON so a
downstream consumer can draw them):
  creature_n, creature_top_label, creature_top_score, creature_boxes
  weapon_n, weapon_top_label, weapon_top_score, weapon_boxes

**Known to be noisy on its own**: a blind Nova Pro QA pass over this
project's real OWLv2 output found roughly 60%+ of its "creature detected"
boxes were false positives (see 21_creature_weapon_dino.py, a second
independent zero-shot detector run on the exact same posters and
vocabulary specifically to cross-check this one -- neither script alone
is the real signal; agreement between them is). Don't treat this script's
output as ground truth by itself.

  python3 20_creature_weapon_owlv2.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("creature_weapon_owlv2")

MODEL_ID = "google/owlv2-base-patch16"
MIN_SCORE = 0.2
MAX_BOXES = 3

CREATURE_QUERIES = {
    "vampire": "vampire", "werewolf": "werewolf", "zombie": "zombie",
    "ghost": "ghost", "demon": "demon", "witch": "witch", "skeleton": "skull",
    "alien": "alien", "giant_monster": "giant monster", "masked_killer": "masked killer",
    "clown": "evil clown", "doll": "creepy doll", "shark": "shark", "spider": "spider",
    "snake": "snake", "wolf_dog": "wolf", "bird": "bird", "insect": "insect",
}
WEAPON_QUERIES = {
    "knife": "knife", "gun": "handgun", "rifle": "rifle", "axe": "axe",
    "sword": "sword", "machete": "machete", "chainsaw": "chainsaw",
    "scissors": "scissors", "syringe": "syringe", "hammer": "hammer",
    "baseball_bat": "baseball bat", "arrow": "arrow",
}

FIELDS = ["id", "title", "year",
          "creature_n", "creature_top_label", "creature_top_score", "creature_boxes",
          "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error"]


def load_owlv2(device: str):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model


def filter_boxes(boxes, scores, labels, w, h, min_score=MIN_SCORE, max_boxes=MAX_BOXES) -> list[dict]:
    rows = []
    for box, score, lab in zip(boxes, scores, labels):
        if float(score) < min_score:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        xywh = [round(max(0.0, x0 / w), 4), round(max(0.0, y0 / h), 4),
                round(max(0.0, (x1 - x0) / w), 4), round(max(0.0, (y1 - y0) / h), 4)]
        area = xywh[2] * xywh[3]
        if area < 0.002 or area > 0.95:
            continue
        rows.append({"label": str(lab), "score": round(float(score), 3), "box": xywh})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:max_boxes]


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "top_label": "", "top_score": "", "boxes": "[]"}
    top = max(rows, key=lambda r: r["score"])
    return {"n": len(rows), "top_label": top["label"], "top_score": top["score"],
            "boxes": json.dumps(rows)}


def detect(processor, model, device: str, img_path: Path) -> dict:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    all_labels = list(CREATURE_QUERIES) + list(WEAPON_QUERIES)
    query_map = {**CREATURE_QUERIES, **WEAPON_QUERIES}
    text = [[query_map[label] for label in all_labels]]

    inputs = processor(text=text, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=MIN_SCORE,
        target_sizes=torch.tensor([[h, w]], device=device), text_labels=[all_labels],
    )[0]
    boxes = results["boxes"].detach().cpu().tolist()
    scores = results["scores"].detach().cpu().tolist()
    if results.get("text_labels") is not None:
        labs = [str(x) for x in results["text_labels"]]
    else:
        labs = [all_labels[int(i)] for i in results["labels"].detach().cpu().tolist()]

    creature_set = set(CREATURE_QUERIES)
    weapon_set = set(WEAPON_QUERIES)
    c_idx = [i for i, lab in enumerate(labs) if lab in creature_set]
    w_idx = [i for i, lab in enumerate(labs) if lab in weapon_set]

    creature_rows = filter_boxes([boxes[i] for i in c_idx], [scores[i] for i in c_idx],
                                  [labs[i] for i in c_idx], w, h)
    weapon_rows = filter_boxes([boxes[i] for i in w_idx], [scores[i] for i in w_idx],
                                [labs[i] for i in w_idx], w, h)

    c = _summarize(creature_rows)
    wp = _summarize(weapon_rows)
    return {
        "creature_n": c["n"], "creature_top_label": c["top_label"],
        "creature_top_score": c["top_score"], "creature_boxes": c["boxes"],
        "weapon_n": wp["n"], "weapon_top_label": wp["top_label"],
        "weapon_top_score": wp["top_score"], "weapon_boxes": wp["boxes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/creature_weapon_owlv2.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    processor, model = load_owlv2(device)
    log.info(f"{MODEL_ID} loaded")

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
                    out.update(detect(processor, model, device, poster_file))
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
