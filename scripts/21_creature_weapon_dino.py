#!/usr/bin/env python3
"""Grounding DINO creature + weapon detection -- a second, independent
zero-shot detector, run on the same text vocabulary as
20_creature_weapon_owlv2.py, specifically to cross-check it.

Why a second model at all: a blind Nova Pro QA pass over this project's
real OWLv2 output found roughly 60%+ of its "creature detected" boxes
were false positives. Neither detector's raw output is the real signal
by itself -- agreement between OWLv2 and Grounding DINO on the same
poster is (see `25_creature_weapon_agreement.py`, which writes that join,
and docs/RESULTS.md, "Creature/weapon detection," for the real numbers).
CREATURE_QUERIES/WEAPON_QUERIES below are intentionally identical to
20's -- keep them in sync if you ever edit either.

Per-poster metrics (same shape as 20, so the two are directly comparable):
  creature_n, creature_top_label, creature_top_score, creature_boxes
  weapon_n, weapon_top_label, weapon_top_score, weapon_boxes

  python3 21_creature_weapon_dino.py --in data/sample_input/sample_100_posters.csv

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

log = get_logger("creature_weapon_dino")

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Pinned to the HF Hub repo's current commit, verified 2026-08-19 via
# curl https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny
# (the "sha" field). Same gap MODELS.md already closed for SigLIP/LAION.
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
MAX_BOXES = 3

# Identical to 20_creature_weapon_owlv2.py's vocab on purpose -- this is a
# cross-check, not an independent taxonomy.
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


def build_prompt(query_map: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Grounding DINO expects a single caption, lowercase, phrases
    separated by '. '. Returns (caption, phrase->label lookup)."""
    phrase_to_label = {}
    phrases = []
    for label, phrase in query_map.items():
        p = phrase.lower().strip()
        phrase_to_label[p] = label
        phrases.append(p)
    return ". ".join(phrases) + ".", phrase_to_label


def load_dino(device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    return processor, model


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "top_label": "", "top_score": "", "boxes": "[]"}
    top = max(rows, key=lambda r: r["score"])
    return {"n": len(rows), "top_label": top["label"], "top_score": top["score"],
            "boxes": json.dumps(rows)}


def detect(processor, model, device: str, img_path: Path, caption: str, all_lookup: dict[str, str]) -> dict:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    inputs = processor(images=img, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(h, w)],
    )[0]
    boxes = results["boxes"].detach().cpu().tolist()
    scores = results["scores"].detach().cpu().tolist()
    phrases = [str(x).strip().lower() for x in results.get("text_labels", results.get("labels", []))]

    creature_rows, weapon_rows = [], []
    for box, score, phr in zip(boxes, scores, phrases):
        label = all_lookup.get(phr)
        if label is None:
            # partial/merged phrase match -- fall back to substring lookup
            for cand_phr, cand_label in all_lookup.items():
                if cand_phr in phr or phr in cand_phr:
                    label = cand_label
                    break
        if label is None:
            continue
        x0, y0, x1, y1 = box
        xywh = [round(max(0.0, x0 / w), 4), round(max(0.0, y0 / h), 4),
                round(max(0.0, (x1 - x0) / w), 4), round(max(0.0, (y1 - y0) / h), 4)]
        area = xywh[2] * xywh[3]
        if area < 0.002 or area > 0.95:
            continue
        row = {"label": label, "score": round(float(score), 3), "box": xywh}
        (creature_rows if label in CREATURE_QUERIES else weapon_rows).append(row)

    creature_rows.sort(key=lambda r: r["score"], reverse=True)
    weapon_rows.sort(key=lambda r: r["score"], reverse=True)
    c = _summarize(creature_rows[:MAX_BOXES])
    wp = _summarize(weapon_rows[:MAX_BOXES])
    return {
        "creature_n": c["n"], "creature_top_label": c["top_label"],
        "creature_top_score": c["top_score"], "creature_boxes": c["boxes"],
        "weapon_n": wp["n"], "weapon_top_label": wp["top_label"],
        "weapon_top_score": wp["top_score"], "weapon_boxes": wp["boxes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/creature_weapon_dino.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    processor, model = load_dino(device)
    log.info(f"{MODEL_ID} loaded")

    creature_caption, creature_lookup = build_prompt(CREATURE_QUERIES)
    weapon_caption, weapon_lookup = build_prompt(WEAPON_QUERIES)
    all_lookup = {**creature_lookup, **weapon_lookup}
    caption = creature_caption[:-1] + ". " + weapon_caption  # merge into one pass
    log.info(f"caption ({len(all_lookup)} phrases): {caption[:200]}...")

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
                    out.update(detect(processor, model, device, poster_file, caption, all_lookup))
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
