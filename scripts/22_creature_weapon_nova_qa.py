#!/usr/bin/env python3
"""Nova vision QA of creature/weapon box detections from 20_creature_weapon_owlv2.py
or 21_creature_weapon_dino.py.

Both detectors are zero-shot (open-vocabulary), run at a loose confidence
threshold on purpose (better to over-detect and filter later) -- but that
means a large share of boxes are likely false positives. This draws the
detected box on the poster and asks Nova Pro to judge whether it's really
there. This is the actual mechanism behind the "roughly 60%+ of OWLv2's
'creature detected' boxes were false positives" finding cited in both
20/21's docstrings and docs/RESULTS.md's "Creature/weapon detection"
section -- porting it makes that finding reproducible from inside this
repo instead of a fact you have to take on faith from the private
project's prior run.

Samples are stratified toward low-confidence detections (where errors
concentrate), with some high-confidence ones kept for calibration.

Unlike every other script in this repo, this ONE needs real AWS access
(Bedrock's Nova Pro, via the standard boto3 credential chain -- an
AWS_PROFILE env var or otherwise configured credentials) and is not free
to run at any real sample size. It is deliberately NOT wired into
statemachine/compute_metrics.asl.json in poster-analysis-infrastructure --
this is a QA/validation tool for spot-checking a detector's output, not a
per-poster metric-producing pipeline stage, the same reason none of the
private project's other qa_*.py scripts (qa_census.py, qa_faces.py, ...)
became pipeline stages either.

The `PROMPT` below is the settled text after several Bedrock runs and
prompt revisions -- we crossed the detector with Nova more than once
before citing a false-positive rate.

  export AWS_PROFILE=sandbox_bedrock
  python3 22_creature_weapon_nova_qa.py --in data/sample_input/sample_100_posters.csv --boxes data/sample_output/creature_weapon_owlv2.csv --source owlv2 --n 50
  python3 22_creature_weapon_nova_qa.py --in data/sample_input/sample_100_posters.csv --boxes data/sample_output/creature_weapon_dino.csv --source dino --n 50 --out data/sample_output/qa_creature_weapon_dino.csv

Resumable: re-running with the same --out skips (id, kind, label, box)
combinations already scored "ok". Shares its poster cache with the other
per-poster scripts -- see utils/posters.py. Not shardable on purpose --
see the module note above.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file

log = get_logger("creature_weapon_nova_qa")

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-pro-v1:0"
MAX_SIDE = 1200

SOURCE_NAMES = {
    "owlv2": "OWLv2, zero-shot/open-vocabulary object detection",
    "dino": "Grounding DINO, zero-shot/open-vocabulary object detection",
}

FIELDS = [
    "id", "source", "kind", "label", "score", "box",
    "model", "status", "verdict", "actual", "reason", "latency_s", "error",
]

PROMPT = """A computer vision model ({source_name}) found "{label}" inside the RED
rectangle drawn on this movie poster, with confidence {score:.2f}.

Judge whether that red rectangle actually contains a "{label}" (or something close enough
to reasonably count -- e.g. "wolf" for a generic canine, "knife" for any bladed weapon).

Return ONLY valid JSON (no markdown):
{{
  "verdict": "correct" | "false_positive" | "uncertain",
  "actual": "what's actually in the box, in a few words",
  "reason": "one short sentence"
}}

Rules:
- correct: the label is a reasonable description of what's in the red box.
- false_positive: the red box contains something clearly different (background, a person's
  face, unrelated object, empty space, text, etc).
- uncertain: the box is too small/blurry/cropped-at-edge to tell confidently.
"""

_write_lock = threading.Lock()


def load_detections(in_path: Path, boxes_path: Path, source: str) -> list[dict]:
    poster_paths = {}
    with in_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("poster_path"):
                poster_paths[row["id"]] = row["poster_path"]

    rows = []
    with boxes_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in poster_paths:
                continue
            for kind in ("creature", "weapon"):
                boxes_json = row.get(f"{kind}_boxes") or "[]"
                try:
                    dets = json.loads(boxes_json)
                except json.JSONDecodeError:
                    continue
                for d in dets:
                    rows.append({
                        "id": pid, "poster_path": poster_paths[pid], "kind": kind,
                        "label": d["label"], "score": float(d["score"]), "box": d["box"],
                    })
    return rows


def pick_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    low = [r for r in rows if r["score"] < 0.3]
    mid = [r for r in rows if 0.3 <= r["score"] < 0.5]
    high = [r for r in rows if r["score"] >= 0.5]
    rng = random.Random(seed)
    for bucket in (low, mid, high):
        rng.shuffle(bucket)
    # 60% low-confidence (most likely wrong), 25% mid, 15% high (calibration)
    n0 = min(len(low), int(round(n * 0.60)))
    n1 = min(len(mid), int(round(n * 0.25)))
    n2 = min(len(high), n - n0 - n1)
    picked = low[:n0] + mid[:n1] + high[:n2]
    if len(picked) < n:
        used = {(r["id"], r["kind"], r["label"], tuple(r["box"])) for r in picked}
        for r in low + mid + high:
            key = (r["id"], r["kind"], r["label"], tuple(r["box"]))
            if key in used:
                continue
            picked.append(r)
            if len(picked) >= n:
                break
    rng.shuffle(picked)
    return picked[:n]


def draw_box(path: Path, box: list[float]) -> bytes:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, MAX_SIDE / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        w, h = im.size
    x, y, bw, bh = box
    x0, y0 = x * w, y * h
    x1, y1 = x0 + bw * w, y0 + bh * h
    draw = ImageDraw.Draw(im)
    lw = max(3, int(min(w, h) * 0.006))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=lw)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def load_done(path: Path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                done.add((r["id"], r["kind"], r["label"], r["box"]))
    return done


def append_row(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with _write_lock:
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)


def append_jsonl(path: Path, obj: dict) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_one(client, r: dict, source: str, posters_dir: Path, session: requests.Session,
                 s3_bucket: str, s3_prefix: str, out_csv: Path, out_jsonl: Path) -> None:
    box_str = json.dumps(r["box"])
    base = {
        "id": r["id"], "source": source, "kind": r["kind"], "label": r["label"], "score": r["score"],
        "box": box_str, "model": "nova-pro", "status": "error",
        "verdict": "", "actual": "", "reason": "", "latency_s": 0.0, "error": "",
    }
    t0 = time.perf_counter()
    try:
        poster_file = posters_dir / f"{r['id']}.jpg"
        if not fetch_poster_file(session, r["poster_path"], poster_file, s3_bucket, s3_prefix):
            raise RuntimeError("download_failed")
        img = draw_box(poster_file, r["box"])
        prompt = PROMPT.format(source_name=SOURCE_NAMES[source], label=r["label"], score=r["score"])
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": img}}},
                {"text": prompt},
            ]}],
            inferenceConfig={"temperature": 0, "maxTokens": 300},
        )
        text = "".join(
            b.get("text", "") for b in resp.get("output", {}).get("message", {}).get("content", [])
            if "text" in b
        ).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        obj = json.loads(text)
        base.update(
            status="ok",
            verdict=obj.get("verdict", ""),
            actual=obj.get("actual", ""),
            reason=obj.get("reason", ""),
            latency_s=round(time.perf_counter() - t0, 3),
        )
        append_jsonl(out_jsonl, {"id": r["id"], "kind": r["kind"], "label": r["label"], "raw": obj, "text": text[:2000]})
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"[:400]
        base["latency_s"] = round(time.perf_counter() - t0, 3)
    append_row(out_csv, base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--boxes", required=True, help="output of 20_creature_weapon_owlv2.py or 21_creature_weapon_dino.py")
    ap.add_argument("--source", choices=["owlv2", "dino"], required=True)
    ap.add_argument("--kind", choices=["all", "creature", "weapon"], default="all")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-interval", type=float, default=0.1)
    ap.add_argument("--out", default="data/sample_output/qa_creature_weapon.csv")
    ap.add_argument("--region", default=REGION)
    add_poster_source_args(ap)
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_jsonl = out_csv.with_suffix(".jsonl")

    rows = load_detections(Path(args.in_path), Path(args.boxes), args.source)
    if args.kind != "all":
        rows = [r for r in rows if r["kind"] == args.kind]
    log.info(f"detections available (source={args.source}, kind={args.kind}): {len(rows):,}")
    sample = pick_sample(rows, args.n, args.seed)
    done = load_done(out_csv)
    todo = [r for r in sample if (r["id"], r["kind"], r["label"], json.dumps(r["box"])) not in done]
    log.info(f"sample={len(sample):,} done={len(done):,} todo={len(todo):,} workers={args.workers}")

    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).client(
        "bedrock-runtime", region_name=args.region,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
    )

    posters_dir = Path(args.posters_dir)
    session = requests.Session()
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_one, client, r, args.source, posters_dir, session,
                      args.posters_s3_bucket, args.posters_s3_prefix, out_csv, out_jsonl): r
            for r in todo
        }
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if args.min_interval:
                time.sleep(args.min_interval / args.workers)
            if n_done % 50 == 0 or n_done == len(todo):
                rate = n_done / max(time.time() - t0, 1e-9)
                log.info(f"  {n_done:,}/{len(todo):,} rate={rate:.1f}/s")

    verdict_counts: dict[str, int] = {}
    with out_csv.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
    log.info(f"wrote {out_csv} | {verdict_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
