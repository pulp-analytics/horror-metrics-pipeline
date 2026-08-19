#!/usr/bin/env python3
"""Nova vision QA of 06_clip_census.py's CLIP-based monster census.

Cross-checks CLIP zero-shot embedding-similarity labels against an
independent Nova Pro judgment on the same poster -- Nova sees the image
and picks its own category from the same taxonomy, blind to nothing
except being told what CLIP guessed (for context, not as an anchor: the
prompt explicitly asks for Nova's OWN independent judgment).

Same category as 22_creature_weapon_nova_qa.py's justification: this is a
semantic classification call (what creature is this?), the kind of
judgment a vision-LLM adds real signal on -- unlike composition/depth/
saliency/pose, which are continuous geometric measurements with no
comparable "Nova, is this right?" question to ask. See that script's
docstring for the full reasoning, and docs/RESULTS.md, "Census," for what
running this against real posters found.

Unlike every other script in this repo except 22, this needs real AWS
access (Bedrock's Nova Pro) and is not free to run at any real sample
size. Deliberately NOT wired into compute_metrics.asl.json -- a QA tool
for spot-checking 06's output, not a per-poster metric-producing stage.

--census also accepts 13_siglip_reanalysis.py's --census-out (same
id/label/score shape, different embedding model) if you want to QA the
SigLIP census instead -- the taxonomy and prompt are the same either way.

  export AWS_PROFILE=sandbox_bedrock
  python3 23_census_nova_qa.py --in data/sample_input/sample_100_posters.csv --census data/sample_output/census.csv --n 50

Resumable: re-running with the same --out skips ids already scored "ok".
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py. Not shardable on purpose -- see the module note above.
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file

log = get_logger("census_nova_qa")

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-pro-v1:0"
MAX_SIDE = 1200

# Identical to 06_clip_census.py's TAXONOMY keys (+ "none") on purpose --
# this is a cross-check against that exact taxonomy, not an independent one.
CATEGORIES = [
    "vampire", "werewolf", "zombie", "ghost", "demon", "witch", "skeleton",
    "alien", "giant_monster", "masked_killer", "clown", "doll", "shark",
    "spider", "snake", "wolf_dog", "bird", "insect", "none",
]

FIELDS = [
    "id", "clip_label", "clip_score",
    "model", "status", "nova_label", "agree", "reason", "latency_s", "error",
]

PROMPT = """Classify the main creature/monster/threat shown on this movie poster into
EXACTLY ONE of these categories: {categories}

- Pick "none" if there's no monster/creature -- just ordinary people, a generic villain
  with no supernatural/creature element, or abstract imagery.
- "masked_killer" = a human-shaped masked/costumed killer with a weapon (slasher-style),
  not a supernatural creature.
- "wolf_dog" = a real/mundane dog or wolf, not a transformed werewolf.
- "giant_monster" = kaiju-scale monster (city-destroying size).

A CLIP zero-shot embedding classifier already predicted "{clip_label}"
(similarity score {clip_score}). Give your OWN independent judgment.

Return ONLY valid JSON (no markdown):
{{
  "label": "<one of the categories above, exact spelling>",
  "reason": "one short sentence"
}}
"""

_write_lock = threading.Lock()


def load_rows(in_path: Path, census_path: Path) -> list[dict]:
    poster_paths = {}
    with in_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("poster_path"):
                poster_paths[row["id"]] = row["poster_path"]

    rows = []
    with census_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in poster_paths:
                continue
            rows.append({
                "id": pid, "poster_path": poster_paths[pid],
                "clip_label": (row.get("label") or "").strip(),
                "clip_score": row.get("score") or "0",
            })
    return rows


def pick_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["clip_label"], []).append(r)
    rng = random.Random(seed)
    labels = list(by_label)
    per_label = max(1, n // max(1, len(labels)))
    picked: list[dict] = []
    for lab in labels:
        bucket = by_label[lab][:]
        bucket.sort(key=lambda r: float(r["clip_score"] or 0))  # low confidence first
        low = bucket[: int(per_label * 0.7)]
        rest = bucket[int(per_label * 0.7):]
        rng.shuffle(rest)
        picked.extend(low + rest[: per_label - len(low)])
    if len(picked) < n:
        used = {r["id"] for r in picked}
        remaining = [r for r in rows if r["id"] not in used]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    rng.shuffle(picked)
    return picked[:n]


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                done.add(r["id"])
    return done


def append_row(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with _write_lock:
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)


def resize_jpeg(path: Path) -> bytes:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, MAX_SIDE / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def process_one(client, r: dict, posters_dir: Path, session: requests.Session,
                 s3_bucket: str, s3_prefix: str, out_csv: Path) -> None:
    base = {
        "id": r["id"], "clip_label": r["clip_label"], "clip_score": r["clip_score"],
        "model": "nova-pro", "status": "error",
        "nova_label": "", "agree": "", "reason": "", "latency_s": 0.0, "error": "",
    }
    t0 = time.perf_counter()
    try:
        poster_file = posters_dir / f"{r['id']}.jpg"
        if not fetch_poster_file(session, r["poster_path"], poster_file, s3_bucket, s3_prefix):
            raise RuntimeError("download_failed")
        img = resize_jpeg(poster_file)
        prompt = PROMPT.format(
            categories=", ".join(CATEGORIES),
            clip_label=r["clip_label"] or "uncertain",
            clip_score=r["clip_score"],
        )
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": img}}},
                {"text": prompt},
            ]}],
            inferenceConfig={"temperature": 0, "maxTokens": 200},
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
        nova_label = obj.get("label", "")
        base.update(
            status="ok",
            nova_label=nova_label,
            agree=str(nova_label == r["clip_label"]),
            reason=obj.get("reason", ""),
            latency_s=round(time.perf_counter() - t0, 3),
        )
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"[:400]
        base["latency_s"] = round(time.perf_counter() - t0, 3)
    append_row(out_csv, base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--census", required=True, help="output of 06_clip_census.py (or 13_siglip_reanalysis.py's --census-out)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/sample_output/qa_census.csv")
    ap.add_argument("--region", default=REGION)
    add_poster_source_args(ap)
    args = ap.parse_args()

    out_csv = Path(args.out)

    rows = load_rows(Path(args.in_path), Path(args.census))
    log.info(f"census rows available: {len(rows):,}")
    sample = pick_sample(rows, args.n, args.seed)
    done = load_done(out_csv)
    todo = [r for r in sample if r["id"] not in done]
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
            ex.submit(process_one, client, r, posters_dir, session,
                      args.posters_s3_bucket, args.posters_s3_prefix, out_csv): r
            for r in todo
        }
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(todo):
                rate = n_done / max(time.time() - t0, 1e-9)
                log.info(f"  {n_done:,}/{len(todo):,} rate={rate:.1f}/s")

    agree = disagree = 0
    with out_csv.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            a = r["agree"] == "True"
            agree += a
            disagree += not a
    total = agree + disagree
    log.info(f"wrote {out_csv} | agree={agree:,} ({agree/max(total,1):.1%}) disagree={disagree:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
