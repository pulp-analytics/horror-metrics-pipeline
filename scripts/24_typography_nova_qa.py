#!/usr/bin/env python3
"""Nova vision QA of 08_clip_typography_axis.py's CLIP-based typography axis.

Cross-checks CLIP's continuous ornate<->minimal axis, binned into 5
registers, against an independent Nova Pro judgment of the same poster's
title lettering style. Same semantic-classification justification as
22/23's docstrings -- see those for the full reasoning on why this
category (and not composition/depth/saliency/pose) is worth a Nova
cross-check at all.

08_clip_typography_axis.py deliberately outputs only a continuous axis
score, not discrete register buckets (see that script's own docstring:
"Not something this script does: quantile-based register binning").
The private project's real clip_typography_axis.py *did* bucket into 5
registers via corpus-wide quantiles (np.quantile over the whole axis
distribution, highest axis = "ornate") specifically to ask this exact
Nova-agreement question -- bin_register() below ports that quantile logic
verbatim, but it lives here rather than in 08, since it's QA-specific
and 08's whole reason for staying continuous is that quantile buckets
are relative to whatever corpus you happen to be scoring, not a
per-poster property. Needs at least a few dozen rows in --typography for
the quantile edges to mean anything -- running this on a 5-poster sample
will produce close to meaningless bucket boundaries.

Unlike every other script in this repo except 22/23, this needs real AWS
access (Bedrock's Nova Pro). Deliberately NOT wired into
compute_metrics.asl.json -- a QA tool for spot-checking 08's output, not
a per-poster metric-producing stage.

The `PROMPT` below is the settled text after several Bedrock runs and
prompt revisions.

  export AWS_PROFILE=sandbox_bedrock
  python3 24_typography_nova_qa.py --in data/sample_input/sample_100_posters.csv --typography data/sample_output/typography.csv --n 50

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

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file

log = get_logger("typography_nova_qa")

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-pro-v1:0"
MAX_SIDE = 1200

REGISTERS = ["ornate", "decorative", "standard", "clean", "minimal"]

FIELDS = [
    "id", "clip_register", "clip_axis",
    "model", "status", "nova_register", "agree", "agree_adjacent", "reason", "latency_s", "error",
]

PROMPT = """Look at the movie TITLE LETTERING on this poster (ignore tagline/credits/small
print). Classify its style into EXACTLY ONE of these 5 points on a spectrum:

- "ornate": heavily decorative, hand-drawn/painted, dripping/textured, elaborate flourishes
- "decorative": stylized display type with some flourish/texture, but not full illustration
- "standard": a normal bold display font, some character but not heavily stylized
- "clean": simple bold sans-serif or serif, minimal styling
- "minimal": plain, thin, minimal-impact typography, almost utilitarian

A CLIP zero-shot embedding classifier already predicted "{clip_register}"
(axis score {clip_axis}, where higher = more ornate). Give your OWN independent judgment.

Return ONLY valid JSON (no markdown):
{{
  "register": "ornate" | "decorative" | "standard" | "clean" | "minimal",
  "reason": "one short sentence"
}}
"""

_write_lock = threading.Lock()


def bin_register(axis_values: list[float]) -> list[str]:
    """Corpus-wide quantile binning, ported verbatim from the real
    project's clip_typography_axis.py -- see module docstring for why
    this lives here instead of in 08_clip_typography_axis.py."""
    arr = np.array(axis_values, dtype=np.float64)
    edges = np.quantile(arr, np.linspace(0, 1, len(REGISTERS) + 1))
    b = np.digitize(arr, edges[1:-1])  # 0..nbins-1, 0=lowest axis (most "minimal")
    return [REGISTERS[::-1][k] for k in b]  # reversed -> 0=ornate at the highest axis


def load_rows(in_path: Path, typography_path: Path) -> list[dict]:
    poster_paths = {}
    with in_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("poster_path"):
                poster_paths[row["id"]] = row["poster_path"]

    ids, axes = [], []
    with typography_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in poster_paths:
                continue
            ids.append(pid)
            axes.append(float(row["axis"]))

    registers = bin_register(axes) if axes else []
    rows = [
        {"id": pid, "poster_path": poster_paths[pid], "clip_register": reg, "clip_axis": round(axis, 4)}
        for pid, axis, reg in zip(ids, axes, registers)
    ]
    return rows


def pick_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_reg: dict[str, list[dict]] = {}
    for r in rows:
        by_reg.setdefault(r["clip_register"], []).append(r)
    rng = random.Random(seed)
    regs = list(by_reg)
    per_reg = max(1, n // max(1, len(regs)))
    picked: list[dict] = []
    for reg in regs:
        bucket = by_reg[reg][:]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_reg])
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
        "id": r["id"], "clip_register": r["clip_register"], "clip_axis": r["clip_axis"],
        "model": "nova-pro", "status": "error",
        "nova_register": "", "agree": "", "agree_adjacent": "", "reason": "", "latency_s": 0.0, "error": "",
    }
    t0 = time.perf_counter()
    try:
        poster_file = posters_dir / f"{r['id']}.jpg"
        if not fetch_poster_file(session, r["poster_path"], poster_file, s3_bucket, s3_prefix):
            raise RuntimeError("download_failed")
        img = resize_jpeg(poster_file)
        prompt = PROMPT.format(clip_register=r["clip_register"] or "standard", clip_axis=r["clip_axis"])
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
        nova_reg = obj.get("register", "")
        clip_idx = REGISTERS.index(r["clip_register"]) if r["clip_register"] in REGISTERS else -99
        nova_idx = REGISTERS.index(nova_reg) if nova_reg in REGISTERS else -99
        base.update(
            status="ok",
            nova_register=nova_reg,
            agree=str(nova_reg == r["clip_register"]),
            agree_adjacent=str(abs(clip_idx - nova_idx) <= 1),
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
    ap.add_argument("--typography", required=True, help="output of 08_clip_typography_axis.py")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/sample_output/qa_typography.csv")
    ap.add_argument("--region", default=REGION)
    add_poster_source_args(ap)
    args = ap.parse_args()

    out_csv = Path(args.out)

    rows = load_rows(Path(args.in_path), Path(args.typography))
    log.info(f"typography rows available: {len(rows):,}")
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

    agree = adjacent = disagree = 0
    with out_csv.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            if r["agree"] == "True":
                agree += 1
            elif r["agree_adjacent"] == "True":
                adjacent += 1
            else:
                disagree += 1
    total = agree + adjacent + disagree
    log.info(f"wrote {out_csv} | exact={agree:,} ({agree/max(total,1):.1%}) adjacent={adjacent:,} disagree={disagree:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
