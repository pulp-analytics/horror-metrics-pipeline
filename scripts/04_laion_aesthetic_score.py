#!/usr/bin/env python3
"""LAION aesthetic predictor (CLIP ViT-L/14 image embedding + a small
linear MLP head trained on human aesthetic ratings) per poster --
originally built to filter LAION-5B before Stable Diffusion training,
repurposed here as a third, independent "does this poster look good"
signal alongside 02's clipiqa and 03's NIMA.

Needs its own fresh ViT-L/14 (768-dim) embeddings -- doesn't reuse any
CLIP embedding cache from a future CLIP/SigLIP semantic-metrics category,
since those use ViT-B/32 (512-dim), a different, incompatible model.

  python3 04_laion_aesthetic_score.py --in data/sample_input/sample_100_posters.csv

Resumable: re-running with the same --out skips ids already processed.
THREADS (default 8) caps OMP/MKL/torch thread count, same reasoning as
03_nima_score.py -- set it to your instance's physical core count when
several scoring processes share one machine.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)

import open_clip
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", _THREADS)))

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("laion_aesthetic_score")

FIELDS = ["id", "title", "year", "aesthetic_score", "error"]


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2), nn.Linear(1024, 128),
            nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


AESTHETIC_HEAD_REPO = "camenduru/improved-aesthetic-predictor"
AESTHETIC_HEAD_FILE = "sac+logos+ava1-l14-linearMSE.pth"
# Pinned to the HF Hub repo's current commit, verified 2026-08-14 via
# curl https://huggingface.co/api/models/camenduru/improved-aesthetic-predictor
# (the "sha" field). See docs/MODELS.md.
AESTHETIC_HEAD_REVISION = "7b2449be1264fcd9a1cf92e3d30dd29af989c836"


def load_aesthetic_head(device):
    path = hf_hub_download(repo_id=AESTHETIC_HEAD_REPO, filename=AESTHETIC_HEAD_FILE,
                            revision=AESTHETIC_HEAD_REVISION)
    m = MLP(768)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/laion_aesthetic_score.csv")
    add_poster_source_args(ap)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    head = load_aesthetic_head(device)
    log.info("models loaded")

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
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start:batch_start + args.batch_size]
            imgs, ok_rows = [], []
            for row in batch:
                poster_file = posters_dir / f"{row['id']}.jpg"
                if not fetch_poster_file(session, row["poster_path"], poster_file,
                                          args.posters_s3_bucket, args.posters_s3_prefix):
                    w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                                "error": "download_failed"})
                    n_err += 1
                    continue
                try:
                    imgs.append(preprocess(Image.open(poster_file).convert("RGB")))
                    ok_rows.append(row)
                except Exception as e:
                    w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                                "error": str(e)[:200]})
                    n_err += 1

            if imgs:
                pixel_batch = torch.stack(imgs).to(device)
                with torch.inference_mode():
                    if device == "cpu":
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            emb = clip_model.encode_image(pixel_batch)
                    else:
                        emb = clip_model.encode_image(pixel_batch)
                    emb = emb.float()
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    scores = head(emb).squeeze(-1).cpu().numpy()
                for row, score in zip(ok_rows, scores):
                    w.writerow({"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""),
                                "aesthetic_score": round(float(score), 4), "error": ""})
                    n_ok += 1

            done_n = batch_start + len(batch)
            if done_n % (args.batch_size * 5) == 0 or done_n >= len(todo):
                rate = done_n / max(time.time() - t0, 1e-9)
                log.info(f"{done_n}/{len(todo)} rate={rate:.2f}/s ok={n_ok} err={n_err}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} scored, {n_err} failed (this run)")


if __name__ == "__main__":
    main()
