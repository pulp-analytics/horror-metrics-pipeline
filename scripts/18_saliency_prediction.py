#!/usr/bin/env python3
"""BLOCKED as of this writing -- see docs/RESULTS.md, "Saliency (blocked),"
before running this. Loading MSI-Net's legacy TF SavedModel via
tf.keras.layers.TFSMLayer hard-crashes the Python process (libprotobuf
FATAL, not a catchable exception) with at least some tensorflow/protobuf
version combinations. The code below is a faithful, otherwise-untested
port of the real project's script -- included so the work isn't lost and
so a future environment with a compatible protobuf can pick it up, not
because it's known to run end to end today.

Visual saliency prediction (MSI-Net) over movie posters -- where does
the eye go first?

alexanderkroner/MSI-Net, a contextual encoder-decoder CNN trained on real
human eye-tracking fixation data. General-purpose attention prediction,
not tied to any horror-specific heuristic.

Per-poster metrics (all from the predicted saliency heatmap, normalized
as a probability-like distribution):
  peak_x, peak_y  normalized location of the single most salient point
                  (where the eye is predicted to land first)
  top10pct_mass   fraction of total saliency concentrated in the hottest
                  10% of pixels (higher = focused attention on one spot,
                  e.g. a face or monster; lower = attention spread across
                  a busy/cluttered composition)
  mean_saliency   average saliency (mostly a sanity/normalization check --
                  should hover near a similar value across posters since
                  the map is a probability-like distribution)

  python3 18_saliency_prediction.py --in data/sample_input/sample_100_posters.csv

Requires tensorflow (see requirements.txt) -- the one script in this repo
that isn't torch-based. The model ships as a legacy TF SavedModel, loaded
via Keras 3's TFSMLayer since tf.keras.models.load_model() dropped
support for that format.

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("saliency_prediction")

FIELDS = ["id", "title", "year", "peak_x", "peak_y", "top10pct_mass", "mean_saliency", "error"]


def load_msinet():
    import tensorflow as tf
    from huggingface_hub import snapshot_download

    hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")
    # Keras 3 dropped tf.keras.models.load_model() support for legacy TF
    # SavedModel dirs -- TFSMLayer is the documented replacement.
    return tf.keras.layers.TFSMLayer(hf_dir, call_endpoint="serving_default")


def predict_saliency(model, img_path: Path) -> dict:
    import tensorflow as tf

    img = tf.keras.utils.load_img(str(img_path))
    arr = np.array(img, dtype=np.float32)
    inp = tf.expand_dims(arr, axis=0)
    inp = tf.image.resize(inp, (320, 320), preserve_aspect_ratio=True)
    result = model(inp)
    sal = result[list(result.keys())[0]].numpy().squeeze()  # key is "layer_from_saved_model", not "output"

    total = sal.sum()
    flat = sal.flatten()
    k = max(1, int(0.10 * flat.size))
    top10_mass = float(np.sort(flat)[-k:].sum() / total) if total > 0 else 0.0
    peak_y, peak_x = np.unravel_index(np.argmax(sal), sal.shape)
    return {
        "peak_x": round(float(peak_x / sal.shape[1]), 4),
        "peak_y": round(float(peak_y / sal.shape[0]), 4),
        "top10pct_mass": round(top10_mass, 4),
        "mean_saliency": round(float(sal.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/saliency_prediction.csv")
    add_poster_source_args(ap)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    log.info("loading MSI-Net...")
    model = load_msinet()
    log.info("MSI-Net loaded")

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
                    out.update(predict_saliency(model, poster_file))
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
