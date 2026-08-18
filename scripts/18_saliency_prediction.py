#!/usr/bin/env python3
"""Visual saliency prediction (MSI-Net) over movie posters -- where does
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
that isn't torch-based. The model ships as a legacy TF SavedModel with its
weights embedded as graph constants rather than a normal variables file --
loading it through Keras 3's tf.keras.layers.TFSMLayer (the documented
replacement for the dropped tf.keras.models.load_model() SavedModel path)
hard-crashes the process outright (libprotobuf FATAL, uncatchable) on at
least some tensorflow/protobuf builds, because restoring that graph's
protobuf-encoded weights trips a bug in protobuf's C++/upb parser. Plain
`tf.saved_model.load()` (the low-level loader, bypassing Keras entirely)
hits the exact same crash restoring the same graph -- it isn't a
Keras-specific bug. What actually avoids it: forcing protobuf's pure-Python
implementation via PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python, set
below before tensorflow is ever imported. That's a runtime backend switch,
not a package change -- nothing in requirements.txt or the shared
environment's installed versions is touched. See docs/RESULTS.md,
"Saliency," for how this was root-caused and confirmed.

Resumable: re-running with the same --out skips ids already processed.
Shares its poster cache with the other per-poster scripts -- see
utils/posters.py.

Shardable: --shard-index/--shard-count split --in's rows by position,
same convention as every other script in this repo.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Must be set before tensorflow (and therefore protobuf) is imported
# anywhere in this process -- see the module docstring for why.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
    # Low-level loader, not tf.keras.layers.TFSMLayer -- see module
    # docstring: the crash this works around isn't Keras-specific, it's
    # triggered by restoring this SavedModel's graph at all under
    # protobuf's default (C++/upb) backend.
    loaded = tf.saved_model.load(hf_dir)
    return loaded.signatures["serving_default"]


def summarize_saliency(sal: np.ndarray) -> dict:
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


def predict_saliency(model, img_path: Path) -> dict:
    import tensorflow as tf

    img = tf.keras.utils.load_img(str(img_path))
    arr = np.array(img, dtype=np.float32)
    inp = tf.expand_dims(arr, axis=0)
    inp = tf.image.resize(inp, (320, 320), preserve_aspect_ratio=True)
    result = model(input_1=inp)
    sal = result[list(result.keys())[0]].numpy().squeeze()  # key is "layer_from_saved_model", not "output"
    return summarize_saliency(sal)


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
