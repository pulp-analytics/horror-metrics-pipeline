#!/usr/bin/env python3
"""THE FEAR AXIS -- dread/nightmarish <-> calm/mundane, over the whole
poster, as a continuous score instead of discrete categories.

Method: axis = cos(embedding, DREAD_prototype) - cos(embedding, CALM_prototype).
Higher = more terrifying/nightmarish; lower = calmer/more mundane. This is
a continuous projection between two prompt-ensemble poles, not a
classifier -- chosen deliberately over discrete categories (see
docs/METHODOLOGY.md for why, and 08_clip_typography_axis.py's docstring
for the real failed experiment that established this pattern).

  python3 07_clip_fear_axis.py --in data/sample_input/sample_100_posters.csv
  python3 07_clip_fear_axis.py --validate   # ranking of known specimens

Not something this script does: binning posters into registers
(nightmarish/dreadful/.../calm) by quantiles of the current batch, or
aggregating by decade -- both are corpus-relative (they depend on what
else is in the run, not just this one poster), which is aggregation logic
out of scope for this repo (see the README's Scope note). The raw `axis`
score is a pure per-poster value; register/decade binning is a downstream
concern.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.clip_backbone import get_tokenizer, load_clip, text_prototype
from utils.logging_setup import get_logger

log = get_logger("clip_fear_axis")

DREAD_PROMPTS = [
    "a deeply unsettling, ominous movie poster that fills the viewer with dread and unease",
    "a horror movie poster with a dark, foreboding, nightmarish atmosphere",
    "a movie poster conveying an unmistakable sense of impending doom and terror",
    "a disturbing movie poster designed to frighten and horrify the viewer",
]
CALM_PROMPTS = [
    "a bright, cheerful, reassuring movie poster with no sense of danger",
    "an upbeat, lighthearted movie poster with a safe, pleasant mood",
    "a movie poster showing a peaceful, comforting, everyday scene",
    "a movie poster with a calm, serene, unthreatening atmosphere",
]

# hand-verified ground truth: +1 extreme dread, -1 calm/light, 0 in between
VALIDATION = [
    ("Hereditary", 2018, +1), ("The Exorcist", 1973, +1), ("Sinister", 2012, +1),
    ("The Conjuring", 2013, +1), ("Terrifier", 2016, +1), ("Get Out", 2017, +1),
    ("Knives Out", 2019, -1), ("Clue", 1985, -1), ("The Nice Guys", 2016, -1),
    ("A Simple Favor", 2018, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/clip_embeddings.npz")
    ap.add_argument("--out", default="data/sample_output/fear_axis.csv")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_clip(device)
    tok = get_tokenizer()
    DREAD = text_prototype(model, tok, DREAD_PROMPTS, device)
    CALM = text_prototype(model, tok, CALM_PROMPTS, device)

    z = np.load(args.embeddings)
    ids, vecs = z["ids"].astype(int), z["vecs"].astype(np.float32)
    axis = (vecs @ DREAD) - (vecs @ CALM)
    df = pd.DataFrame(dict(id=ids, axis=axis))
    meta = pd.read_csv(args.in_path, usecols=["id", "year", "title"])
    df = df.merge(meta, on="id")

    if args.validate:
        want = {(t, y): w for t, y, w in VALIDATION}
        print(f'{"film":18}{"want":6}{"axis":>10}')
        wants, axes = [], []
        for (t, y), w in want.items():
            m = df[(df.title == t) & (df.year == y)]
            if not len(m):
                print(f"{t:18} NOT FOUND")
                continue
            a = float(m.iloc[0]["axis"])
            tag = "dread" if w > 0 else "calm" if w < 0 else "mid"
            print(f"{t:18}{tag:6}{a:+10.4f}")
            wants.append(w)
            axes.append(a)
        r = np.corrcoef(wants, axes)[0, 1] if len(set(wants)) > 1 else float("nan")
        print(f"\ncorr(want, axis) = {r:+.3f}  (real full-corpus run: >=0.6 to trust)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
