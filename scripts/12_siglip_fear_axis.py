#!/usr/bin/env python3
"""THE FEAR AXIS, SigLIP version -- same method as 07_clip_fear_axis.py
(continuous dread<->calm projection: cos(embedding, DREAD_prototype) -
cos(embedding, CALM_prototype)), same prompt wording, run over
11_siglip_embed.py's embeddings instead of CLIP's -- to compare whether
the model with better zero-shot accuracy gives a cleaner signal.

  python3 12_siglip_fear_axis.py --in data/sample_input/sample_100_posters.csv
  python3 12_siglip_fear_axis.py --validate   # ranking of known specimens

Not something this script does: quantile-based register binning or decade
aggregation -- same reasoning as 07_clip_fear_axis.py (both are
corpus-relative, out of scope for this repo; see the README's Scope
note). The real siglip_fear_axis.py script does compute both -- ported
here through the continuous axis only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.siglip_backbone import load_siglip, text_prototype

log = get_logger("siglip_fear_axis")

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
# -- same set the real siglip_fear_axis.py uses (a shorter list than
# 07_clip_fear_axis.py's CLIP version: "The Nice Guys" isn't in it)
VALIDATION = [
    ("Hereditary", 2018, +1), ("The Exorcist", 1973, +1), ("Sinister", 2012, +1),
    ("The Conjuring", 2013, +1), ("Terrifier", 2016, +1), ("Get Out", 2017, +1),
    ("Knives Out", 2019, -1), ("Clue", 1985, -1), ("A Simple Favor", 2018, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/siglip_embeddings.npz")
    ap.add_argument("--out", default="data/sample_output/siglip_fear_axis.csv")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    model, processor = load_siglip()
    DREAD = text_prototype(model, processor, DREAD_PROMPTS)
    CALM = text_prototype(model, processor, CALM_PROMPTS)

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
        print(f"\ncorr(want, axis) = {r:+.3f}  (compare against 0.611 for the CLIP version)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
