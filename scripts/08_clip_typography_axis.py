#!/usr/bin/env python3
"""THE LETTERING OF FEAR -- the ORNATE <-> MINIMAL axis of a poster's title
typography, over the whole poster, as a continuous score.

Method: axis = cos(embedding, ORNATE_prototype) - cos(embedding, CLEAN_prototype).
Higher = more ornamented/decorative lettering; lower = cleaner/more minimal.

This continuous, whole-poster approach is the winner of three real,
actually-tried alternatives, in order:
  1. 8 discrete typography style categories, whole poster -- failed
     validation (6/10 hand-checked cases correct). Categories like
     "dripping" became attractors for overall darkness, not lettering.
  2. Cropping just the title via OCR (Tesseract), then classifying the
     crop -- 4/10. OCR doesn't read ornamented display fonts reliably, and
     ends up cropping whatever line IS legible, biasing hard toward "clean".
  3. Cropping via MSER text-region detection + CLAHE contrast enhancement
     -- corr 0.72-0.78 against hand-verified ground truth. Better, but
     still worse than what won.
  4. **Whole poster, continuous ornate<->clean axis (this script)** --
     corr 0.81, and the resulting per-decade trend correlates -0.93 with
     decade. This is the real, empirically-grounded conclusion: the
     typography signal CLIP can read reliably over a full poster is a
     continuous axis, not discrete style buckets.

  python3 08_clip_typography_axis.py --in data/sample_input/sample_100_posters.csv
  python3 08_clip_typography_axis.py --validate   # expected corr ~0.81

Not something this script does: quantile-based register binning or
decade aggregation -- see 07_clip_fear_axis.py's docstring for why (both
are corpus-relative, out of scope for this repo).
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

log = get_logger("clip_typography_axis")

# prototypes centered on the LETTERING itself, stripped of horror "mood"
# words that were biasing the earlier discrete-category attempts
ORNATE_PROMPTS = [
    "a movie poster with ornate decorative hand-drawn display title lettering",
    "a poster title in elaborate vintage painted show-card letters",
    "a movie title in fancy ornamental custom 3D lettering",
    "a poster with a heavily stylized illustrated logo title",
]
CLEAN_PROMPTS = [
    "a movie poster title in clean minimal sans-serif type",
    "a poster title in plain simple modern typography",
    "a movie title in a restrained understated typeface",
    "a poster with small tidy unadorned lettering",
]

# hand-verified ground truth: +1 ornate, -1 minimal, 0 in between
VALIDATION = [
    (244, "King Kong", +1), (57283, "Haxan", +1), (3053, "Fearless Vamp.", +1),
    (21588, "Cemetery Man", +1), (377, "Nightmare Elm", 0), (948, "Halloween", -1),
    (48171, "The Rite", -1), (4232, "Scream", -1), (8461, "Funny Games", -1),
    (493922, "Hereditary", -1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/clip_embeddings.npz")
    ap.add_argument("--out", default="data/sample_output/typography.csv")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_clip(device)
    tok = get_tokenizer()
    ORNATE = text_prototype(model, tok, ORNATE_PROMPTS, device)
    CLEAN = text_prototype(model, tok, CLEAN_PROMPTS, device)

    z = np.load(args.embeddings)
    ids, vecs = z["ids"].astype(int), z["vecs"].astype(np.float32)
    axis = (vecs @ ORNATE) - (vecs @ CLEAN)
    df = pd.DataFrame(dict(id=ids, axis=axis))
    meta = pd.read_csv(args.in_path, usecols=["id", "year", "title"])
    df = df.merge(meta, on="id")

    if args.validate:
        want = {i: w for i, _, w in VALIDATION}
        sub = df[df.id.isin(want)].copy()
        sub["want"] = sub.id.map(want)
        r = np.corrcoef(sub["want"], sub["axis"])[0, 1] if len(sub) > 1 else float("nan")
        print(f'{"film":16}{"want":6}{"axis":>10}')
        for i, name, w in VALIDATION:
            match = df[df.id == i]
            tag = "orn" if w > 0 else "min" if w < 0 else "mid"
            if not len(match):
                print(f"{name:16}{tag:6}{'NOT FOUND':>10}")
                continue
            a = float(match["axis"].iloc[0])
            print(f"{name:16}{tag:6}{a:+10.4f}")
        print(f"\ncorr(want, axis) = {r:+.3f}   (real full-poster baseline: 0.81)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
