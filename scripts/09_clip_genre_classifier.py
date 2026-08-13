#!/usr/bin/env python3
"""CLIP zero-shot genre classifier: does this poster "look like" its genre,
based purely on the artwork -- no title, no metadata, just the image?

Text prototypes per genre (a few phrasings each, averaged), cosine
similarity against each poster's cached CLIP embedding. Outputs a
similarity score against all four genres plus the predicted (highest-
similarity) genre for every poster, regardless of what its catalog genre
actually is -- if your --in file happens to carry a `genre` column (the
real project ran this once per single-genre corpus: horror, scifi,
thriller, mystery), an `agree` column also gets added comparing predicted
vs. that catalog genre.

  python3 09_clip_genre_classifier.py --in data/sample_input/sample_100_posters.csv
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

log = get_logger("clip_genre_classifier")

GENRE_PROMPTS = {
    "horror": [
        "a horror movie poster, scary and menacing imagery",
        "a horror film poster with blood, monsters, or a masked killer",
        "a dark scary movie poster meant to frighten",
    ],
    "scifi": [
        "a science fiction movie poster with futuristic technology or space",
        "a sci-fi film poster featuring spaceships, aliens, or robots",
        "a science fiction poster with a futuristic or otherworldly setting",
    ],
    "thriller": [
        "a thriller movie poster, tense and suspenseful imagery",
        "a crime thriller poster with a gun, chase, or shadowy figure",
        "a suspense thriller film poster, dramatic and tense",
    ],
    "mystery": [
        "a mystery movie poster with a detective or an unsolved puzzle",
        "a mystery film poster, intriguing and enigmatic imagery",
        "a whodunit mystery poster with clues or investigation themes",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/clip_embeddings.npz")
    ap.add_argument("--out", default="data/sample_output/genre_classifier.csv")
    ap.add_argument("--true-genre-col", default="genre",
                     help="optional column in --in with the catalog genre, to compute 'agree'")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_clip(device)
    tok = get_tokenizer()
    protos = {g: text_prototype(model, tok, prompts, device) for g, prompts in GENRE_PROMPTS.items()}
    genres = list(protos)
    P = np.stack([protos[g] for g in genres])

    z = np.load(args.embeddings)
    ids, vecs = z["ids"].astype(int), z["vecs"].astype(np.float32)
    sims = vecs @ P.T
    pred_idx = sims.argmax(1)
    pred_genre = [genres[i] for i in pred_idx]

    rows = []
    for pid, pg, sim_row in zip(ids, pred_genre, sims):
        rows.append({"id": pid, "pred_genre": pg,
                      **{f"sim_{g}": round(float(sim_row[i]), 4) for i, g in enumerate(genres)}})
    df = pd.DataFrame(rows)

    in_cols = pd.read_csv(args.in_path, nrows=0).columns
    if args.true_genre_col in in_cols:
        meta = pd.read_csv(args.in_path, usecols=["id", args.true_genre_col])
        meta = meta.rename(columns={args.true_genre_col: "true_genre"})
        df = df.merge(meta, on="id", how="left")
        df["agree"] = df["pred_genre"] == df["true_genre"]
        log.info(f"agreement with catalog genre: {df['agree'].mean():.1%}")
    else:
        log.info(f"no '{args.true_genre_col}' column in --in -- only pred_genre/similarities computed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
