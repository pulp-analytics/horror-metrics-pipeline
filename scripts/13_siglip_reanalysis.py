#!/usr/bin/env python3
"""Census + typography + genre_classifier, SigLIP version -- same
taxonomy/prompts as the CLIP originals (06_clip_census.py,
08_clip_typography_axis.py, 09_clip_genre_classifier.py), re-run on
11_siglip_embed.py's embeddings. One script, one model load shared across
all three (the real siglip_reanalysis.py combines them for exactly this
reason: avoids paying SigLIP's load cost three times).

  python3 13_siglip_reanalysis.py --in data/sample_input/sample_100_posters.csv

Not something this script does: quantile-based typography register
binning -- same reasoning as every other axis script in this repo (see
the README's Scope note). The real siglip_reanalysis.py computes it;
ported here through the continuous axis only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.siglip_backbone import load_siglip, text_prototype

log = get_logger("siglip_reanalysis")

# identical taxonomy to 06_clip_census.py -- same labels, same real
# per-label prompt sets from the private pipeline's siglip_reanalysis.py
CENSUS_TAXONOMY = {
    "vampire": ["a horror movie poster featuring a vampire with fangs",
                "a dracula movie poster, vampire, cape, fangs"],
    "werewolf": ["a horror poster featuring a werewolf",
                 "a wolf-man creature on a movie poster"],
    "zombie": ["a horror poster featuring zombies, undead corpses",
               "rotting undead zombie faces on a movie poster"],
    "ghost": ["a horror poster featuring a ghost or spectral figure",
              "a pale spectral apparition on a movie poster"],
    "demon": ["a horror poster featuring a demon or the devil",
              "a demonic possessed figure on a movie poster"],
    "witch": ["a horror poster featuring a witch",
              "occult witchcraft imagery on a movie poster"],
    "skeleton": ["a horror poster featuring a skull or skeleton",
                 "a large skull on a movie poster"],
    "alien": ["a horror poster featuring an alien creature",
              "an extraterrestrial monster on a movie poster"],
    "giant_monster": ["a poster featuring a giant monster attacking, kaiju",
                       "a giant creature destroying a city on a movie poster"],
    "masked_killer": ["a horror poster featuring a masked killer with a weapon",
                       "a slasher villain in a mask on a movie poster"],
    "clown": ["a horror poster featuring an evil clown"],
    "doll": ["a horror poster featuring a creepy doll or puppet"],
    "shark": ["a poster featuring a shark attacking"],
    "spider": ["a poster featuring a giant spider"],
    "snake": ["a poster featuring a snake or serpent attacking"],
    "wolf_dog": ["a poster featuring a menacing dog or wolf (real animal)"],
    "bird": ["a poster featuring attacking birds"],
    "insect": ["a poster featuring insects or bugs swarming"],
    "none": ["a movie poster with only ordinary people, no monster or creature",
             "a movie poster showing a house or landscape, no creature",
             "a movie poster with plain typography, no monster"],
}

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


def run_census(model, processor, ids, vecs, meta, min_score, out_path):
    protos = {label: text_prototype(model, processor, prompts) for label, prompts in CENSUS_TAXONOMY.items()}
    labels = list(protos)
    P = np.stack([protos[l] for l in labels])
    sims = vecs @ P.T
    probs = torch.softmax(torch.tensor(sims * 100.0), dim=1).numpy()
    top = probs.argmax(1)
    df = pd.DataFrame(dict(id=ids, label=[labels[i] for i in top], score=probs.max(1).round(3)))
    df.loc[df.score < min_score, "label"] = "uncertain"
    df = df.merge(meta[["id", "year", "title"]], on="id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"census -> {out_path} ({len(df):,} rows)")
    return df


def run_typography(model, processor, ids, vecs, meta, out_path):
    ORNATE = text_prototype(model, processor, ORNATE_PROMPTS)
    CLEAN = text_prototype(model, processor, CLEAN_PROMPTS)
    axis = (vecs @ ORNATE) - (vecs @ CLEAN)
    df = pd.DataFrame(dict(id=ids, axis=axis))
    df = df.merge(meta[["id", "year", "title"]], on="id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"typography -> {out_path} ({len(df):,} rows)")
    return df


def run_genre(model, processor, ids, vecs, meta, true_genre_col, out_path):
    protos = {g: text_prototype(model, processor, prompts) for g, prompts in GENRE_PROMPTS.items()}
    genres = list(protos)
    P = np.stack([protos[g] for g in genres])
    sims = vecs @ P.T
    pred_idx = sims.argmax(1)
    pred_genre = [genres[i] for i in pred_idx]
    df = pd.DataFrame({"id": ids, "pred_genre": pred_genre})
    if true_genre_col in meta.columns:
        genre_meta = meta[["id", true_genre_col]].rename(columns={true_genre_col: "true_genre"})
        df = df.merge(genre_meta, on="id", how="left")
        df["agree"] = df["pred_genre"] == df["true_genre"]
        log.info(f"agreement with catalog genre: {df['agree'].mean():.1%}")
    else:
        log.info(f"no '{true_genre_col}' column in --in -- only pred_genre computed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"genre -> {out_path} ({len(df):,} rows)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/siglip_embeddings.npz")
    ap.add_argument("--census-out", default="data/sample_output/siglip_census.csv")
    ap.add_argument("--typography-out", default="data/sample_output/siglip_typography.csv")
    ap.add_argument("--genre-out", default="data/sample_output/siglip_genre_classifier.csv")
    ap.add_argument("--min-score", type=float, default=0.5,
                     help="census: below this, label becomes 'uncertain' instead of a low-confidence guess")
    ap.add_argument("--true-genre-col", default="genre",
                     help="optional column in --in with the catalog genre, to compute 'agree'")
    args = ap.parse_args()

    log.info("loading SigLIP...")
    model, processor = load_siglip()
    log.info("model ready")

    z = np.load(args.embeddings)
    ids, vecs = z["ids"].astype(int), z["vecs"].astype(np.float32)
    meta = pd.read_csv(args.in_path)

    run_census(model, processor, ids, vecs, meta, args.min_score, Path(args.census_out))
    run_typography(model, processor, ids, vecs, meta, Path(args.typography_out))
    run_genre(model, processor, ids, vecs, meta, args.true_genre_col, Path(args.genre_out))


if __name__ == "__main__":
    main()
