#!/usr/bin/env python3
"""THE MONSTER CENSUS -- zero-shot creature/monster taxonomy over cached
CLIP embeddings (05_clip_embed.py). Runs in seconds, no image reprocessing.

Method: a prompt-ensemble text prototype per label (see TAXONOMY below,
2-3 phrasings per label, averaged); a poster's embedding gets whichever
label's prototype it's cosine-closest to, via softmax over all labels at
once (temperature 100, sharpens the softmax so the top label usually
dominates unless the image genuinely sits between two labels). Below
--min-score (default 0.5), the label is downgraded to "uncertain" instead
of trusting a low-confidence guess.

  python3 06_clip_census.py --in data/sample_input/sample_100_posters.csv
  python3 06_clip_census.py --validate    # sanity-check against famous, hand-verified posters

Per-poster labels are noisy by design (this is zero-shot, not a trained
classifier) -- see docs/METHODOLOGY.md for what "noisy per-poster, but
correct in aggregate" means here and why that's an acceptable tradeoff.
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

log = get_logger("clip_census")

TAXONOMY = {
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
ANIMALS = {"shark", "spider", "snake", "wolf_dog", "bird", "insect"}

# hand-verified by actually looking at the artwork -- real validation set,
# not a formal benchmark. Only useful if --in/--embeddings happen to cover
# these specific (title, year) pairs.
VALIDATION = [
    ("Godzilla", 1954, {"giant_monster"}),
    ("Jaws", 1975, {"shark"}),
    ("An American Werewolf in London", 1981, {"werewolf"}),
    ("Night of the Living Dead", 1968, {"zombie"}),
    ("Arachnophobia", 1990, {"spider", "none"}),  # tiny spider against the moon; low-conf "none" is fine
    ("It", 2017, {"clown", "doll", "none"}),  # artwork is a kid + balloon, Pennywise is a shadow
    ("Annabelle", 2014, {"doll"}),
    ("Halloween", 1978, {"skeleton", "masked_killer", "none"}),  # knife + jack-o'-lantern, no visible killer
    ("The Birds", 1963, {"bird"}),
    ("Get Out", 2017, {"none"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--embeddings", default="data/sample_output/clip_embeddings.npz")
    ap.add_argument("--out", default="data/sample_output/census.csv")
    ap.add_argument("--temp", type=float, default=100.0, help="softmax temperature")
    ap.add_argument("--min-score", type=float, default=0.5,
                     help="below this, label becomes 'uncertain' instead of a low-confidence guess")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_clip(device)
    tok = get_tokenizer()
    log.info("computing label prototypes...")
    protos = {label: text_prototype(model, tok, prompts, device) for label, prompts in TAXONOMY.items()}
    labels = list(protos)
    P = np.stack([protos[l] for l in labels])

    z = np.load(args.embeddings)
    ids, vecs = z["ids"], z["vecs"].astype(np.float32)
    meta = pd.read_csv(args.in_path, usecols=["id", "year", "title"])

    sims = vecs @ P.T
    probs = torch.softmax(torch.tensor(sims * args.temp), dim=1).numpy()
    top = probs.argmax(1)

    df = pd.DataFrame(dict(id=ids, label=[labels[i] for i in top], score=probs.max(1).round(3)))
    df.loc[df.score < args.min_score, "label"] = "uncertain"
    df = df.merge(meta, on="id")
    df["is_animal"] = df.label.isin(ANIMALS)
    df["is_creature"] = ~df.label.isin(["none", "uncertain"])

    if args.validate:
        print(f'{"title":38} expected -> detected (score)')
        ok = 0
        found = 0
        for t, y, exp in VALIDATION:
            m = df[(df.title == t) & (df.year == y)]
            if not len(m):
                print(f"{t:38} NOT FOUND")
                continue
            found += 1
            r = m.iloc[0]
            hit = "OK" if (r.label in exp or (r.label == "uncertain" and "none" in exp)) else "FAIL"
            ok += hit == "OK"
            print(f"{t:38} {'/'.join(sorted(exp)):28} -> {r.label:14} ({r.score:.2f}) {hit}")
        print(f"VALIDATION: {ok}/{found} found (of {len(VALIDATION)} total in the validation set)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
