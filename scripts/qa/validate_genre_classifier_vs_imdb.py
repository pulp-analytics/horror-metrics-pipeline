#!/usr/bin/env python3
"""Validates 09_clip_genre_classifier.py's zero-shot predictions against
IMDb's own genre tags -- the direct analog to the sibling
poster-corpus-validation repo's `04_bedrock_ocr.py --validate` (see that
repo's docs/MODELS.md, "Building a human ground-truth set"), but for a
question IMDb can answer automatically: unlike "does this poster's visible
title match the catalog" (only a human can judge that), genre is
already-curated metadata two independent catalogs both maintain, so no
human review step is needed here at all.

Ground truth source: IMDb's public, free, non-commercial title.basics.tsv.gz
(https://datasets.imdbws.com/title.basics.tsv.gz -- no API key, no AWS,
matching this repo's existing "no credentials needed" scope). Auto-
downloaded once and cached locally, same pattern as 14_face_detect.py's
YuNet model download.

--in needs an imdb_id column (data/ground_truth/genre_classifier_sample.csv
is a real 200-poster sample -- 50 each of horror/scifi/thriller/mystery,
drawn from the real project's own historical CLIP predictions with a fixed
seed -- that already has one; this repo's own data/sample_input/
sample_100_posters.csv does NOT, since fetching imdb_id needs TMDB's
authenticated API, which is out of scope for this repo by design -- see
README's "No API key or AWS needed anywhere in this repo." Join imdb_id in
yourself from wherever you already have it, e.g. the sibling repo's
03_fetch_alt_titles.py.)

IMDb genres are multi-label (a film can be "Horror,Sci-Fi,Thriller" at
once); this script's classifier is forced single-label. Two different
questions follow from that, both reported:
  - precision per predicted class: of the posters CLIP called "scifi", how
    many does IMDb's genre list actually contain "Sci-Fi" for (lenient --
    a multi-genre film with Sci-Fi anywhere in its list counts as a hit)
  - recall per true class: of the posters IMDb tags "Sci-Fi" (regardless
    of what else it's also tagged), how many did CLIP predict as "scifi"
    specifically (strict -- this is the number that catches "CLIP saw a
    real sci-fi poster and called it horror instead")

  python3 scripts/qa/validate_genre_classifier_vs_imdb.py
  python3 scripts/qa/validate_genre_classifier_vs_imdb.py --imdb-basics /path/to/title.basics.tsv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import sys
import urllib.request
from pathlib import Path

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utils.clip_backbone import get_tokenizer, load_clip, text_prototype
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import write_csv_rows

# GENRE_PROMPTS lives in 09_clip_genre_classifier.py, not duplicated here --
# imported by path since a leading digit makes it an invalid module name.
_spec = importlib.util.spec_from_file_location("clip_genre_classifier", ROOT / "scripts" / "09_clip_genre_classifier.py")
_genre_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_genre_mod)
GENRE_PROMPTS = _genre_mod.GENRE_PROMPTS

log = get_logger("validate_genre_vs_imdb")

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
DEFAULT_IMDB_BASICS_PATH = "data/ground_truth/.imdb_cache/title.basics.tsv.gz"

# CLIP's label -> IMDb's own spelling of the same genre in its `genres` column
GENRE_MAP = {"horror": "Horror", "scifi": "Sci-Fi", "thriller": "Thriller", "mystery": "Mystery"}
CLASSES = list(GENRE_MAP)

# Matches the sibling private-pipeline QA script's own convention
# (compare_tmdb_imdb_horror.py's STRICT_TYPES) -- shorts/videos/TV
# episodes are a different comparison than feature-film posters, not
# just "more data," so they're excluded by default rather than silently
# folded in.
STRICT_TITLE_TYPES = frozenset({"movie", "tvMovie"})


def ensure_imdb_basics(path: Path) -> None:
    if path.exists():
        return
    log.info(f"downloading IMDb title.basics.tsv.gz (~225MB, public non-commercial dataset) to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(IMDB_BASICS_URL, path)
    log.info("downloaded")


def load_imdb_info(basics_path: Path, wanted_tconsts: set[str]) -> dict[str, dict]:
    """One streaming pass over title.basics.tsv.gz, keeping only genres +
    titleType for the tconsts we actually need -- the file covers every
    IMDb title ever (10M+ rows), not just ours."""
    out: dict[str, dict] = {}
    with gzip.open(basics_path, "rt", encoding="utf-8", errors="replace") as f:
        header = next(f).rstrip("\n").split("\t")
        tt_i = header.index("tconst")
        type_i = header.index("titleType")
        g_i = header.index("genres")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tt = parts[tt_i]
            if tt in wanted_tconsts:
                raw = parts[g_i]
                out[tt] = {
                    "titleType": parts[type_i],
                    "genres": set() if (not raw or raw == "\\N") else set(raw.split(",")),
                }
    return out


def compute_genre_metrics(rows: list[dict]) -> dict:
    """Pure function (no I/O): rows are {pred_genre, imdb_genres} where
    imdb_genres is a set of IMDb genre strings. Returns containment
    precision per predicted class and strict recall per true class, plus
    a full predicted-as confusion breakdown for each true class.

    Rows where IMDb tags none of horror/sci-fi/thriller/mystery at all are
    excluded from every count, not just flagged -- inspecting these by hand
    (see the sibling poster-corpus-validation repo's ground-truth work for
    the same judgment call re: "unjudgeable") showed they're a mix of
    content CLIP was never going to get right regardless of model quality
    (shorts, documentaries, animation) and older/obscure titles where
    IMDb's own genre tagging looks incomplete (several "predicted mystery"
    misses here are actually Crime/Drama/Film-Noir films that read as
    mystery-adjacent on inspection) -- neither is evidence the classifier
    is wrong, so forcing them into the denominator would blame the model
    for gaps in the ground truth."""
    valid = [r for r in rows if r["pred_genre"] in GENRE_MAP]
    no_target_genre = [r for r in valid if not (r["imdb_genres"] & set(GENRE_MAP.values()))]
    scored = [r for r in valid if r["imdb_genres"] & set(GENRE_MAP.values())]

    per_class_precision = {}
    for c in CLASSES:
        sub = [r for r in scored if r["pred_genre"] == c]
        hits = sum(1 for r in sub if GENRE_MAP[c] in r["imdb_genres"]) if sub else 0
        per_class_precision[c] = {
            "n_predicted": len(sub),
            "hits": hits,
            "precision": hits / len(sub) if sub else None,
        }

    per_class_recall = {}
    for c in CLASSES:
        imdb_label = GENRE_MAP[c]
        sub = [r for r in scored if imdb_label in r["imdb_genres"]]
        correct = sum(1 for r in sub if r["pred_genre"] == c)
        breakdown = {pc: sum(1 for r in sub if r["pred_genre"] == pc) for pc in CLASSES}
        per_class_recall[c] = {
            "n_true": len(sub),
            "recall": correct / len(sub) if sub else None,
            "predicted_as": breakdown,
        }

    overall_hits = sum(1 for r in scored if GENRE_MAP[r["pred_genre"]] in r["imdb_genres"])

    return {
        "n_scored": len(scored),
        "n_no_target_genre": len(no_target_genre),
        "overall_containment_accuracy": overall_hits / len(scored) if scored else None,
        "precision": per_class_precision,
        "recall": per_class_recall,
    }


def print_report(metrics: dict) -> None:
    n = metrics["n_scored"]
    if n == 0:
        log.info("nothing scored -- can't report metrics")
        return

    print(f"\nScored: {n} posters ({metrics['n_no_target_genre']} excluded -- IMDb tags none of "
          f"horror/sci-fi/thriller/mystery for them at all, so they can't be attributed to the "
          f"classifier being wrong)")
    acc = metrics["overall_containment_accuracy"]
    print(f"Overall containment accuracy: {acc*100:.1f}%\n")

    print(f'{"predicted class":16}{"n_predicted":>13}{"precision":>12}')
    for c in CLASSES:
        p = metrics["precision"][c]
        prec = f"{p['precision']*100:.1f}%" if p["precision"] is not None else "n/a"
        print(f"{c:16}{p['n_predicted']:>13}{prec:>12}")

    print(f'\n{"true class (IMDb)":18}{"n_true":>9}{"recall":>9}   predicted-as breakdown')
    for c in CLASSES:
        r = metrics["recall"][c]
        if r["n_true"] == 0:
            continue
        rec = f"{r['recall']*100:.1f}%" if r["recall"] is not None else "n/a"
        bstr = ", ".join(f"{pc}={r['predicted_as'][pc]} ({100*r['predicted_as'][pc]/r['n_true']:.0f}%)" for pc in CLASSES)
        print(f"{c:18}{r['n_true']:>9}{rec:>9}   {bstr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/ground_truth/genre_classifier_sample.csv")
    ap.add_argument("--out", default="data/ground_truth/genre_classifier_validate_results.csv")
    ap.add_argument("--imdb-basics", default=DEFAULT_IMDB_BASICS_PATH,
                     help="local path to IMDb's title.basics.tsv.gz -- auto-downloaded here if missing")
    ap.add_argument("--include-shorts", action="store_true",
                     help="include shorts/videos/TV episodes/etc, not just movie/tvMovie "
                          "(default: movie/tvMovie only, matching the sibling private-pipeline "
                          "QA script's own convention -- a short's poster isn't really the same "
                          "comparison as a feature film's)")
    add_poster_source_args(ap)
    args = ap.parse_args()

    with open(args.in_path, newline="", encoding="utf-8") as f:
        in_rows = list(csv.DictReader(f))
    if in_rows and "imdb_id" not in in_rows[0]:
        raise SystemExit(f"{args.in_path} has no imdb_id column -- see this script's docstring")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(device)
    tok = get_tokenizer()
    protos = {g: text_prototype(model, tok, prompts, device) for g, prompts in GENRE_PROMPTS.items()}
    genres = list(protos)
    P = np.stack([protos[g] for g in genres])

    posters_dir = Path(args.posters_dir)
    session = requests.Session()
    from PIL import Image

    results = []
    for i, row in enumerate(in_rows, 1):
        poster_file = posters_dir / f"{row['id']}.jpg"
        if not fetch_poster_file(session, row["poster_path"], poster_file,
                                  args.posters_s3_bucket, args.posters_s3_prefix):
            log.info(f"  {row['id']}: poster fetch failed, skipping")
            continue
        img = preprocess(Image.open(poster_file).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(img)
            vec = (vec / vec.norm(dim=-1, keepdim=True))[0].cpu().numpy()
        sims = vec @ P.T
        pred_genre = genres[int(sims.argmax())]
        results.append({"id": row["id"], "title": row.get("title", ""), "imdb_id": row["imdb_id"],
                         "pred_genre": pred_genre})
        if i % 25 == 0 or i == len(in_rows):
            log.info(f"{i}/{len(in_rows)} classified")

    imdb_basics = Path(args.imdb_basics)
    ensure_imdb_basics(imdb_basics)
    wanted = {r["imdb_id"] for r in results}
    imdb_info = load_imdb_info(imdb_basics, wanted)
    log.info(f"IMDb info found for {len(imdb_info)}/{len(wanted)} imdb_ids")

    for r in results:
        info = imdb_info.get(r["imdb_id"], {})
        r["imdb_genres_set"] = info.get("genres", set())
        r["imdb_genres"] = ",".join(sorted(r["imdb_genres_set"]))
        r["imdb_title_type"] = info.get("titleType", "")

    out_path = Path(args.out)
    write_csv_rows(out_path, [{"id": r["id"], "title": r["title"], "imdb_id": r["imdb_id"],
                                "pred_genre": r["pred_genre"], "imdb_genres": r["imdb_genres"],
                                "imdb_title_type": r["imdb_title_type"]}
                               for r in results])
    log.info(f"wrote {out_path} ({len(results)} rows)")

    for_metrics = results
    if not args.include_shorts:
        before = len(for_metrics)
        for_metrics = [r for r in for_metrics if r["imdb_title_type"] in STRICT_TITLE_TYPES]
        log.info(f"--include-shorts not set: {before - len(for_metrics)} row(s) excluded for a "
                 f"non-movie/tvMovie IMDb titleType (shorts, videos, TV episodes, or no IMDb match at all)")

    metrics = compute_genre_metrics([{"pred_genre": r["pred_genre"], "imdb_genres": r["imdb_genres_set"]} for r in for_metrics])
    print_report(metrics)


if __name__ == "__main__":
    main()
