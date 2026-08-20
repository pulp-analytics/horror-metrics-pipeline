#!/usr/bin/env python3
"""Optional, downstream-only join of this repo's per-metric CSVs into one
master_dataset.csv. Deliberately NOT part of the pipeline itself (see
README's "Scope" note) -- this repo's contract is one file per metric,
one row per poster; this script is for whoever wants a single flat table
instead of joining the pieces themselves.

Usage:
  python3 assemble_master_dataset.py --data-dir data/prodtest --out data/prodtest/master_dataset.csv

Every *.csv in --data-dir except --base and --skip files is treated as a
metric file and left-joined onto the base corpus list by "id". Each
metric file's columns (other than id/title/year, which are redundant
with the base) are prefixed with "<stem>_" so same-named columns across
different metric files (e.g. creature_weapon_owlv2.csv,
creature_weapon_dino.csv, and creature_weapon_agreement.csv all have
"creature_n") never collide.

face_expression.csv is a special case: unlike every other file, it has
multiple rows per poster (one per detected face), so it can't be
left-joined 1:1. It's aggregated to one row per id first: a count and a
";"-joined "label:score" summary per poster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# files that define the corpus itself, not a metric -- read once as the base
BASE_CANDIDATES = ["validated_corpus.csv", "metrics_input.csv"]
# not real per-poster metric tables
SKIP_BY_DEFAULT = {"master_dataset.csv"}
PER_FACE_FILES = {"face_expression.csv"}


def load_base(data_dir: Path, base_name: str | None) -> pd.DataFrame:
    if base_name:
        path = data_dir / base_name
        if not path.exists():
            raise SystemExit(f"--base {base_name} not found in {data_dir}")
    else:
        path = next((data_dir / n for n in BASE_CANDIDATES if (data_dir / n).exists()), None)
        if path is None:
            raise SystemExit(f"no base file found in {data_dir} (tried {BASE_CANDIDATES}); pass --base")
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].astype(str)
    return df.drop_duplicates("id")


def aggregate_per_face(df: pd.DataFrame, stem: str) -> pd.DataFrame:
    """One row per poster: face count + a ';'-joined label:score summary."""
    df = df.copy()
    df["id"] = df["id"].astype(str)

    def summarize(g: pd.DataFrame) -> pd.Series:
        parts = [f"{row.label}:{row.score}" for row in g.itertuples() if pd.notna(row.label)]
        return pd.Series({f"{stem}_n": len(g), f"{stem}_summary": ";".join(parts)})

    return df.groupby("id").apply(summarize).reset_index()


def load_metric_file(path: Path) -> pd.DataFrame:
    stem = path.stem
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].astype(str)

    if path.name in PER_FACE_FILES:
        return aggregate_per_face(df, stem)

    df = df.drop_duplicates("id")
    rename = {c: f"{stem}_{c}" for c in df.columns if c not in ("id", "title", "year")}
    df = df.drop(columns=[c for c in ("title", "year") if c in df.columns])
    return df.rename(columns=rename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--base", default="", help="corpus-definition CSV (default: auto-detect validated_corpus.csv / metrics_input.csv)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip", nargs="*", default=[], help="extra filenames in --data-dir to exclude from the join")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    base_name = args.base or None
    base = load_base(data_dir, base_name)
    print(f"base corpus: {len(base):,} ids from {base_name or '(auto-detected)'}")

    skip = SKIP_BY_DEFAULT | set(BASE_CANDIDATES) | set(args.skip)
    metric_files = sorted(p for p in data_dir.glob("*.csv") if p.name not in skip)

    merged = base
    for path in metric_files:
        metric = load_metric_file(path)
        before = merged.shape[1]
        merged = merged.merge(metric, on="id", how="left")
        print(f"  + {path.name}: {len(metric):,} rows, {merged.shape[1] - before} new columns")

    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged):,} rows, {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
