#!/usr/bin/env python3
"""Aggregates 01_color_metrics.py's per-poster output into the two chart
datasets and the go/no-go checkpoint this whole project's first phase
turned on.

This is the actual "Continue / Pivot" decision point from the real
project: before building anything else, the question was whether horror
posters actually get darker after a certain era, or whether that's just a
cultural assumption not borne out in the data. If the brightness gap
between early and modern decades is weak or absent, that's a signal to
rethink the premise, not push forward -- see docs/RESULTS.md for what the
real full-corpus run found.

Outputs:
  yearly.json      mean brightness/dark_share/saturation/red_share per year
  hue_river.json   mean hue-family share per decade (the "Color River")
  darkness_curve.png   5-year rolling mean brightness, if matplotlib is installed
  a CONTINUE/PIVOT verdict printed to the console

  python3 02_aggregate_and_checkpoint.py --in data/sample_output/color_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger

log = get_logger("aggregate_and_checkpoint")

BAND_COLS = ["band_red", "band_warm", "band_green", "band_blue", "band_purple", "band_dark"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_output/color_metrics.csv")
    ap.add_argument("--out-dir", default="data/sample_output")
    ap.add_argument("--chart", default="", help="path for the darkness-curve PNG (skipped if not set)")
    args = ap.parse_args()

    res = pd.read_csv(args.in_path)
    if res.empty:
        sys.exit(f"{args.in_path} has no rows -- run 01_color_metrics.py first")
    res["year"] = res["year"].astype(int)
    res["decade"] = (res.year // 10) * 10
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yearly = (res.groupby("year")
                 .agg(n=("id", "count"), brightness=("brightness", "mean"),
                      dark_share=("dark_share", "mean"),
                      saturation=("saturation", "mean"),
                      red_share=("red_share", "mean"))
                 .round(4).reset_index())
    yearly_path = out_dir / "yearly.json"
    yearly.to_json(yearly_path, orient="records")
    log.info(f"wrote {yearly_path} ({len(yearly)} years)")

    # Color River: mean hue-family share per decade
    river = res.groupby("decade")[BAND_COLS].mean().round(4)
    river["n"] = res.groupby("decade").size()
    river_path = out_dir / "hue_river.json"
    river.reset_index().to_json(river_path, orient="records")
    log.info(f"wrote {river_path} ({len(river)} decades)")

    # ---- Continue / Pivot checkpoint ----
    d = res.groupby("decade")[["brightness", "red_share", "dark_share"]].mean().round(2)
    log.info("=== DARKNESS CURVE CHECKPOINT (by decade) ===")
    print(d.to_string())
    pre70 = res[res.decade < 1970].brightness.mean()
    post70 = res[(res.decade >= 1970) & (res.decade < 2010)].brightness.mean()
    gap = pre70 - post70
    verdict = "CONTINUE -- the curve exists" if gap > 3 else "PIVOT? -- gap is weak, look at what the data IS saying"
    log.info(f"Pre-1970 mean brightness: {pre70:.1f} | 1970-2009: {post70:.1f} | gap: {gap:.1f} -> {verdict}")

    if args.chart:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0a0a0c")
            ax.set_facecolor("#0a0a0c")
            roll = yearly.set_index("year").brightness.rolling(5, min_periods=2).mean()
            ax.plot(roll.index, roll.values, color="#e5a00d", lw=2.5)
            ax.scatter(yearly.year, yearly.brightness, s=8, color="#e5a00d", alpha=.25)
            ax.set_title("The Darkness Curve -- mean poster brightness (L*), 5yr rolling", color="#e8e4da")
            ax.tick_params(colors="#9a958a")
            for sp in ax.spines.values():
                sp.set_color("#2a2a30")
            fig.savefig(args.chart, dpi=150, bbox_inches="tight")
            log.info(f"chart saved: {args.chart}")
        except ImportError:
            log.info("matplotlib not installed -- skipping chart (pip install matplotlib)")


if __name__ == "__main__":
    main()
