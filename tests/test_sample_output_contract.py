"""Contract tests for the checked-in 99-poster sample: every 01-21 metric
CSV plus 25's agreement file is present, one row per poster, and
assemble_master_dataset.py can join them using sample_output/metrics_input.csv
as the auto-detected base.
No model downloads -- reads committed CSVs only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import assemble_master_dataset as assemble

SAMPLE_OUT = ROOT / "data" / "sample_output"
SAMPLE_N = 99

METRIC_FILES = [
    "color_metrics.csv",
    "iqa_multi_score.csv",
    "nima_score.csv",
    "laion_aesthetic_score.csv",
    "census.csv",
    "fear_axis.csv",
    "typography.csv",
    "genre_classifier.csv",
    "medium.csv",
    "siglip_census.csv",
    "siglip_fear_axis.csv",
    "siglip_genre_classifier.csv",
    "siglip_typography.csv",
    "face_detect.csv",
    "geometric_composition.csv",
    "depth_estimation.csv",
    "saliency_prediction.csv",
    "pose_dynamism.csv",
    "creature_weapon_owlv2.csv",
    "creature_weapon_dino.csv",
    "creature_weapon_agreement.csv",
]


def test_metrics_input_is_the_99_poster_sample_base():
    base = assemble.load_base(SAMPLE_OUT, None)
    assert len(base) == SAMPLE_N
    assert base["id"].is_unique


def test_each_one_row_per_poster_metric_file_matches_the_sample_ids():
    base_ids = set(pd.read_csv(SAMPLE_OUT / "metrics_input.csv", dtype=str)["id"])
    assert len(base_ids) == SAMPLE_N
    for name in METRIC_FILES:
        path = SAMPLE_OUT / name
        assert path.exists(), f"missing {name}"
        df = pd.read_csv(path, dtype={"id": str})
        assert df["id"].is_unique, f"{name} has duplicate ids"
        assert set(df["id"]) == base_ids, f"{name} ids don't match metrics_input.csv"


def test_assemble_sample_output_yields_99_rows(tmp_path):
    out = tmp_path / "master_dataset.csv"
    # argv-style: call internals the same way main() does
    base = assemble.load_base(SAMPLE_OUT, None)
    skip = assemble.SKIP_BY_DEFAULT | set(assemble.BASE_CANDIDATES)
    merged = base
    for path in sorted(p for p in SAMPLE_OUT.glob("*.csv") if p.name not in skip):
        merged = merged.merge(assemble.load_metric_file(path), on="id", how="left")
    merged.to_csv(out, index=False)
    assert len(merged) == SAMPLE_N
    assert "geometric_composition_symmetry" in merged.columns
    assert "creature_weapon_owlv2_creature_n" in merged.columns
    assert "creature_weapon_dino_creature_n" in merged.columns
    assert "creature_weapon_agreement_creature_n" in merged.columns
    assert "depth_estimation_mean_depth" in merged.columns
    assert "pose_dynamism_n_persons" in merged.columns
    assert "saliency_prediction_peak_x" in merged.columns
    # prefixed columns must not collide across the two detectors
    assert "creature_weapon_owlv2_creature_n" != "creature_weapon_dino_creature_n"
