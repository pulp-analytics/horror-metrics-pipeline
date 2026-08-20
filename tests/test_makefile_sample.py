"""`make sample` dependency graph -- dry-run only, no model download.

Uses an empty OUT so Make treats every sample file as missing. The
checked-in data/sample_output/ CSVs must not be rebuilt by a plain
`make sample` (that would overwrite the citable sample)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IN = ROOT / "data" / "sample_input" / "sample_100_posters.csv"

BEFORE_AFTER = [
    ("05_clip_embed.py", "06_clip_census.py"),
    ("05_clip_embed.py", "07_clip_fear_axis.py"),
    ("05_clip_embed.py", "08_clip_typography_axis.py"),
    ("05_clip_embed.py", "09_clip_genre_classifier.py"),
    ("11_siglip_embed.py", "12_siglip_fear_axis.py"),
    ("11_siglip_embed.py", "13_siglip_reanalysis.py"),
    ("14_face_detect.py", "15_face_expression.py"),
    ("20_creature_weapon_owlv2.py", "25_creature_weapon_agreement.py"),
    ("21_creature_weapon_dino.py", "25_creature_weapon_agreement.py"),
]


def _dry_run_sample(out: Path) -> str:
    result = subprocess.run(
        ["make", "-n", "sample", f"OUT={out}", f"IN={SAMPLE_IN}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def test_make_sample_dry_run_orders_dependencies(tmp_path):
    log = _dry_run_sample(tmp_path / "out")
    for earlier, later in BEFORE_AFTER:
        assert earlier in log, f"missing {earlier} in dry-run"
        assert later in log, f"missing {later} in dry-run"
        assert log.index(earlier) < log.index(later), f"{earlier} must run before {later}"


def test_make_sample_dry_run_includes_agreement_not_nova(tmp_path):
    log = _dry_run_sample(tmp_path / "out")
    assert "25_creature_weapon_agreement.py" in log
    assert "01_color_metrics.py" in log
    assert "16_geometric_composition.py" in log
    for nova in (
        "22_creature_weapon_nova_qa.py",
        "23_census_nova_qa.py",
        "24_typography_nova_qa.py",
    ):
        assert nova not in log, f"Nova QA {nova} is not a pipeline stage"


def test_make_sample_is_noop_when_checked_in_outputs_exist():
    result = subprocess.run(
        ["make", "-n", "sample"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    log = result.stdout + result.stderr
    assert "01_color_metrics.py" not in log
    assert "20_creature_weapon_owlv2.py" not in log
    assert "25_creature_weapon_agreement.py" not in log


def test_make_sample_is_noop_even_when_scripts_are_newer_than_csvs():
    """GitHub Actions extracts scripts/ after data/, so script mtimes look
    newer than the checked-in CSVs. Timestamp deps would overwrite the
    sample; recipes must not be registered for files that already exist."""
    scripts = list((ROOT / "scripts").glob("*.py"))
    times = {p: p.stat() for p in scripts}
    try:
        for p in scripts:
            os.utime(p, None)
        result = subprocess.run(
            ["make", "-n", "sample"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        for p, st in times.items():
            os.utime(p, (st.st_atime, st.st_mtime))
    log = result.stdout + result.stderr
    assert "01_color_metrics.py" not in log
    assert "25_creature_weapon_agreement.py" not in log
