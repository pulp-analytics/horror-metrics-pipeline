"""Blind review HTML builders -- no network if posters-dir already has jpgs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "qa"))

from review_page import write_review_html  # noqa: E402


def _tiny_jpeg(path: Path) -> None:
    Image.new("RGB", (8, 12), (20, 20, 20)).save(path, format="JPEG")


def test_write_review_html_rejects_model_scores(tmp_path):
    with pytest.raises(ValueError, match="clip_score"):
        write_review_html(
            tmp_path / "x.html",
            title="t", blurb="b", storage_key="k", export_name="e.csv",
            verdicts=[{"v": "a", "label": "A"}],
            rows=[{"id": "1", "key": "1", "question": "q", "img": "", "clip_score": "0.9"}],
        )


def test_census_review_page_is_blind(tmp_path, monkeypatch):
    posters = tmp_path / "posters"
    posters.mkdir()
    _tiny_jpeg(posters / "1.jpg")
    inn = tmp_path / "in.csv"
    with inn.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "year", "poster_path"])
        w.writeheader()
        w.writerow({"id": "1", "title": "Film", "year": "1999", "poster_path": "/x.jpg"})
    out = tmp_path / "census.html"
    import build_census_review_page as b
    monkeypatch.setattr("sys.argv", [
        "build_census_review_page.py",
        "--in", str(inn), "--out", str(out), "--posters-dir", str(posters),
    ])
    assert b.main() == 0
    html = out.read_text(encoding="utf-8")
    assert "clip_label" not in html
    assert "clip_score" not in html
    assert "nova_label" not in html
    assert "What is the main creature" in html
    assert "vampire" in html


def test_box_review_page_draws_claim_not_score(tmp_path, monkeypatch):
    posters = tmp_path / "posters"
    posters.mkdir()
    _tiny_jpeg(posters / "1.jpg")
    inn = tmp_path / "in.csv"
    boxes = tmp_path / "boxes.csv"
    with inn.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "year", "poster_path"])
        w.writeheader()
        w.writerow({"id": "1", "title": "Film", "year": "1999", "poster_path": "/x.jpg"})
    creature = json.dumps([{"label": "vampire", "score": 0.91, "box": [0.1, 0.1, 0.2, 0.2]}])
    with boxes.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "title", "year", "creature_n", "creature_top_label", "creature_top_score",
            "creature_boxes", "weapon_n", "weapon_top_label", "weapon_top_score", "weapon_boxes", "error",
        ])
        w.writeheader()
        w.writerow({
            "id": "1", "title": "Film", "year": "1999", "creature_n": "1",
            "creature_top_label": "vampire", "creature_top_score": "0.91",
            "creature_boxes": creature, "weapon_n": "0", "weapon_top_label": "",
            "weapon_top_score": "", "weapon_boxes": "[]", "error": "",
        })
    out = tmp_path / "boxes.html"
    import build_creature_weapon_review_page as b
    monkeypatch.setattr("sys.argv", [
        "build_creature_weapon_review_page.py",
        "--in", str(inn), "--boxes", str(boxes), "--out", str(out),
        "--posters-dir", str(posters),
    ])
    assert b.main() == 0
    html = out.read_text(encoding="utf-8")
    assert "0.91" not in html
    assert "Does the red rectangle contain a vampire?" in html
    assert "nova" not in html.lower() or "Nova verdict" in html  # blurb may say Nova is not shown
    assert '"score"' not in html
