"""Pure-function tests for scripts/qa/validate_genre_classifier_vs_imdb.py's
compute_genre_metrics -- no CLIP model, no network, no IMDb dataset needed.
Imported by file path since the script lives under scripts/qa/ and the
module itself pulls in torch/CLIP at import time only for load_clip() etc.,
which compute_genre_metrics never calls."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "validate_genre_classifier_vs_imdb", ROOT / "scripts" / "qa" / "validate_genre_classifier_vs_imdb.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

compute_genre_metrics = mod.compute_genre_metrics
GENRE_MAP = mod.GENRE_MAP


def row(pred_genre: str, imdb_genres: set[str]) -> dict:
    return {"pred_genre": pred_genre, "imdb_genres": imdb_genres}


def test_perfect_single_label_agreement():
    rows = [
        row("horror", {"Horror"}),
        row("scifi", {"Sci-Fi"}),
        row("thriller", {"Thriller"}),
        row("mystery", {"Mystery"}),
    ]
    m = compute_genre_metrics(rows)
    assert m["n_scored"] == 4
    assert m["overall_containment_accuracy"] == 1.0
    for c in GENRE_MAP:
        assert m["precision"][c]["precision"] == 1.0
        assert m["recall"][c]["recall"] == 1.0


def test_multi_label_containment_counts_as_a_precision_hit():
    # CLIP predicted "thriller" for a film IMDb tags as both Sci-Fi and
    # Thriller -- the lenient/containment definition should count this as
    # a hit for "thriller" precision, since Thriller really is in there
    rows = [row("thriller", {"Sci-Fi", "Thriller"})]
    m = compute_genre_metrics(rows)
    assert m["precision"]["thriller"]["precision"] == 1.0


def test_recall_is_strict_not_containment():
    # same row: for STRICT per-class recall, "scifi" only counts a hit if
    # CLIP's own prediction was literally "scifi" -- predicting "thriller"
    # on a Sci-Fi+Thriller film is a real recall miss for sci-fi, even
    # though Sci-Fi genuinely is one of the film's real genres
    rows = [row("thriller", {"Sci-Fi", "Thriller"})]
    m = compute_genre_metrics(rows)
    assert m["recall"]["scifi"]["n_true"] == 1
    assert m["recall"]["scifi"]["recall"] == 0.0
    assert m["recall"]["scifi"]["predicted_as"]["thriller"] == 1


def test_no_target_genre_rows_excluded_from_every_count():
    # IMDb tags this film Drama/Comedy only -- none of our 4 classes.
    # CLIP still had to guess one, but there's no way to tell if that
    # guess was right or wrong from IMDb's data, so (after inspecting the
    # real 52 such cases in the actual 200-poster sample -- mostly shorts/
    # documentaries/animation and older titles with incomplete IMDb
    # tagging) these are excluded entirely, the same way Bedrock's
    # ground-truth review excludes "unjudgeable" rows rather than forcing
    # them into the denominator as an automatic miss.
    rows = [row("horror", {"Drama", "Comedy"}), row("mystery", {"Mystery"})]
    m = compute_genre_metrics(rows)
    assert m["n_scored"] == 1  # only the real "mystery" row counts
    assert m["n_no_target_genre"] == 1
    assert m["overall_containment_accuracy"] == 1.0  # not dragged down by the excluded row
    assert m["precision"]["horror"]["n_predicted"] == 0  # the excluded row isn't in horror's denominator either


def test_predicted_as_breakdown_sums_to_n_true():
    rows = [
        row("horror", {"Sci-Fi"}),
        row("thriller", {"Sci-Fi"}),
        row("scifi", {"Sci-Fi"}),
    ]
    m = compute_genre_metrics(rows)
    breakdown = m["recall"]["scifi"]["predicted_as"]
    assert sum(breakdown.values()) == m["recall"]["scifi"]["n_true"] == 3
    assert breakdown["scifi"] == 1


def test_no_support_is_none_not_zero():
    # a class that's never predicted has no precision to report
    rows = [row("horror", {"Horror"})]
    m = compute_genre_metrics(rows)
    assert m["precision"]["mystery"]["n_predicted"] == 0
    assert m["precision"]["mystery"]["precision"] is None


def test_empty_rows():
    m = compute_genre_metrics([])
    assert m["n_scored"] == 0
    assert m["overall_containment_accuracy"] is None
