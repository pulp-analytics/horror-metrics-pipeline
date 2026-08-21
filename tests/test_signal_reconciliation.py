"""Unit tests for scripts/qa/build_signal_reconciliation_review_page.py's
verdict-extraction functions and scripts/qa/compare_signal_engines.py's
scoring math -- no network calls, no AWS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qa"))

import build_signal_reconciliation_review_page as build_page  # noqa: E402
import compare_signal_engines as compare  # noqa: E402


# ---- verdict extraction ----

def test_census_verdict_true():
    assert build_page._census_verdict({"is_animal": "True", "score": "0.8"}) == (True, 0.8)


def test_census_verdict_false():
    assert build_page._census_verdict({"is_animal": "False", "score": "0.3"}) == (False, 0.3)


def test_census_verdict_empty_row_is_none():
    assert build_page._census_verdict({}) is None


def test_weapon_boxes_verdict_positive():
    v = build_page._weapon_boxes_verdict({"weapon_n": "2", "weapon_top_score": "0.55"})
    assert v == (True, 0.55)


def test_weapon_boxes_verdict_zero_boxes():
    v = build_page._weapon_boxes_verdict({"weapon_n": "0", "weapon_top_score": ""})
    assert v == (False, 0.0)


def test_weapon_boxes_verdict_bad_input_is_none():
    assert build_page._weapon_boxes_verdict({"weapon_n": "not-a-number"}) is None


def test_rek_flag_verdict_above_threshold():
    f = build_page._rek_flag_verdict("rek_animal")
    assert f({"rek_animal": "0.7"}) == (True, 0.7)


def test_rek_flag_verdict_below_threshold():
    f = build_page._rek_flag_verdict("rek_animal")
    assert f({"rek_animal": "0.2"}) == (False, 0.2)


def test_rek_flag_verdict_missing_field_is_none():
    f = build_page._rek_flag_verdict("rek_weapon")
    assert f({}) is None
    assert f({"rek_weapon": ""}) is None


# ---- select_sample: disagreement/positive/negative categorization ----

def test_select_sample_stratifies_disagreement_first(tmp_path, monkeypatch):
    base_csv = tmp_path / "base.csv"
    base_csv.write_text("id,title,poster_path\n1,A,/a.jpg\n2,B,/b.jpg\n3,C,/c.jpg\n4,D,/d.jpg\n", encoding="utf-8")

    census_csv = tmp_path / "census.csv"
    census_csv.write_text(
        "id,label,score,is_animal\n"
        "1,dog,0.9,True\n"   # both agree positive (with rek below)
        "2,none,0.1,False\n"  # both agree negative
        "3,dog,0.6,True\n"   # disagreement: census says yes
        "4,none,0.2,False\n",  # disagreement: census says no
        encoding="utf-8",
    )
    rek_csv = tmp_path / "rek.csv"
    rek_csv.write_text(
        "id,rek_animal\n"
        "1,0.9\n"
        "2,0.1\n"
        "3,0.1\n"   # disagrees with census's True on id 3
        "4,0.9\n",  # disagrees with census's False on id 4
        encoding="utf-8",
    )

    monkeypatch.setattr(build_page, "SAMPLE_INPUT", base_csv)
    monkeypatch.setitem(build_page.ENGINES, "animal", {
        "question": build_page.ENGINES["animal"]["question"],
        "sources": [
            ("clip_census", census_csv, build_page._census_verdict),
            ("rekognition", rek_csv, build_page._rek_flag_verdict("rek_animal")),
        ],
    })

    chosen = build_page.select_sample("animal", n=4)
    ids = {r["id"] for r in chosen}
    assert {"3", "4"} <= ids  # both disagreement cases must be included
    assert len(chosen) == 4


def test_select_sample_raises_when_no_engine_has_usable_data(tmp_path, monkeypatch):
    base_csv = tmp_path / "base.csv"
    base_csv.write_text("id,title,poster_path\n1,A,/a.jpg\n", encoding="utf-8")
    monkeypatch.setattr(build_page, "SAMPLE_INPUT", base_csv)
    monkeypatch.setitem(build_page.ENGINES, "animal", {
        "question": "q",
        "sources": [("clip_census", tmp_path / "missing.csv", build_page._census_verdict)],
    })
    try:
        build_page.select_sample("animal", n=4)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


# ---- compare_signal_engines scoring ----

def test_prf_perfect_classifier():
    acc, prec, rec, n = compare.prf(tp=5, fp=0, fn=0, tn=5)
    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert n == 10


def test_prf_all_wrong():
    acc, prec, rec, n = compare.prf(tp=0, fp=5, fn=5, tn=0)
    assert acc == 0.0


def test_prf_no_positive_predictions_precision_is_nan():
    acc, prec, rec, n = compare.prf(tp=0, fp=0, fn=3, tn=7)
    assert prec != prec  # NaN


def test_score_matches_human_labels():
    verdicts = {"1": True, "2": False, "3": True, "4": False}
    human = {"1": True, "2": False, "3": False, "4": False}  # id 3 is a false positive
    acc, prec, rec, n = compare.score(verdicts, human)
    assert n == 4
    assert acc == 0.75  # 3/4 correct


def test_score_skips_ids_engine_never_scored():
    verdicts = {"1": True}  # engine only has an opinion on id 1
    human = {"1": True, "2": False}
    acc, prec, rec, n = compare.score(verdicts, human)
    assert n == 1
    assert acc == 1.0


def test_load_human_excludes_no_seguro(tmp_path):
    csv_path = tmp_path / "human.csv"
    csv_path.write_text(
        "id,title,poster_path,human_verdict\n"
        "1,A,/a.jpg,si\n"
        "2,B,/b.jpg,no\n"
        "3,C,/c.jpg,no_seguro\n"
        "4,D,/d.jpg,\n",
        encoding="utf-8",
    )
    human = compare.load_human(csv_path)
    assert human == {"1": True, "2": False}
