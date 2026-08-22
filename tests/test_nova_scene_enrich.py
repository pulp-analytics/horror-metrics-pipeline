"""Unit tests for 27_nova_scene_enrich.py's pure parsing helpers -- ported
as-is from the real nova_poster_enrich.py's _fear_labels()/_score()/
_join_list(). No AWS calls needed."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "nova_scene_enrich", Path(__file__).resolve().parents[1] / "scripts" / "27_nova_scene_enrich.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_join_list = mod._join_list
_fear_labels = mod._fear_labels
_score = mod._score
_face_count = mod._face_count
_poster_qa_verdict = mod._poster_qa_verdict
_top_from_labels = mod._top_from_labels


def test_join_list_basic():
    assert _join_list(["dread", "gothic", "camp"]) == "dread|gothic|camp"


def test_join_list_strips_empty_entries():
    assert _join_list(["dread", "", "  ", "gothic"]) == "dread|gothic"


def test_join_list_none_is_empty_string():
    assert _join_list(None) == ""


def test_join_list_non_list_scalar():
    assert _join_list("dread") == "dread"


def test_fear_labels_dict_items():
    val = [{"name": "knife", "conf": 0.8}, {"name": "fire", "conf": 0.3}]
    assert _fear_labels(val) == "knife:0.80|fire:0.30"


def test_fear_labels_empty_is_empty_string():
    assert _fear_labels([]) == ""
    assert _fear_labels(None) == ""


def test_fear_labels_skips_entries_with_no_name():
    val = [{"name": "", "conf": 0.5}, {"name": "ghost", "conf": 0.9}]
    assert _fear_labels(val) == "ghost:0.90"


def test_fear_labels_bad_conf_falls_back_to_str():
    val = [{"name": "knife", "conf": "high"}]
    assert _fear_labels(val) == "knife:high"


def test_fear_labels_already_string_passthrough():
    assert _fear_labels("knife:0.80|fire:0.30") == "knife:0.80|fire:0.30"


def test_score_clamps_to_0_1_range():
    assert _score(1.5) == 1.0
    assert _score(-0.3) == 0.0
    assert _score(0.7) == 0.7


def test_score_defaults_on_bad_input():
    assert _score(None) == 0.0
    assert _score("not-a-number") == 0.0


def test_score_rounds_to_4_decimals():
    assert _score(0.123456) == 0.1235


def test_face_count_valid_int():
    assert _face_count(5) == 5


def test_face_count_valid_float_string():
    assert _face_count("3.0") == 3


def test_face_count_negative_clamped_to_zero():
    assert _face_count(-2) == 0


def test_face_count_bad_input_defaults_to_zero():
    assert _face_count(None) == 0
    assert _face_count("not-a-number") == 0


def test_poster_qa_verdict_valid_values_pass_through():
    assert _poster_qa_verdict("poster") == "poster"
    assert _poster_qa_verdict("not_poster") == "not_poster"
    assert _poster_qa_verdict("uncertain") == "uncertain"


def test_poster_qa_verdict_is_case_insensitive():
    assert _poster_qa_verdict("Not_Poster") == "not_poster"


def test_poster_qa_verdict_unrecognized_value_falls_back_to_uncertain():
    assert _poster_qa_verdict("definitely a poster") == "uncertain"
    assert _poster_qa_verdict("") == "uncertain"
    assert _poster_qa_verdict(None) == "uncertain"


def test_top_from_labels_picks_highest_confidence():
    val = [{"name": "person", "conf": 0.6}, {"name": "weapon", "conf": 0.9}, {"name": "sky", "conf": 0.3}]
    assert _top_from_labels(val) == ("weapon", 0.9)


def test_top_from_labels_ignores_first_position_bias():
    val = [{"name": "sky", "conf": 0.2}, {"name": "person", "conf": 0.85}]
    assert _top_from_labels(val) == ("person", 0.85)


def test_top_from_labels_empty_or_missing_is_blank():
    assert _top_from_labels([]) == ("", 0.0)
    assert _top_from_labels(None) == ("", 0.0)


def test_top_from_labels_bad_entries_are_skipped():
    val = ["not-a-dict", {"name": "", "conf": 0.9}, {"name": "person", "conf": "bad"}]
    assert _top_from_labels(val) == ("person", 0.0)
