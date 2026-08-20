"""CSV column contract: sample headers match each script's FIELDS (or the
pandas to_csv shape for 06-09 / 12-13), and docs/SCHEMA.md names every
sample metric file.

Parses FIELDS via ast so CI does not import pyiqa / tensorflow / boto3.
No model downloads -- reads committed CSVs and script source only."""
from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SAMPLE_OUT = ROOT / "data" / "sample_output"
SCHEMA = ROOT / "docs" / "SCHEMA.md"

# Scripts that write FIELDS via csv.DictWriter / write_csv_rows.
FIELDS_TO_SAMPLE = {
    "01_color_metrics.py": "color_metrics.csv",
    "02_iqa_multi_score.py": "iqa_multi_score.csv",
    "03_nima_score.py": "nima_score.csv",
    "04_laion_aesthetic_score.py": "laion_aesthetic_score.csv",
    "10_clip_medium.py": "medium.csv",
    "14_face_detect.py": "face_detect.csv",
    "15_face_expression.py": "face_expression.csv",
    "16_geometric_composition.py": "geometric_composition.csv",
    "17_depth_estimation.py": "depth_estimation.csv",
    "18_saliency_prediction.py": "saliency_prediction.csv",
    "19_pose_dynamism.py": "pose_dynamism.csv",
    "20_creature_weapon_owlv2.py": "creature_weapon_owlv2.csv",
    "21_creature_weapon_dino.py": "creature_weapon_dino.csv",
    "25_creature_weapon_agreement.py": "creature_weapon_agreement.csv",
}

# pandas to_csv -- no FIELDS. Exact headers of the checked-in sample
# (09/13 genre optional columns are present in that sample; see SCHEMA).
PANDAS_SAMPLE_HEADERS = {
    "census.csv": ["id", "label", "score", "title", "year", "is_animal", "is_creature"],
    "fear_axis.csv": ["id", "axis", "title", "year"],
    "typography.csv": ["id", "axis", "title", "year"],
    "genre_classifier.csv": [
        "id", "true_genre", "pred_genre", "agree",
        "sim_horror", "sim_scifi", "sim_thriller", "sim_mystery",
    ],
    "siglip_census.csv": ["id", "label", "score", "year", "title"],
    "siglip_fear_axis.csv": ["id", "axis", "title", "year"],
    "siglip_typography.csv": ["id", "axis", "year", "title"],
    "siglip_genre_classifier.csv": ["id", "pred_genre", "true_genre", "agree"],
}

NOVA_FIELDS = {
    "22_creature_weapon_nova_qa.py": [
        "id", "source", "kind", "label", "score", "box",
        "model", "status", "verdict", "actual", "reason", "latency_s", "error",
    ],
    "23_census_nova_qa.py": [
        "id", "clip_label", "clip_score",
        "model", "status", "nova_label", "agree", "reason", "latency_s", "error",
    ],
    "24_typography_nova_qa.py": [
        "id", "clip_register", "clip_axis",
        "model", "status", "nova_register", "agree", "agree_adjacent",
        "reason", "latency_s", "error",
    ],
}


def _list_of_strings(node: ast.AST, env: dict[str, list[str]]) -> list[str]:
    if not isinstance(node, ast.List):
        raise TypeError(f"expected List, got {type(node).__name__}")
    out: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
        elif isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
            out.extend(env[elt.value.id])
        else:
            raise TypeError(f"unsupported list element {type(elt).__name__}")
    return out


def script_fields(path: Path) -> list[str]:
    """Parse top-level FIELDS; expand *METRIC_NAMES (see 02) without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    env: dict[str, list[str]] = {}
    fields_value: ast.AST | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "METRIC_NAMES":
            env["METRIC_NAMES"] = _list_of_strings(node.value, {})
        elif target.id == "FIELDS":
            fields_value = node.value
    if fields_value is None:
        raise AssertionError(f"{path.name} has no parseable FIELDS list")
    return _list_of_strings(fields_value, env)


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_script_fields_match_sample_csv_headers():
    for script_name, csv_name in FIELDS_TO_SAMPLE.items():
        fields = script_fields(SCRIPTS / script_name)
        header = csv_header(SAMPLE_OUT / csv_name)
        assert header == fields, f"{csv_name} header != {script_name} FIELDS"


def test_pandas_metric_headers_match_checked_in_sample():
    for csv_name, expected in PANDAS_SAMPLE_HEADERS.items():
        header = csv_header(SAMPLE_OUT / csv_name)
        assert header == expected, f"{csv_name} header drifted from SCHEMA/sample"


def test_metrics_input_is_the_corpus_base_columns():
    assert csv_header(SAMPLE_OUT / "metrics_input.csv") == [
        "id", "title", "year", "poster_path",
    ]


def test_nova_qa_fields_match_schema_lists():
    for script_name, expected in NOVA_FIELDS.items():
        assert script_fields(SCRIPTS / script_name) == expected


def test_iqa_fields_expands_metric_names_star():
    """02 is the only FIELDS = [..., *METRIC_NAMES, ...]; keep the AST path honest."""
    assert script_fields(SCRIPTS / "02_iqa_multi_score.py") == [
        "id", "title", "year", "clipiqa", "musiq", "brisque", "error",
    ]


def test_schema_md_names_every_sample_metric_file():
    text = SCHEMA.read_text(encoding="utf-8")
    missing = []
    for csv_name in [
        *FIELDS_TO_SAMPLE.values(),
        *PANDAS_SAMPLE_HEADERS,
        "metrics_input.csv",
    ]:
        if csv_name not in text:
            missing.append(csv_name)
    assert missing == [], f"docs/SCHEMA.md does not mention {missing}"


def test_schema_md_names_nova_qa_columns():
    text = SCHEMA.read_text(encoding="utf-8")
    for needle in ("`verdict`", "`clip_label`", "`agree_adjacent`"):
        assert needle in text, f"docs/SCHEMA.md missing {needle}"
