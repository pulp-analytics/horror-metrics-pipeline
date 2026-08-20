"""pyproject.toml extras -- cpu / tf-saliency / bedrock. No install."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _section(name: str) -> str:
    """Body of [project] or [project.optional-dependencies] through the next [."""
    m = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=\n\[|\Z)", PYPROJECT, re.S | re.M)
    assert m, f"missing [{name}] in pyproject.toml"
    return m.group(1)


def test_optional_extras_cpu_tf_saliency_bedrock_and_all_exist():
    optional = _section("project.optional-dependencies")
    for extra in ("cpu", "tf-saliency", "bedrock", "all"):
        assert re.search(rf"^{re.escape(extra)}\s*=", optional, re.M), extra


def test_default_install_does_not_pull_tensorflow_or_boto3():
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", _section("project"), re.S | re.M)
    assert m, "project.dependencies missing"
    deps = m.group(1)
    assert "tensorflow" not in deps
    assert "boto3" not in deps
    assert "pyiqa" not in deps
    assert "ultralytics" not in deps


def test_cpu_extra_adds_pyiqa_and_ultralytics_not_tensorflow():
    optional = _section("project.optional-dependencies")
    # grab the cpu = [ ... ] block
    m = re.search(r"^cpu\s*=\s*\[(.*?)\]", optional, re.S | re.M)
    assert m, "cpu extra missing"
    body = m.group(1)
    assert "pyiqa" in body
    assert "ultralytics" in body
    assert "tensorflow" not in body
    assert "boto3" not in body


def test_tf_saliency_extra_is_tensorflow():
    m = re.search(r"^tf-saliency\s*=\s*\[(.*?)\]", _section("project.optional-dependencies"), re.S | re.M)
    assert m and "tensorflow" in m.group(1)


def test_bedrock_extra_is_boto3():
    m = re.search(r"^bedrock\s*=\s*\[(.*?)\]", _section("project.optional-dependencies"), re.S | re.M)
    assert m and "boto3" in m.group(1)
