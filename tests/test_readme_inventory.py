"""README Structure must name every tests/test_*.py so the inventory
cannot silently rot the way it did before (8 of 23 files listed)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_structure_names_every_test_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = sorted(
        p.name
        for p in (ROOT / "tests").glob("test_*.py")
        if p.name not in readme
    )
    assert missing == [], f"README.md Structure is missing {missing}"
