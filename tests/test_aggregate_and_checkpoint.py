"""Unit tests for 02_aggregate_and_checkpoint.py's checkpoint logic -- no
files, no network. Confirms the script isn't horror-specific: it works on
any --in with brightness/decade columns, and degrades honestly (not with
a misleading number) when a sample doesn't span both checkpoint eras --
the real case found running this against a genuinely non-horror (mostly
modern) sci-fi sample, where there were zero pre-1970 rows."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib
agg = importlib.import_module("02_aggregate_and_checkpoint")


def make_df(rows: list[tuple[int, float]]) -> pd.DataFrame:
    """rows of (decade, brightness) -> a minimal frame compute_checkpoint() needs."""
    df = pd.DataFrame(rows, columns=["decade", "brightness"])
    return df


def test_checkpoint_continue_when_gap_is_large():
    res = make_df([(1950, 50.0), (1950, 52.0), (1980, 30.0), (1980, 28.0)])
    ck = agg.compute_checkpoint(res)
    assert ck["n_pre70"] == 2 and ck["n_post70"] == 2
    assert ck["gap"] == pytest.approx(22.0)
    assert ck["verdict"].startswith("CONTINUE")


def test_checkpoint_pivot_when_gap_is_weak():
    res = make_df([(1950, 40.0), (1950, 41.0), (1980, 39.0), (1980, 40.0)])
    ck = agg.compute_checkpoint(res)
    assert ck["gap"] == pytest.approx(1.0)
    assert ck["verdict"].startswith("PIVOT")


def test_checkpoint_unavailable_with_no_pre1970_rows():
    # the real case found testing this script against a mostly-modern,
    # non-horror (sci-fi) sample: zero rows before 1970
    res = make_df([(1980, 50.0), (2000, 30.0), (2020, 20.0)])
    ck = agg.compute_checkpoint(res)
    assert ck["n_pre70"] == 0
    assert "verdict" not in ck
    assert "gap" not in ck


def test_checkpoint_unavailable_with_no_1970_2009_rows():
    res = make_df([(1950, 50.0), (1930, 40.0)])
    ck = agg.compute_checkpoint(res)
    assert ck["n_post70"] == 0
    assert "verdict" not in ck
