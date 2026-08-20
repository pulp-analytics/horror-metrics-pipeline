"""Unit tests for utils/device.py -- no GPU, no model download."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils.device import pick_device


def test_pick_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: True)
    assert pick_device() == "cuda"


def test_pick_device_cpu_when_cuda_is_absent(monkeypatch):
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: False)
    assert pick_device() == "cpu"


def test_pick_device_explicit_cpu_even_if_cuda_exists(monkeypatch):
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: True)
    assert pick_device("cpu") == "cpu"


def test_pick_device_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: False)
    with pytest.raises(SystemExit, match="cuda"):
        pick_device("cuda")


def test_pick_device_rejects_unknown_name():
    with pytest.raises(SystemExit, match="unknown"):
        pick_device("tpu")


def test_pick_device_rejects_mps():
    with pytest.raises(SystemExit, match="unknown"):
        pick_device("mps")
