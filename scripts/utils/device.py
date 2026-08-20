"""Pick a torch device: cuda if present, else MPS (Apple Silicon), else CPU.

17/19/20/21 used to be `cuda if cuda else cpu`, so a Mac ran Grounding
DINO on CPU (~27 min for the 99-poster sample). `pick_device()` is the
shared replacement. 18_saliency_prediction.py is TensorFlow, not torch,
and does not use this helper.
"""
from __future__ import annotations

import argparse

import torch


def mps_is_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def pick_device(explicit: str = "auto") -> str:
    """Return 'cuda', 'mps', or 'cpu'.

    `explicit` is `auto` (default) or a forced device. A forced device
    that isn't available exits rather than silently falling back --
    `--device mps` on a machine without Metal should fail, not hide on CPU.
    """
    requested = (explicit or "auto").strip().lower()
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise SystemExit(f"unknown --device {explicit!r} (want auto/cpu/cuda/mps)")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return "cuda"
    if requested == "mps":
        if not mps_is_available():
            raise SystemExit("--device mps requested but MPS is not available")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"


def add_device_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--device", default="auto",
        help="auto (cuda > mps > cpu), or cpu/cuda/mps",
    )
