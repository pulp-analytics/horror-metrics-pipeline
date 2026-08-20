"""Pick a torch device: CUDA if the container has it, else CPU.

AWS Batch/Fargate contract (CPU today; GPU EC2 later). MPS is not a
valid `--device`. 18_saliency_prediction.py is TensorFlow and does not
use this helper.
"""
from __future__ import annotations

import argparse

import torch


def pick_device(explicit: str = "auto") -> str:
    """Return 'cuda' or 'cpu'.

    `explicit` is `auto` (default) or a forced device. `--device cuda`
    on a CPU Fargate task exits rather than silently falling back.
    """
    requested = (explicit or "auto").strip().lower()
    if requested not in ("auto", "cpu", "cuda"):
        raise SystemExit(f"unknown --device {explicit!r} (want auto/cpu/cuda)")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def add_device_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--device", default="auto",
        help="auto (cuda > cpu), or cpu/cuda",
    )
