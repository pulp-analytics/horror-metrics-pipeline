"""Shared SigLIP (google/siglip-base-patch16-224) loading + text-prototype
helper, used by every SigLIP-based script in this repo (11-13). Mirrors
utils/clip_backbone.py's load/text_prototype split, but SigLIP's Hugging
Face `transformers` API shapes things a bit differently than open_clip:
one `AutoModel` exposes both towers, `AutoProcessor` handles both image
and text preprocessing, and text features come back through
`get_text_features()` with fixed `padding="max_length"` (SigLIP's own
tokenizer convention, not optional -- the model was trained on
fixed-length padded text this way).

Real project text-only scripts (siglip_fear_axis.py, siglip_reanalysis.py)
never bother moving the model to a GPU device at all -- a few thousand
text-prompt embeddings is cheap enough on CPU that it wasn't worth the
code. This module keeps that same default (device="cpu") for
text_prototype() while still accepting a device arg for symmetry with
clip_backbone.py; only 11_siglip_embed.py (image embedding, the actually
heavy part) picks cuda when available.
"""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM = 768


def load_siglip(device: str = "cpu"):
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def text_prototype(model, processor, prompts: list[str], device: str = "cpu"):
    """Mean of L2-normalized text embeddings for a prompt ensemble, then
    re-normalized -- identical rationale to clip_backbone.text_prototype,
    just through SigLIP's get_text_features()/pooler_output."""
    inputs = processor(text=prompts, padding="max_length", return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
    p = feats.mean(0)
    return (p / p.norm()).cpu().numpy()
