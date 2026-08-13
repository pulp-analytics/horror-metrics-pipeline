"""Shared CLIP ViT-B/32 loading + text-prototype helper, used by every
CLIP-based script in this repo (census, fear_axis, typography_axis,
genre_classifier). This exact load_clip()/proto() pattern is duplicated
verbatim across all four real private-pipeline scripts this repo ports --
consolidated here since it's genuinely identical code, not different
methodology per script (unlike each script's own taxonomy/prompts, which
stay in the script that owns them)."""
from __future__ import annotations

import torch
import open_clip

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBED_DIM = 512


def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    return model.to(device).eval(), preprocess


def get_tokenizer():
    return open_clip.get_tokenizer(MODEL_NAME)


def text_prototype(model, tok, prompts: list[str], device: str):
    """Mean of L2-normalized text embeddings for a prompt ensemble, then
    re-normalized -- averaging several phrasings of the same concept is
    more robust than trusting any single prompt's embedding."""
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        p = t.mean(0)
        return (p / p.norm()).cpu().numpy()
