"""Unit tests for the shared SigLIP math utilities -- no model loading, no
network. The softmax/cosine-similarity/axis math is identical to CLIP's
(see test_clip_backbone.py); this file exists separately so SigLIP's
scripts (11-13) have their own coverage that doesn't implicitly depend on
CLIP's test file staying unchanged."""
import numpy as np


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = x * temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_softmax_picks_the_closest_prototype():
    sims = np.array([[0.9, 0.1, 0.05]])  # [vampire, zombie, none]
    probs = softmax(sims, temp=100.0)
    assert probs.argmax() == 0
    assert probs[0, 0] > 0.99


def test_axis_score_sign_matches_which_pole_is_closer():
    # axis = cos(emb, DREAD) - cos(emb, CALM), used by 12_siglip_fear_axis.py
    # and the typography axis in 13_siglip_reanalysis.py
    emb = np.array([1.0, 0.0])
    dread = np.array([1.0, 0.0])   # identical to emb -> cos = 1
    calm = np.array([0.0, 1.0])    # orthogonal to emb -> cos = 0
    axis = float(emb @ dread) - float(emb @ calm)
    assert axis > 0


def test_l2_normalization_makes_prototype_a_unit_vector():
    # every text_prototype() call in utils/siglip_backbone.py re-normalizes
    # the mean of several prompt embeddings, same as clip_backbone.py
    vecs = np.array([[3.0, 4.0], [0.0, 5.0]])
    mean = vecs.mean(axis=0)
    unit = mean / np.linalg.norm(mean)
    assert np.isclose(np.linalg.norm(unit), 1.0)


def test_embed_dim_matches_siglip_base_patch16_224():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from utils.siglip_backbone import EMBED_DIM
    assert EMBED_DIM == 768  # vs. 512 for CLIP ViT-B/32 -- not interchangeable
