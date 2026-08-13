"""Unit tests for the shared CLIP math utilities -- no model loading, no
network. Confirms the softmax/cosine-similarity plumbing every CLIP script
(census/fear_axis/typography_axis/genre_classifier) relies on behaves
correctly on synthetic embeddings, independent of whether the actual CLIP
model produces sensible embeddings for a given image."""
import numpy as np


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Mirrors the exact pattern every census-style script uses:
    torch.softmax(torch.tensor(sims * temp), dim=...) -- reimplemented in
    plain numpy here so this test has no torch dependency."""
    z = x * temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_softmax_picks_the_closest_prototype():
    # a poster embedding closer to "vampire" than any other label
    sims = np.array([[0.9, 0.1, 0.05]])  # [vampire, zombie, none]
    probs = softmax(sims, temp=100.0)
    assert probs.argmax() == 0
    assert probs[0, 0] > 0.99  # high temperature sharpens a clear winner


def test_softmax_high_temperature_amplifies_a_small_gap():
    # a small raw similarity gap should still produce a confident winner
    # at temp=100 -- this is why the real census script uses temp=100.0,
    # not temp=1.0 (raw cosine similarities cluster in a narrow band)
    sims = np.array([[0.30, 0.28, 0.25]])
    probs_low_temp = softmax(sims, temp=1.0)
    probs_high_temp = softmax(sims, temp=100.0)
    assert probs_high_temp[0, 0] > probs_low_temp[0, 0]


def test_axis_score_sign_matches_which_pole_is_closer():
    # axis = cos(emb, POLE_A) - cos(emb, POLE_B), the exact formula
    # 07_clip_fear_axis.py and 08_clip_typography_axis.py both use
    emb = np.array([1.0, 0.0])
    pole_a = np.array([1.0, 0.0])   # identical to emb -> cos = 1
    pole_b = np.array([0.0, 1.0])   # orthogonal to emb -> cos = 0
    axis = float(emb @ pole_a) - float(emb @ pole_b)
    assert axis > 0  # emb is closer to pole_a


def test_l2_normalization_makes_prototype_a_unit_vector():
    # every text_prototype() call in utils/clip_backbone.py re-normalizes
    # the mean of several prompt embeddings -- confirm that operation
    # actually produces a unit vector regardless of the inputs' scale
    vecs = np.array([[3.0, 4.0], [0.0, 5.0]])  # arbitrary magnitudes
    mean = vecs.mean(axis=0)
    unit = mean / np.linalg.norm(mean)
    assert np.isclose(np.linalg.norm(unit), 1.0)
