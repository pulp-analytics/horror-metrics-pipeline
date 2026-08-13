# Methodology

## Color metrics

### What gets computed, per poster

`01_color_metrics.py` downloads each poster from TMDB's public image CDN
via the cache shared with the perceptual-quality scripts below (see
utils/posters.py) and, per poster:

1. **Resize to 96×144** before any analysis. Color statistics don't need
   full resolution, and clustering on ~14k pixels instead of a few hundred
   thousand is what keeps this fast enough to run on plain CPU at corpus
   scale — no GPU, no AWS ML service, just numpy/PIL/scikit-learn.
2. **Convert to CIELAB** (D65 illuminant) via the standard sRGB→XYZ→LAB
   matrix transform. Brightness (`brightness`, mean L*) and "near-black
   share" (`dark_share`, fraction of pixels with L* < 20) come directly
   from the L* channel.
3. **Convert to HSV** (a small vectorized implementation, not
   `colorsys` — needs to run over a whole pixel array at once, not pixel
   by pixel) for saturation (`saturation`, mean S) and hue-based
   classification.
4. **"Blood red" share** (`red_share`): pixels with hue in ±15° of pure
   red, saturation > 0.4, and value > 0.15 — the value floor specifically
   excludes near-black pixels that happen to have a reddish hue but read
   as "dark," not "red," to a viewer.
5. **Six hue-family bands** (`band_red/warm/green/blue/purple/dark`):
   every pixel gets bucketed into one of six mutually exclusive families
   (a pixel counts as "dark" if it's too dim, too grey, or too desaturated
   to have a meaningful hue at all — `v < 0.12 OR L* < 15 OR s < 0.15`).
   These six shares always sum to 1.0 per poster; averaged per decade
   across many posters, they're what the project's "Color River" chart is
   built from downstream (aggregation is out of scope for this repo — see
   the README).
6. **Dominant 5-color palette**, via **saturation-weighted k-means in
   CIELAB space** — the one genuinely non-obvious step, based on ["The
   Colour of Horror" (ACM, 2022)](https://dl.acm.org/doi/10.1145/3532719):
   sample 4,000 pixels with probability proportional to `0.25 + saturation`
   (not uniformly), then run k=5 k-means on that weighted sample in LAB
   space, sorted by cluster size. Why weighted: an unweighted sample from
   a poster that's 70% flat dark background returns a palette that's
   mostly near-identical shades of that background — technically the most
   common colors, but not what a human would call "the poster's palette."
   Weighting toward saturated pixels still respects a genuinely monochrome
   poster (the `0.25 +` floor keeps neutrals from being weighted to zero)
   while keeping large flat backgrounds from swallowing every cluster.

### Why saturation-weighted, not uniform, k-means

This is the one methodological choice worth explaining rather than just
stating. A quick way to see the difference: run k-means uniformly on a
poster where a black background covers 70% of the frame and a red title
covers 5% — with 5 clusters and uniform sampling, expected cluster
occupancy roughly follows pixel frequency, so 3-4 of the 5 clusters land
inside the "black" region (as slightly different, functionally
indistinguishable near-black shades), and the red title may not get a
cluster of its own at all. Weighting the *sampling* by saturation (not
reweighting the k-means loss itself, which is a different and more
complex approach) means saturated regions are proportionally
over-represented in the 4,000-pixel sample the clustering actually sees,
so a small saturated region is far more likely to earn its own cluster.

### What this deliberately does NOT do

- No semantic understanding of what's in the poster — this is a pure
  pixel-color analysis, run before (and independent of) anything in
  `vision-validation-framework`'s CLIP/vision-LLM work.
- No perceptual color-difference metric (ΔE) between clusters — clusters
  can end up close together in LAB space if the poster's actual palette is
  genuinely that narrow; this isn't corrected for.
- No color-blindness-aware analysis. The hue bands and "blood red"
  detection use standard hue-wheel boundaries, not a simulated
  color-vision-deficiency transform.
- No AWS Rekognition. The real project also ran Rekognition's
  `IMAGE_PROPERTIES` feature over the same posters (brightness, contrast,
  sharpness, dominant colors as named CSS colors) — that's real,
  already-run data, not something skipped by oversight. It's not ported
  here on purpose: it's a black-box managed service (no published
  algorithm to cite or verify, unlike CIELAB + the ACM k-means method
  above), it costs money per image for a signal that substantially
  overlaps what this script already computes for free and
  deterministically, and the one piece that's genuinely different
  (human-readable color *names*, not just hex) is a cheap local
  post-process on this script's own palette output if it's ever needed,
  not a reason to stand up a second AWS-dependent script.

### Why this metric existed before any of the others

Color was the first metric category built, and not arbitrarily: it was
the real go/no-go gate the project used before investing in any of the
other categories (CLIP/SigLIP, perceptual quality, geometric
composition). The premise being tested: horror posters get visually
darker over time. If the pre-1970 vs. 1970-2009 mean brightness gap
turned out to be weak, that's a signal the cultural assumption isn't
actually supported by the data, and the project's premise needed
rethinking before sinking more time into metrics that assume it's true.

That comparison is aggregation logic (grouping many posters' output by
decade), which is out of scope for this repo on purpose — see the Scope
note in the README — so it isn't something `01_color_metrics.py` computes
or that lives here. See docs/RESULTS.md for what the real 63,127-poster
corpus found, kept here as the historical context for why this category
was built first, not as a description of anything in `scripts/`.

## Perceptual quality

Three no-reference (no "correct" comparison image needed) image-quality
scores per poster, each from a differently-trained model, on purpose —
see each script's own docstring for why that specific model was chosen:

- **`02_iqa_multi_score.py`** — `clipiqa` (CLIP-based quality/naturalness),
  `musiq` (KonIQ-trained, general photographic quality), and `brisque`
  (a classic non-deep-learning statistical baseline), all via the `pyiqa`
  toolbox.
- **`03_nima_score.py`** — NIMA (InceptionV2, trained on the AVA aesthetic
  ratings dataset).
- **`04_laion_aesthetic_score.py`** — the LAION aesthetic predictor (CLIP
  ViT-L/14 embedding + a small trained MLP head), originally built to
  filter LAION-5B for Stable Diffusion training.

This is quality/naturalness/aesthetic-appeal, not "is there a creature in
this poster" — a different axis from color or (eventually) semantic
CLIP/SigLIP metrics, not a replacement for them.

### These are model-based, not deterministic math — verification looks different than color

Color's per-poster verification (docs/RESULTS.md) reproduced the real
project's already-computed values to within floating-point noise, because
CIELAB conversion and k-means with a fixed seed are fully deterministic
given the same input pixels. The three quality scripts here are neural
network inference instead, and a live check against the same reference
data showed noticeably larger deviations (tens of points on musiq/brisque's
0-100ish scales, not the sub-0.2 noise color showed) — plausibly because
the exact poster image bytes originally scored are no longer available to
compare against bit-for-bit (the private project's local poster cache
predates this check and wasn't reachable to verify against), and/or
pretrained model checkpoint versions can drift over the months between
when the original values were computed and when this repo was built. Each
script *is* internally deterministic — re-running it against the same
cached image file reproduces identical output every time, confirmed live
— the open question is only how closely a value computed today matches a
value computed by (possibly) a different model checkpoint months ago, not
whether the script itself is behaving consistently. Treat these three
scripts as correct, faithful ports of the real methodology; don't treat a
specific historical numeric value as something a fresh run is guaranteed
to reproduce exactly, the way color's CIELAB math is.
