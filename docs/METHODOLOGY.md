# Methodology: color metrics

## What gets computed, per poster

`01_color_metrics.py` downloads each poster from TMDB's public image CDN
(`w342` size — enough resolution for color work, small enough to keep
downloads fast) and, per poster:

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
   These six shares always sum to 1.0 per poster and are what
   `02_aggregate_and_checkpoint.py` averages per decade into the "Color
   River" chart.
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

## Why saturation-weighted, not uniform, k-means

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

## What this deliberately does NOT do

- No semantic understanding of what's in the poster — this is a pure
  pixel-color analysis, run before (and independent of) anything in
  `vision-validation-framework`'s CLIP/vision-LLM work.
- No perceptual color-difference metric (ΔE) between clusters — clusters
  can end up close together in LAB space if the poster's actual palette is
  genuinely that narrow; this isn't corrected for.
- No color-blindness-aware analysis. The hue bands and "blood red"
  detection use standard hue-wheel boundaries, not a simulated
  color-vision-deficiency transform.

## The checkpoint: why this matters before building anything else

`02_aggregate_and_checkpoint.py`'s "CONTINUE/PIVOT" verdict isn't
decoration — it's the actual go/no-go gate the real project used before
investing in any of the other metric categories (CLIP/SigLIP, perceptual
quality, geometric composition). The premise being tested: horror posters
get visually darker over time. If the pre-1970 vs. 1970-2009 mean
brightness gap is weak (≤3 L* points, the threshold in the script), that's
a signal the cultural assumption isn't actually supported by the data, and
the project's premise needs rethinking before sinking more time into
metrics that assume it's true. See docs/RESULTS.md for what the real
63,127-poster corpus found.
