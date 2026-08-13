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

## CLIP semantic embeddings

Five scripts, all zero-shot (no training, no labeled examples -- just
comparing a poster's CLIP embedding against text-prompt "prototypes"),
all sharing one embedding cache built once:

- **`05_clip_embed.py`** — embeds every poster with CLIP ViT-B/32 once,
  caches the result (`.npz`, L2-normalized 512-d vectors). Everything
  below except `10_clip_medium.py` reads this cache instead of
  re-embedding, so they run in seconds.
- **`06_clip_census.py`** — "Monster Census": which of ~18 creature/
  monster labels (or "none") a poster's embedding is closest to, via a
  prompt-ensemble prototype per label (2-3 phrasings, averaged) and a
  temperature-100 softmax. Below `--min-score` (default 0.5), the label
  becomes "uncertain" instead of a low-confidence guess. Ships with a
  real, hand-verified validation set (`--validate`) — famous posters
  where someone actually looked at the artwork and recorded what should
  be detected (e.g. Godzilla 1954 → giant_monster, Jaws 1975 → shark).
- **`07_clip_fear_axis.py`** — a continuous dread↔calm score:
  `cos(embedding, DREAD_prototype) - cos(embedding, CALM_prototype)`.
- **`08_clip_typography_axis.py`** — a continuous ornate↔minimal
  lettering score, same formula shape as fear_axis. See the script's own
  docstring for the real methodological story: three other approaches
  (8 discrete style categories; OCR-cropped title classification;
  MSER+CLAHE-cropped classification) were actually tried and scored
  worse (6/10, 4/10, 0.72-0.78 corr respectively) before landing on this
  one (0.81 corr) — a genuine "here's what didn't work and why" result,
  not just the method that happened to get built.
- **`09_clip_genre_classifier.py`** — zero-shot: does a poster's artwork
  alone "look like" horror/scifi/thriller/mystery, compared against
  whatever its catalog genre actually is (if `--in` has a `genre` column).
- **`10_clip_medium.py`** — painted/illustrated vs. photographic, zero-shot.
  Doesn't use the shared embedding cache (embeds fresh) — see its own
  docstring for why that's a faithful port, not an inconsistency.

### Why continuous axes, not discrete categories, for fear/typography

Both fear_axis and typography_axis project a poster's embedding onto a
line between two prompt-ensemble poles rather than classifying into
buckets. This avoids a specific, real failure mode: a discrete category
like "gore" or "dripping" (typography) tends to become an *attractor* that
ends up measuring something else entirely (overall darkness, in
typography's case) rather than the thing it's named after. A continuous
axis has nowhere to hide that confusion — it's always exactly "how much
closer to pole A than pole B," nothing more. This project's own napkin
math backs it up empirically: typography's continuous version scored 0.81
correlation against hand-verified ground truth vs. 6/10 for the discrete
8-category version it replaced.

### What this repo does NOT compute for these axes

Neither `07` nor `08` bins posters into named registers
(nightmarish/dreadful/.../calm, or ornate/decorative/.../minimal) by
quantile, and neither aggregates by decade. Both are corpus-relative —
which register a poster falls into depends on the quantile distribution
of *whatever else is in the same run*, not just that one poster — so
they're aggregation logic, out of scope for this repo (see the README's
Scope note), same reasoning as color's dropped Continue/Pivot checkpoint.
The raw `axis` score this repo outputs is a pure per-poster value,
independent of anything else in the batch.

### Reproduction gap here is similar to perceptual quality, not color

A live check against 5 posters found the two continuous axis scores
(`fear_axis`, `typography_axis`) close to their historical reference
values (differences in the 0.001-0.003 range) — much tighter than
perceptual quality's ML metrics, though not the sub-floating-point-noise
match color achieved. `06_clip_census.py`'s discrete top-label pick
flipped in 1 of 5 sampled posters (a near-threshold case, "clown" vs.
"uncertain," both plausible given the actual scores were close), and
`10_clip_medium.py`'s continuous `p_painted` showed larger absolute
differences (up to ~0.08) though it agreed on the discrete painted/photo
call in all 5 cases. Same likely cause as perceptual quality: the exact
original poster bytes and/or exact library versions aren't reproducible
months later. See docs/RESULTS.md for the specific numbers.

### Genre applicability: what's real methodology vs. this repo's own judgment

Two different situations here, worth being precise about:

- **`07_clip_fear_axis.py`'s cross-genre use is real, not a guess.** The
  private pipeline's `clip_fear_axis.py` (and `siglip_fear_axis.py`)
  deliberately run the dread↔calm axis over all four corpora the real
  project had (horror, scifi, thriller, mystery) — `--genre all` is
  literally the script's default, backed by a `GENRE_FILES` dict mapping
  each genre to its own embeddings/metadata files, and the script's own
  final output is `df.groupby("genre")["axis"].agg(["mean","std","count"])`
  printed to the console. That's a deliberate validation check baked into
  the real methodology, not scope creep: does the dread axis actually
  score horror posters higher than other genres, on average, as a sanity
  check that the axis measures what it claims to? So `07`/`12` in this
  repo are faithfully genre-general, matching how the real project used
  them — not restricted to horror input.
- **`06_clip_census.py`'s creature taxonomy was never run cross-genre in
  production.** The real `clip_census.py` has no `--genre` flag and no
  `GENRE_FILES` equivalent — it only ever reads `data/clip_embeddings.npz`
  (the horror corpus). Several taxonomy labels (`alien`, `giant_monster`)
  are conceptually just as relevant to sci-fi posters, but that's this
  repo's own judgment about where the taxonomy plausibly generalizes, not
  a reproduced result — the real project simply never tested it there.
  Treat `06`/`13`'s census as verified-faithful for horror, and
  plausible-but-unverified for sci-fi; mystery/thriller posters mostly
  don't contain the kind of creature imagery this taxonomy was built to
  detect, so "none"/"uncertain" dominating there (see the cross-genre
  check in docs/RESULTS.md) is an expected, correct outcome, not a
  sign the script needs a different taxonomy for those genres.

## SigLIP semantic embeddings

Three scripts, the SigLIP counterpart to the CLIP category above — same
zero-shot method (prompt-ensemble text prototypes, cosine similarity),
same taxonomies and prompts where the real project reused them verbatim,
just through `google/siglip-base-patch16-224` (768-d) instead of CLIP
ViT-B/32 (512-d) as the backbone:

- **`11_siglip_embed.py`** — embeds every poster with SigLIP once, caches
  the result. Not interchangeable with `05`'s CLIP cache (different
  embedding space and dimensionality) — `12` and `13` read this one.
- **`12_siglip_fear_axis.py`** — the same dread↔calm continuous axis as
  `07_clip_fear_axis.py`, same prompt wording, over SigLIP embeddings.
- **`13_siglip_reanalysis.py`** — census + typography axis + genre
  classifier, all three in one script sharing a single SigLIP model load
  (the real project's own `siglip_reanalysis.py` combines them for
  exactly this reason: loading the model three times separately would
  triple a real, non-trivial cost). Same taxonomy/prompts as `06`, `08`,
  and `09` respectively.

### Why SigLIP at all, given CLIP already works

Not a replacement — a second, independently-trained model run over the
same zero-shot method, checked in `RESULTS.md`'s "SigLIP semantic
embeddings" section against real numbers from both models. SigLIP's
sigmoid loss gives measurably better zero-shot accuracy than plain CLIP
on public benchmarks (Google's own reported numbers: ~85% ImageNet
zero-shot for SigLIP2 vs. ~68-75% typical for CLIP ViT-B/32), which is
the actual reason the real project reran its zero-shot analyses on it —
worth checking whether a stronger general-purpose backbone gives a
cleaner signal on this specific task, not swapping one for the other on
faith.

### What this repo does NOT compute for the SigLIP axes

Same reasoning as CLIP's `07`/`08`: neither `12` nor `13`'s typography
axis bins posters into named registers by quantile, and neither
aggregates by decade — both are corpus-relative, out of scope for this
repo (see the README's Scope note). The real `siglip_fear_axis.py` and
`siglip_reanalysis.py` scripts compute both; ported here through the
continuous per-poster `axis` score only, same as the CLIP versions.

### Genre applicability

Same split as CLIP's `07`/`06` (see that section above): the real
`siglip_fear_axis.py` also has its own `GENRE_FILES`-equivalent
`load_meta()` pulling in `posters.csv`/`posters_scifi.csv`/
`posters_thriller.csv`/`posters_mystery.csv` and ends with `print("\n===
por genero (SigLIP) ===")` grouping the axis by genre — a deliberate
cross-genre validation check in the real methodology, same as the CLIP
version, so `12_siglip_fear_axis.py` here is genre-general by design too.
`13_siglip_reanalysis.py`'s census portion, on the other hand, shares
`06`'s taxonomy verbatim and was never run against sci-fi/thriller/
mystery embeddings in production either — same "verified for horror,
plausible-but-unverified for sci-fi" caveat applies.

### Reproduction: tighter label agreement than CLIP, similar axis-score gap

A live check against 5 posters (see docs/RESULTS.md for the exact table)
found `fear_axis` and `typography_axis` diffs in a similar range to
CLIP's versions (roughly 0.0001-0.013), and — unlike CLIP census's 1/5
label flip — all 5 census labels and all 5 genre predictions matched
their historical reference exactly. Not read as "SigLIP is more
reproducible than CLIP" from 5 samples; read as "no evidence of a
SigLIP-specific reproduction problem," consistent with the same
likely cause discussed above (poster bytes/library versions drifting
over months, not a bug in either port).
