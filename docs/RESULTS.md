# Results

## Color

These are real findings from the project, kept here as context for why
the color-metrics category exists and was built first — not a description
of anything computed by `scripts/`. Aggregating per-poster output into
decade-level trends and a go/no-go verdict is explicitly out of scope for
this repo (see the README's Scope note); that logic lives outside it.

### The real checkpoint: full 63,127-poster corpus

The full color-analysis result for every horror poster in the corpus at
that point (`data/posters.csv` from the real project), aggregated by
decade and compared against the go/no-go threshold the project actually
used: is the pre-1970 vs. 1970-2009 mean-brightness gap bigger than 3 L*
points (CIELAB lightness, 0-100 scale)?

```
        brightness  red_share  dark_share
decade
1890         32.37       0.02        0.47
1900         36.28       0.01        0.36
1910         45.59       0.03        0.27
1920         41.95       0.04        0.34
1930         43.17       0.06        0.32
1940         44.92       0.09        0.29
1950         46.82       0.10        0.25
1960         45.11       0.10        0.31
1970         41.61       0.12        0.35
1980         36.54       0.11        0.39
1990         36.12       0.12        0.39
2000         32.14       0.12        0.46
2010         29.85       0.10        0.49
2020         28.28       0.11        0.52

Pre-1970 mean brightness: 44.8 | 1970-2009 mean brightness: 35.0 | gap: 9.8
-> CONTINUE -- the curve exists
```

A 9.8-point gap — roughly triple the >3 threshold. The trend is also
monotonic decade-over-decade from the 1950s peak (46.8) onward, not just a
before/after step, and `red_share` and `dark_share` move the same
direction over the same span (blood-red content roughly doubling from the
1920s-40s to the 1970s-2020s, near-black pixel share rising from ~30% to
over 50%) — three independent signals from the same pixel data pointing
the same way, not one metric doing all the work.

(There's a stray `decade=9990` row in the raw output, one poster with a
corrupted/placeholder year value in the source dataset — a pre-existing
data-quality artifact in the upstream TidyTuesday horror-movies dataset,
excluded from the numbers above since it falls outside any real decade
range.)

### Sample run (99 posters, this repo's checked-in sample)

`data/sample_output/color_metrics.csv` is a real, stratified-by-decade
sample of 99 posters (9 per decade, 1920s-2020s) drawn directly from the
same corpus, with color metrics computed by the actual production
methodology (this repo's `01_color_metrics.py` was verified to reproduce
these exact values, live, before being checked in — brightness/saturation/
hue-band values matched to within floating-point noise, palettes matched
to within 1-2 hex digits per channel). The same decade-level aggregation
as above, applied to just this sample:

```
        brightness  red_share  dark_share
decade
1920         41.70       0.03        0.38
1930         37.13       0.08        0.42
1940         44.02       0.12        0.28
1950         41.14       0.19        0.29
1960         42.77       0.06        0.31
1970         46.16       0.07        0.28
1980         35.70       0.05        0.36
1990         25.46       0.16        0.51
2000         43.71       0.06        0.31
2010         33.47       0.11        0.46
2020         41.71       0.08        0.33

Pre-1970 mean brightness: 41.4 | 1970-2009 mean brightness: 37.8 | gap: 3.6
-> CONTINUE -- the curve exists
```

At only 99 posters the per-decade means are noisier (9 posters per decade
is not a lot — note the 1970s and 2000s decades bucking the overall trend
here, which washes out at full-corpus scale), so the gap is smaller and
noisier than the full-corpus result, but the verdict direction still
agrees. This is a useful sanity check in itself: a small honest sample
should show the same *direction* as the full run with more noise, not a
contradictory result — if it ever doesn't, that's a signal something in
the sampling or the metric itself is broken, not just noisy.

### Genre-agnostic, verified

`01_color_metrics.py` has no horror-specific logic — pixel color doesn't
know what genre its poster belongs to. Verified live (no code changes)
against a real, pure-sci-fi sample (20 posters, `sources ==
"community_scifi"` in the master dataset, none horror-tagged):
brightness/saturation/hue-band values matched the already-computed
reference to the same floating-point-noise tolerance as the horror sample
above (max abs diff 0.20 on brightness, 0-100 scale).

That same test also surfaced a real edge case in the (out-of-scope-here)
aggregation step this data eventually feeds: the sci-fi sample had zero
posters before 1970 (unsurprising — sci-fi as a poster-heavy genre on TMDB
skews modern), so the pre-1970-vs-1970-2009 comparison above had nothing
to compare for that sample. Worth knowing for whoever builds the
aggregation/front-end layer: the 1970 threshold is tied to horror's own
historical narrative, not a universal color-analysis constant, so a
sample that can't support that specific comparison is an expected
outcome, not a bug in the underlying per-poster color metrics.

## Perceptual quality

`data/sample_output/{iqa_multi_score,nima_score,laion_aesthetic_score}.csv`
are real, already-computed reference values for the same 99-poster sample
as color, pulled from the same master dataset. Unlike color, these are
**not** verified to reproduce those exact historical values on a fresh
run — see docs/METHODOLOGY.md's "These are model-based, not deterministic
math" section for why, but the short version: a live check against a
5-poster subset showed meaningfully larger deviations than color's
sub-0.2 noise:

| metric | scale | max abs diff observed |
|---|---|---|
| `clipiqa` | 0-1 | 0.25 |
| `musiq` | ~0-100 | 15.5 |
| `brisque` | ~0-100 (lower=better) | 25.6 |
| `nima_score` | ~1-10 | 0.16 |

What *was* confirmed live: each script is internally deterministic
(re-running `02_iqa_multi_score.py` against the same cached poster file
twice produced byte-identical output both times), all three scripts run
correctly end to end against real TMDB-downloaded posters with zero
errors, and the values produced are plausible and within each metric's
expected range — this is a faithful, working port of the real
methodology. What's *not* established is that a value computed today by
these scripts will numerically match a specific historical value in
`data/sample_output/` to the same tightness color achieved — the most
likely explanation is that the exact poster image bytes originally scored
aren't available anymore to compare against directly (the private
project's local poster cache predates this check), which matters more for
neural quality metrics sensitive to compression/resolution than for color
math that downsamples to 96×144 regardless of input size.

`04_laion_aesthetic_score.py` was verified live end to end too (correct
box/JSON output, no errors) but wasn't included in the table above since
the ViT-L/14 model download (~1.7GB) took long enough in this environment
that the numeric comparison wasn't re-run after the shared-cache
refactor — nothing about that refactor changes the scoring logic itself,
so the same caveat applies.

## CLIP semantic embeddings

A live 5-poster check against real historical reference values
(`data/master_dataset.csv`'s `clip_fear_axis_axis`, `clip_typography_axis`,
`clip_medium_p_painted`, `clip_census_label`) found:

| metric | scale | max abs diff observed |
|---|---|---|
| `fear_axis` (`07`) | continuous, no fixed range (typically ±0.05) | 0.00302 |
| `typography_axis` (`08`) | same shape | 0.00308 |
| `medium.p_painted` (`10`) | 0-1 | 0.1321 |
| `census` top label (`06`) | discrete | flipped in 1/5 (clown vs. uncertain, both plausible — scores 0.555 vs 0.482) |

The two continuous axis scores reproduce tightly (three-decimal agreement)
— much closer than perceptual quality's ML metrics, though not
color's sub-floating-point-noise match. `10_clip_medium.py`'s
`p_painted` showed the largest gap (up to 0.13 on a 0-1 scale) but agreed
on the discrete painted/photo call in all 5 sampled posters regardless.
`06_clip_census.py`'s single label flip happened on a genuinely
borderline case (both scores sat in the low-confidence 0.4-0.6 range,
where a small embedding difference can flip the softmax argmax) — not a
sign the taxonomy or method is unreliable, but a real illustration of
"per-poster labels are noisy by design" from the script's own docstring.
Same likely root cause as perceptual quality's gap: the exact original
poster bytes and/or open_clip library version aren't reproducible months
later, not a bug in this port.

The full 99-poster `data/sample_output/clip_embeddings.npz` was generated
live for this repo (not copied from the private project) — `06`-`09` in
this repo's own sample commands read directly from it.

### Cross-genre sanity check: does census/fear_axis/genre_classifier make sense on non-horror posters?

A live check against 18 real, pure non-horror posters (6 each of
sci-fi/thriller/mystery, `sources` column in `data/master_dataset.csv`
equal to exactly one of `community_scifi`/`community_thriller`/
`community_mystery`, no horror overlap) — not whether the scripts run
without erroring, but whether the *output is a meaningful answer* for a
poster outside the corpus these taxonomies/prompts were built against:

- **Census (`06`)**: "uncertain" or "none" dominates in all three genres
  (mystery 5/6, sci-fi 4/6, thriller 5/6) — the taxonomy correctly
  reports "no creature detected" rather than forcing a monster label onto
  a poster that doesn't have one. The few non-"none" hits are individually
  plausible (*Time Demon* → demon, *The Running Man* → masked_killer),
  not a systematic false-positive pattern. Reads as evidence the
  taxonomy generalizes safely to genres it was never tested on in
  production — see the applicability caveat in docs/METHODOLOGY.md.
- **Fear axis (`07`)**: mean axis by true genre was mystery −0.0071,
  sci-fi −0.0061, thriller −0.0023 — all slightly negative (calm-leaning),
  not saturated toward "dread" despite some prompt text mentioning
  "horror movie poster." Consistent with the real project's own
  cross-genre validation design (see docs/METHODOLOGY.md) rather than
  contradicting it.
- **Genre classifier (`09`)**: a real, notable finding — 7 of 18 (38.9%)
  non-horror posters were predicted "horror," above the ~25% baseline
  four roughly-even classes would suggest, and sci-fi specifically was
  mispredicted as horror in 4 of 6 cases (67%): *Time Demon*, *The
  Running Man*, *Déchu*, *Eureka!* all scored closer to the horror
  prototype than their own genre's. Plausible cause: the horror prompt
  ensemble ("scary and menacing imagery... blood, monsters, or a masked
  killer") captures dark/tense visual tone as much as horror-specific
  content, and that tone overlaps with moody sci-fi and thriller
  artwork. n=18 (n=6 for sci-fi) is too small to trust as a rate, though
  — just large enough to flag a pattern worth checking properly.

  **Follow-up at real scale** (`scripts/qa/validate_genre_classifier_vs_imdb.py`):
  500 real posters (125 per genre), scored against **IMDb's own genre
  tags** instead of TMDB's — an independent reference the classifier had
  no part in and this repo doesn't otherwise touch. Restricted to IMDb's
  own `movie`/`tvMovie` titleTypes (143 of 500 were shorts, videos, TV
  episodes, or had no IMDb match at all — not a fair comparison against a
  feature-film poster classifier either way), then 92 more excluded for
  tagging none of horror/sci-fi/thriller/mystery in IMDb at all
  (shorts-adjacent content and older titles IMDb itself never finished
  tagging; neither is evidence the classifier is wrong). Across the 265
  that could be scored: **60.0%** of the classifier's genre calls hold up
  against IMDb.

  Recall by true genre (strict / credited for picking a genre the film
  also legitimately has, since IMDb genres are multi-label and this
  classifier is forced to pick one): horror 56.3%/66.2%, thriller
  51.2%/63.6%, mystery 50.6%/69.6%, sci-fi 34.0%/42.0%. An earlier
  200-poster pass at this same question (no titleType filter, n=148
  scored) found a similar shape but different exact numbers — horror
  64.0%/72.0%, sci-fi 23.7%/42.1% — worth flagging on its own: even a few
  hundred posters isn't enough for these specific percentages to fully
  stabilize, though sci-fi's *lenient* recall landed within a point of
  itself across both runs (42.1% vs. 42.0%), some reassurance it's
  measuring something real and not just sampling noise.

  The mechanism keeps getting less tidy the more carefully it's checked.
  At n=148, sci-fi's most common misclassification was thriller. At
  n=265 with the titleType filter, it splits between mystery (30%) and
  thriller (24%) — and horror, the genre the original 18-poster check
  blamed, is now the *least* common misprediction for true sci-fi
  posters (12%). Horror still edges out the other three as the
  classifier's most reliable call, but the margin is modest now (56.3%
  vs. 50-51% for the rest), not the "everything collapses into horror"
  story the small sample suggested. Same bottom line as always: worth
  knowing before trusting the `agree` column across genres. Not a bug in
  this port; this is the same prompts and the same taxonomy the real
  project used, measured at three different sample sizes with three
  different answers about the *exact* mechanism — the general
  unreliability for non-horror genres is real and consistent, the
  specific story about *why* keeps moving, which is itself worth knowing
  before citing a precise number from any single run of this check.

### Nova QA (23/24): does an independent vision-LLM agree with CLIP?

`23_census_nova_qa.py` and `24_typography_nova_qa.py` port the real
project's `qa_census.py`/`qa_typography.py`: draw nothing, just ask Nova
Pro to independently classify the same poster CLIP already scored, then
compare. Same justification as the creature/weapon QA below -- these are
semantic classification calls, the kind of judgment a vision-LLM adds
real signal on, unlike composition/depth/saliency/pose's continuous
geometric measurements.

Live run, 40 real posters, `us.amazon.nova-pro-v1:0`:

| script | metric | result |
|---|---|---|
| `23` (census, `06`) | exact label agreement (raw strings, pre-mapping) | 8/40 (20.0%) |
| `24` (typography, `08`) | exact register agreement | 30/40 (75.0%) |
| `24` (typography, `08`) | exact-or-adjacent-register agreement | 39/40 (97.5%) |

Very different agreement rates, and both make sense on inspection.
Census's 20% was mostly a labeling-convention artifact, not disagreement
about content: CLIP's `06` outputs the literal string `"uncertain"` for
any low-confidence poster (27 of this sample's 40 — most posters
genuinely have no creature), but Nova is never offered `"uncertain"` as
an option and always picks a real category or `"none"`. `23`'s `agree`
column now maps CLIP `"uncertain"` → `"none"` before comparing (same
convention as `06`'s famous-poster validate), so a re-run of this QA
counts those rows as agreement when Nova also says `"none"`. The 8/40
figure above is the historical unmapped exact-string rate from that live
run; it was not recomputed here. Spot-checking the reasons still finds
real remaining disagreements: e.g. poster 870056 ("House of Dracula"),
CLIP said `"uncertain"`, Nova said `"vampire"` because it read the
*title text* on the poster -- a channel CLIP's embedding-similarity
method doesn't use the same way. That row stays a disagreement after
the mapping. Typography's 75%/97.5% is a much more direct, high
agreement result: title lettering style is a lower-ambiguity call than
"is there a monster," and the 5-register spectrum gives partial credit
(`agree_adjacent`) for being one bucket off, which is where most of the
remaining 15/40 landed.

## SigLIP semantic embeddings

Same 5-poster live check as CLIP above, against `data/master_dataset.csv`'s
`siglip_fear_axis_axis`, `siglip_typography_axis`, `siglip_census_label`,
and `siglip_genre_pred_genre`:

| id | fear_axis diff | typography_axis diff | census label | genre |
|---|---|---|---|---|
| 2969 | 0.00009 | 0.00015 | witch = witch | horror = horror |
| 14594 | 0.00547 | 0.01017 | none = none | mystery = mystery |
| 19131 | 0.00909 | 0.00013 | uncertain = uncertain | horror = horror |
| 19169 | 0.01324 | 0.00431 | uncertain = uncertain | horror = horror |
| 19971 | 0.00206 | 0.00159 | witch = witch | mystery = mystery |

Both continuous axes stay in roughly the same tightness band as CLIP's
versions (max abs diff 0.0132 here vs. CLIP's 0.0031 — CLIP's numeric
match happened to be tighter on this particular 5-poster sample, not a
general claim about which model reproduces better). The discrete outputs
matched exactly in all 5 cases: 5/5 census labels and 5/5 genre
predictions agree with the historical reference, where CLIP's census
flipped 1/5. Same likely root cause as CLIP's gap (original poster
bytes/library versions not reproducible months later), not a
SigLIP-specific issue.

`13_siglip_reanalysis.py`'s genre classifier also confirmed live:
48.5% agreement between SigLIP's zero-shot pred_genre and each poster's
actual catalog genre (`siglip_genre_true_genre` from the master dataset,
merged in for this one run since the checked-in `sample_100_posters.csv`
has no genre column of its own — see the script's own `--true-genre-col`
flag) across the same 99-poster sample. Not directly compared against
CLIP's genre-agreement rate here since CLIP's `data/sample_output/
genre_classifier.csv` was built from a different, larger multi-genre real
QA sample, not these same 99 posters.

The full 99-poster `data/sample_output/siglip_embeddings.npz` was
generated live for this repo (not copied from the private project) —
`12` and `13` read it directly.

## Faces

`14_face_detect.py`'s `--validate` mode (7 hand-verified posters, real
expected face counts) run live:

```
title                                          expected  detected   boxes
Resident Evil: Welcome to Raccoon City                4         3 OK
Scream                                                 6         1 FAIL
Get Out                                                1         1 OK
Psycho                                                 3         2 OK
The Exorcist                                           1         0 OK
Us                                                      1         1 OK
Halloween                                              0         0 OK

VALIDATION: 6/7 within tolerance
```

*Scream* (1996) is a real, known YuNet limitation on that specific
poster's composition (detected 1 of an expected 6±1 tolerance) — not
something introduced by this port; the real project's own validation set
carries the same tolerance bands for exactly this reason.

Reproduction against `data/master_dataset.csv`'s real `faces_*` columns,
for the subset of this repo's 99-poster sample that has that data (37 of
99 — the real project hadn't finished running face detection against the
full corpus at export time):

| metric | scale | result |
|---|---|---|
| `n_faces` | integer count | exact match, 37/37 |
| `face_area` | 0-1 | max abs diff 0.034 |

The tightest reproduction in this repo after color — expected, since
YuNet is a deterministic CV model (not a large pretrained transformer
sensitive to library-version drift the way CLIP/SigLIP/pyiqa are).

`15_face_expression.py` against the real historical
`data/qa/face_expression.csv` (matched on `id`+`face_idx`, 159 of 160
live-detected faces found in the historical file):

| metric | result |
|---|---|
| label exact match | 114/159 (71.7%) |
| score diff, agreeing labels | max 0.353, mean 0.046 |
| disagreements near the 0.35 confidence threshold (both scores < 0.45) | 40/45 (89%) |

Lower agreement than the whole-poster CLIP scripts (census, fear_axis,
etc.), but the *shape* of the disagreement explains why: face crops are a
much smaller, lower-resolution image region than a whole poster, so the
same "poster bytes / library versions aren't reproducible months later"
gap this repo's CLIP scripts already show gets amplified here — the large
majority of mismatches are cases where both runs were already
low-confidence, not one run being confidently right and the other
confidently wrong.

## Geometric composition

`16_geometric_composition.py` ports the real project's `multi_analyze.py`
verbatim (same OpenCV calls, same parameters) -- five independent metric
groups from one downsampled frame per poster: `composition` (symmetry,
negative space, visual complexity, center of mass), `typography` (MSER
text coverage), `grid` (layout-block alignment + rule-of-thirds), `aesthetic`
(saliency-based balance, hue-scheme harmony), and `diagonal` (Hough-line
diagonal share, pyramid/funnel weight-shift).

Reproduction against `data/master_dataset.csv`'s real `clip_attributes_*`
columns (30-poster random sample, same OpenCV version, 4.10.0, on both
sides -- posters read directly from the real project's own local files,
not re-downloaded, to rule out a resolution difference from TMDB's CDN):

| metric group | columns | max diff | mean diff | within 0.005 |
|---|---|---:|---:|---:|
| composition/aesthetic/diagonal (Sobel/Canny/saliency/HSV-histogram/Hough) | `symmetry`, `neg_space`, `complexity`, `mass_x/y`, `balance`, `harmony`, `diagonal_score`, `pyramid_shift` | 0.01-0.57* | 0.001-0.07 | 16-29/30 |
| grid's contour-based half | `thirds_dist` | 0.02 | 0.00 | 27/30 |
| **MSER-text-dependent** | `text_area`, `text_y`, `text_regions`, `align_score`, `n_blocks` | up to 614 (region *count*) | large | 1-9/30 |

\* `harmony`'s one 2.0 outlier is a single sentinel-value disagreement
(-1 vs. a real score on a poster with an ambiguous hue histogram), not
representative of the other 29 posters.

**MSER (`cv2.MSER_create`) is the real, unresolved reproduction gap here**,
not the port's logic: every metric computed from the *same* resized
grayscale frame reproduces closely (often exactly) except the ones that
run MSER glyph-region detection on that identical frame. Confirmed this
isn't environment noise -- re-running this repo's own MSER call on the
same file three times in a row gives byte-identical output every time
(deterministic within a run), and forcing single-threaded OpenCV
(`cv2.setNumThreads(1)`) doesn't change the result either. The most
likely explanation: the real project's posters on disk today may not be
byte-identical to what `multi_analyze.py` originally analyzed (e.g. a
later poster-quality backfill pass) -- MSER's blob detection is far more
sensitive to small pixel-level differences (recompression noise, a
slightly different source resolution before downsampling) than a
gradient mean or a color histogram is, which would produce exactly this
pattern: near-exact agreement on aggregate/statistical metrics, wild
disagreement on "count the small blobs" ones. Unresolved, flagged here
rather than silently claimed as a clean port -- same spirit as the
Faces section's YuNet/*Scream* tolerance-band note above.

Not tied to horror specifically, same claim as `01_color_metrics.py`:
none of the five metric groups have genre-specific logic.

## Depth

`17_depth_estimation.py` (MiDaS_small via `torch.hub`) reproduction
against `data/qa/depth_score.csv`'s real values (15-poster random
sample, posters read directly from the real project's own local files):

| metric | max diff | mean diff |
|---|---:|---:|
| `mean_depth` | 0.0000 | 0.0000 |
| `p95_depth` | 0.0000 | 0.0000 |
| `depth_std` | 0.0000 | 0.0000 |
| `close_area_frac` | 0.0000 | 0.0000 |

Byte-exact, 15/15 -- the cleanest reproduction in this repo. MiDaS_small
is a small, fully deterministic CNN (no dropout/sampling at inference,
no MSER-style pixel-noise sensitivity like the geometric composition
category above), run through the exact same `torch.hub` load path and
model version (`MiDaS_small`) the real project used.

## Saliency

`18_saliency_prediction.py` ports the real project's MSI-Net script.
It was blocked for part of this porting effort: loading the model's
legacy TF SavedModel via `tf.keras.layers.TFSMLayer` crashes the whole
Python process outright --

```
[libprotobuf FATAL google/protobuf/message_lite.cc:353] CHECK failed: target + size == res:
libc++abi: terminating due to uncaught exception of type google::protobuf::FatalException
```

-- not a catchable Python exception, so no amount of try/except could
work around it from inside the script. Root cause, confirmed by testing:
this is **not** a Keras-specific bug -- calling the low-level
`tf.saved_model.load()` on the exact same SavedModel directory (bypassing
`TFSMLayer`/Keras entirely) hits the identical crash restoring the same
graph. The model's weights are embedded as graph constants rather than a
normal variables file (`variables.data-00000-of-00001` is only 1.1KB next
to a 99.8MB `saved_model.pb`), and restoring that large embedded-constant
graph trips a real bug in protobuf's default C++/upb parser backend.

**Fix**: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before
`tensorflow` is imported anywhere in the process, forcing protobuf's pure-
Python implementation for this one script. This is a runtime backend
switch, not a package change -- no version in `requirements.txt` or the
shared environment changes, so it can't break any other category sharing
that environment. `18_saliency_prediction.py` sets this itself at the top
of the module, before importing tensorflow, so no extra setup is needed
to run it. Confirmed not a corrupted download (re-fetched cleanly, same
crash before the fix) or a threading issue.

Reproduction against `data/qa/saliency_score.csv`'s real values (8-poster
random sample, posters read directly from the real project's own local
files):

| metric | max diff | mean diff |
|---|---:|---:|
| `peak_x` | 0.0000 | 0.0000 |
| `peak_y` | 0.0000 | 0.0000 |
| `top10pct_mass` | 0.0000 | 0.0000 |
| `mean_saliency` | 0.0000 | 0.0000 |

Byte-exact, 8/8 -- same as depth, MSI-Net is a small deterministic CNN
(no dropout/sampling at inference) and the pure-Python protobuf backend
only changes how the SavedModel is deserialized, not the tensor math run
afterward.

## Pose

`19_pose_dynamism.py` (YOLOv8n person detection + ViTPose COCO-17
keypoints) reproduction against `data/qa/pose_score.csv`'s real values
(12-poster random sample, restricted to posters the real project
recorded at least one detected person for):

| metric | max diff | mean diff | n |
|---|---:|---:|---:|
| `n_persons` | 0 | 0 | 12/12 |
| `kpt_bbox_area_frac` | 0.0001 | 0.0000 | 12/12 |
| `limb_asymmetry` | 0.0001 | 0.0000 | 10/12 (2 posters: both real and this port agree no torso keypoints were confident enough to score it) |
| `mean_kpt_confidence` | 0.0000 | 0.0000 | 12/12 |

Essentially exact -- floating-point rounding noise only, same YOLOv8n
checkpoint (`yolov8n.pt`) and same ViTPose checkpoint
(`usyd-community/vitpose-base-simple`) the real project used, both
deterministic at inference (no sampling).

Unlike the real project's own version, this port doesn't pre-filter
`--in` to ids with a YuNet-detected face -- see the module docstring for
why (a full-corpus compute-cost optimization, not a correctness
requirement; `n_persons=0` is a legitimate answer this port computes
directly instead of skipping).

## Creature/weapon detection

Two independent zero-shot open-vocabulary object detectors, run over
the same 18-phrase creature vocabulary and 12-phrase weapon vocabulary:
`20_creature_weapon_owlv2.py` (OWLv2, `google/owlv2-base-patch16`) and
`21_creature_weapon_dino.py` (Grounding DINO,
`IDEA-Research/grounding-dino-tiny`). Two detectors exist because one
alone isn't trustworthy: a blind Nova Pro QA pass over the real
project's full-corpus OWLv2 output found roughly 60%+ of its "creature
detected" boxes were false positives. Neither script's raw output
should be read as ground truth by itself -- agreement between the two
on the same poster is the real signal. `25_creature_weapon_agreement.py`
writes that join: a same-label box pair is kept only when IoU >= 0.3
(illustrated-poster boxes are looser than COCO; override with `--min-iou`).
`creature_label_agree` / `weapon_label_agree` are a looser check (both
detectors' top_label strings match) that does not require overlap.

Checked-in sample (`data/sample_output/creature_weapon_agreement.csv`,
99 posters, `--min-iou 0.3`). Context: OWLv2 fired a creature box on 72
posters and a weapon box on 28; DINO on 97 and 70; both detectors fired
on 71 creature / 26 weapon. Agreement is a small slice of that:

| signal | posters | of 99 |
|---|---:|---:|
| creature box agreement (`creature_n` > 0) | 9 | 9.1% |
| weapon box agreement (`weapon_n` > 0) | 9 | 9.1% |
| creature top-label agreement | 4 | 4.0% |
| weapon top-label agreement | 6 | 6.1% |

So 9/71 co-detected creature posters (12.7%) and 9/26 co-detected
weapon posters (34.6%) share a same-label overlapping box. Box
agreement can exceed top-label agreement: a matching pair need not be
each detector's highest-scoring box (6 of 9 creature box-agreements
have mismatched `top_label`s). Creature agreed labels are skewed to
the known OWLv2 failure mode (`vampire` 6, `giant_monster`/`bird`/`doll`
1 each) -- two detectors independently drawing a face-sized "vampire"
box still isn't ground truth. Weapon agreed labels look more like real
objects (`gun` 6, `sword`/`axe`/`arrow` 1 each). Cite the agreement
CSV, not `creature_weapon_owlv2.csv` or `creature_weapon_dino.csv`
alone, and don't treat even the intersection as verified presence.

Reproduction against the real project's own output
(`data/creature_boxes.json`/`data/weapon_boxes.json` for OWLv2,
`data/creature_boxes_dino.json`/`data/weapon_boxes_dino.json` for
Grounding DINO), same 8-poster sample, comparing `n` and `top_label`
per category:

| detector | creature match | weapon match |
|---|---:|---:|
| OWLv2 (20) | 7/8 exact | 8/8 exact |
| Grounding DINO (21) | 8/8 exact | 8/8 exact |

OWLv2's one mismatch (id `18405`): real recorded `n=2`, this port got
`n=1`, same `top_label` -- one fewer box past the score threshold,
not a different top detection. Everything else, both detectors, matched
exactly, including agreement on zero-detection posters and on which
specific weapon (e.g. `machete`) was found.

Both detectors run the exact same 30 label phrases, area filter
(0.002-0.95 of image area), and top-`MAX_BOXES=3`-by-score truncation
as the real project's scripts -- the only source of nondeterminism at
inference is floating-point kernel scheduling, which these results show
doesn't move the discrete outputs (`n`, `top_label`) at this sample size.

Not tied to horror specifically -- the vocabulary (vampires, zombies,
knives, chainsaws, etc.) skews toward horror/thriller imagery by
construction, but the detectors themselves are general-purpose.

### Nova QA (22): the ~60%+ false-positive claim, made reproducible

The "roughly 60%+ of OWLv2's boxes were false positives" claim above is
a citation from the real project's prior run of its own `qa_creature_weapon_boxes.py`
-- until now, that script itself had never been ported, so the finding
could only be taken on faith. `22_creature_weapon_nova_qa.py` ports it:
draws the detected box in red on the poster, asks Nova Pro whether it's
really there.

Live run, 29 real OWLv2 detections (15 posters spanning multiple genres,
not exclusively horror -- see caveat below), `us.amazon.nova-pro-v1:0`:

| verdict | n | % |
|---|---:|---:|
| `false_positive` | 23 | 79.3% |
| `correct` | 4 | 13.8% |
| `uncertain` | 2 | 6.9% |

Higher than the original ~60% figure, and the reasons why line up with a
real, honest difference in sample composition rather than a worse
detector: this 15-poster sample was pulled from the real project's own
`creature_boxes.json`/`weapon_boxes.json` history (the same ids used for
20/21's reproduction test above) to guarantee real detections to QA, not
filtered to horror/genre content the way a stratified full-corpus sample
would be -- it includes *Citizen Kane*, *Brazil*, *2001: A Space
Odyssey*, *Memento*: films with essentially zero real creature content,
where OWLv2's hits are almost certainly spurious by construction. Nova's
reasoning on those is exactly the failure mode the finding describes --
e.g. id 15 (*Citizen Kane*), OWLv2 said `"vampire"` (score 0.218-0.289),
Nova's `actual`: `"man"` / `"woman"`; id 77 (*Memento*), `"vampire"`
(0.203) → `"man's face"`. The two `correct` creature hits both came from
posters that genuinely have the thing (*Brazil* → real `"vampire"` visual
motif at 0.456; *Mars Attacks!* → real `"zombie"`/alien imagery at
0.408), and the one weapon detection scored (*A History of Violence*,
`"gun"`) was also confirmed correct -- so the mechanism isn't rubber-
stamping "false_positive," it's tracking something real.

Not wired into `compute_metrics.asl.json` -- see the module docstring
for why (a QA/spot-check tool, not a pipeline stage), and this needs real
AWS credentials, unlike every other script in this repo except 23/24.

**At real scale**: the 15-poster run above was a mechanism check, not a
finding -- too small and deliberately not stratified toward genre
content. A second run against `data/creature_boxes.json`/
`weapon_boxes.json`'s real values (130,093 posters with a local poster
file, every OWLv2 detection the real project ever recorded, not a
fresh re-run of the model), `--n 1000` (this script's own stratified
sampling: 60% low-confidence/25% mid/15% high, same as the private
original), `us.amazon.nova-pro-v1:0`:

| verdict | n | % |
|---|---:|---:|
| `false_positive` | 625 | 62.5% |
| `correct` | 266 | 26.6% |
| `uncertain` | 109 | 10.9% |

(1 of 1,001 sampled rows errored on a transient failure, excluded above.)
**62.5% -- this is now a real, precise, reproducible number, not a
citation of the private project's own prior estimate**, and it lands
almost exactly on the "~60%+" figure that original run reported. Split
by kind, creature detections are noisier than weapon:

| kind | n | `false_positive` | `correct` | `uncertain` |
|---|---:|---:|---:|---:|
| creature | 661 | 64.9% | 20.0% | 15.1% |
| weapon | 339 | 57.8% | 39.5% | 2.7% |

Makes sense on reflection: the creature vocabulary includes ambiguous
categories that overlap with ordinary poster content by construction
(`vampire`/`zombie`/`masked_killer` can all plausibly just be "a person"
in a bad crop), where weapon labels (`knife`, `gun`, `chainsaw`) describe
concrete objects with much less room for a defensible "well, sort of"
verdict -- consistent with `uncertain` being 5.6x rarer for weapons than
creatures.
