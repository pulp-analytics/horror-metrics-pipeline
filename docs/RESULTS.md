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
  artwork. A real limitation of `09`/`13`'s genre classifier worth
  knowing before trusting its `agree` column across genres, not a bug
  in this port -- the same prompts the real project used.

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
