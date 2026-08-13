# Results

These are real findings from the project, kept here as context for why
the color-metrics category exists and was built first — not a description
of anything computed by `scripts/`. Aggregating per-poster output into
decade-level trends and a go/no-go verdict is explicitly out of scope for
this repo (see the README's Scope note); that logic lives outside it.

## The real checkpoint: full 63,127-poster corpus

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

## Sample run (99 posters, this repo's checked-in sample)

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

## Genre-agnostic, verified

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
