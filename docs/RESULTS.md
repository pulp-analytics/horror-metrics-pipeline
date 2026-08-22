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

### Reconciling `is_animal`, `weapon`, `monster`, and `person` across engines -- the real rule

Several questions this repo's engines can now each independently answer
-- "is there a real animal," "is there a real weapon," "is there a real
monster/supernatural creature," "is there a real person" -- got a real,
blind human-reviewed answer for which engine(s) to actually trust, scored
against the full 145,492-poster private corpus (not just this repo's
99-poster sample; the private project had already run every relevant
engine -- CLIP census, OWLv2, Grounding DINO, Rekognition `DetectLabels`,
YOLOv8n (`19_pose_dynamism.py`'s own person count), and Nova's scene-enrich
call -- across its full corpus, so this reuses that real, already-computed
data rather than waiting on this repo's own `26_rekognition_enrich.py`/
`27_nova_scene_enrich.py` to get AWS access). Same discipline as
`25_creature_weapon_agreement.py`'s own justification: don't assume
agreement helps, or that any given engine is trustworthy alone -- measure
it, per question, against real human judgment.

**Corpus-wide positive rates first** (145,492 posters, ~131k with every
engine present):

| question | CLIP census | OWLv2 | DINO | Rekognition | Nova |
|---|---:|---:|---:|---:|---:|
| animal | 2.8% | -- | -- | 19.1% | 3.1% |
| weapon | -- | 32.7% | 63.4% | 16.4% | 12.4% |
| monster | 18.6% | 55.2% | 96.8% | n/a | 14.3% |

DINO flagging 96.8% of the *entire corpus* as containing a monster, and
63.4% as containing a weapon, is the first real tell -- no real corpus is
that saturated with monsters or weapons. Rekognition has no
monster/creature field at all (its label taxonomy is general-purpose
object/scene detection, not a supernatural-creature vocabulary), so it
was never a candidate for that question.

**Blind human review, 50 posters each** (`scripts/qa/build_signal_reconciliation_review_page.py`,
stratified toward the specific pattern "exactly one engine says yes, every
other available engine says no," plus a few agreed-positive/agreed-negative
anchors for calibration):

| question | engine | n | accuracy | precision | recall |
|---|---|---:|---:|---:|---:|
| animal | CLIP census | 50 | 90.0% | 80.0% | 50.0% |
| animal | **Rekognition** | 50 | **26.0%** | **17.8%** | 100.0% |
| animal | Nova | 50 | 90.0% | 80.0% | 50.0% |
| weapon | OWLv2 | 50 | 100.0% | 100.0% | 100.0% |
| weapon | **DINO** | 50 | **20.0%** | **11.1%** | 100.0% |
| weapon | Rekognition | 50 | 100.0% | 100.0% | 100.0% |
| weapon | Nova | 50 | 100.0% | 100.0% | 100.0% |
| monster | CLIP census | 50 | 96.0% | 60.0% | 100.0% |
| monster | OWLv2 | 50 | 96.0% | 60.0% | 100.0% |
| monster | **DINO** | 50 | **16.0%** | **6.7%** | 100.0% |
| monster | Nova | 50 | 96.0% | 60.0% | 100.0% |
| person | Rekognition | 50 | 80.0% | 85.7% | 85.7% |
| person | Nova | 50 | 76.0% | 82.9% | 82.9% |
| person | **pose (YOLOv8n)** | 50 | **40.0%** | 100.0% | **14.3%** |

**Correction, monster row**: this table's monster numbers came from a
50-poster sample with only 3 real monsters in it -- too few to trust, the
same problem water and silhouette's round 1 hit later. A properly powered
round 2 overturned the CLIP/OWLv2/Nova tie shown above: CLIP census and
OWLv2 both turned out to perform close to DINO, not close to Nova. See
"Monster: a correction" below for the real numbers and the corrected rule.
The table above is left as originally measured (not edited in place) so
the correction itself stays visible as a documented event, not a silently
rewritten history.

**Correction, animal and weapon rows too**: both had the same underpowered
problem -- animal's round 1 (8 real positives) was stratified toward
Rekognition's own disagreement pattern, not CLIP/Nova's, so its "90%/80%/
50%" CLIP/Nova numbers came from a handful of anchor cases, not a real
test; weapon's round 1 had only 5 real positives. Properly powered round
2s (36 and 56 real positives respectively) overturned both: CLIP census
scored 20.0% precision on real illustrated/anime posters (confidently
predicting "bird"/"snake" where there's no animal at all), and OWLv2
scored 63.9% precision on weapon with only 13% of its unique-disagreement
positives real -- both much closer to their era's "noisy" engines than to
Nova. Rekognition, dropped from both rows above, turned out strong on
both. See "Animal: a correction" and "Weapon: a correction" below.

The pattern repeats across animal/weapon/monster: the outlier engine has
near-perfect *recall* (it rarely misses a real positive) because it says
"yes" almost everywhere, and near-worthless *precision* as the direct
consequence. Scored specifically on the disagreement subset each review
was stratified toward (the one engine says yes, every other available
engine says no):

| question | outlier engine | n disputed | outlier was right |
|---|---|---:|---:|
| animal | Rekognition | 40 | 4/40 (10.0%) |
| weapon | DINO | 40 | 0/40 (0.0%) |
| monster | DINO | 40 | 0/40 (0.0%) |

**Person is the mirror-image failure mode, not the same one**: pose's
YOLOv8n person count has 100% precision (never wrong when it fires) but
only 14.3% recall -- it's not noisy, it's blind. Of the 20 posters in the
review where Rekognition and Nova both said yes and pose said no, 19/20
(95%) were real people YOLOv8n simply never detected -- consistent with a
detector tuned on photographic content missing illustrated/stylized
poster art. This has a real downstream consequence for
`19_pose_dynamism.py`: `pose_n_persons == 0` does not mean "no person in
this poster" the way it might be read -- it usually means detection
failed on a person who is there. The pose *dynamism* score itself (when
YOLOv8n does fire) isn't shown unreliable by this review; what's
unreliable is treating a zero as a negative rather than a missing value
when computing coverage/denominators downstream.

**The rule this repo adopted**: one deterministic model + Nova (the LLM)
+ this blind human review, not "more engines voting is safer." The
specific engine that's unreliable is a per-question finding, not a fixed
"always drop engine X" rule -- Rekognition failed animal but scored
perfectly on weapon and person; DINO failed weapon and monster; pose
failed person by omission, the opposite failure shape from DINO/
Rekognition's over-triggering. Concretely:

- **animal**: ~~CLIP census (`06`) + Nova (`27`)~~ **corrected to
  Rekognition (`26`) + Nova (`27`)** -- the round-1 CLIP numbers were a
  sample-size artifact; see "Animal: a correction" below
- **weapon**: ~~OWLv2 (`20`) + Nova (`27`)~~ **corrected to Rekognition
  (`26`) + Nova (`27`)** -- OWLv2 was never actually as good as round 1
  suggested, and the cost argument for dropping Rekognition doesn't hold
  once that's known; see "Weapon: a correction" below
- **monster**: ~~OWLv2 (`20`) + Nova (`27`)~~ **corrected to Nova (`27`)
  alone** -- the tie shown above was a round-1, 3-real-positive artifact;
  see "Monster: a correction" below
- **person**: Rekognition (`26`) + Nova (`27`) -- `pose_n_persons`
  (`19`) dropped from this vote specifically; still computed and used for
  its own purpose (pose dynamism), just not trusted as a presence signal

A real, specific example from the animal review: CLIP missed the one real
animal in an earlier, smaller 35-poster pass of this same test (*The Thin
Man Goes Home*, id 14594, a dog on the poster CLIP scored
`is_animal=False` at 0.582) and produced 2 false positives on titles
containing the word "Dragon" -- "The Dragon Murder Case" (id 51565) and
"Lady Dragon 2" (id 80342) -- consistent with CLIP's embedding picking up
the word from the title/poster text rather than genuinely detecting
animal content in the image. Neither engine in the final animal/weapon/
monster pairs above is assumed perfect going forward; each is trusted
specifically because a real blind review said so, on this specific
question, not by default.

Caveat on sample size: n=50 per question (n=40 for each disputed subset)
is real, live-verified evidence, not a definitive population statistic --
the 100%/0% results for weapon and monster in particular are clean enough
to act on (a single true positive would have changed 0% to 2.5%, not
overturned the finding), but a larger confirmatory sample is a reasonable
future step before treating these numbers as exact.

### A fifth candidate engine tested and rejected: real segmentation data, still not good enough

The private project separately ran real semantic/material/concept
segmentation over 65,201 posters (`data/segmentation.csv` in the private
repo, never ported here or merged into `master_dataset.csv`, but real,
live-computed data, not a stub) -- SegFormer-b0 trained on ADE20K (14
scene classes, % of image area per class, including `ade_animal`/
`ade_water`), a SigLIP2-based MINC-Materials read (23 material classes,
including `minc_water`), and 10 CLIP zero-shot concept scores including
`clip_weapon` and `clip_water`. On paper this looked like a promising
extra vote: corpus-wide, `ade_animal`/`clip_weapon` agreed with the
already-trusted engines 83-92% of the time.

That aggregate agreement rate was misleading -- the same lesson as
DINO/Rekognition above, that the disagreement subset is where the real
signal (and the real errors) concentrate, not the overall agreement
percentage. Blind human review, 50 posters each, same method:

| question | segmentation engine | n | accuracy | precision | recall |
|---|---|---:|---:|---:|---:|
| weapon | `clip_weapon` | 50 | **26.0%** | **26.7%** | 34.8% |
| animal | `ade_animal` (>0.01 area) | 50 | **22.0%** | **20.0%** | 28.6% |

Both catastrophically bad, and bad in *both* directions at once -- unlike
DINO (all false positives, 0% right on its own disagreements) or pose
(all false negatives, never wrong but rarely fires), the segmentation
signals got nearly everything wrong on both sides of the disagreement:
of the posters where segmentation alone said yes (OWLv2/Rekognition/Nova,
or CLIP census/Nova, all said no), only 3/25 (weapon) and 1/25 (animal)
were real; of the posters where segmentation alone said no while the
already-trusted engines unanimously said yes, 15/15 (weapon) and 15/15
(animal) -- 100% both times -- were real weapons/animals the segmentation
read completely missed. **Not added to either rule.** Flagged here
specifically because it's a real negative result on a data source that
looked reasonable at a glance (real model, real corpus-scale run, plausible
aggregate agreement) -- exactly the kind of assumption this repo's whole
reconciliation discipline exists to catch before it ships, not just the
positive results.

### Monster: a correction

The monster row in the table above -- CLIP census, OWLv2, and Nova tied
exactly at 96.0% accuracy / 60.0% precision / 100% recall -- turned out to
rest on only **3 real monsters** in the entire 50-poster sample. That's
thinner than water's round 1 (4 real positives) and silhouette's round 1
(2), both of which got caught and fixed the same way this one now is.
Corpus-wide, the numbers already looked suspicious in hindsight: OWLv2
fires on 55.3% of the full corpus, vs. CLIP census's 18.5% and Nova's
14.3% -- much closer to DINO's 96.8% over-triggering shape than to the two
engines it was supposedly tied with.

**Round 2** (100-poster review: 50 from "2+ of {clip, owlv2, nova} agree,"
25 from a dedicated "OWLv2 alone says yes" slice built specifically to test
this suspicion, 10 clip-only, 10 nova-only, 5 negative anchors -- 28 real
positives, properly powered):

| method | accuracy | precision | recall |
|---|---:|---:|---:|
| CLIP census alone | 46.0% | 22.9% | 39.3% |
| OWLv2 alone | 41.0% | 26.9% | 64.3% |
| DINO alone (unchanged from round 1) | 30.0% | 28.6% | 100.0% |
| **Nova alone** | **80.0%** | **60.0%** | 85.7% |
| OWLv2 AND Nova | 82.0% | 72.7% | 57.1% |

Of the 25 posters in this round where OWLv2 alone said yes (CLIP and Nova
both said no), only **2 (8%)** were real monsters -- OWLv2 behaves like a
second noisy engine here, not the trustworthy partner round 1 suggested.
CLIP census performs almost as poorly. Nova alone is clearly the best
single engine, well ahead of either deterministic candidate. OWLv2 AND
Nova reaches higher precision (72.7%) but at real recall cost -- it misses
12 of 28 real monsters (57.1% recall) -- the same trade this repo already
rejected for water's and fire's AND combinations, for the same reason:
losing that much recall isn't worth the precision gain when Nova alone
already clears 80% accuracy on its own.

**The corrected rule: Nova (`27`) alone**, no partner -- monster joins
water and fire as a "no deterministic engine earns a seat next to Nova"
signal, not weapon's "one deterministic model + Nova." This is the second
time this document has had to walk back a round-1 conclusion after a
properly powered round 2 (water was the first) -- worth naming plainly:
**a 50-poster review with fewer than ~10 real positives in either class
should be treated as provisional, not final**, until a bigger, resampled
round confirms it. Weapon (5 real positives) and animal (8) are the
remaining reviews in this document thin enough to warrant the same
scrutiny; person (35) is not.

**A caveat round 2 itself couldn't see: genre.** Round 2's 100 posters
skewed 52% horror / 23% scifi vs. the corpus's real 43.2% / 12.5% -- and
thriller (14% of the sample vs. 25.5% of the corpus) and mystery (8% vs.
7.4%) were both underrepresented relative to how much of the corpus they
actually are. That mattered here specifically because real monsters are
genuinely much rarer in thriller/mystery by genre convention (psychological
thrillers and detective mysteries don't usually have supernatural
creatures) -- round 2's own per-genre breakdown showed a 38.5% real-monster
rate in horror, 26.1% in scifi, but only 7.1% in thriller and **0%** in
mystery (on just 8 posters, too few to trust on their own). Since round
2's overall 60.0% Nova precision was averaged across a sample tilted
toward the genres where monsters are common, it risked overstating how
well "Nova alone" holds up specifically where they aren't.

**Round 3** (95 posters, sampled only from thriller/mystery genres --
`siglip_genre_true_genre` -- specifically to test this: 50 from Nova's own
positives in that genre pair, the direct precision test; 25 from "OWLv2
says yes, Nova says no" to check whether Nova is missing real monsters
there; 10 CLIP-only; 10 negative anchors; 92 scored after dropping "not
sure," 15 real positives):

| engine | accuracy | precision | recall |
|---|---:|---:|---:|
| CLIP census | 71.7% | 23.8% | 33.3% |
| OWLv2 | 40.2% | 21.4% | 100.0% |
| **Nova** | 65.2% | **31.9%** | **100.0%** |

The direct test confirms the concern: of the 47 thriller/mystery posters
where Nova itself said "monster," only 15 (31.9%) actually were one --
well below the 60.0% precision round 2 measured on a horror/scifi-heavy
sample. But two things hold up: Nova still beats both deterministic
engines here too (CLIP 23.8%, OWLv2 21.4% -- neither is a fix), and Nova's
*recall* stays perfect (100%; of the 25 posters where OWLv2 alone flagged
something and Nova didn't, zero were real monsters Nova missed). This
looks like a base-rate effect, not a genre-specific Nova bug: when the
true prevalence of a concept is very low, any imperfect classifier's
precision degrades mechanically, because false positives make up a larger
share of everything it flags. No engine tested here escapes that; Nova is
just the least bad of the three.

**The rule stays Nova (`27`) alone** -- nothing tested beats it in this
genre pair either -- but **its real-world precision is genre-dependent**:
roughly 60% in horror/scifi, roughly 32% in thriller/mystery. Anything
downstream that treats `nova_monster >= 0.5` as equally trustworthy across
all four genres is over-trusting it specifically on thriller and mystery
posters.

### Animal: a correction

The animal row above -- CLIP census and Nova tied at 90.0% accuracy /
80.0% precision / 50.0% recall, Rekognition rejected at 26.0% / 17.8% /
100% -- rested on a round 1 with only **8 real animals** in a 50-poster
sample. Worse than thin: that sample was stratified toward *Rekognition's*
disagreement pattern specifically ("Rekognition says yes, CLIP+Nova say
no"), not toward CLIP/Nova's own behavior -- meaning CLIP and Nova's
reported 90%/80%/50% numbers came from whatever handful of anchor/
agreement cases happened to be in that sample, never a real test of CLIP
or Nova against a population that actually matters for the animal
question.

**Round 2** (97 scored after dropping "not sure," resampled from "CLIP OR
Nova says yes" -- the rule that was actually in production -- for a real
test of that population; 36 real positives):

| method | accuracy | precision | recall |
|---|---:|---:|---:|
| CLIP census alone | 41.2% | 20.0% | 19.4% |
| Rekognition alone | 78.4% | 65.3% | 88.9% |
| **Nova alone** | 81.4% | 68.0% | 94.4% |
| CLIP OR Nova *(old rule)* | 57.7% | 46.8% | 100.0% |
| **Rekognition AND Nova** | **85.6%** | **78.9%** | 83.3% |

CLIP census isn't just weak here, it's confidently wrong in a specific,
checkable way: on illustrated/anime posters, it predicts "bird" or "snake"
with 0.5-0.7 confidence and no conflict flag on posters that have no
animal in them at all -- e.g. id 1274262 (*Space Battleship Yamato
2199*, a sci-fi anime spaceship poster) scored "bird." This is the same
domain-mismatch shape as YOLOv8n's failure on person (a detector doing
fine on the content it was tuned for, badly on stylized/illustrated
art) -- just showing up as false positives here instead of false
negatives. Rekognition, dropped in the original rule, turns out to be the
second-best single engine. **The corrected rule: Rekognition (`26`) +
Nova (`27`)**, not CLIP census -- back to a "one deterministic + Nova"
shape, just a different deterministic engine than originally published.

### Weapon: a correction

The weapon row above -- OWLv2/Rekognition/Nova all at 100.0% -- rested on
a round 1 with only **5 real weapons** in the sample, the thinnest
positive count of any signal in this document at the time (monster's 3
came later and was worse, but weapon was already thin). A clean 100% on
5 true positives and 45 true negatives isn't wrong, exactly, but it's not
enough evidence to trust the *precision* difference between engines that
all happened to score perfectly on such a small draw.

**Round 2** (97 scored after dropping "not sure," 50 from the high-
confidence "OWLv2 AND Nova agree" pool, 25 a dedicated "OWLv2 alone says
yes" slice, 15 "Nova alone," 10 negative anchors -- 56 real positives,
well powered):

| method | accuracy | precision | recall |
|---|---:|---:|---:|
| OWLv2 alone | 62.9% | 63.9% | 82.1% |
| DINO alone | 67.0% | 64.3% | 96.4% |
| Rekognition alone | 82.5% | 88.2% | 80.4% |
| Nova alone | 85.6% | 82.8% | 94.6% |
| OWLv2 AND Nova *(old rule)* | 80.4% | 87.8% | 76.8% |
| **Rekognition OR Nova** | **86.6%** | 82.1% | **98.2%** |

Of 23 posters in this round where OWLv2 alone said yes (Nova said no),
only 3 (13%) were real weapons -- the same "OWLv2 behaves like a second
noisy engine on its own unique-disagreement cases" finding that also
showed up for monster (8% there). DINO, previously rejected outright,
actually scores closer to usable here (67.0%/64.3%/96.4%) than round 1's
"20.0% accuracy" suggested -- still not good enough to adopt, but a
reminder that DINO's *monster*-specific rejection doesn't automatically
transfer to weapon at full strength either. **The corrected rule:
Rekognition (`26`) + Nova (`27`)**, not OWLv2 -- the original cost
argument for dropping Rekognition ("OWLv2 is free and already running, no
reason to also pay for Rekognition") assumed OWLv2 was Rekognition's
equal on accuracy; it isn't.

**A separate, still-open question this correction surfaces: presence vs.
localization.** All the numbers above (here and in every other signal in
this document) answer "is there a weapon/animal/monster," a yes/no
question. They say nothing about whether a given engine's *bounding box*
correctly locates that weapon in the image -- a different, still-untested
axis. Nova cannot help with this at all (a text/JSON-out LLM has no
mechanism for reliable pixel localization, the same structural gap
documented for `rek_n_boxes` in `27_nova_scene_enrich.py`'s docstring).
If the downstream need is marking or cropping the weapon's actual
location, not just flagging its presence, the candidates are
`rek_label_boxes` (`26_rekognition_enrich.py`) or OWLv2's own boxes
(`weapon_owlv2_top_label`/`_top_score` and the full box list in
`creature_weapon_owlv2.csv`) -- and which of those localizes *correctly*,
given a real weapon is present, is a real, separate reconciliation
question this document hasn't answered yet.

### Water: another signal with no deterministic partner

Water had four real candidates: `rek_water` (Rekognition), `ade_water`/
`minc_water`/`clip_water` (three separate segmentation reads -- ADE20K
scene area, MINC material, CLIP concept), and Nova's own read (parsed out
of `nova_fear_labels`, since `27_nova_scene_enrich.py`'s prompt didn't
have a standalone `water` field the way it has `weapon`/`monster`/
`person`/`animal`). Corpus-wide (65,003 posters): Rekognition 3.7%
positive, the three segmentation reads 9.9-11.3%, Nova 2.2% -- segmentation
running noticeably hotter than the other two, same shape as the animal/
weapon segmentation failures above.

**Round 1** (50-poster blind review, disagreement-stratified): all three
segmentation reads failed the same way segmentation failed on animal/
weapon (7-11% precision on their own unique-positive subset, 100%
false-negative on posters the trusted pair unanimously caught). But this
round only had 4 real positives in the whole sample -- too few to trust
Rekognition/Nova's own precision numbers (28.6%/26.7%), which looked
mediocre.

**Round 2** (100-poster review, re-sampled specifically from "Rekognition
OR Nova says yes" for a properly powered read on those two specifically):

| method | n | accuracy | precision | recall |
|---|---:|---:|---:|---:|
| Rekognition alone | 96 | 54.2% | 45.0% | 71.1% |
| **Nova alone** | 96 | **83.3%** | **80.6%** | 76.3% |
| either (OR) | 96 | 60.4% | 50.0% | 100.0% |
| both (AND) | 96 | 77.1% | 90.0% | 47.4% |

With a properly balanced sample (38 real positives, not 4), the picture
flips: **Nova alone clearly beats Rekognition alone**, and beats both
combination rules too -- OR keeps the recall but tanks precision (every
one of Rekognition's extra false positives comes along for free), AND
gets better precision than Nova alone but at real recall cost (misses
half the real water). **Water doesn't fit this repo's "deterministic +
Nova" pattern** -- there's no deterministic engine worth pairing Nova
with here (segmentation is rejected above, Rekognition alone is worse
than Nova alone). The rule: **Nova (`27`) alone**, no partner.

### Rekognition's overall cost-benefit ledger, once all five signals were tested

With animal/weapon/monster/person/water all scored, `26_rekognition_enrich.py`'s
`DetectLabels`+`DetectFaces` call earns its keep on some fields and not
others -- worth listing plainly rather than leaving as a single "worth it
or not" verdict, since the answer differs by field:

**Kept, real reason:**
- `rek_person` -- one of the 2 trusted engines for person (80.0% accuracy)
- `rek_age_lo`/`rek_age_hi`/`rek_gender`/`rek_emotion` (face demographics)
  -- no other engine in this repo computes this at all
- `rek_bright` -- cross-validated against `01_color_metrics.py`'s
  independent CIELAB brightness (Pearson 0.929, Spearman 0.948 across
  129,072 posters, both free of human review since it's an objective
  numeric check, not a subjective judgment)

**Dropped, tested and beaten by a free alternative:**
- `rek_animal`, `rek_weapon` -- see the reconciliation rule above
- `rek_water` -- beaten by Nova alone (see above)
- `rek_colors` -- compared against `color_palette` (CIELAB k-means, free)
  across 129,072 posters: 87.6% land in the same general color tone
  (RGB Euclidean distance < 100), luminance correlation 0.789. Decent but
  not tight enough to justify the API cost when a free method already
  agrees with it most of the time. A third, independently-implemented
  classical algorithm (PIL's built-in median-cut quantization, 300-poster
  sample, real fetch+compute, no model download) agrees with the free
  CIELAB k-means method more often (66.3% of posters) than with
  Rekognition (33.3%) -- two free, independent classical algorithms
  converge on each other more than either converges on the paid API,
  which is real evidence the free method is the more standard answer,
  not that Rekognition is simply "different."
- `rek_n_faces` -- compared against `faces_n_faces` (YuNet, local/free)
  across 131,035 posters: 74.1% exact match, Pearson correlation 0.879.
  Real disagreement concentrates on crowded/collage posters. A 20-poster
  human recount of the most extreme disagreements found the *higher* of
  the two numbers is usually (not always) closer to the truth -- both
  engines tend to undercount on busy posters rather than invent faces,
  with two clean, documented counter-examples: *Opus Sanguinis* (id
  1719230, YuNet said 22, Rekognition said 6, real count is 1 -- both
  fooled by numerous background skulls, a real, understandable failure
  mode: skull anatomy pattern-matches "face" for both detectors) and
  *Beringin* (id 455460, YuNet said 10, Rekognition correctly said 0 --
  a real YuNet false-positive burst Rekognition avoided). `rek_n_faces`
  doesn't independently justify the API call, but is free additional
  signal once the call is already happening for other reasons (a
  `max(faces_n_faces, rek_n_faces)` rule beats either alone on the
  disagreement tail).

**No competitor exists to test against at all:**
- `rek_sharp`, `rek_contrast` -- confirmed via a full search of the
  private project's ~144 scripts: nothing else in this pipeline computes
  image sharpness or contrast as its own metric. Neither validated nor
  contradicted; just genuinely unique.
- `rek_n_boxes` (count of localized object *instances* from `DetectLabels`'
  `Instances` field) -- and this one stays that way structurally, not for
  lack of trying: a vision-language model prompted for JSON text output
  can't do per-object pixel localization the way a real detector does.
  `27_nova_scene_enrich.py`'s docstring documents this explicitly so it
  doesn't get re-proposed later.

**Already reconciled, but in the sibling `poster-corpus-validation` repo,
not here:** `rek_gore`/`rek_violence`/`rek_mod_weapons`/`rek_nudity`/
`rek_suggestive` (Rekognition `DetectModerationLabels`) already have real
Nova counterparts (`nova_blood_gore`, `nova_violence`, `nova_sexual_content`)
in that repo's gate 15 -- live-verified with real AWS calls (672 posters,
0 errors, 2026-08-16) and blind human review (85.5%/87.9% Nova precision
on blood_gore/violence). Nothing to port here; check that repo's
`docs/RESULTS.md` ("Gate 15: content moderation") before re-deriving any
of this.

**Not real detections at all:** `rek_title`/`rek_year` look like
Rekognition output columns but aren't -- `build_master_dataset.py`
auto-prefixes every non-`id` column from `rekognition_enrich.py`'s CSV
with `rek_`, including its pass-through `title`/`year` metadata columns
(the same convention `27_nova_scene_enrich.py`'s own `FIELDS` list uses).
No detection happened; nothing to reconcile.

**Dropped, rejected on its own merits (didn't need a Nova comparison to
lose):** `rek_labels`/`rek_top`/`rek_top_conf` -- Rekognition's general
open-vocabulary object/scene labels (`DetectLabels`' single highest-
confidence pick per poster). Corpus-wide, the two most common `rek_top`
values are **"Book" (26.1%, 34,414 posters) and "Advertisement" (30.9%,
40,697 posters)** -- together 57% of the entire corpus, for a corpus of
movie posters. `rek_top_conf` is useless for filtering: median 0.9998,
essentially saturated. A 60-poster verification review (not blind --
shown Rekognition's actual claimed label, asked "is this an accurate
description of something visible here" -- the same style as
`qa_title_ocr.py`'s existing QA pass, since a general label *list* isn't
a yes/no presence question the usual blind format can score): **"Book"
0/20 (0%), "Advertisement" 0/20 (0%), everything else 9/20 (45%) --
15.0% overall, roughly 19% weighted by real corpus share.** Rekognition's
general-purpose label model appears to be reading movie posters as a
*document type* (rectangular, text-heavy image → "Book"/"Advertisement")
rather than describing their actual illustrated content. This is worse
than DINO's rejected weapon/monster precision (11.1%/6.7%) and rejected
on the same terms DINO was: bad enough on its own that no comparison
engine is needed to justify dropping it. `27_nova_scene_enrich.py`'s new
`nova_labels`/`nova_top_label`/`nova_top_label_conf` fields (added before
this review ran, to test rek_labels against) are still pending an AWS
run, but that original motivating question is already answered --
they're now just Nova's own independent general-label read, not a
head-to-head rek_labels needs to win.

Separately: the same "is this even a poster" ground truth mentioned above
(`poster-corpus-validation`'s `poster_type_human_labels.csv`) covers
1,904 of this corpus's 4,915 `no_title_on_poster` ids already --
`27_nova_scene_enrich.py`'s `poster_qa_verdict`/`poster_qa_reason` fields
(see that script's docstring) only need to cover the other 3,011 plus the
rest of the corpus, not re-answer what's already answered.

**Net**: if the only reason to call Rekognition were presence-detection
signals (animal/weapon/monster/water) or color/face-count, it would not
be worth the cost -- free alternatives beat or tie it on every one of
those, and its general-purpose `rek_labels`/`rek_top` read is actively
wrong most of the time it matters (15.0% precision, 0% on 57% of the
corpus). The real, defensible reasons to keep calling it are
`rek_person`, face demographics, and (weakly) brightness cross-validation
-- everything else the same API call happens to also return is free
bonus signal once that call is already justified, not an independent
reason to make it.

### Fire: a sixth signal, same shape as water

Same overlap check that found the animal/weapon/monster/person/water
candidates surfaced one more real (poster, name) match worth the same
treatment: `rek_fire` (Rekognition) vs. `clip_fire` (segmentation) vs.
Nova's own `fire` tag (parsed from `nova_fear_labels`, same pattern as
water). Corpus-wide (65,003 posters) all three land in a tighter, more
similar range than water did (4.7-6.6% positive, 92-94% pairwise
agreement) -- a real difference from the lopsided rates that gave away
animal/weapon/monster/water's bad actors before any human review was
needed. That tighter aggregate agreement turned out to be the same
misleading-aggregate trap segmentation already sprang twice, not evidence
fire is an easier signal for all three -- the blind review below settled
it.

**50-poster blind review** (disagreement-stratified, real private corpus;
13 real positives, 26.0%):

| method | accuracy | precision | recall |
|---|---:|---:|---:|
| Rekognition alone | 58.0% | 27.8% | 38.5% |
| clip_fire (segmentation) alone | 54.0% | 22.2% | 30.8% |
| **Nova alone** | **78.0%** | **55.6%** | 76.9% |
| either (OR) | 56.0% | 36.4% | 92.3% |
| both (AND) | 80.0% | 100.0% | 23.1% |

Segmentation's `clip_fire` fails the same way it failed on animal/weapon/
water -- rejected on the same grounds. Rekognition alone is weak too
(27.8% precision -- it flags plenty of red/orange lighting, explosions,
and lava that aren't fire). AND reaches 100% precision but only 23.1%
recall -- it misses 10 of the 13 real fires in the sample, so it's not
usable as the sole rule even though every positive it does return is
real. OR is worse than Nova alone on every axis except recall. **Fire
doesn't fit this repo's "deterministic + Nova" pattern either** -- same
conclusion as water, no deterministic engine survived to pair with Nova.
The rule: **Nova (`27`) alone**, no partner.

A specialized alternative was considered and rejected without importing:
real fire/smoke-detection models exist publicly (e.g. YOLOv26/YOLOv10
fire detection, SigLIP2-based Forest-Fire-Detection, ViT-Forest-Fire-
Detection checkpoints on Hugging Face), but every one found is trained on
real photographic wildfire/surveillance footage, not illustrated poster
art -- the same domain mismatch that already broke YOLOv8n's
person-detection recall on this corpus (14.3% recall, see above). Since
Nova already beat every deterministic candidate that *was* tested here,
there was no longer a gap left for a specialized model to fill.

### Silhouette: the one signal where Nova is the noisy engine, not the anchor

`rek_silhouette` (Rekognition) vs. `clip_shadow` (segmentation) vs. Nova's own
`silhouette` tag (parsed from `nova_fear_labels`, same pattern as water/fire)
flips the pattern every other signal in this document follows. Corpus-wide
(65,003 posters): Rekognition 9.6% positive, `clip_shadow` 14.5%, **Nova
30.8%** -- Nova is the one firing far more often than the other two here, not
the trustworthy anchor.

**Round 1** (50-poster blind review, disagreement-stratified, same method as
every signal above): only 2 real positives in the whole sample -- even worse
underpowered than water's round 1 (4 positives). But it was informative on
its own: with all three engines individually firing on 36% of the sample (18
of 50, by stratification design), only 4% (2/50) turned out to be a real
silhouette under the strict definition ("backlit outline, no interior detail
visible" -- not just a dim or shadowy figure whose features are still
visible). The single poster where all three engines agreed was a real
silhouette; almost nothing in the three single-engine-disagreement pools was.
That gap -- 36% trigger rate vs. 4% real rate -- is what round 2 was built to
explain: is silhouette just a much rarer concept than any engine's threshold
suggests, or is the disagreement-only sampling strategy itself the problem
here (missing the cases where 2 of 3 engines already agree)?

**Round 2** (100-poster review, resampled from "2 or more of the 3 engines
agree" -- 7,457 posters in that pool, vs. round 1's disagreement-only
sampling; 96 scored after dropping "not sure," 28 real positives):

| method | accuracy | precision | recall |
|---|---:|---:|---:|
| Rekognition alone | 71.9% | 51.0% | 89.3% |
| clip_shadow (segmentation) alone | 33.3% | 20.0% | 42.9% |
| Nova alone | 46.9% | 35.4% | **100.0%** |
| **Rekognition AND Nova** | **79.2%** | **59.5%** | 89.3% |

`clip_shadow` is rejected the same way segmentation was rejected for animal/
weapon/water/fire. Nova alone never misses a real silhouette (100% recall)
but fires on plenty of ordinary dim or moody lighting that isn't one (35.4%
precision) -- the opposite failure shape from every other signal, where Nova
was the precise one and something else over-triggered. Rekognition alone is
better balanced (51.0% precision, 89.3% recall) but still wrong nearly half
the time it says yes. Requiring **both** to agree -- Rekognition AND Nova,
not "pick one deterministic + trust Nova alone" -- gives the best result of
any single engine or combination: Nova's recall covers what Rekognition
misses, Rekognition's higher precision filters out most of what Nova
over-fires on. **The rule: `rek_silhouette` (`26`) AND Nova's `silhouette`
tag (`27`)**, both required -- the only AND rule in this document; every
other signal above resolved to a single trusted engine, an OR, or Nova alone.

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

### Reconciling `weapon_n` with a second engine: does agreement actually help?

The 62.5%-false-positive finding above (Nova QA against OWLv2's own raw
output) already established that OWLv2 alone isn't trustworthy. What
hadn't been checked with real numbers on this repo's own corpus: does
requiring OWLv2 *and* DINO to agree actually improve on either alone, the
way the real project's own methodology assumes?

Both detectors run fresh against the full 99-poster sample
(`20_creature_weapon_owlv2.py`/`21_creature_weapon_dino.py`, no AWS
needed -- these are local, open-vocabulary detectors), OWLv2 flagging
28/99 posters `weapon_n > 0` and DINO 70/99 -- already a striking gap, no
overlap check needed to see these two disagree a lot (46/99 posters are
real OWLv2/DINO disagreements). Blind human review
(`scripts/qa/build_signal_reconciliation_review_page.py --signal weapon`):
the reviewed 58-poster sample was actually generated *before* DINO's run
completed, so it's stratified on OWLv2 alone (28 agreed-positive/30
agreed-negative per OWLv2, no disagreement-aware stratification) rather
than the disagreement-first sampling the tool normally does once 2+
engines have data -- of the 58 reviewed, only 15 happen to be real
OWLv2/DINO disagreements, not a targeted majority. 55 scoreable of 58
(3 `no_seguro`):

| method | n | accuracy | precision | recall |
|---|---:|---:|---:|---:|
| OWLv2 alone | 55 | 65.5% | 34.6% | 81.8% |
| DINO alone | 55 | 47.3% | 26.3% | 90.9% |
| ANY (either flags) | 55 | 45.5% | 25.6% | 90.9% |
| **ALL (both agree)** | 55 | **67.3%** | **36.0%** | 81.8% |

Confirms the real project's methodology with real numbers on this repo's
own corpus, not just a citation: **requiring both detectors to agree
beats either alone on both accuracy and precision**, at the cost of
identical recall to OWLv2 alone (some real weapons only DINO catches are
lost when both must agree). ANY is worse than either engine individually
-- summing two detectors' false positives instead of filtering them out.
DINO alone is notably weaker than OWLv2 alone here (47.3% vs. 65.5%
accuracy) -- it flags weapons on 70% of the corpus, high recall but at a
real precision cost.

**Closed with a third and fourth opinion**: Rekognition (`rek_weapon`)
and Nova (`nova_weapon`) were both added to this test and scored against
a fresh 50-poster blind review on the full private corpus -- see "CLIP
semantic embeddings," "Reconciling `is_animal`, `weapon`, and `monster`
across engines," above, for the final numbers and the rule this repo
settled on (OWLv2 + Nova, DINO dropped entirely: 0/40 real weapons on its
own disagreement subset).
