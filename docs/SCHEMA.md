# Schema

Column contract for every CSV this repo writes. Methodology (why a
metric exists) is [METHODOLOGY.md](METHODOLOGY.md); findings are
[RESULTS.md](RESULTS.md); model pins are [MODELS.md](MODELS.md).

**Scope of the files themselves:** one CSV per metric, one row per
poster -- except `face_expression.csv` (one row per detected face) and
the Nova QA files (one row per sampled judgment, not the full corpus).
`tests/test_schema_contract.py` checks sample headers against each
script's `FIELDS` (parsed without importing, so CI does not need pyiqa
or TensorFlow) and against the pandas `to_csv` shape for 06-09 / 12-13.

## Conventions

- **`id`** -- TMDB-style film id, string. Join key everywhere.
- **`title`, `year`** -- copied from `--in` when the script writes them.
  Some CLIP/SigLIP tables omit them (`genre_classifier.csv`) or put
  them after the metric columns. Do not join on title.
- **`error`** -- empty on success. Present on most per-poster inference
  scripts; `01` and `16` skip failed ids instead of writing an error
  row. `06`–`09` / `12`–`13` / `25` have no `error` column (they read a
  cache or join CSVs).
- **Sentinels.** `-1` means "couldn't compute" on `16`'s `text_y`,
  `thirds_dist`, `balance`, `harmony`. Empty string on `20`/`21`/`25`
  `*_top_label` / `*_top_score` when `n=0`. `n_persons=0` on `19` is a
  real answer, not a failure.
- **Boxes.** Normalized `[x, y, w, h]` in 0–1 of the poster, JSON in
  the `*_boxes` / `face_boxes` / `box` columns unless noted (pose
  `box` is `[x0, y0, x1, y1]` in **pixels**).
- **`uncertain` vs `none`.** Census (`06`/`13`): `uncertain` is CLIP/
  SigLIP's low-confidence sentinel (score < `--min-score`, default)
  0.5); `none` is a real taxonomy label ("no creature"). Nova QA (`23`)
  never emits `uncertain`; its `agree` column treats CLIP `uncertain` as
  `none`. Do not rewrite `06`/`13` to match Nova.
- **Assemble prefix.** `assemble_master_dataset.py` left-joins by `id`
  and prefixes every non-`id`/`title`/`year` column with `<stem>_`
  (e.g. `creature_weapon_owlv2_creature_n`). `face_expression.csv`
  becomes `face_expression_n` + `face_expression_summary` (one row per
  id). Cite the unprefixed names below; the flat table is a derived
  view, not the contract.

CLIP/SigLIP embedding caches (`clip_embeddings.npz`,
`siglip_embeddings.npz`) are not CSVs: arrays `ids` and `vecs`
(float16, L2-normalized). They are generated on demand, not committed.

Nova QA outputs (`qa_*.csv`) are not in `data/sample_output/`. They are
sampled methodology (a Step Functions `--n` state, not in the ASL yet),
not metric CSVs.

## Corpus list

`data/sample_input/sample_100_posters.csv` and
`data/sample_output/metrics_input.csv` (copy used as assemble base):

| column | meaning |
|---|---|
| `id` | film id |
| `title` | title |
| `year` | release year |
| `poster_path` | TMDB poster path (`/….jpg`); required for any script that downloads |

## Color -- `color_metrics.csv` (`01`)

| column | meaning |
|---|---|
| `brightness` | mean CIELAB L* |
| `dark_share` | fraction of pixels with L* < 20 |
| `saturation` | mean HSV S |
| `red_share` | "blood red" share (hue ±15° of red, S>0.4, V>0.15) |
| `palette` | JSON list of 5 hex colors, k-means in LAB, size-sorted |
| `palette_share` | JSON list of those clusters' pixel shares |
| `band_red`, `band_warm`, `band_green`, `band_blue`, `band_purple`, `band_dark` | mutually exclusive hue-family shares; sum to 1.0 |

## Perceptual quality

`iqa_multi_score.csv` (`02`): `clipiqa` (0–1, higher=better), `musiq`
(KonIQ scale, higher=better), `brisque` (lower=better), `error`.

`nima_score.csv` (`03`): `nima_score` (AVA aesthetic, higher=better),
`error`.

`laion_aesthetic_score.csv` (`04`): `aesthetic_score` (LAION MLP on
CLIP ViT-L/14, higher=better), `error`.

These are model inference. Re-running the same cached JPEG is
deterministic; matching a historical number from months ago is not
guaranteed -- see METHODOLOGY.

## CLIP semantic -- `05` cache plus `06`–`10`

`census.csv` (`06`):

| column | meaning |
|---|---|
| `label` | taxonomy key, `none`, or `uncertain` |
| `score` | softmax confidence of the (pre-downgrade) top label |
| `is_animal` | `label` in `{shark, spider, snake, wolf_dog, bird, insect}` |
| `is_creature` | `label` is not `none` or `uncertain` |

`fear_axis.csv` (`07`) and `typography.csv` (`08`): `axis` -- continuous
cosine difference (dread−calm, ornate−minimal). No register column;
quantile bins are QA-only in `24`.

`genre_classifier.csv` (`09`): always `id`, `pred_genre`,
`sim_horror`, `sim_scifi`, `sim_thriller`, `sim_mystery`. If `--in`
has a `genre` column (or `--true-genre-col`): also `true_genre`,
`agree`. The checked-in sample includes those optional columns.
No `title`/`year`.

`medium.csv` (`10`): `p_painted` (0–1), `painted` (boolean), `error`.
Embeds fresh; does not read `05`'s cache.

## SigLIP semantic -- `11` cache plus `12`–`13`

Same metric names as CLIP where they overlap, different files:

| file | script | columns |
|---|---|---|
| `siglip_fear_axis.csv` | `12` | `id`, `axis`, `title`, `year` |
| `siglip_census.csv` | `13` | `id`, `label`, `score`, `year`, `title` -- **no** `is_animal` / `is_creature` |
| `siglip_typography.csv` | `13` | `id`, `axis`, `year`, `title` |
| `siglip_genre_classifier.csv` | `13` | `id`, `pred_genre`; optional `true_genre`, `agree` (no `sim_*` columns) |

`13` writes three CSVs from one model load. Not interchangeable with
the CLIP tables (768-d vs 512-d space).

## Faces

`face_detect.csv` (`14`): `n_faces`, `face_area` (mean face-box area
share, 0–1), `max_conf`, `face_boxes` (JSON, normalized xywh, largest
first), `error`.

`face_expression.csv` (`15`) -- **one row per face**, not per poster:

| column | meaning |
|---|---|
| `id` | poster id (repeated) |
| `face_idx` | 0-based, largest-face-first as `14` ordered them |
| `box` | that face's normalized xywh |
| `label` | one of terrified/screaming/shocked/menacing/angry/sad/in_pain/calm, or `uncertain` |
| `score` | softmax confidence |

Assemble collapses this to `face_expression_n` +
`face_expression_summary` (`label:score` joined by `;`).

## Geometric composition -- `geometric_composition.csv` (`16`)

No `error` column; failed posters are omitted.

| column | group | meaning |
|---|---|---|
| `symmetry` | composition | left-right symmetry, 0–1 |
| `neg_space` | composition | low-gradient pixel fraction |
| `complexity` | composition | Canny edge density |
| `mass_x`, `mass_y` | composition | visual-mass centroid, 0–1 |
| `text_area` | typography | MSER coverage fraction |
| `text_y` | typography | vertical centroid of MSER mask; `-1` if none |
| `text_regions` | typography | MSER box count (integer) |
| `align_score` | grid | layout-block edge alignment |
| `thirds_dist` | grid | distance of main mass to nearest thirds point; `-1` if no main box |
| `n_blocks` | grid | detected layout-block count |
| `balance` | aesthetic | saliency-centroid distance from geometric center; `-1` if no map |
| `harmony` | aesthetic | hue-scheme agreement, 0–1; `-1` if histogram too flat |
| `diagonal_score` | diagonal | share of Hough length that is diagonal |
| `pyramid_shift` | diagonal | bottom-third minus top-third gradient spread |

MSER columns are the unresolved reproduction gap in this category --
see RESULTS.

## Depth -- `depth_estimation.csv` (`17`)

All four metrics are unitless closeness after per-image min-max to
[0, 1]. Higher = closer / more foreground.

`mean_depth`, `p95_depth`, `depth_std`, `close_area_frac` (fraction
above 0.7), `error`.

## Saliency -- `saliency_prediction.csv` (`18`)

`peak_x`, `peak_y` -- 0–1 location of the heatmap argmax (origin top-
left). `top10pct_mass` -- fraction of saliency in the hottest 10% of
pixels. `mean_saliency` -- mean of the probability-like map. `error`.

## Pose -- `pose_dynamism.csv` (`19`)

| column | meaning |
|---|---|
| `n_persons` | YOLOv8n person count; `0` is valid |
| `kpt_bbox_area_frac` | confident-keypoint box over the person box; empty string when `n_persons=0` or no usable pose |
| `limb_asymmetry` | left/right limb imbalance; empty string if torso keypoints are weak |
| `mean_kpt_confidence` | mean ViTPose score |
| `box` | JSON `[x0, y0, x1, y1]` of the person box used, **pixels** |
| `keypoints` | JSON `[[x, y, score], …]` length 17, COCO order (NOSE…R_ANKLE), pixels |
| `error` | empty on success |

## Creature/weapon -- `20`, `21`, `25`

`creature_weapon_owlv2.csv` and `creature_weapon_dino.csv` share:

| column | meaning |
|---|---|
| `creature_n` / `weapon_n` | boxes kept (0–3) |
| `creature_top_label` / `weapon_top_label` | highest-score label; empty if n=0 |
| `creature_top_score` / `weapon_top_score` | that score; empty if n=0 |
| `creature_boxes` / `weapon_boxes` | JSON list of `{label, score, box}` with `box` = normalized xywh |
| `error` | empty on success |

Cite `creature_weapon_agreement.csv` (`25`), not 20 or 21 alone. Same
`n` / `top_*` / `boxes` shape, where each agreed box is
`{label, iou, owlv2_score, dino_score, owlv2_box, dino_box}`. Extra:

| column | meaning |
|---|---|
| `creature_label_agree` / `weapon_label_agree` | `1` iff both detectors' non-empty top labels match (no IoU required) |

No `error` on 25 (it's a join). `--min-iou` default 0.3.

## Nova QA -- sampled methodology, not metric CSVs

Layer 2–3 of [METHODOLOGY, "Validation methodology"](METHODOLOGY.md#validation-methodology).
The prompts in `22`/`23`/`24` are settled after several Bedrock runs;
do not treat a one-shot QA CSV as the citable finding.

`22` creature/weapon boxes: `id`, `source` (`owlv2`/`dino`), `kind`
(`creature`/`weapon`), `label`, `score`, `box`, `model`, `status`,
`verdict` (`correct`/`false_positive`/`uncertain`), `actual`, `reason`,
`latency_s`, `error`. One row per sampled detection.

`23` census: `id`, `clip_label`, `clip_score`, `model`, `status`,
`nova_label`, `agree` (`True`/`False` after mapping CLIP `uncertain` →
`none`), `reason`, `latency_s`, `error`.

`24` typography: `id`, `clip_register`, `clip_axis`, `model`, `status`,
`nova_register`, `agree`, `agree_adjacent` (±1 register), `reason`,
`latency_s`, `error`. Registers: ornate / decorative / standard /
clean / minimal.

## Adding a metric

New pipeline stage: next unused script number (25 is the join; 22–24
are QA -- do not insert a stage as 26 that belongs in `make sample`
without updating the Makefile graph). One row per poster, `FIELDS`
declared in the script, pin the model in MODELS.md, check in a sample
CSV, document columns here and the why in METHODOLOGY. Heavy new
dependency goes in a pip extra, not the default install. Do not add
decade aggregates. Nova (`22`–`24`) belongs in the Step Function as a
sampled `--n` state, not a 145k loop, and does not write into 06/08/20/21
CSVs; that ASL state is not there yet. A new **per-poster metric**
stage must also land in `compute_metrics.asl.json` in
poster-analysis-infrastructure. The full contributor checklist is
[CONTRIBUTING.md](../CONTRIBUTING.md).
