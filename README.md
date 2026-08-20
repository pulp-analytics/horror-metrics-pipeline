# poster-metrics-pipeline

Python pipeline for computing per-poster metrics at scale: color palettes,
CLIP/SigLIP semantic embeddings (fear axis, monster census, typography,
medium/painted-vs-photo classification), perceptual quality scores
(pyiqa, NIMA, LAION aesthetic), face detection + expression, and
geometric composition metrics.

Part of the [Pulp Analytics](https://github.com/pulp-analytics) horror poster
analysis project ("The Anatomy of Fear").

**Scope: this repo analyzes one poster at a time and stops there.**
Aggregating those per-poster metrics into charts/trends/decisions (e.g.
the yearly brightness curve, the Continue/Pivot checkpoint, decade-level
register shares) is a presentation concern for whatever consumes this
data downstream, not something computed here — that logic will live in a
separate front-end/presentation repo once one exists, not in this one.

**Status: color, perceptual quality, CLIP semantic embeddings, SigLIP
semantic embeddings, faces, geometric composition, depth, saliency,
pose, and creature/weapon detection are all built and documented
(below) -- see docs/RESULTS.md.**

## Quickstart

```bash
pip install -r requirements.txt
make sample                               # 01-21 + 25, in dependency order; skips files that already exist
# rebuild one metric: delete its CSV, then make sample
python3 scripts/01_color_metrics.py       # or run one script: color: CIELAB + k-means palette
python3 scripts/02_iqa_multi_score.py         # quality: clipiqa/musiq/brisque
python3 scripts/03_nima_score.py              # quality: NIMA aesthetic score
python3 scripts/04_laion_aesthetic_score.py   # quality: LAION aesthetic predictor
python3 scripts/05_clip_embed.py              # CLIP: build the embedding cache first
python3 scripts/06_clip_census.py             # CLIP: zero-shot creature/monster census
python3 scripts/07_clip_fear_axis.py          # CLIP: dread<->calm axis
python3 scripts/08_clip_typography_axis.py    # CLIP: ornate<->minimal lettering axis
python3 scripts/09_clip_genre_classifier.py   # CLIP: zero-shot genre agreement
python3 scripts/10_clip_medium.py             # CLIP: painted-vs-photographic
python3 scripts/11_siglip_embed.py            # SigLIP: build its own embedding cache
python3 scripts/12_siglip_fear_axis.py        # SigLIP: dread<->calm axis
python3 scripts/13_siglip_reanalysis.py       # SigLIP: census + typography + genre
python3 scripts/14_face_detect.py             # faces: YuNet detection (local, no AWS)
python3 scripts/15_face_expression.py         # faces: CLIP zero-shot expression per face
python3 scripts/16_geometric_composition.py   # composition: symmetry/grid/balance/diagonal (local, no AWS)
python3 scripts/17_depth_estimation.py        # depth: MiDaS monocular depth (torch.hub, first run downloads weights)
python3 scripts/18_saliency_prediction.py     # saliency: MSI-Net eye-tracking prediction (huggingface_hub, first run downloads weights)
python3 scripts/19_pose_dynamism.py           # pose: YOLOv8n person detection + ViTPose skeleton
python3 scripts/20_creature_weapon_owlv2.py   # creature/weapon: OWLv2 zero-shot detection
python3 scripts/21_creature_weapon_dino.py    # creature/weapon: Grounding DINO cross-check
python3 scripts/25_creature_weapon_agreement.py  # creature/weapon: OWLv2 ∩ DINO (the citable signal)
python3 -m pytest tests/ -v -m "not slow"     # -m "not slow" skips tests needing a model download
make test-fast                               # same as the pytest line above
```

`make sample` is the dependency graph the individual script lines above
encode by hand: 05 before 06-09, 11 before 12-13, 14 before 15, 20 and 21
before 25. It fills in missing files under `data/sample_output/` and is a
no-op on a clone that already has the checked-in CSVs (CLIP/SigLIP `.npz`
caches are *not* committed; they are only rebuilt when a CSV that needs
them is itself missing). Nova QA (22/23/24) is not in `make sample`.
Override paths with `IN=... OUT=...`. Independent scripts can run in
parallel: `make -j4 sample`.

```bash
# optional Nova QA scripts (22/23/24) -- need real AWS/Bedrock access, see below
python3 scripts/22_creature_weapon_nova_qa.py --boxes data/sample_output/creature_weapon_owlv2.csv --source owlv2 --n 50
python3 scripts/23_census_nova_qa.py --census data/sample_output/census.csv --n 50
python3 scripts/24_typography_nova_qa.py --typography data/sample_output/typography.csv --n 50
```

GitHub Actions runs `make test-fast` on every push and PR to `main`
(`.github/workflows/test.yml`). It installs `requirements-ci.txt`, a
subset of `requirements.txt` that skips tensorflow/pyiqa/ultralytics --
those are only needed by `@pytest.mark.slow` tests or by `load_*`
helpers the fast suite never calls.

`18_saliency_prediction.py` (MSI-Net) loads a legacy TF SavedModel that
crashes under TensorFlow/protobuf's default C++ backend -- the script
works around this itself by forcing protobuf's pure-Python
implementation before importing tensorflow, no environment changes
needed. See docs/RESULTS.md, "Saliency," for how this was root-caused.

`20_creature_weapon_owlv2.py` and `21_creature_weapon_dino.py` are meant
to be run together, not standalone: a blind QA pass over the real
project's OWLv2-only output found roughly 60%+ of its "creature
detected" boxes were false positives, so treat each script's raw output
as a candidate, not a verdict. `25_creature_weapon_agreement.py` is the
join that materializes that signal (same label + box IoU) into
`creature_weapon_agreement.csv` -- cite that file, not 20 or 21 alone.
See docs/RESULTS.md, "Creature/weapon detection."

`22_creature_weapon_nova_qa.py`, `23_census_nova_qa.py`, and
`24_typography_nova_qa.py` are Nova Pro vision-LLM QA tools, not pipeline
stages -- they cross-check a detector/classifier's raw output against an
independent judgment on the same poster, the same methodology behind the
"roughly 60%+ false positives" claim above. Unlike scripts 01-21, they
don't compute anything new for the corpus; they grade output that
already exists. To actually run one:

1. Run the detector/classifier it grades first (its raw output is the
   `--boxes`/`--census`/`--typography` input these three scripts read):
   `20`/`21` for `22`, `06_clip_census.py` for `23`,
   `08_clip_typography_axis.py` for `24`.
2. Get AWS credentials with `bedrock:InvokeModel` access to
   `us.amazon.nova-pro-v1:0` in whatever region you pass via `--region`
   (default `us-east-1`) -- set `AWS_PROFILE` (or any other credential
   source boto3's default chain picks up). If your account has never
   called a Nova model before, Bedrock model access has to be enabled
   once per account/region first (AWS Console -> Bedrock -> Model
   access) -- an `AccessDeniedException` mentioning the model ID is the
   usual symptom if that step's still pending, not a bug in these
   scripts.
3. Run it: `--n` controls sample size (small for a spot-check, e.g. 50;
   larger, e.g. 1000+, for a real citable finding -- see
   docs/RESULTS.md's "Nova QA" subsections for what running these against
   real posters found, at both scales).

Not wired into `compute_metrics.asl.json` in
poster-analysis-infrastructure, and never will be -- these are a human
deciding whether to trust a detector before citing it, the same as the
private project's own `qa_*.py` scripts never became pipeline stages
either. No Step Functions execution runs them automatically.

No API key or AWS needed for scripts 01-21 or 25 — posters come from TMDB's
public image CDN, and the one non-CLIP/SigLIP model (`14`'s YuNet face
detector) is a small local ONNX file, not a cloud service. Every
download-capable script (everything except `06`-`09`, `12`-`13`, `15`,
and `25`, which read `05`'s/`11`'s embedding cache, `14`'s face boxes,
or 20+21's CSVs -- `16`-`21` all download fresh, same as `01`-`04`/`14`) shares one poster cache
(`data/posters_cache/`, see `utils/posters.py`): whichever script runs
first downloads a given poster, the others reuse that file. That cache
can optionally check S3 first (`--posters-s3-bucket`, matching the real
project's own storage pattern) before falling back to TMDB — entirely
optional, off by default.

`17_depth_estimation.py`, `19_pose_dynamism.py`, `20_creature_weapon_owlv2.py`,
and `21_creature_weapon_dino.py` pick a torch device with `cuda` > `mps` >
`cpu` (`--device` to override). On Apple Silicon that means Metal instead
of the CPU fallback that made the 99-poster Grounding DINO sample take
~27 minutes. `18_saliency_prediction.py` is TensorFlow, not torch.

Not tied to horror specifically: `01_color_metrics.py` has no
genre-specific logic and was verified live against a real, non-horror
(sci-fi) sample with zero code changes — see "Genre-agnostic, verified"
in docs/RESULTS.md.

## Joining the outputs into one table

This repo's own contract stays one file per metric, one row per poster
(see "Scope" above) -- but if you want a single flat table instead of
joining the pieces yourself, `assemble_master_dataset.py` does exactly
that:

```bash
python3 assemble_master_dataset.py --data-dir data/sample_output --out master_dataset.csv
make assemble-sample   # same join
```

`data/sample_output/metrics_input.csv` is the 99-poster corpus list
(same ids as `data/sample_input/sample_100_posters.csv`), so that
command auto-detects the base -- no `--base` needed on the checked-in
sample. It left-joins every metric CSV found in `--data-dir` onto the
corpus base (`validated_corpus.csv` or `metrics_input.csv`), prefixing each
file's columns with its own stem so same-named columns across files
(e.g. `creature_n` in `creature_weapon_owlv2.csv`,
`creature_weapon_dino.csv`, and `creature_weapon_agreement.csv`) never
collide. `face_expression.csv` is
aggregated first since it's the one output with multiple rows per
poster (one per detected face).

## Structure

```
Makefile                           `make sample` / `make test-fast` / `make assemble-sample`
scripts/
  01_color_metrics.py          Per-poster brightness/saturation/hue-bands/
                                dominant-palette (CIELAB, saturation-weighted
                                k-means)
  02_iqa_multi_score.py        clipiqa/musiq/brisque via pyiqa
  03_nima_score.py             NIMA aesthetic score
  04_laion_aesthetic_score.py  LAION aesthetic predictor (CLIP ViT-L/14 + MLP head)
  05_clip_embed.py             CLIP ViT-B/32 embedding cache -- 06-09 read this
  06_clip_census.py            zero-shot creature/monster taxonomy
  07_clip_fear_axis.py         continuous dread<->calm axis
  08_clip_typography_axis.py   continuous ornate<->minimal lettering axis
  09_clip_genre_classifier.py  zero-shot genre similarity/agreement
  10_clip_medium.py            painted-illustration vs. photographic (embeds fresh)
  11_siglip_embed.py           SigLIP embedding cache -- 12/13 read this, not 05's
  12_siglip_fear_axis.py       SigLIP version of 07, same axis/prompts
  13_siglip_reanalysis.py      SigLIP version of 06+08+09, one shared model load
  14_face_detect.py            YuNet face detection (local ONNX, not Rekognition)
  15_face_expression.py        CLIP zero-shot expression per detected face -- reads 14's output
  16_geometric_composition.py  symmetry/negative-space/mass, MSER text coverage,
                                grid+thirds alignment, balance/harmony, diagonal/
                                pyramid weight-shift -- pure OpenCV, no model
  17_depth_estimation.py       MiDaS_small monocular depth (torch.hub) -- how
                                close/foreground the threat reads as
  18_saliency_prediction.py    MSI-Net predicted eye-tracking saliency -- where
                                the eye lands first (huggingface_hub-served model)
  19_pose_dynamism.py          YOLOv8n person detection + ViTPose skeleton --
                                static portrait vs. dynamic action pose
  20_creature_weapon_owlv2.py  OWLv2 zero-shot creature/weapon detection --
                                noisy alone, see docs/RESULTS.md before using
  21_creature_weapon_dino.py   Grounding DINO zero-shot creature/weapon
                                detection -- same vocabulary as 20, run both
                                and treat agreement as the signal
  25_creature_weapon_agreement.py  join of 20 ∩ 21: same-label boxes with
                                IoU >= 0.3 -- the citable creature/weapon
                                output; no model, reads the two CSVs
  22_creature_weapon_nova_qa.py  Nova Pro QA of 20/21's boxes -- needs AWS,
                                  not a pipeline stage, see docs/RESULTS.md
  23_census_nova_qa.py           Nova Pro QA of 06's CLIP census -- needs AWS,
                                  not a pipeline stage
  24_typography_nova_qa.py       Nova Pro QA of 08's CLIP typography axis --
                                  needs AWS, not a pipeline stage
  utils/
    logging_setup.py, resumable.py   shared conventions with the sibling
                                      poster-corpus-validation repo
    posters.py                       shared TMDB/S3/local-disk poster cache
    clip_backbone.py                 shared CLIP model loading + text-prototype
                                      helper, used by 06/07/08/09
    siglip_backbone.py               shared SigLIP model loading + text-prototype
                                      helper, used by 12/13
    device.py                        cuda > mps > cpu for 17/19/20/21
data/
  sample_input/    99 real posters, stratified by decade (1920s-2020s)
  sample_output/   real, already-computed metrics for those same posters
                    (01-21 plus 25's OWLv2 ∩ DINO agreement), including
                    geometric/depth/saliency/pose and both creature/weapon
                    detectors, plus metrics_input.csv as the join base
                    (CLIP/SigLIP .npz caches are generated on demand by
                    05/11, not committed -- see docs/RESULTS.md for what's
                    verified to reproduce exactly vs. what isn't, and why)
docs/
  METHODOLOGY.md   what's computed and why, per category
  RESULTS.md        real findings, per category
  MODELS.md         every model this repo loads, what it resolves to, and
                     how tight the version pin is (SigLIP/LAION/OWLv2/DINO/
                     ViTPose/MSI-Net by HF revision, YuNet/YOLOv8n by sha256,
                     MiDaS by torch.hub GitHub commit, CLIP already pinned
                     by open_clip itself)
tests/
  test_color_metrics.py     pure-function tests for the color math
  test_clip_backbone.py     softmax/cosine-similarity math underlying every
                             CLIP script, on synthetic embeddings
  test_siglip_backbone.py   same math, SigLIP's side, on synthetic embeddings
  test_face_expression.py   pure crop/box-parsing geometry for 15, on synthetic images
  test_posters.py           shared poster-cache logic -- confirms the local-
                             cache-hit path never touches the network, and
                             that no AWS import happens when S3 isn't configured
  test_makefile_sample.py   `make sample` dry-run: 05 before 06-09, 20+21
                             before 25, Nova QA not in the graph, no-op when
                             the checked-in CSVs already exist
  test_device.py            cuda > mps > cpu pick, including --device override
```

## License

MIT — see [LICENSE](LICENSE).
