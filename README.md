# poster-metrics-pipeline

Python pipeline for computing per-poster metrics at scale: color palettes,
CLIP/SigLIP semantic embeddings (fear axis, monster census, typography,
medium/painted-vs-photo classification), perceptual quality scores
(pyiqa, NIMA, LAION aesthetic), face detection + expression, and
geometric composition metrics.

Part of the [Pulp Analytics](https://github.com/pulp-analytics) horror poster
analysis project ("The Anatomy of Fear"). Corpus-scale runs are orchestrated
by [poster-analysis-infrastructure](https://github.com/pulp-analytics/poster-analysis-infrastructure);
this repo is the code that image executes. See [Where this runs](#where-this-runs).

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

Docs: [METHODOLOGY](docs/METHODOLOGY.md) · [SCHEMA](docs/SCHEMA.md) ·
[RESULTS](docs/RESULTS.md) · [MODELS](docs/MODELS.md)

- [Where this runs](#where-this-runs)
- [Validation methodology](#validation-methodology)
- [Quickstart](#quickstart)
- [Join into one table](#joining-the-outputs-into-one-table)
- [Structure](#structure)
- [Contributing](CONTRIBUTING.md)
- [License](#license)

## Where this runs

This repo is the **code the cloud job runs**, not a laptop app. Corpus-scale
scoring is orchestrated by
[poster-analysis-infrastructure](https://github.com/pulp-analytics/poster-analysis-infrastructure)
(`statemachine/compute_metrics.asl.json`): Step Functions + AWS Batch
array jobs for every script that loops per poster (01–05, 10–11, 14,
16–21) + Fargate ECS tasks for the vectorized CLIP/SigLIP/expression
passes (06–09, 12–13, 15). Shared EFS holds `--in` / `--out`. The
container is `docker/Dockerfile.metrics` in that repo, which clones
**this** repo at a pinned commit.

**CPU today; GPU is an infrastructure switch, not a flag here.** The
metrics Batch compute environment is Fargate, which has no GPUs — on
purpose. That repo's `docs/ARCHITECTURE.md` ("GPU is a real option this
design doesn't take") and `batch/compute-environment-metrics.json` spell
it out: the 145k-poster corpus was scored on self-terminating EC2; Fargate
exists so a forgotten instance cannot bill again (see that repo's
`docs/COST_SAFETY.md`). If wall-clock at full scale needs GPUs, change
that compute environment to EC2 (`g4dn` / `g5`) and add GPU
`resourceRequirements` on the job definition. These scripts already pick
`cuda` when the container has it (`utils.device.pick_device` on 17/19/20/21;
the rest are `cuda` if `torch.cuda.is_available()`). Nothing in *this*
repo turns GPU on or off.

`make sample` and `pip install -e ".[cpu]"` below are for developing and
for the checked-in 99-poster sample. The extra name `[cpu]` means "no
TensorFlow, no boto3" — not "this pipeline is CPU-only." Nova QA (22/23/24)
is methodology ([Validation methodology](#validation-methodology)): a sampled
Bedrock pass, intended as a Step Functions state, not in the ASL yet.
`25` is a no-model join of 20+21 (cite that CSV); it is not currently a
state in `compute_metrics.asl.json` either.

## Validation methodology

Same three-layer check as the sibling
[poster-corpus-validation](https://github.com/pulp-analytics/poster-corpus-validation)
(`README.md`, "Validation methodology") before a number is trusted, not
just built and assumed correct. Not every metric uses all three.
Details: [METHODOLOGY](docs/METHODOLOGY.md#validation-methodology).
Tables: [RESULTS](docs/RESULTS.md) "Nova QA."

1. **Deterministic first, where one exists.** Pixel math and small CNNs
   that re-run identically on the same file: color (`01`), composition
   (`16`), YuNet (`14`), depth, saliency. Quality scores and pose are
   neural but continuous — re-run, not a vision-LLM. There is no "Nova,
   is this L* right?" question, so those categories stop here.

2. **Vision-LLM cross-check (Amazon Nova Pro).** Semantic calls (CLIP
   census, typography register, creature/weapon boxes) over-detect on
   purpose. We cross them with something that is not the same model:
   `20` ∩ `21` → `25`, and Nova Pro (`22`/`23`/`24`) on a **sample**.
   Prompts are pinned in those scripts (`temperature: 0`), settled after
   several Bedrock runs and prompt revisions — isolated prompts, not a
   combined mega-prompt. Each asks for Nova's own judgment; the
   CLIP/detector guess is context, not an instruction to agree.

   | script | prompt (settled in the file) | live finding |
   |---|---|---|
   | `22` | red box: is `"{label}"` really there? (`correct` / `false_positive` / `uncertain`) | n=1000 OWLv2: **62.5% false_positive** (citable; a 15-poster mechanism check hit 79% on a non-horror-stratified sample) |
   | `23` | one census label or `none` (never `uncertain`) | 40 posters: 8/40 exact string vs CLIP — mostly CLIP `uncertain` vs Nova `none` |
   | `24` | title lettering: ornate → minimal (5 registers) | 40 posters: 75% exact register, 97.5% ±1 |

   Raw `qa_*.csv` from those runs are **not** in this repo. RESULTS is
   the citable record. Re-running needs Bedrock (`us.amazon.nova-pro-v1:0`).
   Nova belongs in `compute_metrics.asl.json` as a sampled `--n` state
   after the metric it grades; that state is **not in the ASL yet**.
   `make sample` stays Bedrock-free.

3. **Human ground truth (blind HTML).** Same layer as
   poster-corpus-validation: `scripts/qa/build_*_review_page.py` write a
   self-contained page (`data/ground_truth/*_review.html`, generated,
   gitignored) that shows the poster and a plain question — never CLIP
   scores or Nova verdicts. Export CSV, then join. `--validate` sets on
   famous posters remain a small in-script check. Genre-vs-IMDb
   (`scripts/qa/validate_genre_classifier_vs_imdb.py`) uses curated
   catalog tags, so it skips this leg — the same exception the sibling
   makes for IMDb `isAdult`.

To run a sampled Nova pass: extra `[bedrock]`, then `22`/`23`/`24` with
`--n`. To collect the human layer on the 99-poster sample (no Bedrock):

```bash
python3 scripts/qa/build_census_review_page.py
python3 scripts/qa/build_typography_review_page.py
python3 scripts/qa/build_creature_weapon_review_page.py
# open data/ground_truth/*_review.html  — file://, labels stay in the browser until Export CSV
```

## Quickstart

Local development and the checked-in 99-poster sample. Corpus-scale runs
go through [Where this runs](#where-this-runs).

```bash
pip install -e ".[cpu]"                   # 01-17, 19-21, 25 (no TensorFlow, no AWS)
# pip install -e ".[cpu,tf-saliency]"     # plus 18 MSI-Net
# pip install -e ".[all]"                 # plus Nova QA / optional S3 poster cache
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
them is itself missing). Nova QA (22/23/24) is not in `make sample`
(Bedrock; see [Validation methodology](#validation-methodology)).
Override paths with `IN=... OUT=...`. Independent scripts can run in
parallel: `make -j4 sample`.

GitHub Actions runs `make test-fast` on every push and PR to `main`
(`.github/workflows/test.yml`). It installs the default extra-free
dependencies (`pip install -e .` / `requirements-ci.txt`): no
tensorflow, pyiqa, ultralytics, or boto3 -- those are only needed by
`18`, `02`/`03`, `19`, and Nova QA/`--posters-s3-bucket` respectively,
or by `@pytest.mark.slow` tests. `pip install -e ".[cpu]"` is the
**local** extra (no TensorFlow, no boto3); `".[all]"` matches the old
flat `requirements.txt`. Production installs whatever
`docker/Dockerfile.metrics` in poster-analysis-infrastructure pins.
Lower bounds live in `pyproject.toml`. There is no platform lock file:
torch wheels differ across macOS / CPU Linux / CUDA, so a single
`uv.lock` / `pip freeze` would lie to the other two.

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
See docs/RESULTS.md, "Creature/weapon detection." Nova's role in that
~62.5% figure is [Validation methodology](#validation-methodology).

The Python for 01-21 and 25 does not call AWS ML APIs — posters come from
TMDB's public image CDN (optional `--posters-s3-bucket`). Production still
*runs on* AWS (Fargate/Batch); that is orchestration, not Rekognition. The
one non-CLIP/SigLIP model (`14`'s YuNet) is a small local ONNX file. Every
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
`cpu` (`--device` to override). On a CUDA Batch/EC2 worker that is GPU;
on Apple Silicon (local sample) that is Metal. `18_saliency_prediction.py`
is TensorFlow, not torch.

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
poster (one per detected face). Column names, units, and sentinels
(before that prefix) are listed in docs/SCHEMA.md.

## Structure

```
Makefile                           `make sample` / `make test-fast` / `make assemble-sample`
assemble_master_dataset.py         join per-metric CSVs (not a metric stage)
CONTRIBUTING.md                    invariants (one poster at a time; Nova is sampled QA)
pyproject.toml                     extras: cpu / tf-saliency / bedrock
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
  22_creature_weapon_nova_qa.py  Nova Pro QA of 20/21's boxes -- sampled
                                  methodology; prompts + rates in README
  23_census_nova_qa.py           Nova Pro QA of 06's CLIP census -- sampled;
                                  agree maps CLIP uncertain → none
  24_typography_nova_qa.py       Nova Pro QA of 08's CLIP typography axis --
                                  sampled methodology
  qa/
    build_census_review_page.py           blind HTML for 06 (no CLIP labels)
    build_typography_review_page.py       blind HTML for 08 (no CLIP axis)
    build_creature_weapon_review_page.py  blind HTML for 20/21 boxes (no scores)
    review_page.py                        shared page template (localStorage + CSV export)
    validate_genre_classifier_vs_imdb.py  09 vs IMDb genres -- catalog GT,
                                  skips human-review leg
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
  ground_truth/    generated `*_review.html` (gitignored) + exported human
                    CSVs; genre-vs-IMDb sample (catalog tags)
docs/
  METHODOLOGY.md   what's computed and why, per category
  SCHEMA.md         column names, units, sentinels -- the CSV contract
  RESULTS.md        real findings, per category
  MODELS.md         every model this repo loads, what it resolves to, and
                     how tight the version pin is (SigLIP/LAION/OWLv2/DINO/
                     ViTPose/MSI-Net by HF revision, YuNet/YOLOv8n by sha256,
                     MiDaS by torch.hub GitHub commit, CLIP already pinned
                     by open_clip itself)
tests/                             `make test-fast` = pytest -m "not slow"
  test_census_nova_qa.py            23: uncertain→none agree, pick_sample
  test_clip_backbone.py             CLIP softmax/cosine math (synthetic)
  test_color_metrics.py             01 color math
  test_creature_weapon_agreement.py 25: IoU + greedy box match
  test_creature_weapon_dino.py      21: prompt build / box filter
  test_creature_weapon_nova_qa.py   22: load_detections / pick_sample
  test_creature_weapon_owlv2.py     20: filter_boxes
  test_depth_estimation.py          17: min-max closeness (slow: live MiDaS)
  test_device.py                    cuda > mps > cpu
  test_face_expression.py           15: crop/box geometry
  test_geometric_composition.py     16: OpenCV heuristics
  test_makefile_sample.py           make sample graph; Nova not in it
  test_model_pins.py                Hub/GitHub/file loads have a revision
  test_pose_dynamism.py             19: compute_metrics
  test_posters.py                   cache hit never touches the network
  test_pyproject_extras.py          cpu / tf-saliency / bedrock extras
  test_readme_inventory.py          README Structure names every test_*.py
  test_resumable.py                 flock, 0-byte header, no duplicate ids
  test_review_pages.py              blind HTML builders omit CLIP/Nova scores
  test_saliency_prediction.py       18: heatmap summary (slow: live MSI-Net)
  test_sample_output_contract.py    99-poster sample: one row per id, assemble
  test_schema_contract.py           sample headers match FIELDS / SCHEMA
  test_siglip_backbone.py           SigLIP softmax/cosine math (synthetic)
  test_typography_nova_qa.py        24: bin_register / agree_adjacent
  test_validate_genre_classifier_vs_imdb.py  genre-vs-IMDb metrics (no CLIP)
```

## License

MIT — see [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
