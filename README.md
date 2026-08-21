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
is not a Step Functions state. `25` is a no-model join of 20+21 (cite that
CSV); it is not currently a state in `compute_metrics.asl.json`.

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

The Python for 01-21 and 25 does not call AWS ML APIs — posters come from
TMDB's public image CDN (optional `--posters-s3-bucket`). Production still
*runs on* AWS (Fargate/Batch); that is orchestration, not Rekognition. The
one non-CLIP/SigLIP model (`14`'s YuNet) is a small local ONNX file.
`26_rekognition_enrich.py` is the one exception that does call an AWS ML
API directly -- scene/object labels, image-quality properties, and face
demographics via AWS Rekognition, the same real, already-run signal
family as scripts 22-24's AWS dependency, just a metric rather than a QA
tool. See its own docstring for what it deliberately drops (moderation --
that's poster-corpus-validation's gate 15's job) and why; not yet
live-verified (AWS credentials unavailable) or wired into
`compute_metrics.asl.json`. Every download-capable script (everything
except `06`-`09`, `12`-`13`, `15`, and `25`, which read `05`'s/`11`'s
embedding cache, `14`'s face boxes, or 20+21's CSVs -- `16`-`21` and `26`
all download fresh, same as `01`-`04`/`14`) shares one poster cache
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

## Reconciling overlapping signals

Several questions have more than one engine answering them -- "is there
an animal," "is there a weapon," "is there a monster/supernatural
creature," "is there a person," "is there water," "is there fire."
`scripts/qa/build_signal_reconciliation_review_page.py` +
`compare_signal_engines.py` exist to test each one for real rather than
assume more engines voting is automatically safer:

```bash
python3 scripts/qa/build_signal_reconciliation_review_page.py --signal animal   # or weapon, monster, person, water, fire
# review the generated page, export its CSV
python3 scripts/qa/compare_signal_engines.py --signal animal --human data/qa/animal_reconciliation_human_review.csv
```

**The rule this repo settled on, after real blind human review on the
full private corpus (145,492 posters -- see docs/RESULTS.md, "Reconciling
`is_animal`, `weapon`, `monster`, and `person` across engines" for the
numbers): one deterministic model + Nova + human review, not "every
available engine votes."** Concretely:

- **animal**: CLIP census (`06`) + Nova (`27`) -- Rekognition (`26`)
  scored 26.0% accuracy / 17.8% precision on a 50-poster review; of the
  40 posters where Rekognition alone said yes, only 4 (10%) were real
  animals.
- **weapon**: OWLv2 (`20`) + Nova (`27`) -- DINO (`21`) scored 20.0%
  accuracy / 11.1% precision; 0/40 of its own-disagreement posters were
  real weapons. Rekognition also scored 100% here but is dropped anyway
  (free local model already running for `25`, no cost reason to also
  pay for a Rekognition call).
- **monster**: OWLv2 (`20`) + Nova (`27`) -- same DINO failure (16.0%
  accuracy, 0/40 on its own disagreement posters). Rekognition was never
  a candidate -- it has no monster/creature field at all.
- **person**: Rekognition (`26`) + Nova (`27`) -- `19_pose_dynamism.py`'s
  own YOLOv8n person count scored 100% precision but only 14.3% recall:
  never wrong when it fires, but of 20 posters where Rekognition+Nova
  both said yes and pose said no, 19 (95%) were real people it simply
  missed (illustrated poster art, not the photographic content it's
  tuned for). `pose_n_persons` is still used for its own purpose (pose
  dynamism), just not trusted as a presence vote.
- **water**: Nova (`27`) alone -- no deterministic partner survived.
  Rekognition alone scored 54.2% accuracy / 45.0% precision on a
  properly-powered 100-poster review (38 real positives); Nova alone hit
  83.3% / 80.6%. Segmentation's three water reads (`ade_water`/
  `minc_water`/`clip_water`) were rejected the same way as animal/weapon
  (7-11% precision on their own disagreements).
- **fire**: Nova (`27`) alone -- same shape as water. On a 50-poster
  review (13 real positives), Nova alone hit 78.0% accuracy / 55.6%
  precision, ahead of Rekognition (58.0% / 27.8%) and segmentation's
  `clip_fire` (54.0% / 22.2%). Public Hugging Face fire-detection models
  were found and considered but not imported -- they're trained on real
  photographic wildfire footage, a domain mismatch with illustrated
  poster art, and Nova already beat every deterministic candidate that
  was actually tested.

The pattern that emerged: DINO and Rekognition are each unreliable for
exactly one question (DINO for weapon and monster, Rekognition for
animal) while being fine or excellent on the others -- *which* engine is
the weak link is a real, per-question finding from blind human review,
not something to guess or apply uniformly. Person adds a different
failure shape entirely: pose isn't noisy like DINO/Rekognition, it's
blind -- high precision, terrible recall, the opposite problem. Water and
fire add a third shape: no engine failed outright, but no *deterministic*
engine was good enough to earn a place next to Nova either -- both
signals ended up as Nova alone, breaking the "one deterministic + Nova"
default that held for the other four. This is exactly what the tool is
for: re-run it for any new overlapping pair rather than assume the last
question's answer generalizes.

**A fifth candidate that looked promising and wasn't**: the private
project separately ran real semantic/material/concept segmentation over
65,201 posters (SegFormer-b0/ADE20K + a SigLIP2 material read + CLIP
zero-shot concepts) -- corpus-wide it agreed with the already-trusted
engines 83-92% of the time, which looked like a viable fourth vote for
animal/weapon. Scored the same way, both failed badly (26.0%/22.0%
accuracy) -- worse, they were wrong in *both* directions (false positives
on the disagreement subset it uniquely flagged, and 100% false negatives
on posters the already-trusted engines unanimously caught). Not added to
either rule. See docs/RESULTS.md for the full numbers -- flagged
specifically as a negative result worth documenting, since aggregate
agreement rate turned out to be a bad predictor of trustworthiness here,
the same lesson DINO/Rekognition already taught.

Sample size caveat: each of these is n=50 (n=40 on the specific
disagreement subset) -- real, live-verified evidence, but a larger
confirmatory sample is a reasonable next step given how clean the 0%/100%
results are, before treating them as exact population statistics.

`27_nova_scene_enrich.py`'s own docstring raises a related, still-open
methodological question: its `nova_weapon`/`nova_monster`/`nova_animal`
fields are combined in one call with descriptive fields (mood, fear
labels, a text description) rather than isolated the way title-text
(gate 6) and moderation (gate 15) were after evidence showed combining
hurt those specifically. The reconciliation results above are good news
on this question, not proof it's resolved -- Nova scored well across all
three signals as currently combined, but that's consistent with (not
proof against) isolating it scoring even better.

## Structure

```
Makefile                           `make sample` / `make test-fast` / `make assemble-sample`
CONTRIBUTING.md                    invariants (one poster at a time, Nova not a stage)
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
  22_creature_weapon_nova_qa.py  Nova Pro QA of 20/21's boxes -- needs AWS,
                                  not a pipeline stage, see docs/RESULTS.md
  23_census_nova_qa.py           Nova Pro QA of 06's CLIP census -- needs AWS,
                                  not a pipeline stage; agree maps CLIP
                                  uncertain → none (Nova has no uncertain)
  24_typography_nova_qa.py       Nova Pro QA of 08's CLIP typography axis --
                                  needs AWS, not a pipeline stage
  26_rekognition_enrich.py       AWS Rekognition labels/image-quality/face-demographics
                                  -- needs AWS, a real per-poster metric (unlike 22-24),
                                  not yet live-verified or wired into compute_metrics.asl.json
  27_nova_scene_enrich.py        Nova Pro mood/fear-labels/description + its own
                                  weapon/monster/person/animal presence reads -- the
                                  remaining fields from the real project's combined
                                  enrich call not already isolated as gate 6/15 --
                                  needs AWS, not yet live-verified or wired into
                                  compute_metrics.asl.json
  utils/
    logging_setup.py, resumable.py   shared conventions with the sibling
                                      poster-corpus-validation repo
    posters.py                       shared TMDB/S3/local-disk poster cache
    clip_backbone.py                 shared CLIP model loading + text-prototype
                                      helper, used by 06/07/08/09
    siglip_backbone.py               shared SigLIP model loading + text-prototype
                                      helper, used by 12/13
    device.py                        cuda > mps > cpu for 17/19/20/21
  qa/
    validate_genre_classifier_vs_imdb.py   scores 09 against IMDb's own genre tags
    build_signal_reconciliation_review_page.py, compare_signal_engines.py
                                            blind human review + scoring for signals
                                            two or more engines both claim to answer
                                            (animal, weapon, monster) -- see "Reconciling
                                            overlapping signals" below. For
                                            creature/weapon specifically, prefer
                                            25_creature_weapon_agreement.py's
                                            IoU-based join -- this tool's
                                            weapon_n>0 boolean check is a
                                            coarser, complementary signal
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
  test_nova_scene_enrich.py         27: _fear_labels/_score/_join_list
  test_pose_dynamism.py             19: compute_metrics
  test_posters.py                   cache hit never touches the network
  test_pyproject_extras.py          cpu / tf-saliency / bedrock extras
  test_readme_inventory.py          README Structure names every test_*.py
  test_rekognition_enrich.py        26: _flag() label-presence scorer
  test_resumable.py                 flock, 0-byte header, no duplicate ids
  test_saliency_prediction.py       18: heatmap summary (slow: live MSI-Net)
  test_sample_output_contract.py    99-poster sample: one row per id, assemble
  test_schema_contract.py           sample headers match FIELDS / SCHEMA
  test_signal_reconciliation.py     qa/: verdict extraction, agreement scoring
  test_siglip_backbone.py           SigLIP softmax/cosine math (synthetic)
  test_typography_nova_qa.py        24: bin_register / agree_adjacent
  test_validate_genre_classifier_vs_imdb.py  genre-vs-IMDb metrics (no CLIP)
```

## License

MIT — see [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).
