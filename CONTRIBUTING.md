# Contributing

Thanks for your interest in contributing.

## How to contribute

1. Fork the repo and create your branch from `main`.
2. Make your changes, with a clear commit message (why, not what; match
   recent `git log` — one or two sentences in English).
3. Open a pull request describing what changed and why.
   CI runs `pytest -m "not slow"` on every push and PR (`make test-fast`).
4. Link any related issue.

## Installing (local development)

Corpus runs use the image in poster-analysis-infrastructure, not this
venv. For a laptop or CI:

```bash
pip install -e ".[cpu]"                 # 01-17, 19-21, 25 (no TensorFlow, no boto3)
pip install -e ".[cpu,tf-saliency]"     # plus 18 (TensorFlow / MSI-Net)
pip install -e ".[bedrock]"             # Nova QA (22/23/24) and S3 poster cache
pip install -e ".[all]"                 # everything
```

The extra name `cpu` is "skip TF/boto3", not "CPU-only instances." CI
(`make test-fast`) uses the extra-free default: `pip install -e .`

## Invariants

The contract this repo will not silently break. Scope is also in the
README; columns are in [docs/SCHEMA.md](docs/SCHEMA.md).

**This repo analyzes one poster at a time and stops there.** Decade
curves, Continue/Pivot checkpoints, and quantile register bins of a
batch are presentation logic. They do not belong in `01`–`21` or `25`.
(`24` bins 08's axis for Nova QA only; 08 itself stays continuous.)

Do not:

- Wire `22`/`23`/`24` into `make sample` or into
  `compute_metrics.asl.json` in poster-analysis-infrastructure. Nova QA
  is a vision-LLM cross-check plus human review of disagreements, not a
  pipeline stage.
- Cite `creature_weapon_owlv2.csv` or `creature_weapon_dino.csv` as
  ground truth. The citable file is `creature_weapon_agreement.csv`
  (`25`).
- Rewrite census `uncertain` to `none` in `06` or `13`. That sentinel
  means low CLIP/SigLIP confidence. `23` maps it only when scoring
  `agree` against Nova.
- Put tensorflow, boto3, pyiqa, or ultralytics in the default install.
  New heavy deps go in a pip extra (`cpu` / `tf-saliency` / `bedrock`).
- Add a GPU Batch compute environment in *this* repo. CPU/Fargate vs
  EC2 GPU is poster-analysis-infrastructure
  (`docs/ARCHITECTURE.md`). These scripts already use CUDA when the
  container has it.
- Commit `data/posters_cache/`, `data/models/`, CLIP/SigLIP `.npz`
  caches, `master_dataset.csv`, or `*.csv.lock`.

Do:

- One CSV per metric, one row per poster (`face_expression.csv` is the
  exception: one row per face). Declare `FIELDS` in the script.
- Pin any new model in [docs/MODELS.md](docs/MODELS.md).
- Document columns in SCHEMA and the why in
  [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Semantic metrics must
  stay compatible with the three trust layers (deterministic or
  reproducible compute, second model / Nova on a sample after prompt
  iteration, human `--validate` / cite decision). Check in a sample CSV
  under `data/sample_output/` and keep
  `tests/test_schema_contract.py` green.
- Test pure functions without model downloads. `@pytest.mark.slow` is
  for tests that fetch weights.
- Use `utils.device.pick_device` (`cuda` > `mps` > `cpu`) on new torch
  GPU scripts. `18` is TensorFlow; leave it.

Adding a metric: next unused script number, Makefile graph if it belongs
in `make sample`, extras/pin/SCHEMA/METHODOLOGY/sample/FIELDS as above.
A new **pipeline stage** also needs a state in
`compute_metrics.asl.json` in poster-analysis-infrastructure (and a pin
bump of `METRICS_PIPELINE_REF` in that repo's `Dockerfile.metrics`).
22–24 are QA — do not insert a pipeline stage in that gap without
updating both graphs. Details: [docs/SCHEMA.md](docs/SCHEMA.md#adding-a-metric).

## Reporting issues

Use the Issues tab. Include steps to reproduce, expected vs. actual behavior,
and relevant environment details (OS, Python version, etc.) when applicable.

## Code style

Keep changes focused and readable. Match the existing style in the file
you're editing rather than introducing a new one. Module docstrings
explain *why*; don't add line-by-line comments that restate the code.

## License

By contributing, you agree that your contributions will be licensed under
the project's MIT License.
