# horror-metrics-pipeline

Python pipeline for computing per-poster metrics at scale: color palettes,
CLIP/SigLIP semantic embeddings (fear axis, monster census, typography,
medium/painted-vs-photo classification), perceptual quality scores
(pyiqa, NIMA, LAION aesthetic), and geometric composition metrics.

Part of the [Pulp Analytics](https://github.com/pulp-analytics) horror poster
analysis project ("The Anatomy of Fear").

**Status: color metrics are built and documented (below). CLIP/SigLIP,
perceptual quality, and geometric composition are real, already-run parts
of the project (see the private pipeline) but not yet ported to this
public repo — don't assume scripts for those exist here yet.**

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/01_color_metrics.py          # per-poster color analysis
python3 scripts/02_aggregate_and_checkpoint.py  # yearly/decade aggregates + the Continue/Pivot verdict
python3 -m pytest tests/ -v
```

No API key needed — posters are fetched from TMDB's public image CDN, no
AWS involved anywhere in this category.

## Structure

```
scripts/
  01_color_metrics.py            Per-poster brightness/saturation/hue-bands/
                                  dominant-palette (CIELAB, saturation-weighted
                                  k-means -- see docs/METHODOLOGY.md)
  02_aggregate_and_checkpoint.py  Yearly + decade aggregates, the "Color River"
                                  dataset, and the real Continue/Pivot go/no-go
                                  checkpoint from the project's first phase
  utils/                          Shared logging, resumability, sharding
                                  (same conventions as the sibling
                                  horror-corpus-validation repo)
data/
  sample_input/    99 real posters, stratified by decade (1920s-2020s)
  sample_output/   real, already-computed color metrics for those same posters
docs/
  METHODOLOGY.md   what's computed and why (incl. why saturation-weighted
                    k-means instead of uniform k-means)
  RESULTS.md        the real Continue/Pivot checkpoint result, full corpus
                     (63,127 posters) and this repo's 99-poster sample
tests/
  test_color_metrics.py   pure-function tests for the color math -- no
                           network calls
```

## License

MIT — see [LICENSE](LICENSE).
