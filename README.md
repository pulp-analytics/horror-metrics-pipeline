# horror-metrics-pipeline

Python pipeline for computing per-poster metrics at scale: color palettes,
CLIP/SigLIP semantic embeddings (fear axis, monster census, typography,
medium/painted-vs-photo classification), perceptual quality scores
(pyiqa, NIMA, LAION aesthetic), and geometric composition metrics.

Part of the [Pulp Analytics](https://github.com/pulp-analytics) horror poster
analysis project ("The Anatomy of Fear").

**Scope: this repo analyzes one poster at a time and stops there.**
Aggregating those per-poster metrics into charts/trends/decisions (e.g.
the yearly brightness curve, the Continue/Pivot checkpoint) is a
presentation concern for whatever consumes this data downstream, not
something computed here — that logic will live in a separate front-end/
presentation repo once one exists, not in this one.

**Status: color metrics are built and documented (below). CLIP/SigLIP,
perceptual quality, and geometric composition are real, already-run parts
of the project (see the private pipeline) but not yet ported to this
public repo — don't assume scripts for those exist here yet.**

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/01_color_metrics.py    # per-poster color analysis
python3 -m pytest tests/ -v
```

No API key needed — posters are fetched from TMDB's public image CDN, no
AWS involved anywhere in this category. Not tied to horror specifically:
the script has no genre-specific logic, and was verified live against a
real, non-horror (sci-fi) sample with zero code changes — see
"Genre-agnostic, verified" in docs/RESULTS.md.

## Structure

```
scripts/
  01_color_metrics.py   Per-poster brightness/saturation/hue-bands/
                         dominant-palette (CIELAB, saturation-weighted
                         k-means -- see docs/METHODOLOGY.md)
  utils/                 Shared logging, resumability, sharding (same
                         conventions as the sibling horror-corpus-
                         validation repo)
data/
  sample_input/    99 real posters, stratified by decade (1920s-2020s)
  sample_output/   real, already-computed color metrics for those same posters
docs/
  METHODOLOGY.md   what's computed and why (incl. why saturation-weighted
                    k-means instead of uniform k-means)
  RESULTS.md        the real Continue/Pivot checkpoint finding that
                     motivated this category, for context -- computed by
                     aggregation logic that lives outside this repo (see
                     Scope above), not by anything checked in here
tests/
  test_color_metrics.py   pure-function tests for the color math -- no
                           network calls
```

## License

MIT — see [LICENSE](LICENSE).
