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

**Status: color and perceptual-quality metrics are built and documented
(below). CLIP/SigLIP and geometric composition are real, already-run
parts of the project but not yet ported to this public repo.**

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/01_color_metrics.py       # color: CIELAB + k-means palette (no AWS, no heavy deps)
python3 scripts/02_iqa_multi_score.py     # quality: clipiqa/musiq/brisque via pyiqa
python3 scripts/03_nima_score.py          # quality: NIMA aesthetic score
python3 scripts/04_laion_aesthetic_score.py  # quality: LAION aesthetic predictor
python3 -m pytest tests/ -v
```

No API key or AWS needed anywhere in this repo — posters come from TMDB's
public image CDN. The four scripts above share one poster cache
(`data/posters_cache/`, see `utils/posters.py`): whichever script runs
first downloads a given poster, the others reuse that file instead of
re-fetching it. That cache can optionally check S3 first
(`--posters-s3-bucket`, matching the real project's own storage pattern)
before falling back to TMDB — entirely optional, off by default.

Not tied to horror specifically: `01_color_metrics.py` has no
genre-specific logic and was verified live against a real, non-horror
(sci-fi) sample with zero code changes — see "Genre-agnostic, verified"
in docs/RESULTS.md.

## Structure

```
scripts/
  01_color_metrics.py         Per-poster brightness/saturation/hue-bands/
                               dominant-palette (CIELAB, saturation-weighted
                               k-means)
  02_iqa_multi_score.py       clipiqa/musiq/brisque via pyiqa
  03_nima_score.py            NIMA aesthetic score
  04_laion_aesthetic_score.py LAION aesthetic predictor (CLIP ViT-L/14 + MLP head)
  utils/
    logging_setup.py, resumable.py   shared conventions with the sibling
                                      horror-corpus-validation repo
    posters.py                       shared TMDB/S3/local-disk poster cache,
                                      used by all four scripts above
data/
  sample_input/    99 real posters, stratified by decade (1920s-2020s)
  sample_output/   real, already-computed metrics for those same posters
                    (color: verified to reproduce exactly; quality scores:
                    see docs/RESULTS.md for why exact reproduction isn't
                    guaranteed the same way for model-based metrics)
docs/
  METHODOLOGY.md   what's computed and why, per category
  RESULTS.md        real findings, per category (the color Continue/Pivot
                     checkpoint; the perceptual-quality reproduction caveat)
tests/
  test_color_metrics.py   pure-function tests for the color math
  test_posters.py         shared poster-cache logic -- confirms the local-
                           cache-hit path never touches the network, and
                           that no AWS import happens when S3 isn't configured
```

## License

MIT — see [LICENSE](LICENSE).
