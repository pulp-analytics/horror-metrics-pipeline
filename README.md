# poster-metrics-pipeline

Python pipeline for computing per-poster metrics at scale: color palettes,
CLIP/SigLIP semantic embeddings (fear axis, monster census, typography,
medium/painted-vs-photo classification), perceptual quality scores
(pyiqa, NIMA, LAION aesthetic), and geometric composition metrics.

Part of the [Pulp Analytics](https://github.com/pulp-analytics) horror poster
analysis project ("The Anatomy of Fear").

**Scope: this repo analyzes one poster at a time and stops there.**
Aggregating those per-poster metrics into charts/trends/decisions (e.g.
the yearly brightness curve, the Continue/Pivot checkpoint, decade-level
register shares) is a presentation concern for whatever consumes this
data downstream, not something computed here — that logic will live in a
separate front-end/presentation repo once one exists, not in this one.

**Status: color, perceptual quality, CLIP semantic embeddings, and SigLIP
semantic embeddings are built and documented (below). Geometric
composition is a real, already-run part of the project but not yet
ported to this public repo.**

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/01_color_metrics.py           # color: CIELAB + k-means palette
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
python3 -m pytest tests/ -v
```

No API key or AWS needed anywhere in this repo — posters come from TMDB's
public image CDN. All nine download-capable scripts (everything except
`06`-`09` and `12`-`13`, which read `05`'s or `11`'s cache) share one
poster cache (`data/posters_cache/`, see `utils/posters.py`): whichever
script runs first downloads a given poster, the others reuse that file.
That cache can optionally check S3 first (`--posters-s3-bucket`, matching
the real project's own storage pattern) before falling back to TMDB —
entirely optional, off by default.

Not tied to horror specifically: `01_color_metrics.py` has no
genre-specific logic and was verified live against a real, non-horror
(sci-fi) sample with zero code changes — see "Genre-agnostic, verified"
in docs/RESULTS.md.

## Structure

```
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
  utils/
    logging_setup.py, resumable.py   shared conventions with the sibling
                                      poster-corpus-validation repo
    posters.py                       shared TMDB/S3/local-disk poster cache
    clip_backbone.py                 shared CLIP model loading + text-prototype
                                      helper, used by 06/07/08/09
    siglip_backbone.py               shared SigLIP model loading + text-prototype
                                      helper, used by 12/13
data/
  sample_input/    99 real posters, stratified by decade (1920s-2020s)
  sample_output/   real, already-computed metrics for those same posters,
                    plus real 99-poster clip_embeddings.npz and
                    siglip_embeddings.npz generated for this repo (see
                    docs/RESULTS.md for what's verified to reproduce
                    exactly vs. what isn't, and why)
docs/
  METHODOLOGY.md   what's computed and why, per category
  RESULTS.md        real findings, per category
tests/
  test_color_metrics.py     pure-function tests for the color math
  test_clip_backbone.py     softmax/cosine-similarity math underlying every
                             CLIP script, on synthetic embeddings
  test_siglip_backbone.py   same math, SigLIP's side, on synthetic embeddings
  test_posters.py           shared poster-cache logic -- confirms the local-
                             cache-hit path never touches the network, and
                             that no AWS import happens when S3 isn't configured
```

## License

MIT — see [LICENSE](LICENSE).
