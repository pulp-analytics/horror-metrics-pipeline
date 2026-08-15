# Models

Every model this repo loads, what it actually resolves to, and how tight
that pin is. Written after finding that two of these (SigLIP, the LAION
aesthetic head) were loading from a Hugging Face Hub model_id with no
`revision` pin -- meaning `from_pretrained(MODEL_ID)` silently tracks
whatever's on that repo's `main` branch, not a fixed artifact. Fixed here;
this doc exists so the next model added to this repo gets the same check
instead of the gap reappearing.

## CLIP (ViT-B-32, "openai") -- 05-10

Already pinned, no action needed. `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")`
resolves through open_clip's own registry to a fixed URL with a sha256
fragment baked into the filename:

```
>>> open_clip.pretrained.get_pretrained_cfg("ViT-B-32", "openai")["url"]
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

open_clip's downloader (`open_clip/pretrained.py::download_pretrained_from_url`)
checks that hash against the downloaded file every time, so this one was
already integrity-verified by the library -- confirmed by inspecting the
installed package's registry directly, not assumed.

## SigLIP (`google/siglip-base-patch16-224`) -- 11-13

Was unpinned; fixed. `utils/siglip_backbone.py` now passes an explicit
`MODEL_REVISION` to both `AutoModel.from_pretrained()` and
`AutoProcessor.from_pretrained()`:

```python
MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
```

Verified 2026-08-14 via `curl https://huggingface.co/api/models/google/siglip-base-patch16-224`
(the response's `"sha"` field is the repo's current commit on the Hub).
To move this pin forward deliberately, re-run that curl, confirm the new
commit is what you expect, and update `MODEL_REVISION`.

## LAION aesthetic predictor head (`camenduru/improved-aesthetic-predictor`) -- 04

Was unpinned; fixed. `04_laion_aesthetic_score.py` now passes a `revision`
to `hf_hub_download()`:

```python
AESTHETIC_HEAD_REPO = "camenduru/improved-aesthetic-predictor"
AESTHETIC_HEAD_FILE = "sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_HEAD_REVISION = "7b2449be1264fcd9a1cf92e3d30dd29af989c836"
```

Verified 2026-08-14 the same way, against that repo's API endpoint. Note
this repo is a third-party mirror (`camenduru`, not the original LAION/CLIP
team) of the linear-probe weights from the original aesthetic predictor
project -- the revision pin protects against that specific mirror changing
underneath us, not against the mirror having been wrong to begin with.

## YuNet (`face_detection_yunet_2023mar.onnx`) -- 14

Was unpinned (worse: no version control at all, since it's not on a
registry with commit history); fixed via content hash instead of a
revision, since that's the only thing available for a bare file download.
See `14_face_detect.py::MODEL_SHA256` and `ensure_model()`, which now
refuses to proceed -- on first download or on a pre-existing file -- if
the sha256 doesn't match.

## Not pinned here, and why that's fine

- **pyiqa metrics** (clipiqa, musiq, brisque -- `02_iqa_multi_score.py`)
  and **NIMA** (`03_nima_score.py`): weights are fetched by the `pyiqa`
  library itself from URLs hardcoded per pyiqa release, not something this
  repo's code calls directly. The lever here is the `pyiqa` version in
  `requirements.txt`, not a model_id this repo owns.
- **Amazon Bedrock** (Nova Pro, in the sibling `poster-corpus-validation`
  repo's `04_bedrock_ocr.py`): a managed model behind a versioned-looking
  ID (`us.amazon.nova-pro-v1:0`) that AWS can still update server-side
  without a client-visible changelog. Not pinnable from the caller's side
  at all -- see that repo's own docs for what's captured instead
  (the model_id per output row, as the closest available provenance
  signal).
