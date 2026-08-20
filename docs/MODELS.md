# Models

Every model this repo loads, what it actually resolves to, and how tight
that pin is. Written after finding that two of these (SigLIP, the LAION
aesthetic head) were loading from a Hugging Face Hub model_id with no
`revision` pin -- meaning `from_pretrained(MODEL_ID)` silently tracks
whatever's on that repo's `main` branch, not a fixed artifact. The same
gap later showed up on MiDaS, MSI-Net, ViTPose, YOLOv8n, OWLv2, and
Grounding DINO when those scripts were ported; those are pinned below
too. This doc exists so the next model added to this repo gets the same
check instead of the gap reappearing.

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

## MiDaS_small (`intel-isl/MiDaS`, torch.hub) -- 17

Was unpinned; fixed. `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")`
clones that GitHub repo's default branch. The GitHub repo was renamed to
`isl-org/MiDaS` (intel-isl still redirects). `17_depth_estimation.py`
now loads a pinned commit:

```python
MIDAS_GITHUB = "intel-isl/MiDaS"
MIDAS_REVISION = "1645b7e1675301fdfac03640738fe5a6531e17d6"  # tag v3_1
```

Verified 2026-08-19 via `https://api.github.com/repos/isl-org/MiDaS/git/refs/tags/v3_1`.
The nested `torch.hub.load()` of `rwightman/gen-efficientnet-pytorch`
inside MiDaS's hubconf is still not pinned from this repo -- same class
of gap as pyiqa (a library we don't own fetching its own weights). The
lever we do own is the MiDaS repo ref.

## MSI-Net (`alexanderkroner/MSI-Net`) -- 18

Was unpinned; fixed. `snapshot_download(repo_id=...)` without `revision`
tracks that Hub repo's `main`. `18_saliency_prediction.py` now passes:

```python
MODEL_ID = "alexanderkroner/MSI-Net"
MODEL_REVISION = "d950b35945db961ae63f84bc2b23f6bd578d0b8f"
```

Verified 2026-08-19 via `curl https://huggingface.co/api/models/alexanderkroner/MSI-Net`.

## YOLOv8n + ViTPose -- 19

Two loads, both were unpinned; both fixed.

**ViTPose** (`usyd-community/vitpose-base-simple`): HF Hub `revision`, same
pattern as SigLIP.

```python
VITPOSE_ID = "usyd-community/vitpose-base-simple"
VITPOSE_REVISION = "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"
```

Verified 2026-08-19 against that repo's API `sha`.

**YOLOv8n**: `YOLO("yolov8n.pt")` downloads whatever file currently sits at
that name on `ultralytics/assets`' latest GitHub release -- no Hub
revision, no hash check. Pinned like YuNet: a specific release URL plus
content hash. See `19_pose_dynamism.py::YOLO_SHA256` and `ensure_yolo()`.

```python
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
```

Verified 2026-08-19 by downloading that asset and hashing it.

## OWLv2 (`google/owlv2-base-patch16`) -- 20

Was unpinned; fixed. `20_creature_weapon_owlv2.py` now passes
`MODEL_REVISION` to both `Owlv2Processor.from_pretrained()` and
`Owlv2ForObjectDetection.from_pretrained()`:

```python
MODEL_ID = "google/owlv2-base-patch16"
MODEL_REVISION = "2a1560802f8cf3c408fec9b809d705f56a2f7146"
```

Verified 2026-08-19 via `curl https://huggingface.co/api/models/google/owlv2-base-patch16`.

## Grounding DINO (`IDEA-Research/grounding-dino-tiny`) -- 21

Was unpinned; fixed. Same Hub-revision pattern as OWLv2/SigLIP:

```python
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
```

Verified 2026-08-19 via `curl https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny`.

## Not pinned here, and why that's fine

- **pyiqa metrics** (clipiqa, musiq, brisque -- `02_iqa_multi_score.py`)
  and **NIMA** (`03_nima_score.py`): weights are fetched by the `pyiqa`
  library itself from URLs hardcoded per pyiqa release, not something this
  repo's code calls directly. The lever here is the `pyiqa` version in
  `requirements.txt`, not a model_id this repo owns.
- **Amazon Bedrock Nova Pro** (`us.amazon.nova-pro-v1:0`, scripts 22-24):
  a managed model behind a versioned-looking ID that AWS can still update
  server-side without a client-visible changelog. Not pinnable from the
  caller's side -- each output row records the `model_id` as the closest
  available provenance signal. Calls use `temperature: 0`. The prompts
  in those scripts are the settled text after several runs; see
  [METHODOLOGY, "How we trust a metric"](METHODOLOGY.md#how-we-trust-a-metric).
- **MiDaS nested EfficientNet backbone** (see MiDaS section above): fetched
  by intel-isl/MiDaS's own hubconf, not by a model_id this repo owns.
