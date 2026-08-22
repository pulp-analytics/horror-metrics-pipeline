#!/usr/bin/env python3
"""Nova Pro scene enrichment: mood tags, visual fear concepts, a short
neutral description, and Nova's own weapon/monster/person/animal presence
reads -- the remaining fields from the real project's combined
`nova_poster_enrich.py` call that this port hasn't isolated into their own
gate yet.

`face_count` (PENDING -- not in the real project's original prompt, added
here and not yet live-verified): a third opinion on how many real human
faces are on a poster, next to `faces_n_faces` (YuNet, local/free) and
`rek_n_faces` (Rekognition, `26_rekognition_enrich.py`). Added after a
real finding on 21 high-disagreement posters (blind human review, see
docs/RESULTS.md): YuNet and Rekognition often disagree sharply on crowded/
collage posters, and the *higher* of the two is usually (not always) the
one closer to a real human count -- both engines tend to undercount, not
invent faces. Whether Nova adds real value here (a third read, or the
tiebreaker when YuNet/Rekognition disagree) is an open question this
field exists to answer once this script has a real AWS run -- scored the
same way as everything else in this repo, not assumed.

`poster_qa_verdict`/`poster_qa_reason` (PENDING -- also not in the real
project's original prompt, also not yet live-verified): asks whether the
image is actually poster art at all, a question this repo never had a
field for. The real pipeline's `qa_title_ocr.py` already runs Nova with
vision on every poster, but only to judge whether OCR's title reading is
accurate -- its `no_title_on_poster` verdict fires identically whether
the image (a) is a real, designed poster with an illegible/artistic title
treatment (checked three real examples: *The Keep*, *Cypher*, *O* -- all
famous posters where the title is worked into the art itself, not
plain text) or (b) isn't poster art at all (checked one real example:
*Dr. Jekyll and Mr. Hyde*, id 3006, whose `poster_path` is a plain
production/actor photo with no poster design of any kind). Nova's own
`title_ocr_qa_reason` text was the same boilerplate ("no visible title
text") for all four -- proof the existing pass cannot tell these apart,
because it was never asked to. This field asks the separate question
directly. Rule out at your leisure: of the corpus's 4,915 posters
(3.4%) with `title_ocr_qa_verdict == "no_title_on_poster"`, an unknown
fraction are case (b), not case (a) -- this field is how that gets
measured, once run, against a blind human spot-check the same way every
other signal in this repo was validated.

**Update -- real ground truth for this already exists in the sibling
`poster-corpus-validation` repo**, discovered after this field was drafted:
its `data/ground_truth/poster_type_human_labels.csv` is a real 2,539-poster
blind human review of zero-OCR-text posters (`es_poster` / `no_es_poster` /
`no_seguro`) -- **73.1% "no es poster."** Far higher than the informal
4-poster spot check above suggested (1/4). It even covers the *Dr. Jekyll
and Mr. Hyde* example cited above -- and the real human reviewer marked it
`no_seguro`, not the clean "not a poster" call made above, a useful check
on how confident this docstring should sound. That ground truth covers
1,904 of this corpus's 4,915 `no_title_on_poster` ids (the other 3,011
were never reviewed there -- different source filter, `ocr_chars == 0` in
`poster_title_match.csv` vs. this corpus's `title_ocr_qa_verdict`). This
field is still useful for covering the remaining 3,011 plus the rest of
the 145k-poster corpus outside the zero-OCR-text set entirely, but check
that ground truth file first before spending a new AWS call re-answering
a question 1,904 posters already have a real human answer for.

`labels`/`labels_top` (PENDING, also not in the real project's original
prompt): a general open-vocabulary object/scene read, the direct
counterpart to `26_rekognition_enrich.py`'s `rek_labels`/`rek_top`/
`rek_top_conf` (Rekognition `DetectLabels`, up to 10 labels with
confidence). No engine in this repo currently gives Nova's own read on
this -- `nova_fear_labels` is deliberately horror-specific (weapon, blood,
silhouette, etc.), not a general-purpose label list, so it's not a
substitute. Added specifically so `rek_labels` can get the same
reconcile-before-trust treatment every other Rekognition field in this
document did.

**Update -- `rek_labels`/`rek_top` already lost, on its own merits, no
Nova run needed to reach that verdict:** a 60-poster verification review
(shown Rekognition's actual claimed top label, asked whether it accurately
describes something visible) found "Book" and "Advertisement" -- 57% of
the corpus's `rek_top` values -- score 0/20 and 0/20, everything else
9/20 (45%), 15.0% overall. Rekognition's general-purpose label model
appears to read movie posters as a document *type* (rectangular,
text-heavy → "Book"/"Advertisement") rather than describe their actual
content. See docs/RESULTS.md, "Rekognition's overall cost-benefit
ledger." `nova_labels` is still worth running once AWS access exists --
it's now an independent read on its own merits, not a benchmark
`rek_labels` needs a chance to win.

**Not attempted: `rek_n_boxes`** (Rekognition's count of localized object
*instances* across all detected labels, from `DetectLabels`'
`Instances` field) has no Nova counterpart and isn't getting one -- this
is a structural gap, not an oversight. `rek_n_boxes` requires per-object
pixel localization, something a vision-language model prompted for JSON
text output cannot do reliably (unlike a naming/counting judgment, which
is exactly what `face_count` above already tests Nova on). Treat
`rek_n_boxes` the same way this repo already treats `rek_sharp`/
`rek_contrast`: no competitor exists to test it against, so it's kept on
Rekognition's say-so alone, not reconciled.
Two fields from that same real prompt are deliberately NOT here, because
they already have a better home:
- `title_text` -> poster-corpus-validation's gate 6 (`06_bedrock_ocr.py`)
- `blood_gore`/`violence`/`sexual_content`/`sensitive`/`moderation_notes`
  -> poster-corpus-validation's gate 15 (`15_content_moderation.py`)
Both were split out specifically because the real project's own combined-
vs-isolated test found the combined prompt scores dramatically worse on
those numeric judgments (isolated won 3-6x on moderation, see this
repo's docs/RESULTS.md and the sibling repo's). `languages` is also
dropped -- gate 7 (Comprehend) already answers that from OCR'd text.

**Open methodological question, not yet resolved**: `weapon`/`monster`/
`person`/`animal` below are the same *kind* of field as the moderation
scores that benefited from isolation (a numeric 0-1 presence judgment),
but they're kept combined here with the descriptive fields (mood, fear
labels, description) rather than split into their own call, because there
is not yet a real, measured reason to believe combining them with
descriptive text generation hurts the same way combining them with
*other* numeric judgments did. This is exactly checkable with
`scripts/qa/build_signal_reconciliation_review_page.py` /
`compare_signal_engines.py` once this script has a real run: if
`nova_weapon`/`nova_animal` score meaningfully worse against blind human
review than the isolated moderation fields did, that's evidence this
prompt should be split further. Don't assume either way -- measure.

Why this matters for reconciliation specifically: `nova_weapon`/
`nova_animal`/`nova_monster` are a *fourth* independent opinion on
questions this repo already cross-checks two or three ways --
`06_clip_census.py` (`is_animal`/`is_creature`), `20`/`21`/`25`
(OWLv2/DINO/agreement, weapon+creature), and `26_rekognition_enrich.py`
(`rek_animal`/`rek_weapon`, also not yet live-verified). Add it to
`scripts/qa/build_signal_reconciliation_review_page.py`'s `ENGINES` dict
once this has real data.

Needs real AWS access (Bedrock's Nova Pro), same exception as scripts
22-24/26. Not yet live-verified (AWS credentials unavailable) or wired
into `compute_metrics.asl.json`.

  export AWS_PROFILE=sandbox_bedrock
  python3 27_nova_scene_enrich.py --in data/sample_input/sample_100_posters.csv --workers 8

Resumable: re-running with the same --out skips ids already processed.
Shares this repo's poster cache -- see utils/posters.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_setup import get_logger
from utils.posters import add_poster_source_args, fetch_poster_file
from utils.resumable import load_done_ids, open_for_append, shard_rows

log = get_logger("nova_scene_enrich")

DEFAULT_MODEL_ID = "us.amazon.nova-pro-v1:0"
REGION = "us-east-1"

SCENE_PROMPT = """You analyze a movie poster image. Return ONLY valid JSON (no markdown) with this exact schema:
{
  "credits_text": "tagline, cast, director, studio, billing block text (empty if none)",
  "other_text": "any other visible text not covered above",
  "mood": ["up to 5 short mood/atmosphere tags, e.g. dread, camp, gothic, erotic, surreal"],
  "fear_labels": [{"name": "label", "conf": 0.0}],
  "weapon": 0.0,
  "monster": 0.0,
  "person": 0.0,
  "animal": 0.0,
  "face_count": 0,
  "poster_qa_verdict": "poster" | "not_poster" | "uncertain",
  "poster_qa_reason": "one short sentence",
  "labels": [{"name": "label", "conf": 0.0}],
  "description": "1-2 sentence neutral visual description for search/embeddings"
}

Rules:
- fear_labels: up to 12 visual concepts useful for horror analysis (weapon, knife, gun, monster, creature, ghost, skull, blood, fire, water, silhouette, face, crowd, house, forest, vehicle, text-heavy, etc.). conf in 0..1.
- labels: up to 10 GENERAL objects/scene elements you see, ordinary open-vocabulary
  terms (e.g. person, car, building, tree, sky, animal, weapon, text) -- not
  horror-specific like fear_labels above. Order from most to least confident;
  conf in 0..1. This is a general-purpose read, the same kind of thing an
  object/scene classifier would return, not a curated or thematic list.
- weapon/monster/person/animal: likelihood 0..1 that the poster artwork genuinely shows that subject, from the image itself.
- face_count: integer count of distinct real human faces visible (count each repeated
  instance in a pattern/collage separately; do not count skulls, monster faces, or
  faces implied only by a silhouette -- see docs/RESULTS.md's face-count reconciliation
  for the exact convention this follows, ported from a real blind human review).
- poster_qa_verdict: "poster" if this image is genuine promotional poster/key-art design
  intended to advertise the film -- illustrated art, a photographic composite, collage,
  or any kind of title treatment, EVEN IF the title is stylized or illegible (letters
  worked into the artwork itself, not plain rendered text -- this is not asking whether
  the title is readable, a separate check already covers that). Use "not_poster" only
  when the image is something else entirely: a plain production/behind-the-scenes
  photo, an actor portrait or headshot, a video screenshot, a blank/placeholder image,
  or artwork unrelated to promoting this film. Use "uncertain" only if genuinely
  ambiguous after looking closely. poster_qa_reason: one short sentence naming what you
  actually see (e.g. "black-and-white studio portrait of an actor, no poster design" or
  "illustrated key art with the title worked into the building's architecture").
- Keep description factual and concise (<= 45 words). No spoilers beyond what the poster shows.
"""

FIELDS = [
    "id", "title", "year",
    "nova_credits_text", "nova_other_text", "nova_mood", "nova_fear_labels",
    "nova_weapon", "nova_monster", "nova_person", "nova_animal", "nova_face_count",
    "nova_poster_qa_verdict", "nova_poster_qa_reason",
    "nova_labels", "nova_top_label", "nova_top_label_conf",
    "nova_description",
    "error",
]


def _join_list(val) -> str:
    if not val:
        return ""
    if isinstance(val, list):
        return "|".join(str(x).strip() for x in val if str(x).strip())
    return str(val).strip()


def _fear_labels(val) -> str:
    """Ported as-is from the real nova_poster_enrich.py's _fear_labels()."""
    if not val:
        return ""
    if isinstance(val, str):
        return val
    parts = []
    for item in val:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            conf = item.get("conf", item.get("confidence", ""))
            try:
                conf_s = f"{float(conf):.2f}"
            except (TypeError, ValueError):
                conf_s = str(conf)
            if name:
                parts.append(f"{name}:{conf_s}")
        else:
            parts.append(str(item))
    return "|".join(parts)


def _score(val, default: float = 0.0) -> float:
    """Ported as-is from the real nova_poster_enrich.py's _score()."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, x)), 4)


def _face_count(val, default: int = 0) -> int:
    """New field (not in the real project's original prompt) -- added
    specifically to compare against faces_n_faces (YuNet, local) and
    rek_n_faces (Rekognition DetectFaces) once this script has a real run.
    See docs/RESULTS.md's face-count reconciliation for why: on 21 real
    high-disagreement posters where YuNet and Rekognition differ sharply,
    the higher of the two is usually (though not always -- 2 real
    counter-examples found) closer to a human's actual count, suggesting
    both undercount on crowded/collage posters rather than either
    over-counting -- Nova's own read is untested and worth adding."""
    try:
        x = int(float(val))
    except (TypeError, ValueError):
        return default
    return max(0, x)


def _poster_qa_verdict(val) -> str:
    """New field -- normalizes Nova's poster/not_poster/uncertain verdict;
    anything unrecognized (including a missing/malformed response) falls
    back to "uncertain" rather than silently defaulting to "poster"."""
    v = str(val or "").strip().lower()
    return v if v in ("poster", "not_poster", "uncertain") else "uncertain"


def _top_from_labels(val) -> tuple[str, float]:
    """New field -- pulls the highest-confidence (name, conf) pair out of
    the raw `labels` list, the direct counterpart to rek_top/rek_top_conf
    (26_rekognition_enrich.py's DetectLabels top pick). Nova is asked to
    already order labels most-to-least confident, but this re-sorts
    defensively rather than trusting that ordering blindly."""
    if not val or not isinstance(val, list):
        return "", 0.0
    best_name, best_conf = "", -1.0
    for item in val:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        try:
            conf = float(item.get("conf", item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            conf = 0.0
        if name and conf > best_conf:
            best_name, best_conf = name, conf
    return (best_name, round(max(0.0, best_conf), 4)) if best_name else ("", 0.0)


def call_nova(bedrock, img_bytes: bytes, model_id: str) -> dict:
    result = bedrock.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": "jpeg", "source": {"bytes": img_bytes}}},
                {"text": SCENE_PROMPT},
            ],
        }],
        inferenceConfig={"maxTokens": 850, "temperature": 0},
    )
    text = result["output"]["message"]["content"][0]["text"].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    import json
    data = json.loads(text)
    top_label, top_label_conf = _top_from_labels(data.get("labels"))
    return {
        "nova_credits_text": str(data.get("credits_text") or "").replace("\n", " ").strip()[:500],
        "nova_other_text": str(data.get("other_text") or "").replace("\n", " ").strip()[:500],
        "nova_mood": _join_list(data.get("mood")),
        "nova_fear_labels": _fear_labels(data.get("fear_labels")),
        "nova_weapon": _score(data.get("weapon")),
        "nova_monster": _score(data.get("monster")),
        "nova_person": _score(data.get("person")),
        "nova_animal": _score(data.get("animal")),
        "nova_face_count": _face_count(data.get("face_count")),
        "nova_poster_qa_verdict": _poster_qa_verdict(data.get("poster_qa_verdict")),
        "nova_poster_qa_reason": str(data.get("poster_qa_reason") or "").replace("\n", " ").strip()[:300],
        "nova_labels": _fear_labels(data.get("labels")),
        "nova_top_label": top_label,
        "nova_top_label_conf": top_label_conf,
        "nova_description": str(data.get("description") or "").replace("\n", " ").strip()[:400],
    }


def process_one(row: dict, bedrock, session, posters_dir: Path, s3_bucket: str, s3_prefix: str, model_id: str) -> dict:
    out = {"id": row["id"], "title": row.get("title", ""), "year": row.get("year", ""), "error": ""}
    poster_file = posters_dir / f"{row['id']}.jpg"
    if not fetch_poster_file(session, row.get("poster_path", ""), poster_file, s3_bucket, s3_prefix):
        out["error"] = "download_failed"
        return out
    try:
        out.update(call_nova(bedrock, poster_file.read_bytes(), model_id))
    except Exception as e:
        out["error"] = str(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/sample_input/sample_100_posters.csv")
    ap.add_argument("--out", default="data/sample_output/nova_scene_enrich.csv")
    add_poster_source_args(ap)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1, help="split --in across N parallel shards (default 1: no sharding)")
    args = ap.parse_args()

    import boto3
    bedrock = boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).client("bedrock-runtime", region_name=args.region)

    with open(args.in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = shard_rows(rows, args.shard_index, args.shard_count)

    out_path = Path(args.out)
    done = load_done_ids(out_path)
    todo = [row for row in rows if row["id"] not in done and row.get("poster_path")]
    if done:
        log.info(f"resuming: {len(done)} already done, {len(todo)} remaining")

    posters_dir = Path(args.posters_dir)
    import requests
    t0 = time.time()
    n_ok = n_err = 0

    f, w = open_for_append(out_path, FIELDS)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(process_one, row, bedrock, requests.Session(), posters_dir,
                          args.posters_s3_bucket, args.posters_s3_prefix, args.model): row
                for row in todo
            }
            n_done = 0
            for fut in as_completed(futs):
                out = fut.result()
                w.writerow(out)
                n_done += 1
                if out.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    rate = n_done / max(time.time() - t0, 1e-9)
                    log.info(f"{n_done}/{len(todo)} rate={rate:.2f}/s ok={n_ok} err={n_err}")
    finally:
        f.close()

    log.info(f"wrote {out_path}: {n_ok} scored, {n_err} failed (this run)")


if __name__ == "__main__":
    main()
