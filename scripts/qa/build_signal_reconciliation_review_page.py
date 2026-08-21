#!/usr/bin/env python3
"""Builds a local, self-contained HTML page for blind human review of a
single yes/no presence question ("is there really an animal/weapon on
this poster?"), used to reconcile two or more engines that independently
answer the same question -- e.g. CLIP's zero-shot census (`is_animal`)
vs. Rekognition's `DetectLabels` flag (`rek_animal`), or OWLv2/DINO/
Rekognition's three independent weapon reads.

Why this exists: creature/weapon detection already has one real,
documented precedent for what "reconciling two engines" should look like
-- OWLv2 alone measured 58-67% false positives (see docs/RESULTS.md,
"Creature/weapon detection"), and agreement between OWLv2 and DINO turned
out to be the trustworthy signal, not either engine alone. Adding
25_rekognition_enrich.py's rek_animal/rek_weapon as a further candidate
signal needs the same treatment before trusting it -- scored against a
real blind human review, not assumed to help just because it's a third
opinion.

Deliberately blind: shows only the poster and the question, never any
engine's verdict or score, so the human judgment isn't anchored.

Reads whichever engine files exist for --signal (see ENGINES below) --
if 25_rekognition_enrich.py hasn't been run yet (no AWS access), this
still builds a useful page from CLIP/OWLv2/DINO's existing sample_output
data alone; re-run once Rekognition data lands to fold that engine's
disagreements into the stratified sample too.

Sample selection: every case where the available engines disagree (this
is where errors concentrate -- same reasoning as the sibling
poster-corpus-validation repo's mega-prompt review), plus enough
agreed-positive and agreed-negative anchors to calibrate against, up to
--n total.

Same review-tool pattern (self-contained HTML, base64-embedded images,
localStorage autosave, CSV export/import) as poster-corpus-validation's
scripts/qa/build_*_review_page.py.

  python3 scripts/qa/build_signal_reconciliation_review_page.py --signal animal
  python3 scripts/qa/build_signal_reconciliation_review_page.py --signal weapon
  open data/qa/animal_reconciliation_review.html
"""
from __future__ import annotations

import base64
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.posters import IMG_BASE  # noqa: E402

SAMPLE_INPUT = ROOT / "data" / "sample_input" / "sample_100_posters.csv"
POSTER_CACHE = ROOT / "data" / "qa" / ".signal_reconciliation_cache"
MAX_WORKERS = 16
SEED = 42


def _census_flag_verdict(field: str):
    """CLIP census's is_animal/is_creature columns are the string 'True'/'False'."""
    def _f(row: dict) -> tuple[bool, float] | None:
        if not row:
            return None
        return (str(row.get(field, "")).strip() == "True", float(row.get("score") or 0))
    return _f


def _weapon_boxes_verdict(row: dict) -> tuple[bool, float] | None:
    if not row:
        return None
    try:
        n = int(float(row.get("weapon_n") or 0))
    except ValueError:
        return None
    score = float(row.get("weapon_top_score") or 0) if row.get("weapon_top_score") else 0.0
    return (n > 0, score)


def _creature_boxes_verdict(row: dict) -> tuple[bool, float] | None:
    if not row:
        return None
    try:
        n = int(float(row.get("creature_n") or 0))
    except ValueError:
        return None
    score = float(row.get("creature_top_score") or 0) if row.get("creature_top_score") else 0.0
    return (n > 0, score)


def _score_field_verdict(field: str):
    """Generic 0-1 score column, thresholded at 0.5 -- used for both
    Rekognition's rek_* flags and Nova's nova_* presence scores; the
    threshold/shape is the same regardless of which vision system produced it."""
    def _f(row: dict) -> tuple[bool, float] | None:
        if not row or row.get(field) in (None, ""):
            return None
        score = float(row[field])
        return (score >= 0.5, score)
    return _f


# signal -> question text + list of (engine name, csv path, id-keyed loader, verdict fn)
ENGINES = {
    "animal": {
        "question": "Does this poster show a real, non-human animal (not a costume/mask/silhouette implying one)?",
        "sources": [
            ("clip_census", ROOT / "data" / "sample_output" / "census.csv", _census_flag_verdict("is_animal")),
            ("rekognition", ROOT / "data" / "sample_output" / "rekognition_enrich.csv", _score_field_verdict("rek_animal")),
            ("nova", ROOT / "data" / "sample_output" / "nova_scene_enrich.csv", _score_field_verdict("nova_animal")),
        ],
    },
    "weapon": {
        "question": "Does this poster show a real weapon (not a hand/silhouette that merely implies one)?",
        "sources": [
            # For box-level agreement (IoU-based), prefer
            # scripts/25_creature_weapon_agreement.py's join instead -- that's the
            # stricter, citable signal (see docs/RESULTS.md, "Creature/weapon
            # detection"). This tool's weapon_n>0 boolean is a coarser, complementary
            # presence-only check, useful for calibrating how much a looser rule
            # (either detector fires) vs. box IoU actually differs.
            ("owlv2", ROOT / "data" / "sample_output" / "creature_weapon_owlv2.csv", _weapon_boxes_verdict),
            ("dino", ROOT / "data" / "sample_output" / "creature_weapon_dino.csv", _weapon_boxes_verdict),
            ("rekognition", ROOT / "data" / "sample_output" / "rekognition_enrich.csv", _score_field_verdict("rek_weapon")),
            ("nova", ROOT / "data" / "sample_output" / "nova_scene_enrich.csv", _score_field_verdict("nova_weapon")),
        ],
    },
    "monster": {
        "question": "Does this poster show a real monster/supernatural creature (vampire, zombie, demon, giant creature, etc. -- not a masked human killer with no supernatural element)?",
        "sources": [
            ("clip_census", ROOT / "data" / "sample_output" / "census.csv", _census_flag_verdict("is_creature")),
            ("owlv2", ROOT / "data" / "sample_output" / "creature_weapon_owlv2.csv", _creature_boxes_verdict),
            ("dino", ROOT / "data" / "sample_output" / "creature_weapon_dino.csv", _creature_boxes_verdict),
            ("nova", ROOT / "data" / "sample_output" / "nova_scene_enrich.csv", _score_field_verdict("nova_monster")),
        ],
    },
}

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Signal reconciliation: blind review</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 780px; margin: 0 auto; padding: 24px 20px 220px; background: #fafafa; color: #1a1a1a; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 20px; }
  .progress { position: sticky; top: 0; background: #fafafa; padding: 10px 0; border-bottom: 1px solid #ddd; margin-bottom: 16px; z-index: 10; }
  .progress-bar { height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin-top: 6px; }
  .progress-fill { height: 100%; background: #2f7d5c; transition: width .2s; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 20px; }
  .poster-wrap { text-align: center; margin-bottom: 16px; }
  img { max-width: 100%; max-height: 48vh; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.15); background: #eee; }
  .catalog-title { font-size: 20px; font-weight: 600; text-align: center; margin: 4px 0 4px; }
  .catalog-title .label { display: block; font-size: 11px; font-weight: 400; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }
  .meta { text-align: center; font-size: 13px; color: #888; margin-bottom: 8px; }
  .action-bar { position: fixed; left: 0; right: 0; bottom: 0; background: #fff;
                border-top: 1px solid #ddd; box-shadow: 0 -4px 12px rgba(0,0,0,.08);
                padding: 12px 20px calc(12px + env(safe-area-inset-bottom)); z-index: 20; }
  .action-bar-inner { max-width: 740px; margin: 0 auto; }
  .axis-label { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  .buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px; }
  button.verdict { flex: 1; min-width: 70px; padding: 10px 6px; font-size: 13px; border-radius: 8px; border: 2px solid #ccc; background: #fff; color: #1a1a1a; cursor: pointer; font-weight: 600; }
  button.verdict:hover { border-color: #999; }
  button.verdict.active[data-v="si"] { background: #fbeae5; border-color: #bf3f24; color: #9c331d; }
  button.verdict.active[data-v="no"] { background: #e7f2ec; border-color: #3f7d5c; color: #316447; }
  button.verdict.active[data-v="no_seguro"] { background: #f0f0f0; border-color: #888; color: #444; }
  .nav { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
  .nav button.plain { padding: 8px 16px; border-radius: 6px; border: 1px solid #ccc; background: #fff; color: #1a1a1a; cursor: pointer; }
  .nav button.plain:disabled { opacity: .4; cursor: default; }
  .counter { font-size: 13px; color: #666; }
  .footer { margin-top: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .export, .import-label { padding: 10px 18px; border-radius: 6px; border: none; background: #2f5f8a; color: #fff; cursor: pointer; font-weight: 600; display: inline-block; }
  .export:hover, .import-label:hover { background: #24486b; }
  .import-label { background: #666; }
  .import-label:hover { background: #4d4d4d; }
  .hint { font-size: 12px; color: #999; }
  .jump { font-size: 12px; }
  .jump input { width: 70px; padding: 4px 6px; border-radius: 4px; border: 1px solid #ccc; }
</style>
</head>
<body>

<h1>Blind review: __SIGNAL__ presence</h1>
<div class="sub">__QUESTION__ Engine verdicts are NOT shown -- judge only from the poster.
Progress saves only in this browser (localStorage).</div>

<div class="progress">
  <span class="counter" id="counter"></span>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
</div>

<div class="card">
  <div class="poster-wrap"><img id="poster-img" src="" alt=""></div>
  <div class="catalog-title"><span class="label">Catalog title</span><span id="catalog-title-text"></span></div>
  <div class="meta" id="meta-text"></div>
</div>

<div class="footer">
  <span class="hint" id="export-hint"></span>
  <span class="jump">go to # <input type="number" id="jump-input" min="1" onkeydown="if(event.key==='Enter') jumpTo()"> <button class="plain" onclick="jumpTo()">Go</button></span>
  <span>
    <label class="import-label" for="import-input">Import progress</label>
    <input type="file" id="import-input" accept=".csv" style="display:none" onchange="importCSV(this.files[0])">
    <button class="export" onclick="exportCSV()">Export CSV</button>
  </span>
</div>

<div class="action-bar">
  <div class="action-bar-inner">
    <div class="axis-label">__QUESTION__</div>
    <div class="buttons">
      <button class="verdict" data-v="si" onclick="setVerdict('si')">S&iacute;</button>
      <button class="verdict" data-v="no" onclick="setVerdict('no')">No</button>
      <button class="verdict" data-v="no_seguro" onclick="setVerdict('no_seguro')">No seguro</button>
    </div>
  </div>
  <div class="nav">
    <button class="plain" id="prev-btn" onclick="go(-1)">&larr; Prev</button>
    <span class="hint">&larr; &rarr; = navigate (auto-advances on answer)</span>
    <button class="plain" id="next-btn" onclick="go(1)">Next &rarr;</button>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const STORAGE_KEY = "signal_reconciliation_review_v1_" + "__SIGNAL__";

function loadState() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch (e) { return {}; }
}
function saveState(state) { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

let state = loadState();
let idx = 0;

function render() {
  const row = DATA[idx];
  document.getElementById("poster-img").src = row.img;
  document.getElementById("poster-img").alt = row.title;
  document.getElementById("catalog-title-text").textContent = row.title;
  document.getElementById("meta-text").textContent = `id ${row.id}`;

  const v = state[row.id];
  document.querySelectorAll("button.verdict").forEach(b => {
    b.classList.toggle("active", b.dataset.v === v);
  });

  const reviewedCount = DATA.filter(r => state[r.id]).length;
  document.getElementById("counter").textContent =
    `${idx + 1} / ${DATA.length}   (${reviewedCount} reviewed so far)`;
  document.getElementById("progress-fill").style.width = (reviewedCount / DATA.length * 100) + "%";
  document.getElementById("prev-btn").disabled = idx === 0;
  document.getElementById("next-btn").disabled = idx === DATA.length - 1;
  document.getElementById("export-hint").textContent =
    reviewedCount < DATA.length ? `${DATA.length - reviewedCount} left before export is complete` : "all reviewed -- ready to export";
}

function setVerdict(v) {
  const row = DATA[idx];
  state[row.id] = v;
  saveState(state);
  render();
  if (idx < DATA.length - 1) setTimeout(() => go(1), 150);
}
function go(delta) {
  idx = Math.max(0, Math.min(DATA.length - 1, idx + delta));
  render();
}
function jumpTo() {
  const v = parseInt(document.getElementById("jump-input").value, 10);
  if (!v) return;
  idx = Math.max(0, Math.min(DATA.length - 1, v - 1));
  render();
}
document.addEventListener("keydown", (e) => {
  if (document.activeElement.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
});

function parseCSVLine(line) {
  const out = [];
  let cur = "", inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') { inQuotes = false; }
      else { cur += c; }
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { out.push(cur); cur = ""; }
      else cur += c;
    }
  }
  out.push(cur);
  return out;
}

function importCSV(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const lines = reader.result.split(/\\r?\\n/).filter(l => l.length > 0);
    if (!lines.length) return;
    const header = parseCSVLine(lines[0]);
    const idCol = header.indexOf("id");
    const vCol = header.indexOf("human_verdict");
    if (idCol === -1) { alert("This doesn't look like an exported CSV (missing id column)."); return; }
    let imported = 0;
    for (let i = 1; i < lines.length; i++) {
      const cols = parseCSVLine(lines[i]);
      const id = cols[idCol];
      if (!id) continue;
      if (vCol !== -1 && cols[vCol]) { state[id] = cols[vCol]; imported++; }
    }
    saveState(state);
    const firstUnreviewed = DATA.findIndex(row => !state[row.id]);
    idx = firstUnreviewed === -1 ? 0 : firstUnreviewed;
    render();
    alert(`Imported ${imported} row(s). Jumped to ${firstUnreviewed === -1 ? "the first poster" : "#" + (idx + 1)}.`);
  };
  reader.readAsText(file);
}

function exportCSV() {
  const header = ["id", "title", "poster_path", "human_verdict"];
  const lines = [header.join(",")];
  for (const row of DATA) {
    const v = state[row.id] || "";
    const esc = (val) => '"' + String(val || "").replace(/"/g, '""') + '"';
    lines.push([row.id, esc(row.title), row.poster_path, v].join(","));
  }
  const blob = new Blob([lines.join("\\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "__SIGNAL___reconciliation_human_review.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

render();
</script>
</body>
</html>
"""


def load_id_keyed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def select_sample(signal: str, n: int) -> list[dict]:
    spec = ENGINES[signal]
    base = {r["id"]: r for r in csv.DictReader(SAMPLE_INPUT.open(newline="", encoding="utf-8"))}

    per_engine: dict[str, dict[str, tuple[bool, float] | None]] = {}
    available = []
    for name, path, verdict_fn in spec["sources"]:
        table = load_id_keyed(path)
        if not table:
            print(f"  {name}: no data at {path} -- skipping (run its script first, or run this later)")
            continue
        overlap = len(set(table) & set(base))
        if overlap == 0:
            print(f"  {name}: {path} has {len(table)} rows, but ZERO ids overlap with --in's "
                  f"{len(base)} -- skipping. Likely a different sample than --in (e.g. an older "
                  f"spot-check run); re-run that script against the same --in to align it.")
            continue
        available.append(name)
        per_engine[name] = {pid: verdict_fn(row) for pid, row in table.items()}
        print(f"  {name}: {overlap}/{len(base)} ids overlap with --in")

    if not available:
        raise SystemExit(f"no engine data usable for --signal {signal} against {SAMPLE_INPUT} -- "
                          f"run at least one of {[str(s[1]) for s in spec['sources']]} against "
                          f"that same --in file first")
    print(f"engines with data: {available}")

    disagree, positive, negative = [], [], []
    for pid in base:
        verdicts = [per_engine[e].get(pid) for e in available if per_engine[e].get(pid) is not None]
        if not verdicts:
            continue
        bools = [v[0] for v in verdicts]
        if len(set(bools)) > 1:
            disagree.append(pid)
        elif all(bools):
            positive.append(pid)
        else:
            negative.append(pid)

    rng = random.Random(SEED)
    rng.shuffle(positive)
    rng.shuffle(negative)

    chosen = list(disagree)
    remaining = max(0, n - len(chosen))
    chosen += positive[: remaining // 2]
    chosen += negative[: remaining - remaining // 2]
    chosen = chosen[:n]
    rng.shuffle(chosen)

    print(f"selected {len(chosen)}: {len(disagree)} disagreements (all included, "
          f"{min(len(disagree), n)} kept), {len(positive[:remaining//2])} agreed-positive anchors, "
          f"{len(negative[:remaining-remaining//2])} agreed-negative anchors")

    return [{"id": pid, "title": base[pid]["title"], "poster_path": base[pid]["poster_path"]} for pid in chosen if pid in base]


def fetch_poster_b64(session: requests.Session, poster_path: str) -> str:
    cache_file = POSTER_CACHE / (poster_path.lstrip("/").replace("/", "_"))
    if cache_file.exists():
        content = cache_file.read_bytes()
    else:
        resp = session.get(f"{IMG_BASE}{poster_path}", timeout=20)
        resp.raise_for_status()
        content = resp.content
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(content)
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def fetch_one(row: dict) -> tuple[str, str]:
    session = requests.Session()
    try:
        return row["id"], fetch_poster_b64(session, row["poster_path"])
    except Exception as e:
        print(f"  {row['id']}: poster fetch failed ({e}) -- leaving img blank", file=sys.stderr)
        return row["id"], ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=list(ENGINES), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = select_sample(args.signal, args.n)
    if not rows:
        print("nothing to review")
        return

    imgs: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch_one, row) for row in rows]
        for fut in as_completed(futures):
            id_, img = fut.result()
            imgs[id_] = img
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"{done}/{len(rows)} posters embedded")

    n_failed = 0
    for row in rows:
        row["img"] = imgs.get(row["id"], "")
        if not row["img"]:
            n_failed += 1

    out_path = Path(args.out) if args.out else ROOT / "data" / "qa" / f"{args.signal}_reconciliation_review.html"
    html = (HTML_TEMPLATE
            .replace("__DATA_JSON__", json.dumps(rows))
            .replace("__SIGNAL__", args.signal)
            .replace("__QUESTION__", ENGINES[args.signal]["question"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MB, {len(rows)} posters, {n_failed} failed to fetch)")


if __name__ == "__main__":
    main()
