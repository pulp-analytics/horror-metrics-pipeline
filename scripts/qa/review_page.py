"""Shared blind human-review HTML page -- same pattern as
poster-corpus-validation's scripts/qa/build_*_review_page.py.

The page shows a poster and a plain question. It never embeds CLIP/Nova
scores or verdicts (those stay in the metric/QA CSVs). Reviewer labels
export to CSV via a real download; progress autosaves in localStorage.
Generated HTML is a build artifact (gitignored): large, with base64 images.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from utils.posters import fetch_poster_file

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
 max-width: 780px; margin: 0 auto; padding: 24px 20px 180px; background: #fafafa; color: #1a1a1a; }
h1 { font-size: 18px; margin-bottom: 4px; }
.sub { color: #666; font-size: 13px; margin-bottom: 20px; }
.progress { position: sticky; top: 0; background: #fafafa; padding: 10px 0;
 border-bottom: 1px solid #ddd; margin-bottom: 16px; z-index: 10; }
.progress-bar { height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin-top: 6px; }
.progress-fill { height: 100%; background: #2f7d5c; transition: width .2s; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 20px; }
.poster-wrap { text-align: center; margin-bottom: 16px; }
img { max-width: 100%; max-height: 48vh; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.15); background: #eee; }
.question { font-size: 18px; font-weight: 600; text-align: center; margin: 8px 0; }
.meta { text-align: center; font-size: 13px; color: #888; margin-bottom: 12px; }
textarea { width: 100%; box-sizing: border-box; padding: 8px; border-radius: 6px; border: 1px solid #ccc;
 font-family: inherit; font-size: 13px; resize: vertical; min-height: 44px; }
.action-bar { position: fixed; left: 0; right: 0; bottom: 0; background: #fff;
 border-top: 1px solid #ddd; box-shadow: 0 -4px 12px rgba(0,0,0,.08);
 padding: 12px 20px calc(12px + env(safe-area-inset-bottom)); z-index: 20; }
.action-bar-inner { max-width: 740px; margin: 0 auto; }
.buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
button.verdict { flex: 1; min-width: 88px; padding: 10px 8px; font-size: 13px; border-radius: 8px;
 border: 2px solid #ccc; background: #fff; color: #1a1a1a; cursor: pointer; font-weight: 600; }
button.verdict:hover { border-color: #999; }
button.verdict.active { background: #e7f2ec; border-color: #3f7d5c; color: #316447; }
.nav { display: flex; justify-content: space-between; align-items: center; }
.nav button.plain { padding: 8px 16px; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
.nav button.plain:disabled { opacity: .4; }
.counter { font-size: 13px; color: #666; }
.footer { margin-top: 24px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.export, .import-label { padding: 10px 18px; border-radius: 6px; border: none; background: #2f5f8a;
 color: #fff; cursor: pointer; font-weight: 600; display: inline-block; }
.hint { font-size: 12px; color: #999; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="sub">__BLURB__</p>
<div class="progress">
  <div class="counter" id="counter"></div>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
</div>
<div class="card">
  <div class="poster-wrap"><img id="poster-img" alt=""/></div>
  <div class="question" id="question"></div>
  <div class="meta" id="meta"></div>
  <textarea id="note" placeholder="optional note" oninput="setNote(this.value)"></textarea>
</div>
<div class="footer">
  <span class="hint" id="export-hint"></span>
  <span>
    <label class="import-label">Import progress<input type="file" accept=".csv" hidden onchange="importCSV(this.files[0])"/></label>
    <button class="export" onclick="exportCSV()">Export CSV</button>
  </span>
</div>
<div class="action-bar"><div class="action-bar-inner">
  <div class="buttons" id="buttons"></div>
  <div class="nav">
    <button class="plain" id="prev-btn" onclick="go(-1)">← Prev</button>
    <span class="hint">1/2/3 = first three verdicts, ← → navigate. No model scores on this page.</span>
    <button class="plain" id="next-btn" onclick="go(1)">Next →</button>
  </div>
</div></div>
<script>
const DATA = __DATA_JSON__;
const VERDICTS = __VERDICTS_JSON__;
const STORAGE_KEY = __STORAGE_KEY__;
const EXPORT_NAME = __EXPORT_NAME__;
function loadState() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch (e) { return {}; } }
function saveState(state) { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
let state = loadState();
let idx = 0;
function keyOf(row) { return row.key; }
function render() {
  const row = DATA[idx];
  document.getElementById("poster-img").src = row.img || "";
  document.getElementById("question").textContent = row.question;
  document.getElementById("meta").textContent = (row.title || "") + (row.year ? " · " + row.year : "") + " · id " + row.id;
  const s = state[keyOf(row)] || {};
  document.getElementById("note").value = s.note || "";
  const box = document.getElementById("buttons");
  box.innerHTML = "";
  VERDICTS.forEach((v, i) => {
    const b = document.createElement("button");
    b.className = "verdict" + (s.verdict === v.v ? " active" : "");
    b.textContent = v.label;
    b.onclick = () => setVerdict(v.v);
    box.appendChild(b);
  });
  const reviewed = Object.keys(state).filter(k => state[k].verdict).length;
  document.getElementById("counter").textContent = (idx + 1) + " / " + DATA.length + " (" + reviewed + " reviewed)";
  document.getElementById("progress-fill").style.width = (reviewed / DATA.length * 100) + "%";
  document.getElementById("prev-btn").disabled = idx === 0;
  document.getElementById("next-btn").disabled = idx === DATA.length - 1;
  document.getElementById("export-hint").textContent = reviewed < DATA.length
    ? (DATA.length - reviewed) + " left" : "all reviewed — ready to export";
}
function setVerdict(v) {
  const k = keyOf(DATA[idx]);
  state[k] = state[k] || {};
  state[k].verdict = v;
  saveState(state);
  render();
  if (idx < DATA.length - 1) setTimeout(() => go(1), 120);
}
function setNote(v) {
  const k = keyOf(DATA[idx]);
  state[k] = state[k] || {};
  state[k].note = v;
  saveState(state);
}
function go(delta) {
  idx = Math.max(0, Math.min(DATA.length - 1, idx + delta));
  render();
}
document.addEventListener("keydown", (e) => {
  if (document.activeElement.tagName === "TEXTAREA") return;
  if (e.key === "1" && VERDICTS[0]) setVerdict(VERDICTS[0].v);
  else if (e.key === "2" && VERDICTS[1]) setVerdict(VERDICTS[1].v);
  else if (e.key === "3" && VERDICTS[2]) setVerdict(VERDICTS[2].v);
  else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
});
function parseCSVLine(line) {
  const out = []; let cur = "", inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else cur += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") { out.push(cur); cur = ""; }
    else cur += c;
  }
  out.push(cur); return out;
}
function importCSV(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const lines = reader.result.split(/\\r?\\n/).filter(l => l.length);
    const header = parseCSVLine(lines[0]);
    const keyCol = header.indexOf("key");
    const verdictCol = header.indexOf("human_verdict");
    const noteCol = header.indexOf("human_note");
    if (keyCol < 0 || verdictCol < 0) { alert("CSV needs key,human_verdict"); return; }
    for (let i = 1; i < lines.length; i++) {
      const cols = parseCSVLine(lines[i]);
      const k = cols[keyCol]; if (!k) continue;
      state[k] = { verdict: cols[verdictCol] || "", note: noteCol >= 0 ? (cols[noteCol] || "") : "" };
    }
    saveState(state); render();
  };
  reader.readAsText(file);
}
function exportCSV() {
  const extra = DATA[0] ? Object.keys(DATA[0]).filter(k => k !== "img" && k !== "question") : [];
  const header = extra.concat(["human_verdict", "human_note"]);
  const lines = [header.join(",")];
  const esc = (v) => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  for (const row of DATA) {
    const s = state[keyOf(row)] || {};
    lines.push(extra.map(k => esc(row[k])).concat([esc(s.verdict), esc(s.note)]).join(","));
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n")], { type: "text/csv" }));
  a.download = EXPORT_NAME;
  document.body.appendChild(a); a.click(); a.remove();
}
render();
</script>
</body>
</html>
"""


def poster_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def fetch_to_cache(session: requests.Session, poster_path: str, dest: Path,
                   s3_bucket: str = "", s3_prefix: str = "") -> bool:
    return fetch_poster_file(session, poster_path, dest, s3_bucket, s3_prefix)


def write_review_html(
    out: Path,
    *,
    title: str,
    blurb: str,
    storage_key: str,
    export_name: str,
    verdicts: list[dict],
    rows: list[dict],
) -> None:
    """rows must include id, key, question, img. Do not put model scores in rows."""
    for banned in ("clip_label", "clip_score", "clip_axis", "clip_register",
                   "nova_label", "nova_register", "verdict", "score"):
        for row in rows:
            if banned in row:
                raise ValueError(f"blind page cannot embed {banned!r}")
    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__BLURB__", blurb)
        .replace("__DATA_JSON__", json.dumps(rows))
        .replace("__VERDICTS_JSON__", json.dumps(verdicts))
        .replace("__STORAGE_KEY__", json.dumps(storage_key))
        .replace("__EXPORT_NAME__", json.dumps(export_name))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
