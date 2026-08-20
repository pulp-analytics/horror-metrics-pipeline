"""Shared resumability helper: if --out already has rows for some ids
(e.g. a previous run got interrupted), skip those and append the rest
instead of starting over and re-downloading/re-analyzing posters already
done. Same helper (and same shard_rows() convention) as the sibling
poster-corpus-validation repo -- kept consistent on purpose.

`open_for_append` exclusive-locks writes so two processes on the same
--out cannot duplicate an id. That happened on the 99-poster Grounding
DINO sample: a 900s timeout plus the original still running in
background both appended, and 25 ids landed twice. `load_done_ids` at
startup does not prevent that -- both processes snapshot the same "not
done" set, both infer, both append. writerow() re-checks under the lock
and no-ops if the id is already on disk. Inference may still run twice
(wasted work); the CSV stays unique.
"""
from __future__ import annotations

import csv
import fcntl
import io
import os
from contextlib import contextmanager
from pathlib import Path


def load_done_ids(path: Path, id_col: str = "id") -> set[str]:
    """ids already present in an existing output CSV, or empty set if none."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        try:
            return {row[id_col] for row in csv.DictReader(f) if row.get(id_col)}
        except (KeyError, csv.Error):
            return set()


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


@contextmanager
def _exclusive_lock(path: Path):
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class _ExclusiveRowWriter:
    """DictWriter stand-in: writerow() is serialized and skips duplicate ids."""

    def __init__(self, path: Path, fieldnames: list[str], fh, writer,
                 id_col: str = "id"):
        self._path = path
        self._fieldnames = fieldnames
        self._fh = fh
        self._writer = writer
        self._id_col = id_col
        self._done = load_done_ids(path, id_col)
        self._pos = path.stat().st_size if path.exists() else 0

    def writerow(self, row: dict) -> bool:
        rid = str(row.get(self._id_col) or "").strip()
        with _exclusive_lock(self._path):
            self._ingest_new_ids()
            if rid and rid in self._done:
                return False
            self._writer.writerow(row)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            if rid:
                self._done.add(rid)
            self._pos = self._path.stat().st_size
            return True

    def _ingest_new_ids(self) -> None:
        """Read rows another process appended since our last check.

        Called under the exclusive lock, so those rows are complete.
        """
        if not self._path.exists():
            return
        size = self._path.stat().st_size
        if size <= self._pos:
            return
        with self._path.open(newline="", encoding="utf-8", errors="replace") as f:
            f.seek(self._pos)
            chunk = f.read()
        if not chunk.strip():
            self._pos = size
            return
        reader = csv.DictReader(io.StringIO(chunk), fieldnames=self._fieldnames)
        for row in reader:
            i = (row.get(self._id_col) or "").strip()
            if i:
                self._done.add(i)
        self._pos = size


def open_for_append(path: Path, fieldnames: list[str], id_col: str = "id") -> tuple:
    """Returns (file_handle, writer). Writes the header only if the
    file didn't already exist -- so re-running after an interruption
    appends cleanly instead of duplicating a header mid-file.

    Treats a 0-byte file as "no header yet", not just a missing file: a
    process killed (OOM, Batch/Fargate stop, spot interruption) right after
    creating the file but before its header line reached disk leaves an
    empty file behind. Without this check, the next run sees the file
    "already exists", opens in append mode, and never writes a header --
    producing a shard whose first data row silently gets read as the
    header downstream (real prodtest-3000 failure, 2026-08-19: 7 of 10 IQA
    shards ended up headerless this way after a retried Batch array job).

    writer.writerow() exclusive-locks the file and skips an id that is
    already present, so two processes sharing --out (timeout retry vs.
    the original still running) cannot duplicate a row. Callers do not
    need to change: the skip is inside writerow, not in load_done_ids.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(path):
        needs_header = not path.exists() or path.stat().st_size == 0
        f = path.open("a" if path.exists() else "w", newline="", encoding="utf-8")
        inner = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            inner.writeheader()
            f.flush()
            os.fsync(f.fileno())
        w = _ExclusiveRowWriter(path, fieldnames, f, inner, id_col=id_col)
    return f, w


def shard_rows(rows: list[dict], shard_index: int, shard_count: int) -> list[dict]:
    """Deterministic partition of --in's rows by position, for running N
    copies of a script in parallel over disjoint slices of the same file
    (e.g. one per AWS Batch array job index -- see the sibling
    poster-analysis-infrastructure repo). shard_count=1 (the default)
    returns every row unchanged, so this is a no-op unless you opt in."""
    if shard_count <= 1:
        return rows
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"shard_index {shard_index} out of range for shard_count {shard_count}")
    return rows[shard_index::shard_count]


def write_csv_rows(out_path: Path | str, rows: list[dict]) -> None:
    """Writes a full --out CSV from an in-memory list of dicts (as opposed
    to open_for_append's incremental per-row writes), inferring fieldnames
    from the first row. No-op if rows is empty -- callers that always want
    a file on disk should check for that themselves. Same helper as the
    sibling poster-corpus-validation repo."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
