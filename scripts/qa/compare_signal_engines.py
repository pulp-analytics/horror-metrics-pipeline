#!/usr/bin/env python3
"""Scores every available engine for a presence signal (--signal animal or
weapon) against the blind human review CSV exported from
build_signal_reconciliation_review_page.py, and reports whether an
agreement-based combination beats any single engine alone -- the same
question already answered once for creature/weapon (OWLv2 alone: 58-67%
false positives; OWLv2+DINO agreement was the trustworthy signal, see
docs/RESULTS.md) applied here to whichever engines have real data,
including 25_rekognition_enrich.py's rek_animal/rek_weapon once that's
been run against real AWS access.

Missing engines are skipped, not treated as errors -- this is meant to be
run again as each engine's data becomes available (e.g. once AWS access
returns and 25_rekognition_enrich.py has a real run), not just once with
everything already in place.

  python3 scripts/qa/compare_signal_engines.py --signal animal --human data/qa/animal_reconciliation_human_review.csv
  python3 scripts/qa/compare_signal_engines.py --signal weapon --human data/qa/weapon_reconciliation_human_review.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_signal_reconciliation_review_page import ENGINES, load_id_keyed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_human(path: Path) -> dict[str, bool]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("human_verdict") or "").strip()
            if v == "si":
                out[r["id"]] = True
            elif v == "no":
                out[r["id"]] = False
            # no_seguro / blank: excluded, not scoreable
    return out


def prf(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return acc, prec, rec, n


def score(verdicts: dict[str, bool], human: dict[str, bool]) -> tuple[float, float, float, int]:
    tp = fp = fn = tn = 0
    for pid, h in human.items():
        v = verdicts.get(pid)
        if v is None:
            continue
        if h and v:
            tp += 1
        elif h and not v:
            fn += 1
        elif not h and v:
            fp += 1
        else:
            tn += 1
    acc, prec, rec, n = prf(tp, fp, fn, tn)
    return acc, prec, rec, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=list(ENGINES), required=True)
    ap.add_argument("--human", required=True)
    args = ap.parse_args()

    human = load_human(Path(args.human))
    if not human:
        raise SystemExit(f"no scoreable rows (si/no) in {args.human} -- review isn't done yet?")
    print(f"human ground truth: {len(human)} scoreable rows "
          f"({sum(human.values())} si, {len(human) - sum(human.values())} no)")

    spec = ENGINES[args.signal]
    engine_verdicts: dict[str, dict[str, bool]] = {}
    for name, path, verdict_fn in spec["sources"]:
        table = load_id_keyed(path)
        if not table:
            print(f"  {name}: no data at {path} -- skip (not run yet)")
            continue
        parsed = {pid: verdict_fn(row) for pid, row in table.items()}
        engine_verdicts[name] = {pid: v[0] for pid, v in parsed.items() if v is not None}

    if not engine_verdicts:
        raise SystemExit("no engine data available at all -- nothing to score")

    print(f"\n{'engine':20}{'n':>6}{'accuracy':>10}{'precision':>11}{'recall':>9}")
    for name, verdicts in engine_verdicts.items():
        acc, prec, rec, n = score(verdicts, human)
        print(f"{name:20}{n:6}{acc:9.1%} {prec:10.1%} {rec:8.1%}")

    engines = list(engine_verdicts)
    if len(engines) >= 2:
        print(f"\n{'combination (agreement rule)':32}{'n':>6}{'accuracy':>10}{'precision':>11}{'recall':>9}")
        for k in range(2, len(engines) + 1):
            for combo in combinations(engines, k):
                # ANY-agree: flag if any engine in the combo says yes -- only ids all combo
                # engines actually scored are counted (fair N across combos)
                ids = set.intersection(*(set(engine_verdicts[e]) for e in combo))
                any_agree = {pid: any(engine_verdicts[e][pid] for e in combo) for pid in ids}
                all_agree = {pid: all(engine_verdicts[e][pid] for e in combo) for pid in ids}
                label = "+".join(combo)
                acc, prec, rec, n = score(any_agree, human)
                print(f"{label + ' (any)':32}{n:6}{acc:9.1%} {prec:10.1%} {rec:8.1%}")
                acc, prec, rec, n = score(all_agree, human)
                print(f"{label + ' (all)':32}{n:6}{acc:9.1%} {prec:10.1%} {rec:8.1%}")
    else:
        print(f"\nonly {len(engines)} engine has data -- run the others (or wait for AWS access "
              f"for 25_rekognition_enrich.py) to compare agreement combinations")


if __name__ == "__main__":
    main()
