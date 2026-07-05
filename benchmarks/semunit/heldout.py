#!/usr/bin/env python
"""HELD-OUT generalization check for the Semantic-Unit layer.

Scores the unit layer on documents that are NEVER tuned against — CUAD-class
contracts, credit agreements, and S-1 exhibits disjoint from the 204-page
clause golden — using the numbering-derived oracle (the same text-intrinsic
unit derivation as ``scripts/build_semunit_golden.py``).

GOVERNANCE (the point of this file):
  1. This suite is SCORED, never tuned. No rule change may be justified by
     improving these numbers; they exist to detect generalization damage
     while rules are hill-climbed on ``golden_204_clause.json``.  If a rule
     improves the 204-bench but regresses here, the rule is overfit —
     fix the rule, don't touch the baseline.
  2. ``heldout_baseline.json`` may only be re-baselined (``--rebaseline``) in
     a commit whose message explains the deliberate behavior change, and
     never merely to absorb a regression.
  3. When this set influences a fix (as any failure investigation must), the
     docs it exposed should be named in the commit — the next auditor can
     then judge how much signal the set has left.  If it ever gets tuned
     against in earnest, retire it and cut a fresh one (unused S-1 bodies
     remain in ``tests/fixtures/s1/``).

Known oracle limits (do NOT chase these to zero): the oracle groups
strictly by leading enumerators / ordinal headings / short ALL-CAPS lines,
so it cannot see redaction-split sibling clauses (eikon ex-10.11 legitimately
scores worse here than a human golden would judge), letter/salutation
granularity, or signature-stack grouping.  It is a drift detector, not truth.

Usage:
    python benchmarks/semunit/heldout.py               # score + compare
    python benchmarks/semunit/heldout.py --rebaseline  # rewrite baseline
"""

import contextlib
import importlib.util
import io
import json
import multiprocessing as mp
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "heldout_manifest.json"
BASELINE = HERE / "heldout_baseline.json"

# floors must not drop / ceilings must not rise by more than TOL
FLOOR_KEYS = ("unit_coverage", "mean_unit_iou")
CEIL_KEYS = (
    "fragmentation_frac",
    "merge_frac",
    "spurious_unit_frac",
    "windowdiff",
    "pk",
)
TOL = 0.02


def _oracle():
    spec = importlib.util.spec_from_file_location(
        "semunit_golden_builder", REPO / "scripts" / "build_semunit_golden.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_doc(relpath):
    from tests.oc_golden_eval import score_units
    from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

    oracle = _oracle()
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        export = parse_to_opencontracts(
            str(REPO / relpath), parse_options={"semantic_units": True}
        )
    for page in range(export.get("page_count", 0)):
        gold = oracle._gold_units_for_page(export, page)
        if not gold:
            continue
        m = score_units(export, page, gold)
        m["doc"] = relpath
        m["page"] = page
        rows.append(m)
    return rows


def aggregate(rows):
    keys = FLOOR_KEYS + CEIL_KEYS
    return {k: round(st.mean([r[k] for r in rows]), 4) for k in keys} | {
        "n_pages": len(rows)
    }


def main():
    docs = json.loads(MANIFEST.read_text())
    with mp.Pool(min(6, mp.cpu_count() - 2 or 1)) as pool:
        per_doc_rows = pool.map(score_doc, docs)
    all_rows = [r for rows in per_doc_rows for r in rows]
    result = {"overall": aggregate(all_rows)}
    for relpath, rows in zip(docs, per_doc_rows):
        if rows:
            result[Path(relpath).stem[:48]] = aggregate(rows)

    print(f"{'doc':<50} cov    iou    frag   merge  wd     pk")
    for name, agg in result.items():
        print(
            f"{name[:50]:<50} {agg['unit_coverage']:.3f}  {agg['mean_unit_iou']:.3f}  "
            f"{agg['fragmentation_frac']:.3f}  {agg['merge_frac']:.3f}  "
            f"{agg['windowdiff']:.3f}  {agg['pk']:.3f}"
        )

    if "--rebaseline" in sys.argv or not BASELINE.exists():
        BASELINE.write_text(json.dumps(result, indent=1, sort_keys=True))
        print(f"\nwrote baseline {BASELINE}")
        return 0

    base = json.loads(BASELINE.read_text())
    failures = []
    for name, b in base.items():
        live = result.get(name)
        if live is None:
            failures.append(f"{name}: missing from live run")
            continue
        for k in FLOOR_KEYS:
            if live[k] < b[k] - TOL:
                failures.append(f"{name}.{k}: {live[k]} < floor {b[k]} - {TOL}")
        for k in CEIL_KEYS:
            if live[k] > b[k] + TOL:
                failures.append(f"{name}.{k}: {live[k]} > ceil {b[k]} + {TOL}")
    if failures:
        print("\nHELD-OUT REGRESSIONS (fix the rule, do not re-baseline):")
        for f in failures:
            print("  ", f)
        return 1
    print("\nheld-out check PASSED (within tolerance of committed baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
