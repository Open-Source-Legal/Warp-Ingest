#!/usr/bin/env python
"""Freeze the heterogeneous-100 structural-correctness regression baseline.

Parses each hetero fixture with Warp (capped at ``--jobs``; >4 OOMs), scores it
against the golden (``tests/oc_hetero100_compat``), and writes the per-page
metrics to ``tests/fixtures/hetero100_baseline.json``. The committed regression
test (``tests/test_oc_hetero100_regression.py``) recomputes these live and floors
the agreement metrics / ceils the smells against this baseline — improvements
pass, regressions fail.

    python scripts/build_hetero100_baseline.py --jobs 2
"""

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests import oc_hetero100_compat as C  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "hetero100_baseline.json"


def _one(task):
    rel, regions = task
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            export = parse_to_opencontracts(str(REPO / rel))
        return rel, C.page_metrics(export, regions), None
    except Exception:
        return rel, None, traceback.format_exc().splitlines()[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)
    jobs = max(1, min(args.jobs, 4))

    golden = C.load_golden()
    tasks = [(rel, golden[rel]["regions"]) for rel in sorted(golden)]
    baseline, errors = {}, []
    with Pool(processes=jobs) as pool:
        for rel, metrics, err in pool.imap_unordered(_one, tasks):
            if err:
                errors.append((rel, err))
                print(f"  !! {rel}: {err}", file=sys.stderr)
            else:
                baseline[rel] = metrics
            if (len(baseline) + len(errors)) % 20 == 0:
                print(f"  {len(baseline) + len(errors)}/{len(tasks)}")

    OUT.write_text(json.dumps(baseline, indent=1, sort_keys=True))
    n = len(baseline)

    def mean(k):
        vals = [m[k] for m in baseline.values() if m.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print(f"\nwrote {OUT.relative_to(REPO)}: {n} pages, {len(errors)} errors")
    print(
        "means:",
        {
            k: mean(k)
            for k in (
                "struct_macro_f1",
                "heading_f1",
                "list_f1",
                "gold_coverage",
                "body_as_tablerow_frac",
                "head_ancestor_agreement",
            )
        },
    )
    return baseline


if __name__ == "__main__":
    main()
