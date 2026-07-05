#!/usr/bin/env python
"""Regenerate the 60-page structural-quality regression baseline.

For each document in ``tests.oc_batch_compat.BATCH_PAGES`` this parses it once,
runs ``validate_export`` (spec §6), and records the **deterministic per-page**
``page_metrics`` for the selected pages plus a per-doc ``::_doc`` entry
(``page_count`` + ``relationship_validity``). Output::

    tests/fixtures/oc_batch_baseline.json

The committed baseline is what ``tests/test_oc_batch_regression.py`` floors
against (counts ceiled, fractions/tightness floored). Parsing is parallel across
documents because the batch includes several 250-340pp prospectus bodies.

Usage::

    python scripts/build_oc_batch_fixtures.py [--jobs 8]
"""

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import traceback
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.oc_batch_compat import BATCH_PAGES, page_key, page_metrics  # noqa: E402
from warp_ingest.ingestor.opencontracts_exporter import (  # noqa: E402
    ExportValidationError,
    validate_export,
)
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

BASELINE = ROOT / "tests" / "fixtures" / "oc_batch_baseline.json"


def _worker(entry):
    relpath, pages, is_slow, doc_class, note = entry
    path = ROOT / relpath
    if not path.exists():
        return {"relpath": relpath, "missing": True}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            export = parse_to_opencontracts(str(path))
    except Exception:
        return {"relpath": relpath, "error": traceback.format_exc()}
    try:
        validate_export(export)
        rel_valid = True
    except ExportValidationError:
        rel_valid = False
    out = {page_key(relpath, p): page_metrics(export, p) for p in pages}
    out[f"{relpath}::_doc"] = {
        "page_count": export["page_count"],
        "relationship_validity": rel_valid,
    }
    return {"relpath": relpath, "entries": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 4)))
    args = ap.parse_args()

    baseline = {}
    with Pool(processes=args.jobs) as pool:
        for res in pool.imap_unordered(_worker, BATCH_PAGES):
            rp = res["relpath"]
            if res.get("missing"):
                print(f"  SKIP (missing): {rp}")
                continue
            if res.get("error"):
                print(f"  !! FAILED: {rp}\n{res['error']}", file=sys.stderr)
                continue
            baseline.update(res["entries"])
            print(f"  exported: {rp}")

    BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    n_pages = sum(1 for k in baseline if not k.endswith("::_doc"))
    print(f"\nwrote baseline for {n_pages} pages -> {BASELINE}")


if __name__ == "__main__":
    main()
