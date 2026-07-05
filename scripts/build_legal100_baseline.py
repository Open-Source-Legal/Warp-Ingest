#!/usr/bin/env python
"""Freeze the legal-100 structural-correctness regression baseline.

Parses each distinct legal document once (capped at ``--jobs``; >4 OOMs), scores
every manifested page against the vision-adjudicated golden
(``tests/oc_legal100_compat``), and writes per-page metrics to
``tests/fixtures/legal100_baseline.json``. Reuses cached exports under
``audit_out/legal/exports`` when present (the eval runner / vision prep already
parsed them), so this is cheap to re-run.

    python scripts/build_legal100_baseline.py --jobs 2
"""

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests import oc_legal100_compat as C  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "legal100_baseline.json"
EXPORT_CACHE = REPO / "audit_out" / "legal" / "exports"


def _slug(relpath):
    return relpath.replace("/", "__").replace(" ", "_")


def _one(task):
    relpath, pages = task
    try:
        cache = EXPORT_CACHE / f"{_slug(relpath)}.json"
        if cache.exists():
            export = json.loads(cache.read_text())
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                export = parse_to_opencontracts(str(REPO / relpath))
        return relpath, pages, export, None
    except Exception:
        return relpath, pages, None, traceback.format_exc().splitlines()[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)
    jobs = max(1, min(args.jobs, 4))

    golden = C.load_golden()
    pages_by_doc = defaultdict(list)
    for relpath, page, _slow, _cls in C.manifest_pairs():
        pages_by_doc[relpath].append(page)
    tasks = [(rel, sorted(set(pgs))) for rel, pgs in sorted(pages_by_doc.items())]

    baseline, errors = {}, []
    with Pool(processes=jobs) as pool:
        for relpath, pages, export, err in pool.imap_unordered(_one, tasks):
            if err:
                errors.append((relpath, err))
                print(f"  !! {relpath}: {err}", file=sys.stderr)
                continue
            for page in pages:
                key = C.page_key(relpath, page)
                if key not in golden:
                    continue
                baseline[key] = C.page_metrics(export, page, golden[key]["regions"])
            print(f"  scored {relpath} ({len(pages)} pages)")

    OUT.write_text(json.dumps(baseline, indent=1, sort_keys=True))

    def mean(k):
        vals = [m[k] for m in baseline.values() if m.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print(
        f"\nwrote {OUT.relative_to(REPO)}: {len(baseline)} pages, {len(errors)} errors"
    )
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
                "parent_class_agreement",
            )
        },
    )
    return baseline


if __name__ == "__main__":
    main()
