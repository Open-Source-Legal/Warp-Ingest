#!/usr/bin/env python
"""Regenerate the OpenContractDocExport regression baseline (and optional dumps).

For each fixture PDF in ``tests/oc_compat.FIXTURE_DOCS`` this parses the document
to an ``OpenContractDocExport``, runs ``validate_export`` (spec §6 invariants),
records the regression metrics, and writes them to::

    tests/fixtures/oc_export_baseline.json

The committed baseline is what ``tests/test_opencontracts_regression.py`` floors
against. Pass ``--dump-dir DIR`` to also write the full export JSON per document
(useful for manual inspection; not committed).

Usage::

    python scripts/build_opencontracts_fixtures.py
    python scripts/build_opencontracts_fixtures.py --dump-dir /tmp/oc_exports
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.oc_compat import (  # noqa: E402
    FIXTURE_DOCS,
    FIXTURE_PARSE_OPTIONS,
    export_metrics,
)
from warp_ingest.ingestor import pdf_ingestor  # noqa: E402
from warp_ingest.ingestor.opencontracts_exporter import validate_export  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
BASELINE = FIXTURES / "oc_export_baseline.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", help="also write full export JSON per doc")
    args = ap.parse_args()

    dump_dir = pathlib.Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    baseline = {}
    for name, is_slow in FIXTURE_DOCS:
        path = FIXTURES / name
        if not path.exists():
            print(f"  SKIP (missing): {name}")
            continue
        print(f"  exporting{' [slow]' if is_slow else ''}: {name}")
        export = pdf_ingestor.parse_to_opencontracts(
            str(path), parse_options=FIXTURE_PARSE_OPTIONS.get(name)
        )
        validate_export(export)
        baseline[name] = export_metrics(export)
        if dump_dir:
            out = dump_dir / (pathlib.Path(name).stem + ".json")
            out.write_text(json.dumps(export, indent=1))
            print(f"    dumped: {out}")

    BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote baseline for {len(baseline)} docs -> {BASELINE}")


if __name__ == "__main__":
    main()
