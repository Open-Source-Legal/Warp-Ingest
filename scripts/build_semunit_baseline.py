#!/usr/bin/env python
"""Freeze the Semantic-Unit regression baseline.

Parses each manifested doc once (semantic_units on), scores every page against
the numbering-derived golden, and writes per-page metrics to
``tests/fixtures/semunit_baseline.json`` (floored/ceiled by
``oc_golden_eval.unit_regressions``).

    python scripts/build_semunit_baseline.py
"""

import contextlib
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests import oc_semunit_compat as C  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402


def main():
    golden = C.load_golden()
    baseline, seen = {}, {}
    for relpath, page, _slow in C.manifest_pairs():
        key = C.page_key(relpath, page)
        if key not in golden:
            continue
        if relpath not in seen:
            with contextlib.redirect_stdout(io.StringIO()):
                seen[relpath] = parse_to_opencontracts(
                    os.path.join(REPO, relpath),
                    parse_options={"semantic_units": True},
                )
        baseline[key] = C.page_metrics(seen[relpath], page, golden[key]["units"])
    with open(C.BASELINE, "w") as fh:
        json.dump(baseline, fh, indent=1, sort_keys=True)
    print("wrote", C.BASELINE, len(baseline), "pages")


if __name__ == "__main__":
    main()
