"""Manifest + metric glue for the heterogeneous-100 structural-correctness suite.

The golden answer set (``tests/fixtures/hetero100_golden.json``) is derived from
ParseBench's **human-verified** ``layout.jsonl`` (see
``scripts/build_hetero100_fixtures.py``). Each entry is a single-page enterprise
PDF under ``tests/fixtures/hetero100/pages/`` with a list of gold regions. This
module loads the golden and scores a live Warp export against it via the shared
``oc_golden_eval`` core; the regression test floors the agreement metrics and
ceils the smells against ``tests/fixtures/hetero100_baseline.json``.

ParseBench has true reading order, so reading-order agreement is scored
(``HAVE_RO = True``).
"""

import json
import os

from tests import oc_golden_eval as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "tests", "fixtures", "hetero100_golden.json")
BASELINE = os.path.join(REPO, "tests", "fixtures", "hetero100_baseline.json")
HAVE_RO = True


def load_golden():
    with open(GOLDEN) as fh:
        return json.load(fh)


def load_baseline():
    with open(BASELINE) as fh:
        return json.load(fh)


def manifest():
    """``[(relpath, page, is_slow)]`` — one single-page fixture per entry."""
    golden = load_golden()
    return [(rel, 0, False) for rel in sorted(golden)]


def page_metrics(export, regions):
    """Golden-agreement metrics for a single-page hetero export."""
    return G.score_page(export, 0, regions, have_ro=HAVE_RO)
