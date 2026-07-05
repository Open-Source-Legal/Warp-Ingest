"""Heterogeneous-100 structural-correctness regression.

For every page in the ParseBench-derived golden, parse the fixture live with
Warp, score it against the human-verified golden, and assert the agreement
metrics have not dropped (floors) and the structural smells have not grown
(ceils) versus the committed baseline. Improvements pass; regressions fail.

This is one of two golden-answer-set suites (legal counterpart:
``test_oc_legal100_regression``). All hetero fixtures are single-page, so the
whole suite runs by default (none are ``@slow``). Regenerate the baseline with
``scripts/build_hetero100_baseline.py`` after a deliberate, reviewed change.
"""

import contextlib
import io
import os

import pytest

from tests import oc_golden_eval as G
from tests import oc_hetero100_compat as C
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GOLDEN = C.load_golden()
_BASELINE = C.load_baseline() if os.path.exists(C.BASELINE) else {}


@pytest.mark.skipif(not _BASELINE, reason="hetero100 baseline not built yet")
@pytest.mark.parametrize("relpath", sorted(_GOLDEN), ids=lambda r: os.path.basename(r))
def test_hetero_page_structural_quality(relpath):
    base = _BASELINE.get(relpath)
    if base is None:
        pytest.skip(f"no baseline entry for {relpath}")
    with contextlib.redirect_stdout(io.StringIO()):
        export = parse_to_opencontracts(os.path.join(REPO, relpath))
    live = C.page_metrics(export, _GOLDEN[relpath]["regions"])
    problems = G.page_regressions(live, base)
    assert not problems, f"{relpath} regressed:\n  " + "\n  ".join(problems)
