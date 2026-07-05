"""Per-page Semantic-Unit segmentation regression against a floored baseline.

Scores the live Semantic-Unit layer for each manifested page against the
numbering-derived golden and floors/ceils the metrics against
``tests/fixtures/semunit_baseline.json`` (improvements pass, regressions fail).
Also re-asserts ``validate_export`` on the augmented export (additivity guard).
"""

import contextlib
import io
import os

import pytest

import tests.oc_golden_eval as G
import tests.oc_semunit_compat as C
from warp_ingest.ingestor.opencontracts_exporter import validate_export
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLDEN = C.load_golden()
_BASELINE = C.load_baseline() if os.path.exists(C.BASELINE) else {}


@pytest.mark.skipif(not _BASELINE, reason="semunit baseline not built")
@pytest.mark.parametrize(
    "relpath,page",
    [(r, p) for r, p, _ in C.manifest_pairs()],
)
def test_semunit_page_does_not_regress(relpath, page):
    key = C.page_key(relpath, page)
    base = _BASELINE.get(key)
    if base is None:
        pytest.skip(f"no baseline for {key}")
    with contextlib.redirect_stdout(io.StringIO()):
        export = parse_to_opencontracts(
            os.path.join(REPO, relpath), parse_options={"semantic_units": True}
        )
    validate_export(export)  # additivity: still a valid export with the SU layer
    live = C.page_metrics(export, page, _GOLDEN[key]["units"])
    problems = G.unit_regressions(live, base)
    assert not problems, f"{key} regressed:\n  " + "\n  ".join(problems)
