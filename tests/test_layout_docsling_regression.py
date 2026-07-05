"""Cross-engine layout regression: Warp-Ingest vs the Docsling structure oracle.

For each study fixture we run Warp-Ingest's ``parse_to_opencontracts`` live and
floor its layout agreement with a committed, slimmed Docling oracle
(``tests/fixtures/docsling_targets/``) against
``tests/fixtures/docsling_layout_baseline.json``. Improvements pass; regressions
fail. Only Warp runs here — the Docling microservice is needed solely to
regenerate the oracle (``scripts/build_docsling_fixtures.py``), mirroring the
S-1 suite.

Docling is an oracle, not ground truth (it drops tables, under-labels numbered
legal headings, over-applies ``list_item``); see ``docs/docsling_layout_oracle.md``.
So agreement metrics are floored with a margin rather than required to be high,
and the one hard ceiling is ``overlong_heading_count`` — the run-in "header"
defect the study justified fixing must never come back.

Large docs are marked ``slow`` (run with ``--runslow``).
"""

import json
import pathlib

import pytest

from tests.docsling_compat import FIXTURE_DOCS, layout_metrics, regressions
from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.opencontracts_exporter import validate_export

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
TARGETS = FIXTURES / "docsling_targets"
_BASELINE_PATH = FIXTURES / "docsling_layout_baseline.json"
BASELINE = json.loads(_BASELINE_PATH.read_text()) if _BASELINE_PATH.exists() else {}


def _params():
    return [
        pytest.param(slug, rel, marks=[pytest.mark.slow] if is_slow else [], id=slug)
        for slug, rel, is_slow in FIXTURE_DOCS
    ]


@pytest.fixture(scope="module")
def _metrics_cache():
    return {}


def _metrics(slug, rel, cache):
    if slug not in cache:
        warp = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / rel))
        validate_export(warp)
        oracle = json.loads((TARGETS / f"{slug}.json").read_text())
        cache[slug] = layout_metrics(warp, oracle)
    return cache[slug]


@pytest.mark.parametrize("slug,rel", _params())
def test_layout_does_not_regress(slug, rel, _metrics_cache):
    if not (FIXTURES / rel).exists():
        pytest.skip(f"fixture PDF missing: {rel}")
    if not (TARGETS / f"{slug}.json").exists() or slug not in BASELINE:
        pytest.skip(
            f"no docsling oracle/baseline for {slug}; run build_docsling_fixtures.py"
        )
    failures = regressions(_metrics(slug, rel, _metrics_cache), BASELINE[slug])
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("slug,rel", _params())
def test_no_overlong_header_blocks(slug, rel, _metrics_cache):
    """The run-in 'header' relabel (exporter._HEADER_MAX_WORDS) holds: no block
    labeled 'Section Header' carries a whole section body. Invariant, not floor."""
    if not (FIXTURES / rel).exists():
        pytest.skip(f"fixture PDF missing: {rel}")
    if not (TARGETS / f"{slug}.json").exists():
        pytest.skip(f"no docsling oracle for {slug}; run build_docsling_fixtures.py")
    assert _metrics(slug, rel, _metrics_cache)["overlong_heading_count"] == 0
