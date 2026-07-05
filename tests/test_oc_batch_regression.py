"""Per-page structural-quality regression over the 60-page audit batch.

For every audited ``(doc, page)`` the live exporter must (a) keep the document's
``validate_export`` invariants + relationship integrity intact and (b) never
regress below the committed per-page baseline
(``tests/fixtures/oc_batch_baseline.json``): smell counts ceiled (must not grow),
coverage/anchored/tightness fractions floored (must not drop). Improvements pass;
regressions fail. Regenerate with ``scripts/build_oc_batch_fixtures.py``.

Large prospectus bodies are marked ``slow`` (run with ``--runslow``).
"""

import json
import pathlib

import pytest

from tests.oc_batch_compat import (
    BATCH_PAGES,
    manifest_pairs,
    page_key,
    page_metrics,
    page_regressions,
)
from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.opencontracts_exporter import (
    ExportValidationError,
    validate_export,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "tests" / "fixtures" / "oc_batch_baseline.json"
BASELINE = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}


def _slug(relpath):
    return pathlib.Path(relpath).stem[:30]


def _params():
    return [
        pytest.param(
            relpath,
            page,
            marks=[pytest.mark.slow] if is_slow else [],
            id=f"{_slug(relpath)}__p{page}",
        )
        for relpath, page, is_slow, _cls, _note in manifest_pairs()
    ]


@pytest.fixture(scope="module")
def _cache():
    return {}


def _export(relpath, cache):
    if relpath not in cache:
        cache[relpath] = pdf_ingestor.parse_to_opencontracts(str(ROOT / relpath))
    return cache[relpath]


@pytest.mark.parametrize("relpath,page", _params())
def test_batch_page_does_not_regress(relpath, page, _cache):
    if not (ROOT / relpath).exists():
        pytest.skip(f"fixture missing: {relpath}")
    key = page_key(relpath, page)
    if key not in BASELINE:
        pytest.skip(f"no baseline for {key}; run build_oc_batch_fixtures.py")

    export = _export(relpath, _cache)

    # (a) document-level invariants + relationship integrity must hold
    try:
        validate_export(export)
    except ExportValidationError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{relpath}: validate_export failed: {exc}")

    doc_base = BASELINE.get(f"{relpath}::_doc", {})
    if "page_count" in doc_base:
        assert export["page_count"] == doc_base["page_count"], (
            f"{relpath}: page_count {export['page_count']} != "
            f"baseline {doc_base['page_count']}"
        )
    assert doc_base.get("relationship_validity", True) is True

    # (b) per-page metric floors/ceils
    live = page_metrics(export, page)
    failures = page_regressions(live, BASELINE[key])
    assert not failures, f"{key}:\n" + "\n".join(failures)


def test_manifest_is_sixty_pages():
    assert len(manifest_pairs()) == 60, "audit batch must cover exactly 60 pages"
    # every doc resolves to an in-repo fixture path
    for relpath, _pages, _slow, _cls, _note in BATCH_PAGES:
        assert (ROOT / relpath).exists(), f"missing committed fixture: {relpath}"
