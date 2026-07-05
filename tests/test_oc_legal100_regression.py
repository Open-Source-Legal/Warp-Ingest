"""Legal-100 structural-correctness regression (vision-adjudicated golden).

For every manifested ``(doc, page)`` the live Warp export must keep
``validate_export`` intact and never regress below the committed per-page
baseline (``tests/fixtures/legal100_baseline.json``): label-agreement metrics +
relationship agreement floored, structural smells ceiled, against the
vision-adjudicated golden (``tests/fixtures/legal100_golden.json``). Improvements
pass; regressions fail.

This is suite #2 of 2 (heterogeneous counterpart:
``test_oc_hetero100_regression``). Large prospectus bodies are ``@slow``. Each
document parses once (module-scoped cache). Regenerate the golden +
baseline with ``scripts/adjudicate_legal_golden`` →
``scripts/assemble_legal_golden.py`` → ``scripts/build_legal100_baseline.py``.
"""

import os

import pytest

from tests import oc_golden_eval as G
from tests import oc_legal100_compat as C
from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.opencontracts_exporter import (
    ExportValidationError,
    validate_export,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GOLDEN = C.load_golden() if os.path.exists(C.GOLDEN) else {}
_BASELINE = C.load_baseline() if os.path.exists(C.BASELINE) else {}


def _params():
    # one param per (doc, page); a document parses once via the module cache
    seen = []
    for relpath, page, is_slow, doc_class in C.manifest_pairs():
        seen.append(
            pytest.param(
                relpath,
                page,
                marks=[pytest.mark.slow] if is_slow else [],
                id=f"{os.path.basename(relpath)[:28]}__p{page}",
            )
        )
    return seen


@pytest.fixture(scope="module")
def _cache():
    return {}


def _export(relpath, cache):
    if relpath not in cache:
        cache[relpath] = pdf_ingestor.parse_to_opencontracts(
            os.path.join(REPO, relpath)
        )
    return cache[relpath]


@pytest.mark.skipif(
    not _GOLDEN or not _BASELINE, reason="legal100 golden/baseline not built yet"
)
@pytest.mark.parametrize("relpath,page", _params())
def test_legal_page_does_not_regress(relpath, page, _cache):
    if not os.path.exists(os.path.join(REPO, relpath)):
        pytest.skip(f"fixture missing: {relpath}")
    key = C.page_key(relpath, page)
    if key not in _GOLDEN or key not in _BASELINE:
        pytest.skip(f"no golden/baseline for {key}")

    export = _export(relpath, _cache)
    try:
        validate_export(export)
    except ExportValidationError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{relpath}: validate_export failed: {exc}")

    live = C.page_metrics(export, page, _GOLDEN[key]["regions"])
    problems = G.page_regressions(live, _BASELINE[key])
    assert not problems, f"{key} regressed:\n  " + "\n  ".join(problems)


def test_manifest_is_complete():
    pairs = C.manifest_pairs()
    assert len(pairs) >= 95, f"legal batch should be ~100 pages, got {len(pairs)}"
    for relpath, _page, _slow, _cls in pairs:
        assert os.path.exists(
            os.path.join(REPO, relpath)
        ), f"missing fixture: {relpath}"
