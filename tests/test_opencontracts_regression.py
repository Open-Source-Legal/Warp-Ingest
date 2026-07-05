"""Cross-fixture regression for the OpenContractDocExport exporter.

For every sample PDF the live exporter must (a) always satisfy the spec §6
invariants and (b) never regress below the committed metric baseline
(``tests/fixtures/oc_export_baseline.json``). Improvements pass; regressions
fail. Regenerate the baseline with ``scripts/build_opencontracts_fixtures.py``.

The scanned ``needs_ocr.pdf`` is marked ``slow`` (run with ``--runslow``).
"""

import json
import pathlib

import pytest

from tests.oc_compat import (
    FIXTURE_DOCS,
    FIXTURE_PARSE_OPTIONS,
    export_metrics,
    regressions,
)
from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.opencontracts_exporter import validate_export

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BASELINE = json.loads((FIXTURES / "oc_export_baseline.json").read_text())


def _params():
    return [
        pytest.param(
            name,
            marks=[pytest.mark.slow] if is_slow else [],
            id=pathlib.Path(name).stem[:30],
        )
        for name, is_slow in FIXTURE_DOCS
    ]


@pytest.fixture(scope="module")
def _export_cache():
    return {}


def _export(name, cache):
    if name not in cache:
        cache[name] = pdf_ingestor.parse_to_opencontracts(
            str(FIXTURES / name), parse_options=FIXTURE_PARSE_OPTIONS.get(name)
        )
    return cache[name]


@pytest.mark.parametrize("name", _params())
def test_export_is_valid(name, _export_cache):
    if not (FIXTURES / name).exists():
        pytest.skip(f"fixture missing: {name}")
    validate_export(_export(name, _export_cache))


@pytest.mark.parametrize("name", _params())
def test_export_does_not_regress(name, _export_cache):
    if not (FIXTURES / name).exists():
        pytest.skip(f"fixture missing: {name}")
    if name not in BASELINE:
        pytest.skip(f"no baseline for {name}; run build_opencontracts_fixtures.py")
    live = export_metrics(_export(name, _export_cache))
    failures = regressions(live, BASELINE[name])
    assert not failures, "\n".join(failures)
