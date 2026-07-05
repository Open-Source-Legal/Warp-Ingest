"""Guard tests for the multi-column discriminator probe.

The probe's *fire verdict* comes straight from ``_detect_column_gutters``; its
``reject_reason`` is a staged re-implementation used only to explain non-fires.
The load-bearing invariant these tests pin: ``reject_reason(words, bbox) is None``
**iff** the real gate fires — so the reason attribution can never disagree with
the headline recall/precision numbers.

Pure-unit (synthetic word dicts); the same builders as
``tests/test_pdf_plumber_columns.py``. An opt-in sweep over real olmOCR-bench
pages runs only when ``$OLMOCR_BENCH_DIR`` is present.
"""

import os
from pathlib import Path

import pytest

from benchmarks.multicolumn import probe
from warp_ingest.file_parser import pdf_plumber_parser as P

BBOX = (0.0, 0.0, 600.0, 800.0)
PROSE = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"]


def W(text, x0, top, size=10.0):
    return {
        "text": text,
        "x0": float(x0),
        "x1": float(x0 + len(text) * size * 0.5),
        "top": float(top),
        "bottom": float(top + size),
        "size": float(size),
        "fontname": "Times",
    }


def row(tokens, x_start, top, size=10.0, gap=4.0):
    out, x = [], x_start
    for t in tokens:
        w = W(t, x, top, size)
        out.append(w)
        x = w["x1"] + gap
    return out


def single_col(n=12):
    words = []
    for i in range(n):
        words += row(PROSE, 40, 100.0 + i * 16)
    return words


def two_col(n=12):
    words = []
    for i in range(n):
        top = 100.0 + i * 16
        words += row(PROSE, 40, top)
        words += row(PROSE, 320, top)
    return words


def data_table(n=12):
    """Rows spanning the full width in 5 narrow cells (crosses any gutter)."""
    words = []
    for i in range(n):
        top = 100.0 + i * 16
        for c in range(5):
            words += row(["x1", "x2"], 40 + c * 110, top)
    return words


CASES = {
    "single_col": single_col(),
    "two_col": two_col(),
    "data_table": data_table(),
    "empty": [],
    "tiny": row(["a", "b", "c"], 40, 100),
}


@pytest.mark.parametrize("name", list(CASES))
def test_reason_none_iff_gate_fires(name):
    """The core guard: reject_reason is None exactly when the real gate fires."""
    words = CASES[name]
    gate_fires = bool(P._detect_column_gutters(words, BBOX))
    reason = probe.reject_reason(words, BBOX)
    assert (reason is None) == gate_fires, f"{name}: reason={reason} gate={gate_fires}"


@pytest.mark.parametrize("name", list(CASES))
def test_probe_page_verdict_matches_gate(name):
    words = CASES[name]
    rec = probe.probe_page(words, BBOX)
    assert rec["fired"] == bool(P._detect_column_gutters(words, BBOX))
    if rec["fired"]:
        assert rec["reason"] is None and rec["n_cols"] >= 2
    else:
        assert rec["reason"] in probe.REASONS


def test_two_col_fires_and_single_col_does_not():
    assert probe.probe_page(two_col(), BBOX)["fired"] is True
    single = probe.probe_page(single_col(), BBOX)
    assert single["fired"] is False
    # single column: no XY-cut region yields a column gutter
    assert single["reason"] == "no_region_gutter"


def test_reason_is_a_known_stage_for_all_cases():
    for words in CASES.values():
        r = probe.reject_reason(words, BBOX)
        assert r is None or r in probe.REASONS


# --- opt-in equivalence sweep over real pages (skipped without the dataset) --- #
def _dataset_pdfs(limit=40):
    root = Path(
        os.environ.get("OLMOCR_BENCH_DIR", Path.home() / "Code" / "olmocr-bench")
    )
    base = root / "olmOCR-bench" / "bench_data" / "pdfs"
    if not base.is_dir():
        return []
    out = []
    for sub in ("multi_column", "tables", "arxiv_math"):
        out += sorted((base / sub).glob("*.pdf"))[:limit]
    return out


@pytest.mark.parametrize(
    "pdf",
    _dataset_pdfs()
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="olmOCR-bench dataset not present (set OLMOCR_BENCH_DIR)"
            ),
        )
    ],
)
def test_probe_matches_gate_on_real_pages(pdf):
    words, bbox = probe.extract_page_words(pdf)
    gate_fires = bool(P._detect_column_gutters(words, bbox))
    assert (probe.reject_reason(words, bbox) is None) == gate_fires
