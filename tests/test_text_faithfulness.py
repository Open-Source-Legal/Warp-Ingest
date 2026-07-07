"""Text-faithfulness regression: Warp must never LOSE text.

Locks in the result of the exhaustive corpus evaluation
(``docs/text_faithfulness_eval.md``): on the OpenContracts PAWLS token surface
(the export's designed no-loss layer), effective token recall against the
independent PDFium oracle stays >= 0.999 on the canonical text fixtures.

"Effective" recall forgives re-tokenization (different word segmentation of
the same characters) but counts real absences — see
``scripts/text_faithfulness_eval.py`` for the scoring definition.
"""

import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX_DIR = REPO / "tests" / "fixtures"

_spec = importlib.util.spec_from_file_location(
    "text_faithfulness_eval", REPO / "scripts" / "text_faithfulness_eval.py"
)
tfe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfe)

TEXT_DOCS = [
    ("USC Title 1 - CHAPTER 1.pdf", 0.999),
    (
        "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_"
        "EX-10.1_Development_Agreement_ZrZJLLv.pdf",
        0.999,
    ),
]


@pytest.mark.parametrize("filename,floor", TEXT_DOCS)
def test_pawls_effective_recall_vs_pdfium(filename, floor):
    rec = tfe.eval_doc(str(FIX_DIR / filename), ["pdfium"])
    assert "error" not in rec, rec.get("error")
    pawls = rec["oracles"]["pdfium"]["pawls"]
    assert pawls["n_oracle"] > 1000, "oracle extracted implausibly little text"
    assert pawls["recall_effective"] >= floor, (
        f"{filename}: PAWLS effective recall {pawls['recall_effective']} "
        f"< {floor}; missing pages: {pawls['flagged_pages'][:3]}"
    )


def test_scanned_doc_does_not_regress_text_layer():
    """needs_ocr.pdf has no embedded text layer, so the oracle sees ~nothing;
    the eval must degrade to a trivial pass rather than crash, and Warp's OCR
    should still produce a substantial token stream."""
    rec = tfe.eval_doc(str(FIX_DIR / "needs_ocr.pdf"), ["pdfium"])
    assert "error" not in rec, rec.get("error")
    pawls = rec["oracles"]["pdfium"]["pawls"]
    assert pawls["recall_effective"] >= 0.999
