"""End-to-end parsing tests for every PDF fixture using the pure-Python
(Java/Tika-free) pipeline.

Covers the regular text-layer PDFs and the scanned ``needs_ocr.pdf`` which
exercises the optional OCR backend.  These assert that each document is parsed
into a non-trivial block structure and that text recall against a pdfplumber
baseline stays high for the text PDFs.
"""

import os

import pdfplumber
import pytest

from warp_ingest.file_parser import ocr_parser
from warp_ingest.ingestor.pdf_ingestor import PDFIngestor

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

TEXT_PDFS = [
    "sample.pdf",
    "USC Title 1 - CHAPTER 1.pdf",
    "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_EX-10.1_"
    "Development_Agreement_ZrZJLLv.pdf",
]
OCR_PDF = "needs_ocr.pdf"

PARSE_OPTIONS = {
    "apply_ocr": False,  # auto-detection routes scanned pages to OCR
    "render_format": "all",
}


def _ingest(filename):
    return PDFIngestor(os.path.join(FIX_DIR, filename), dict(PARSE_OPTIONS))


def _block_words(blocks):
    words = set()
    for b in blocks:
        for w in b.get("block_text", "").lower().split():
            words.add(w)
    return words


def _pdfplumber_words(path):
    words = set()
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for w in (page.extract_text() or "").lower().split():
                words.add(w)
    return words


@pytest.mark.parametrize("filename", TEXT_PDFS)
def test_text_pdf_parses(filename):
    ing = _ingest(filename)
    blocks = ing.blocks
    # structural sanity
    assert len(blocks) > 30, f"{filename}: too few blocks ({len(blocks)})"
    assert ing.return_dict.get("result"), f"{filename}: empty result document"
    assert ing.return_dict.get("num_pages", 0) >= 1

    # multiple block types should be detected (headers + paragraphs at least)
    types = {b.get("block_type") for b in blocks}
    assert "para" in types
    assert "header" in types

    # text recall vs pdfplumber baseline (our front-end IS pdfplumber, so this is
    # effectively a self-consistency / no-text-dropped check)
    baseline = _pdfplumber_words(os.path.join(FIX_DIR, filename))
    got = _block_words(blocks)
    recall = len(baseline & got) / max(1, len(baseline))
    assert recall > 0.93, f"{filename}: text recall too low ({recall:.2%})"


@pytest.mark.skipif(
    not ocr_parser.ocr_available(),
    reason="OCR backend (rapidocr-onnxruntime) not installed",
)
def test_scanned_pdf_parses_with_ocr():
    ing = _ingest(OCR_PDF)
    blocks = ing.blocks
    assert len(blocks) > 5, f"{OCR_PDF}: OCR produced too few blocks ({len(blocks)})"
    words = _block_words(blocks)
    assert len(words) > 50, f"{OCR_PDF}: OCR produced too few words ({len(words)})"
    assert ing.return_dict.get("result"), f"{OCR_PDF}: empty result document"


def test_scanned_pdf_does_not_crash_without_ocr(monkeypatch):
    """Even if the OCR backend is unavailable, a scanned PDF must parse (yielding
    little/no text) rather than raising."""
    monkeypatch.setattr(ocr_parser, "ocr_available", lambda: False)
    ing = _ingest(OCR_PDF)
    assert ing.return_dict is not None
    assert "result" in ing.return_dict
    assert ing.return_dict.get("page_dim")  # page geometry still extracted
