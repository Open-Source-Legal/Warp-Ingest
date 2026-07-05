"""Tests for the parallel (multi-process) front-end.

The contract under test: parallel page extraction is a pure wall-clock
optimization — the emitted XHTML is byte-identical to the serial loop, short
documents stay serial, and any pool-level failure falls back to the serial
path transparently.
"""

import os

import pytest

from warp_ingest.file_parser import pdf_plumber_parser as ppp

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DOC = os.path.join(FIXTURES, "USC Title 1 - CHAPTER 1.pdf")  # 9pp
ONE_PAGE_DOC = os.path.join(FIXTURES, "needs_ocr.pdf")  # 1pp


def test_parallel_output_byte_identical(monkeypatch):
    monkeypatch.setenv("WARP_DISABLE_OCR", "1")
    monkeypatch.setenv("WARP_FE_WORKERS", "1")
    serial = ppp.pdf_to_xhtml(DOC)
    monkeypatch.setenv("WARP_FE_WORKERS", "4")
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "2")
    parallel = ppp.pdf_to_xhtml(DOC)
    assert parallel == serial


def test_short_document_stays_serial(monkeypatch):
    monkeypatch.setenv("WARP_DISABLE_OCR", "1")
    monkeypatch.setenv("WARP_FE_WORKERS", "4")
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "8")

    def _fail(*args, **kwargs):
        raise AssertionError("parallel path used for a short document")

    monkeypatch.setattr(ppp, "_pdf_to_page_divs_parallel", _fail)
    out = ppp.pdf_to_xhtml(ONE_PAGE_DOC)
    assert '<div class="page"' in out


def test_pool_failure_falls_back_to_serial(monkeypatch):
    monkeypatch.setenv("WARP_DISABLE_OCR", "1")
    monkeypatch.setenv("WARP_FE_WORKERS", "1")
    serial = ppp.pdf_to_xhtml(DOC)

    monkeypatch.setenv("WARP_FE_WORKERS", "4")
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "2")

    def _boom(*args, **kwargs):
        raise RuntimeError("pool exploded")

    monkeypatch.setattr(ppp, "_pdf_to_page_divs_parallel", _boom)
    assert ppp.pdf_to_xhtml(DOC) == serial


def test_worker_count_env_parsing(monkeypatch):
    monkeypatch.setenv("WARP_FE_WORKERS", "3")
    assert ppp._fe_worker_count() == 3
    monkeypatch.setenv("WARP_FE_WORKERS", "0")
    assert ppp._fe_worker_count() == 1  # floor: 0 means serial, never crashes
    monkeypatch.setenv("WARP_FE_WORKERS", "not-a-number")
    assert ppp._fe_worker_count() >= 1  # falls back to the default
    monkeypatch.delenv("WARP_FE_WORKERS")
    assert 1 <= ppp._fe_worker_count() <= 8


def test_min_pages_env_parsing(monkeypatch):
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "3")
    assert ppp._fe_parallel_min_pages() == 3
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "1")
    assert ppp._fe_parallel_min_pages() == 2  # floored: a 1-page doc is serial
    monkeypatch.setenv("WARP_FE_PARALLEL_MIN_PAGES", "junk")
    assert ppp._fe_parallel_min_pages() == 8
    monkeypatch.delenv("WARP_FE_PARALLEL_MIN_PAGES")
    assert ppp._fe_parallel_min_pages() == 8
