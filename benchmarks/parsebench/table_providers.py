"""Pluggable table-cell extraction for the ParseBench renderer.

warp owns region detection, reading order, prose, charts and visual grounding;
the *cell grid* of each table is delegated to a swappable provider so a
best-in-class extractor can be dropped in (or replaced later) without touching the
engine. A provider maps ``(pdf_path, regions_by_page)`` to
``{page_index: [<table>, ...]}`` where each table is either a bare HTML string
(legacy, page-granular replacement) or a ``(bbox, html)`` pair with
``bbox = (x0, top, x1, bottom)`` in PDF points (region-aware replacement: only
warp table blocks overlapping the bbox are substituted).

``get_table_provider()`` returns the active provider, or ``None`` (then the
renderer falls back to warp's own table rendering). Selection is explicit via
the ``WARP_TABLE_PROVIDER`` env var:

* ``native`` (default) — warp's own table engine
  (``warp_ingest.ingestor.table_engine``: pdfplumber ruled grids + region-local
  whitespace-channel grid inference over warp's table regions). No external
  parser involved.
* ``none`` — no provider; warp's raw table rendering.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

TableProvider = Callable[..., dict[int, list[Any]]]


def _native_provider(
    pdf_path: str, regions_by_page: Optional[dict] = None
) -> dict[int, list[tuple[tuple, str]]]:
    """Warp's own table engine (license-clean, region-aware)."""
    from warp_ingest.ingestor.table_engine import extract_pdf_tables

    return extract_pdf_tables(pdf_path, regions_by_page=regions_by_page)


def get_table_provider() -> Optional[TableProvider]:
    choice = os.environ.get("WARP_TABLE_PROVIDER", "native").strip().lower()
    if choice in ("none", "off", "0", "warp"):
        return None
    if choice in ("native", "auto", "warp_native", "table_engine"):
        return _native_provider
    return None
