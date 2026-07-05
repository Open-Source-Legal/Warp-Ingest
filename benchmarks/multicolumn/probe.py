"""Read-only probe of the front-end multi-column gate.

The fire/not-fire verdict is taken **directly** from the parser's
``_detect_column_gutters`` (so the headline recall/precision numbers can never
drift from the real gate). A staged re-implementation, ``reject_reason``, is used
*only* to explain *why* a page that did not fire was rejected — attributing it to
the first of the gate's decision stages that fails. A guard test
(``tests/test_multicolumn_probe.py``) asserts the two agree on the fire boundary
(``reject_reason(...) is None`` iff the gate fires) over real dataset pages.

Nothing here edits the parser; it imports the gate's own constants and helpers so
the staged logic stays in lock-step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from warp_ingest.file_parser import pdf_plumber_parser as P

# Ordered decision stages of the XY-cut _detect_column_gutters, for reason
# attribution (mirrors the detector so reject_reason(...) is None iff it fires).
REASONS = (
    "empty",
    "too_few_words",
    "no_region_gutter",
    "prose_gate",
    "strict_tabular",
)


def extract_page_words(pdf_path: str | Path, page_index: int = 0):
    """Return ``(words, bbox)`` for a page using the parser's exact extraction."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as doc:
        page = doc.pages[page_index]
        words = page.extract_words(
            x_tolerance=P.WORD_X_TOLERANCE,
            y_tolerance=P.WORD_Y_TOLERANCE,
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=["fontname", "size"],
        )
        return words, tuple(page.bbox)


def reject_reason(words, bbox) -> Optional[str]:
    """The first XY-cut stage that rejects this page, or ``None`` if it fires.

    Mirrors the XY-cut ``_detect_column_gutters`` stage-for-stage using the same
    module helpers/constants. Diagnostic only — the fire verdict itself comes from
    the real gate (see ``probe_page``); a guard test pins ``reject_reason(...) is
    None`` iff the gate fires.
    """
    if not words or bbox is None:
        return "empty"
    x0, x1 = bbox[0], bbox[2]
    page_w = x1 - x0
    if page_w <= 0 or len(words) < 2 * P.COL_MIN_LINES:
        return "too_few_words"

    # Pass 1: the global gate fires -> the augment gate fires.
    if P._global_column_gutters(words, bbox):
        return None
    # Pass 2: mirror the XY-cut fallback stages.
    h_gap = P.COL_XYCUT_HGAP_FONT_MULT * P._median_line_height(words)
    regions: list = []
    P._xycut_regions(words, x0, page_w, h_gap, regions)
    if not regions:
        return "no_region_gutter"

    _, gutters = max(regions, key=lambda t: len(t[0]))
    if P._count_prose_columns(words, gutters) < 2:
        return "prose_gate"
    if not P._columns_low_tabular(words, gutters, P.COL_STRICT_TABULAR_FRACTION):
        return "strict_tabular"
    return None


def probe_page(words, bbox) -> dict[str, Any]:
    """Verdict (from the real gate) + reason (diagnostic) + shape signals."""
    gutters = P._detect_column_gutters(words, bbox)  # authoritative verdict
    fired = bool(gutters)
    rows = P._group_words_into_lines(words) if words else []
    return {
        "fired": fired,
        "n_cols": (len(gutters) + 1) if fired else 0,
        "reason": None if fired else reject_reason(words, bbox),
        "n_words": len(words),
        "n_rows": len(rows),
    }


def probe_pdf(pdf_path: str | Path, page_index: int = 0) -> dict[str, Any]:
    words, bbox = extract_page_words(pdf_path, page_index)
    rec = probe_page(words, bbox)
    rec["pdf"] = str(pdf_path)
    return rec
