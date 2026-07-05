"""Pure-Python PDF -> Tika-compatible XHTML parser.

This module replaces the Java/Apache-Tika front-end.  It uses ``pdfplumber``
(MIT, pure-Python on top of pdfminer.six) to extract real per-word bounding
boxes and font metadata, and emits the *exact* intermediate XHTML format that
the nlm-modified Tika produced and that ``visual_ingestor`` consumes.

The contract (one ``<p>`` per visual-line segment) reproduced here:

    <div class="page" style="height:792.0px; width:612.0px; position: relative;">
      <svg ...><line x1=.. y1=.. x2=.. y2=../> ...</svg>
      <p style="height:..;font-size:12.0px;font-family:Times;font-style:normal;
                font-weight:normal;top:176.7px;position:absolute;text-indent:121.1px;
                word-start-positions:[(x,y), ...];
                word-end-positions:[(x,y), ...];
                word-fonts:[(family,weight,style,size,size,space_width), ...]">text</p>
    </div>

The word boxes are pdfplumber's real per-word boxes in absolute, top-left-origin
PDF points -- the same technique OpenContracts uses (docs/bbox_architecture.md).
"""

import html
import logging
import multiprocessing
import os
import statistics
import threading
from concurrent.futures import ProcessPoolExecutor

import pdfplumber
from pdfminer.layout import LTChar, LTContainer, LTLine, LTRect
from pdfplumber.utils import extract_words as _pp_extract_words

from warp_ingest.file_parser.file_parser import FileParser

logger = logging.getLogger(__name__)

# pdfplumber word extraction tolerances.  x_tolerance controls how far apart two
# characters can be before pdfplumber treats them as separate words; y_tolerance
# clusters characters onto the same baseline.
WORD_X_TOLERANCE = 1.5
WORD_Y_TOLERANCE = 3.0
# A new <p> segment starts when the horizontal gap between consecutive words on
# the same line exceeds this multiple of the font size (separates columns, e.g.
# a table-of-contents "Sec." number column from the chapter-title column).
SEGMENT_GAP_FONT_MULTIPLE = 2.5
# Two words belong to the same visual line if their tops differ by less than this
# fraction of the (larger) font size.
LINE_TOP_FONT_FRACTION = 0.5
# A page with fewer than this many text lines is treated as scanned/sparse and is
# routed to OCR (when an OCR engine is available) only when it also has very few
# extracted words.  Some PDFs have a dense text layer collapsed onto one or two
# geometric lines; OCRing those pages loses word boundaries and regresses text
# fidelity.
MIN_TEXT_LINES_FOR_PAGE = 4
MIN_TEXT_WORDS_FOR_PAGE = 25
# pdfplumber splits a word whenever an ``extra_attrs`` value (fontname/size)
# changes, shattering a single display word whose glyphs carry different subset
# font names into *touching* fragments.  Two fragments are re-joined as one word
# when the gap between them is at most this fraction of the font size.  Genuine
# inter-word spaces are ~0.20-0.28*size across font sizes (measured), while these
# attribute-split fragments abut (gap ~= 0), so a 0.10*size threshold separates
# them with a safe margin below the real-space floor.
FRAGMENT_MERGE_GAP_FONT_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Multi-column reading order (gutter-aware line grouping).
#
# The default line grouping sorts ALL page words by (top, x0) and grows each
# visual line from the first word's top, so on a 2-column page it interleaves the
# two columns row-by-row (newspaper-read-across) and fragments words whose column
# baseline differs from a same-top neighbour in the other column.  When a
# confident multi-column *prose* layout is detected we partition the page into
# columns first and group each column independently, fixing both reading order
# and fragmentation.  Detection is geometry-only and deliberately conservative:
# a real data table (rows that span the columns) and a single-column page never
# qualify, so their XHTML is byte-identical to the non-column path.
# ---------------------------------------------------------------------------
# --- global gate (first pass: high precision, conservative) -----------------
# A vertical whitespace band is a column gutter only if it is at least this
# fraction of page width wide ...
COL_GUTTER_MIN_WIDTH_FRACTION = 0.025
# ... and is crossed by at most this fraction of the page's text rows (a true
# gutter is empty for nearly every row; a single full-width header/footer that
# crosses it is tolerated).
COL_GUTTER_MAX_ROWCROSS_FRACTION = 0.08
# --- XY-cut region gate (fallback: recovers spanning-header pages) -----------
# Within one XY-cut region a gutter may be only ~1.3% of page width -- justified /
# ragged column edges nibble a real 2-column "river" narrow; 0.013*W (~7.7pt on a
# 595pt page) is still >3 space-widths so it never catches inter-word spacing.
COL_XYCUT_GUTTER_MIN_WIDTH_FRACTION = 0.013
# ... and may be crossed by up to this fraction of the region's rows (a region
# gutter tolerates spanning rows -- equations, run-in figures -- that survive the
# horizontal peel).
COL_XYCUT_ROWCROSS_FRACTION = 0.20
# Each resulting column must be at least this fraction of page width (a data
# table's narrow cells fail this) ...
COL_MIN_WIDTH_FRACTION = 0.20
# ... and the columns must be roughly balanced in width: a clean newspaper /
# report layout has near-equal columns, whereas a prose-column-beside-an-
# infographic-matrix is lopsided.  Reject when widest/narrowest exceeds this.
COL_MAX_WIDTH_IMBALANCE = 1.6
# ... contain at least this many text lines ...
COL_MIN_LINES = 5
# ... and read like flowing prose: median words-per-line at least this.  Paired
# with the tabular-row guard below (which excludes in-column tables that share
# prose's width/balance), this admits genuine 2-column prose while a data-table
# column (few words/line) is rejected.
COL_MIN_MEDIAN_WORDS_PER_LINE = 6
# Resolution of the x-projection used to locate gutters.
COL_PROJECTION_BINS = 300
# A column whose rows are this fraction (or more) "tabular" -- splitting into 3+
# segments, i.e. multiple internal column gaps -- contains a table; splitting the
# page would disturb the engine's in-column table/list detection, so don't.
COL_MAX_TABULAR_ROW_FRACTION = 0.30
# Words this far (fraction of page width) from the text's left/right edge are the
# page margins, not interior gutters.
COL_MARGIN_FRACTION = 0.06
# XY-cut: split the page's rows into horizontal bands wherever a full-width
# vertical gap exceeds this multiple of the median font size (a MAJOR region
# break -- masthead / header / figure / footer).  Conservative (2.0) so ordinary
# paragraph spacing does not over-split a column region.
COL_XYCUT_HGAP_FONT_MULT = 2.0
# Strict per-column tabular-row ceiling applied (on top of the 0.30 gate inside
# _count_prose_columns) to the chosen XY-cut gutters: the ONE geometric signal
# that separates 2-column prose (tabular-row frac ~0) from a dense 2-column data
# table (which can share prose's width / balance / words-per-line).
COL_STRICT_TABULAR_FRACTION = 0.08

# --- grid gate (two-column form/signature grids: the table-safe cell split) --
# The prose gates above deliberately exclude tables/grids, so on a two-column
# signature/approval grid the left and right cells of each row stay stream-
# adjacent and the downstream engine (which welds same-top adjacent <p>
# segments into one block) fuses them ("By:  By:", "Dana Burghdoff  Gary L.
# Wert").  A *grid* region is a banded run of rows sharing exactly ONE wide
# interior gutter that (almost) no row crosses, with BOTH sides populated and
# prose-like column widths / balance -- two independent cell stacks, never a
# row-associative structure.  Such a region's rows are routed column-major
# (like the prose split) so each cell becomes its own <p> / block.  Data
# tables (2+ real gutters), TOC and label|value layouts (narrow or one-sided
# columns) and single-column prose never qualify.
# A band must have at least this many text rows ...
GRID_MIN_ROWS = 5
# ... a candidate gutter is real only when at least this many rows sit
# entirely on EACH side of it (kills spurious in-column slivers, one-sided
# margin content, and cerebras-style label|value tables with few labels) ...
GRID_MIN_SIDE_ROWS = 3
# ... and at most this fraction of the band's rows may cross it (stricter
# than the prose XY-cut's 0.20: a data table bleeding into the band shows up
# as crossings; a lone spanning header is tolerated and flushed full-width).
GRID_MAX_CROSS_FRACTION = 0.10
# Quiet-zone identification is more tolerant than the final crossing budget:
# two overlapping single-crossing rows (a signature line ending at the gutter
# edge under a run-in "APPROVED AS TO FORM AND By:" line) stack a 2-high wall
# in the projection that would split the true gutter in two.  Zones found at
# this tolerance are then TRIMMED past the cheapest crossers down to
# GRID_MAX_CROSS_FRACTION.
GRID_ZONE_CROSS_FRACTION = 0.20
# Rows are banded into regions at vertical gaps exceeding the LARGER of this
# multiple of the page's median font size (the XY-cut peel) and this multiple
# of the band's median row pitch -- grid rows (By:/Name:/Title:) sit further
# apart than their nominal font size suggests, and font-size-only banding
# shatters a signature grid into sub-GRID_MIN_ROWS fragments.
GRID_HGAP_FONT_MULT = 2.0
GRID_HGAP_PITCH_MULT = 1.5
# Finally, the region must LOOK like a form: cells whose first/second or last
# word ends with ":" ("By:", "Name:", "VENDOR:", "Quote Date:", "APPROVAL
# RECOMMENDED:") are form labels -- a *format* signal (punctuation, like the
# colon-terminated list lead-in the exporter keys on), never word content.  A
# signature/approval/form grid is DENSE with them on both sides of the gutter
# and its cells are SPARSE (a few words each), whereas the generic 2-column
# enterprise regions the hetero-100 suite guards are not: measured form-label
# fractions are 0.32-0.90 on the legal grid targets vs 0.07-0.23 on the
# board-bio / statistical-annex / 2-col-paper pages that mis-fired (and the
# one dense-labelled hetero page, a people grid, has 6.5-word cells).  Without
# these gates the splitter fired on 49/100 hetero pages and regressed 36.
GRID_MIN_FORM_LABELS_PER_SIDE = 1  # each side carries at least one label ...
GRID_MIN_FORM_LABEL_FRACTION = 0.28  # ... labelled cells >= this frac of rows
GRID_MAX_MEDIAN_CELL_WORDS = 5  # sparse label/value cells, not flowing prose


def _clean_font_family(fontname):
    """Strip pdfminer subset prefixes ('ABCDEF+Times-Roman' -> 'Times-Roman')."""
    if not fontname:
        return "Default"
    name = fontname.split("+", 1)[1] if "+" in fontname else fontname
    # commas would corrupt the word-fonts tuple serialization
    return name.replace(",", "-").strip() or "Default"


def _font_style(fontname):
    low = (fontname or "").lower()
    return "italic" if ("italic" in low or "oblique" in low) else "normal"


def _font_weight(fontname):
    low = (fontname or "").lower()
    return (
        "bold" if ("bold" in low or low.endswith(".b") or "black" in low) else "normal"
    )


def _round(v, n=3):
    return round(float(v), n)


def _group_words_into_lines(words):
    """Group pdfplumber words (already left-to-right within a line) into visual
    lines by their ``top`` coordinate, then sort each line left-to-right."""
    if not words:
        return []
    # sort by (top, x0) so we sweep top-to-bottom, left-to-right
    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    current = [sorted_words[0]]
    cur_top = sorted_words[0]["top"]
    cur_size = sorted_words[0].get("size") or 10.0
    for w in sorted_words[1:]:
        size = w.get("size") or cur_size or 10.0
        tol = max(LINE_TOP_FONT_FRACTION * max(size, cur_size), 1.5)
        if abs(w["top"] - cur_top) <= tol:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
            cur_top = w["top"]
            cur_size = size
    if current:
        lines.append(sorted(current, key=lambda x: x["x0"]))
    return lines


def _median_line_height(words):
    """Median font size -- a robust proxy for line height, used to size the
    'major horizontal region break' gap in the XY-cut peel."""
    sizes = [w.get("size") or 10.0 for w in words]
    return statistics.median(sizes) if sizes else 10.0


def _region_column_gutters(words, x0, page_w):
    """Row-crossing gutter detection restricted to one XY-cut region (a
    horizontal band with no MAJOR full-width vertical gap).  Same projection as
    the original global gate, but region-scoped and using the crossing-tolerant /
    narrow-valley constants -- so a spanning element that survives the peel does
    not mask the gutter, and a river nibbled narrow by ragged edges is still
    found.  Returns ``[]`` if no balanced, wide-enough gutter set exists.
    """
    rows = _group_words_into_lines(words)
    nrows = len(rows)
    if nrows < COL_MIN_LINES:
        return []
    nbins = COL_PROJECTION_BINS
    binw = page_w / nbins
    rowcross = [0] * nbins
    for r in rows:
        covered = set()
        for w in r:
            b0 = max(0, int((w["x0"] - x0) / binw))
            b1 = min(nbins - 1, int((w["x1"] - x0) / binw))
            covered.update(range(b0, b1 + 1))
        for b in covered:
            rowcross[b] += 1
    wmin = min(w["x0"] for w in words)
    wmax = max(w["x1"] for w in words)
    thr = int(COL_XYCUT_ROWCROSS_FRACTION * nrows)
    min_gw = COL_XYCUT_GUTTER_MIN_WIDTH_FRACTION * page_w
    margin = COL_MARGIN_FRACTION * page_w
    gutters = []
    i = 0
    while i < nbins:
        if rowcross[i] <= thr:
            j = i
            while j < nbins and rowcross[j] <= thr:
                j += 1
            xa = x0 + i * binw
            xb = x0 + j * binw
            if (xb - xa) >= min_gw and xa > wmin + margin and xb < wmax - margin:
                gutters.append((round(xa, 2), round(xb, 2)))
            i = j
        else:
            i += 1
    if not gutters:
        return []
    # Each column between gutters must be wide enough to hold prose ...
    min_cw = COL_MIN_WIDTH_FRACTION * page_w
    cols, prev = [], wmin
    for xa, xb in gutters:
        cols.append((prev, xa))
        prev = xb
    cols.append((prev, wmax))
    widths = [c[1] - c[0] for c in cols]
    if any(w < min_cw for w in widths):
        return []
    # ... and roughly balanced (rejects lopsided prose-beside-infographic pages).
    if max(widths) / min(widths) > COL_MAX_WIDTH_IMBALANCE:
        return []
    return gutters


def _global_column_gutters(words, bbox):
    """The original high-precision GLOBAL-projection gate (first pass).

    A single page-wide row-crossing x-projection with the conservative
    ``COL_GUTTER_*`` thresholds: fires only when a clean gutter is empty across
    nearly every page row.  It correctly handles the pages it already handled
    (e.g. in-column-table + prose report pages) -- so running it *first* preserves
    that behavior, and the XY-cut fallback only adds recall on the pages this gate
    misses (spanning-header ``no_gutter`` layouts).
    """
    if not words or bbox is None:
        return []
    x0, x1 = bbox[0], bbox[2]
    page_w = x1 - x0
    if page_w <= 0 or len(words) < 2 * COL_MIN_LINES:
        return []
    rows = _group_words_into_lines(words)
    nrows = len(rows)
    if nrows < COL_MIN_LINES:
        return []
    nbins = COL_PROJECTION_BINS
    binw = page_w / nbins
    rowcross = [0] * nbins
    for r in rows:
        covered = set()
        for w in r:
            b0 = max(0, int((w["x0"] - x0) / binw))
            b1 = min(nbins - 1, int((w["x1"] - x0) / binw))
            covered.update(range(b0, b1 + 1))
        for b in covered:
            rowcross[b] += 1
    wmin = min(w["x0"] for w in words)
    wmax = max(w["x1"] for w in words)
    thr = int(COL_GUTTER_MAX_ROWCROSS_FRACTION * nrows)
    min_gw = COL_GUTTER_MIN_WIDTH_FRACTION * page_w
    margin = COL_MARGIN_FRACTION * page_w
    gutters = []
    i = 0
    while i < nbins:
        if rowcross[i] <= thr:
            j = i
            while j < nbins and rowcross[j] <= thr:
                j += 1
            xa = x0 + i * binw
            xb = x0 + j * binw
            if (xb - xa) >= min_gw and xa > wmin + margin and xb < wmax - margin:
                gutters.append((round(xa, 2), round(xb, 2)))
            i = j
        else:
            i += 1
    if not gutters:
        return []
    min_cw = COL_MIN_WIDTH_FRACTION * page_w
    cols, prev = [], wmin
    for xa, xb in gutters:
        cols.append((prev, xa))
        prev = xb
    cols.append((prev, wmax))
    widths = [c[1] - c[0] for c in cols]
    if any(w < min_cw for w in widths):
        return []
    if max(widths) / min(widths) > COL_MAX_WIDTH_IMBALANCE:
        return []
    if _count_prose_columns(words, gutters) >= 2:
        return gutters
    return []


def _xycut_regions(words, x0, page_w, h_gap, out, depth=0):
    """Recursive XY-cut.  Split the block's rows into horizontal bands at MAJOR
    full-width vertical gaps (>= *h_gap*), recursing until a band has no further
    horizontal split; that leaf band is a *region* whose column gutters are
    collected into *out* (as ``(region_words, gutters)``).  Peeling a spanning
    masthead / header / figure / footer into its own band lets the header-free
    body region reveal the clean vertical gutter a single global projection
    misses (the dominant ``no_gutter`` recall loss).
    """
    if len(words) < 2 * COL_MIN_LINES or depth > 6:
        return
    rows = _group_words_into_lines(words)
    if not rows:
        return
    bands = [[rows[0]]]
    prev_bottom = max(w["bottom"] for w in rows[0])
    for r in rows[1:]:
        top = min(w["top"] for w in r)
        bottom = max(w["bottom"] for w in r)
        if top - prev_bottom > h_gap:
            bands.append([r])
        else:
            bands[-1].append(r)
        prev_bottom = max(prev_bottom, bottom)
    if len(bands) == 1:  # leaf region -- no further horizontal split
        gutters = _region_column_gutters(words, x0, page_w)
        if gutters:
            out.append((words, gutters))
        return
    for band in bands:
        band_words = [w for r in band for w in r]
        _xycut_regions(band_words, x0, page_w, h_gap, out, depth + 1)


def _columns_low_tabular(words, gutters, max_tab_frac):
    """True iff every non-empty column bounded by *gutters* is low-tabular
    (tabular-row fraction < *max_tab_frac*).  A stricter companion to
    ``_count_prose_columns`` keyed on the one signal that separates 2-column
    prose (tabular-row frac ~0) from a dense 2-column data table.
    """
    splits = [(xa + xb) / 2 for xa, xb in gutters]
    cols = [[] for _ in range(len(gutters) + 1)]
    for w in words:
        if any(w["x0"] < xb and w["x1"] > xa for xa, xb in gutters):
            continue  # word sits in a gutter band (full-width element)
        cx = (w["x0"] + w["x1"]) / 2
        cols[_column_index(cx, splits)].append(w)
    for cw in cols:
        if not cw:
            continue
        lines = _group_words_into_lines(cw)
        if len(lines) < COL_MIN_LINES:
            continue
        if _tabular_row_fraction(lines) >= max_tab_frac:
            return False
    return True


def _detect_column_gutters(words, bbox):
    """Detect confident multi-column *prose* gutters (global gate + XY-cut fallback).

    Returns a left-to-right list of ``(xa, xb)`` gutter bands for the page's
    dominant multi-column region, or ``[]`` when the page is single-column, a
    data table, or a label/value table (-> XHTML byte-identical to the non-column
    path).

    Two passes (augment, not replace):

    1. ``_global_column_gutters`` -- the original conservative page-wide
       projection.  High precision; PRESERVES the pages it already handled (e.g.
       in-column-table + prose report pages), so those never regress.
    2. XY-cut fallback (only when pass 1 finds nothing) -- recursively cut the
       page at MAJOR full-width horizontal gaps so a spanning masthead / header /
       figure is peeled into its own region; the header-free body region then
       yields the clean vertical gutter the global projection misses (the
       dominant ``no_gutter`` recall loss).  The dominant (largest) region's
       gutters are gated by the prose test plus a strict per-column tabular
       ceiling so data tables never qualify.

    The existing ``_group_words_into_lines_columns`` partition + spanning-row
    flush consumes the gutters unchanged (a header + 2-column page routes the
    header full-width and the body into columns).  Design + validation:
    docs/superpowers/specs/2026-07-01-multicolumn-xycut-detector-design.md.
    """
    if not words or bbox is None:
        return []
    # Pass 1: conservative global gate (preserves pages it already handled).
    g = _global_column_gutters(words, bbox)
    if g:
        return g
    # Pass 2: XY-cut fallback for spanning-header 'no_gutter' pages.
    x0, x1 = bbox[0], bbox[2]
    page_w = x1 - x0
    if page_w <= 0 or len(words) < 2 * COL_MIN_LINES:
        return []
    h_gap = COL_XYCUT_HGAP_FONT_MULT * _median_line_height(words)
    regions = []
    _xycut_regions(words, x0, page_w, h_gap, regions)
    if not regions:
        return []
    # the dominant (largest) multi-column region drives the page split
    _, gutters = max(regions, key=lambda t: len(t[0]))
    if _count_prose_columns(words, gutters) >= 2 and _columns_low_tabular(
        words, gutters, COL_STRICT_TABULAR_FRACTION
    ):
        return gutters
    return []


def _column_index(cx, splits):
    """Index of the column an x-centre falls into, given gutter mid-points."""
    return sum(1 for s in splits if cx >= s)


def _tabular_row_fraction(lines):
    """Fraction of a column's lines that are 'tabular' -- split into >= 3 segments
    (>= 2 internal column gaps), i.e. table/grid rows rather than prose."""
    if not lines:
        return 0.0
    tabular = sum(1 for ln in lines if len(_split_line_into_segments(ln)) >= 3)
    return tabular / len(lines)


def _count_prose_columns(words, gutters):
    """Number of columns that read like flowing prose: >= COL_MIN_LINES lines,
    median words/line >= COL_MIN_MEDIAN_WORDS_PER_LINE, and not table-heavy
    (tabular-row fraction < COL_MAX_TABULAR_ROW_FRACTION).  A column containing a
    table is rejected so the split never disturbs in-column tables/lists."""
    splits = [(xa + xb) / 2 for xa, xb in gutters]
    col_words = [[] for _ in range(len(gutters) + 1)]
    for w in words:
        if any(w["x0"] < xb and w["x1"] > xa for xa, xb in gutters):
            continue  # word sits in a gutter band (full-width element)
        cx = (w["x0"] + w["x1"]) / 2
        col_words[_column_index(cx, splits)].append(w)
    prose = 0
    for cw in col_words:
        if not cw:
            continue
        lines = _group_words_into_lines(cw)
        if len(lines) < COL_MIN_LINES:
            continue
        if statistics.median(len(ln) for ln in lines) < COL_MIN_MEDIAN_WORDS_PER_LINE:
            continue
        if _tabular_row_fraction(lines) >= COL_MAX_TABULAR_ROW_FRACTION:
            continue  # column contains a table -> not safe to split
        prose += 1
    return prose


def _group_words_into_lines_columns(words, bbox, gutters=None):
    """Column-aware variant of :func:`_group_words_into_lines`.

    Returns the same shape (a list of word-rows, each sorted by x0).  When no
    confident multi-column layout is detected the result is *identical* to
    :func:`_group_words_into_lines` (single-column / table pages are byte-for-byte
    unchanged).  Otherwise the page is partitioned into columns: a row whose text
    crosses a gutter is emitted full-width (flushing the buffered columns in
    reading order first); other rows are routed to their column and each column is
    grouped independently.

    *gutters* may be supplied by the caller (``_render_page`` computes them once
    to also emit the ``data-columns`` reading-order signal); when ``None`` they
    are detected here.
    """
    if gutters is None:
        gutters = _detect_column_gutters(words, bbox)
    if not gutters:
        return _group_words_into_lines(words)

    splits = [(xa + xb) / 2 for xa, xb in gutters]
    col_bufs = [[] for _ in range(len(gutters) + 1)]
    out = []

    def flush():
        for buf in col_bufs:
            if buf:
                out.extend(_group_words_into_lines(buf))
                buf.clear()

    for r in sorted(_group_words_into_lines(words), key=lambda ln: ln[0]["top"]):
        spanning = any(
            any(w["x0"] < xb and w["x1"] > xa for w in r) for xa, xb in gutters
        )
        if spanning:
            flush()
            out.append(sorted(r, key=lambda w: w["x0"]))
        else:
            for w in r:
                col_bufs[_column_index((w["x0"] + w["x1"]) / 2, splits)].append(w)
    flush()
    return out


def _cell_is_form_labeled(cell_words):
    """True when a grid cell reads like a form label: its first or second
    word, or its last word, ends with ":" ("By:", "M& C: N/A", "APPROVAL
    RECOMMENDED:").  Punctuation format only -- never word content."""
    if not cell_words:
        return False
    if cell_words[0]["text"].endswith(":"):
        return True
    if len(cell_words) > 1 and cell_words[1]["text"].endswith(":"):
        return True
    return cell_words[-1]["text"].endswith(":")


def _grid_trim_zone(rows, xa, xb, budget, min_gw):
    """Trim a quiet zone past its crossing rows so they become routable.

    A quiet zone found by the (tolerant) projection often extends past a
    column's ragged edge, turning a routable row ("Title: Vice President of
    Operations" ending just inside the zone) into a crossing row that would be
    flushed full-width and stay welded.  So: while any row crosses ``[xa,
    xb]``, exclude the cheapest one -- raise ``xa`` past a left intrusion or
    lower ``xb`` past a right one, keeping the widest surviving gutter.  A
    genuinely spanning row (words densely across the zone) cannot be trimmed
    away without falling under *min_gw*; up to *budget* such rows are accepted
    (they flush full-width), more mean this is no grid.  Returns ``(xa, xb)``
    or ``None``.
    """

    def crossers(a, b):
        return [r for r in rows if any(w["x0"] < b and w["x1"] > a for w in r)]

    cs = crossers(xa, xb)
    while cs:
        best = None  # (width, new_xa, new_xb)
        for r in cs:
            in_l = [w["x1"] for w in r if xa < w["x1"] < xb]
            if in_l:  # push xa right past this row's intrusion
                cand = (xb - max(in_l), max(in_l), xb)
                if best is None or cand[0] > best[0]:
                    best = cand
            in_r = [w["x0"] for w in r if xa < w["x0"] < xb]
            if in_r:  # pull xb left past this row's intrusion
                cand = (min(in_r) - xa, xa, min(in_r))
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None or best[0] < min_gw:
            break  # only untrimmable (spanning) rows remain
        xa, xb = best[1], best[2]
        cs = crossers(xa, xb)
    if len(cs) > budget:
        return None
    # no rounding: a trimmed edge sits exactly on a word boundary, and rounding
    # it inward would re-flag that word's row as crossing
    return (xa, xb)


def _grid_band_gutter(rows, x0, page_w):
    """The single real gutter of a two-column grid band, or ``None``.

    Same row-crossing x-projection as the column detectors, but gated for
    *grids* (independent cell stacks) rather than prose: quiet zones are found
    at the tolerant ``GRID_ZONE_CROSS_FRACTION``, trimmed to the strict
    ``GRID_MAX_CROSS_FRACTION`` budget, and a zone is real only when >=
    ``GRID_MIN_SIDE_ROWS`` rows sit entirely on each side of it.  Exactly ONE
    real gutter may exist (a data table has 2+), and the two columns must be
    prose-wide and balanced (a TOC's page-number column or a label|value
    table's label column fails).
    """
    n = len(rows)
    if n < GRID_MIN_ROWS:
        return None
    words = [w for r in rows for w in r]
    nbins = COL_PROJECTION_BINS
    binw = page_w / nbins
    rowcross = [0] * nbins
    for r in rows:
        covered = set()
        for w in r:
            b0 = max(0, int((w["x0"] - x0) / binw))
            b1 = min(nbins - 1, int((w["x1"] - x0) / binw))
            covered.update(range(b0, b1 + 1))
        for b in covered:
            rowcross[b] += 1
    wmin = min(w["x0"] for w in words)
    wmax = max(w["x1"] for w in words)
    zone_thr = max(1, int(GRID_ZONE_CROSS_FRACTION * n))
    budget = int(GRID_MAX_CROSS_FRACTION * n)
    min_gw = COL_XYCUT_GUTTER_MIN_WIDTH_FRACTION * page_w
    margin = COL_MARGIN_FRACTION * page_w
    zones = []
    i = 0
    while i < nbins:
        if rowcross[i] <= zone_thr:
            j = i
            while j < nbins and rowcross[j] <= zone_thr:
                j += 1
            xa = x0 + i * binw
            xb = x0 + j * binw
            if (xb - xa) >= min_gw and xa > wmin + margin and xb < wmax - margin:
                zones.append((xa, xb))
            i = j
        else:
            i += 1
    min_cw = COL_MIN_WIDTH_FRACTION * page_w
    real = []
    for xa, xb in zones:
        trimmed = _grid_trim_zone(rows, xa, xb, budget, min_gw)
        if trimmed is None:
            continue
        xa, xb = trimmed
        # Column width / balance gates apply PER ZONE: a sliver inside a
        # column's own text (word gaps aligning across rows, an OCR'd seal
        # beside the true gutter) splits the band lopsidedly and dies here,
        # and a 3-column table dies because each of its gutters splits ~1:2.
        # Only a genuine two-stack boundary survives.
        lw, rw = xa - wmin, wmax - xb
        if lw < min_cw or rw < min_cw:
            continue
        if max(lw, rw) / max(min(lw, rw), 1.0) > COL_MAX_WIDTH_IMBALANCE:
            continue
        mid = (xa + xb) / 2.0
        left = right = 0
        for r in rows:
            if any(w["x0"] < xb and w["x1"] > xa for w in r):
                continue
            if any((w["x0"] + w["x1"]) / 2.0 <= mid for w in r):
                left += 1
            if any((w["x0"] + w["x1"]) / 2.0 > mid for w in r):
                right += 1
        if left >= GRID_MIN_SIDE_ROWS and right >= GRID_MIN_SIDE_ROWS:
            real.append((xa, xb))
    if len(real) != 1:
        return None
    xa, xb = real[0]
    # Form gate: the region must carry colon-terminated label cells on both
    # sides of the gutter, densely, and its cells must be sparse (see the
    # GRID_*FORM*/GRID_MAX_MEDIAN_CELL_WORDS rationale above) -- or it is a
    # generic 2-column region (board-bio grid, statistical annex, 2-column
    # paper) this splitter must not touch.
    mid = (xa + xb) / 2.0
    left_form = right_form = 0
    cell_words = []
    for r in rows:
        if any(w["x0"] < xb and w["x1"] > xa for w in r):
            continue
        lcell = [w for w in r if (w["x0"] + w["x1"]) / 2.0 <= mid]
        rcell = [w for w in r if (w["x0"] + w["x1"]) / 2.0 > mid]
        if _cell_is_form_labeled(lcell):
            left_form += 1
        if _cell_is_form_labeled(rcell):
            right_form += 1
        cell_words.extend(len(c) for c in (lcell, rcell) if c)
    if left_form < GRID_MIN_FORM_LABELS_PER_SIDE:
        return None
    if right_form < GRID_MIN_FORM_LABELS_PER_SIDE:
        return None
    if (left_form + right_form) / n < GRID_MIN_FORM_LABEL_FRACTION:
        return None
    if cell_words and statistics.median(cell_words) > GRID_MAX_MEDIAN_CELL_WORDS:
        return None
    return (xa, xb)


def _detect_grid_regions(lines, bbox):
    """Detect two-column *grid* regions (signature / approval / form-field
    stacks) in a row-major line stream.

    Bands the rows at major vertical gaps (the XY-cut peel, index-preserving)
    and keeps each band whose rows share exactly one real gutter per
    :func:`_grid_band_gutter`.  Returns ``[(start, end, (xa, xb))]`` over line
    indices, empty for prose / table / TOC pages (stream then unchanged).
    """
    if not lines or bbox is None:
        return []
    x0, x1 = bbox[0], bbox[2]
    page_w = x1 - x0
    if page_w <= 0:
        return []
    all_words = [w for ln in lines for w in ln]
    # Off-page words (a rotated sidebar / infographic bleeding past the media
    # box) mean the page's geometry cannot be trusted for gutter routing.
    if any(w["x0"] < x0 - 2.0 or w["x1"] > x1 + 2.0 for w in all_words):
        return []
    tops = [min(w["top"] for w in ln) for ln in lines]
    pitches = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0.5]
    pitch = statistics.median(pitches) if pitches else 0.0
    h_gap = max(
        GRID_HGAP_FONT_MULT * _median_line_height(all_words),
        GRID_HGAP_PITCH_MULT * pitch,
    )
    bands = []
    start = 0
    prev_bottom = max(w["bottom"] for w in lines[0])
    for i in range(1, len(lines)):
        top = min(w["top"] for w in lines[i])
        if top - prev_bottom > h_gap:
            bands.append((start, i))
            start = i
        prev_bottom = max(prev_bottom, max(w["bottom"] for w in lines[i]))
    bands.append((start, len(lines)))

    regions = []
    for s, e in bands:
        gutter = _grid_band_gutter(lines[s:e], x0, page_w)
        if gutter:
            regions.append((s, e, gutter))
    return regions


def _apply_grid_regions(lines, regions):
    """Route each grid region's rows column-major: all left cells (top to
    bottom), then all right cells, so the engine forms per-cell / per-stack
    blocks instead of welding same-top cells.  A row that crosses the gutter
    flushes the buffered columns and is emitted full-width in place; rows
    outside every region pass through unchanged.  With no regions the stream
    is returned as-is (byte-identical page).
    """
    if not regions:
        return lines
    out = []
    pos = 0
    for start, end, (xa, xb) in regions:
        out.extend(lines[pos:start])
        mid = (xa + xb) / 2.0
        left_buf, right_buf = [], []

        def flush():
            out.extend(left_buf)
            out.extend(right_buf)
            del left_buf[:], right_buf[:]

        for ln in lines[start:end]:
            if any(w["x0"] < xb and w["x1"] > xa for w in ln):
                flush()
                out.append(ln)
                continue
            left = [w for w in ln if (w["x0"] + w["x1"]) / 2.0 <= mid]
            right = [w for w in ln if (w["x0"] + w["x1"]) / 2.0 > mid]
            if left:
                left_buf.append(left)
            if right:
                right_buf.append(right)
        flush()
        pos = end
    out.extend(lines[pos:])
    return out


def _split_line_into_segments(line_words):
    """Split a visual line into segments wherever a large horizontal gap appears
    (mirrors Tika emitting separate <p> tags for separate columns)."""
    if not line_words:
        return []
    segments = []
    current = [line_words[0]]
    for prev, w in zip(line_words, line_words[1:]):
        size = w.get("size") or 10.0
        gap = w["x0"] - prev["x1"]
        if gap > SEGMENT_GAP_FONT_MULTIPLE * size:
            segments.append(current)
            current = [w]
        else:
            current.append(w)
    segments.append(current)
    return segments


def _merge_word_fragments(words):
    """Re-join word fragments that pdfplumber split *only* because of a font /
    size attribute change, not because of a real space.

    ``page.extract_words(..., extra_attrs=["fontname", "size"])`` begins a new
    word on every change of an extra attribute, so a single display word whose
    glyphs carry slightly different subset font names (``ABCDEF+`` vs ``GHIJKL+``)
    or sub-pixel sizes is shattered into several touching fragments.  Joining
    those with spaces garbles text (``"About"`` -> ``"A bou t"``).

    Genuine inter-word spaces measure ~0.20-0.28*size (a space glyph) regardless
    of the absolute font size, whereas these attribute-split fragments *abut*
    (gap ~= 0).  So two adjacent fragments whose gap is at most
    ``FRAGMENT_MERGE_GAP_FONT_FRACTION * size`` are a single display word split by
    an attribute boundary; we re-join exactly those, reconstructing the true word
    boundaries while keeping each merged word's representative (first-fragment)
    font.  The threshold is font-relative (not the absolute ``WORD_X_TOLERANCE``)
    so it does not glue real words in small-font, tightly-set text where a real
    space is only ~1.2pt.
    """
    if not words:
        return []
    merged = [dict(words[0])]
    for w in words[1:]:
        prev = merged[-1]
        gap = w["x0"] - prev["x1"]
        # Use the smaller of the two fonts so the threshold stays conservative
        # (smaller threshold -> less merging -> safer against gluing real words).
        size = min(prev.get("size") or 0.0, w.get("size") or 0.0) or (
            prev.get("size") or w.get("size") or 10.0
        )
        if gap <= FRAGMENT_MERGE_GAP_FONT_FRACTION * size:
            # Same display word: concatenate text and extend the bounding box.
            # Keep prev's fontname/size as the word's representative font.
            prev["text"] = prev["text"] + w["text"]
            prev["x1"] = w["x1"]
            prev["top"] = min(prev["top"], w["top"])
            prev["bottom"] = max(prev["bottom"], w["bottom"])
        else:
            merged.append(dict(w))
    return merged


def _render_p(segment):
    """Render one line-segment of words into a Tika-format <p> tag string."""
    words = _merge_word_fragments(segment)
    # representative line metrics
    tops = [w["top"] for w in words]
    sizes = [w.get("size") or 10.0 for w in words]
    line_top = _round(min(tops))
    # use the most common / max size as the line font-size (matches Tika's single
    # font-size per <p>)
    line_size = _round(max(sizes), 1)
    heights = [(w["bottom"] - w["top"]) for w in words]
    line_height = _round(max(heights) if heights else line_size, 3)

    first = words[0]
    line_family = _clean_font_family(first.get("fontname"))
    line_style = _font_style(first.get("fontname"))
    line_weight = _font_weight(first.get("fontname"))
    left = _round(first["x0"])

    starts = []
    ends = []
    fonts = []
    texts = []
    for w in words:
        wy = _round(w["top"])
        starts.append(f"({_round(w['x0'])},{wy})")
        ends.append(f"({_round(w['x1'])},{wy})")
        wsize = _round(w.get("size") or line_size, 1)
        wfam = _clean_font_family(w.get("fontname"))
        wweight = _font_weight(w.get("fontname"))
        wstyle = _font_style(w.get("fontname"))
        space_w = _round(wsize / 4.0, 2)
        fonts.append(f"({wfam},{wweight},{wstyle},{wsize},{wsize},{space_w})")
        texts.append(w["text"])

    text = " ".join(texts)
    style = (
        f"height:{line_height};margin-top: 0px;"
        f"font-size:{line_size}px;font-family:{line_family};"
        f"font-style:{line_style};font-weight:{line_weight};"
        f"top:{line_top}px;position:absolute;text-indent:{left}px;"
        f"word-start-positions:[{', '.join(starts)}];"
        f"word-end-positions:[{', '.join(ends)}];"
        f"word-fonts:[{', '.join(fonts)}]"
    )
    return f'<p style="{style}">{html.escape(text)}</p>'


def _iter_layout_leaves(objs):
    """Depth-first walk of a pdfminer layout tree yielding leaf objects."""
    for o in objs:
        if isinstance(o, LTContainer):
            yield from _iter_layout_leaves(o)
        else:
            yield o


def _fast_page_objects(page):
    """Read chars + rule lines/rects straight off pdfminer's layout tree.

    This reproduces, field-for-field, the subset of pdfplumber's ``page.chars`` /
    ``page.lines`` / ``page.rects`` that the XHTML contract needs (text + geometry
    + font), but skips pdfplumber's per-object ``process_object`` machinery
    (``resolve_all`` over unused colour/colourspace attributes, unicode
    normalization, graphicstate colour decoding).  The ``top``/``bottom``
    mediabox formula is pdfplumber's own, so the resulting word boxes are
    byte-identical to ``page.extract_words(...)`` -- only faster (≈1.8x on
    text-dense pages, where ``process_object`` is ~45% of extraction cost).

    Returns ``(char_dicts, line_dicts, rect_dicts)``.
    """
    height = page.height
    mb_x0, mb_top = page.mediabox[:2]
    initial_doctop = page.initial_doctop
    chars = []
    lines = []
    rects = []
    for o in _iter_layout_leaves(page.layout._objs):
        cls = o.__class__
        if cls is LTChar:
            top = (height - o.y1) + mb_top
            chars.append(
                {
                    "text": o.get_text(),
                    "x0": o.x0 + mb_x0,
                    "x1": o.x1 + mb_x0,
                    "top": top,
                    "bottom": (height - o.y0) + mb_top,
                    "doctop": initial_doctop + top,
                    "upright": o.upright,
                    "size": o.size,
                    "fontname": o.fontname,
                }
            )
        elif cls is LTLine:
            lines.append(
                {
                    "x0": o.x0 + mb_x0,
                    "x1": o.x1 + mb_x0,
                    "top": (height - o.y1) + mb_top,
                    "bottom": (height - o.y0) + mb_top,
                }
            )
        elif cls is LTRect:
            rects.append(
                {
                    "x0": o.x0 + mb_x0,
                    "x1": o.x1 + mb_x0,
                    "top": (height - o.y1) + mb_top,
                    "bottom": (height - o.y0) + mb_top,
                }
            )
    return chars, lines, rects


def _words_from_chars(chars):
    """Run pdfplumber's WordExtractor over pre-built char dicts (same params as
    ``page.extract_words`` used elsewhere) -> byte-identical word list."""
    return _pp_extract_words(
        chars,
        x_tolerance=WORD_X_TOLERANCE,
        y_tolerance=WORD_Y_TOLERANCE,
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["fontname", "size"],
    )


def _render_svg_from_paths(lines, rects, page_width, page_height):
    """Build the table-detection <svg> from pre-extracted line/rect dicts.

    Byte-identical to :func:`_render_svg` but consumes the dicts produced by
    :func:`_fast_page_objects` so the page's objects are walked only once."""
    parts = []
    for ln in lines:
        try:
            x1 = _round(ln["x0"])
            x2 = _round(ln["x1"])
            y1 = _round(ln["top"])
            y2 = _round(ln["bottom"])
        except Exception:
            continue
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'style="stroke:rgb(0,0,0);stroke-width:1"/>'
        )
    for rc in rects:
        try:
            x = _round(rc["x0"])
            y = _round(rc["top"])
            w = _round(rc["x1"] - rc["x0"])
            h = _round(rc["bottom"] - rc["top"])
        except Exception:
            continue
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'style="fill:none;stroke:rgb(0,0,0);stroke-width:1"/>'
        )
    if not parts:
        return ""
    return (
        f'<svg width="{_round(page_width)}" height="{_round(page_height)}">'
        f'{"".join(parts)}</svg>'
    )


def _render_svg(page):
    """Emit page rule lines / rectangles as an <svg> for table detection."""
    parts = []
    for ln in page.lines:
        try:
            x1 = _round(ln["x0"])
            x2 = _round(ln["x1"])
            y1 = _round(ln["top"])
            y2 = _round(ln["bottom"])
        except Exception:
            continue
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'style="stroke:rgb(0,0,0);stroke-width:1"/>'
        )
    for rc in page.rects:
        try:
            x = _round(rc["x0"])
            y = _round(rc["top"])
            w = _round(rc["x1"] - rc["x0"])
            h = _round(rc["bottom"] - rc["top"])
        except Exception:
            continue
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'style="fill:none;stroke:rgb(0,0,0);stroke-width:1"/>'
        )
    if not parts:
        return ""
    return (
        f'<svg width="{_round(page.width)}" height="{_round(page.height)}">'
        f'{"".join(parts)}</svg>'
    )


def _lines_to_p_tags(lines):
    p_tags = []
    for line_words in lines:
        for seg in _split_line_into_segments(line_words):
            if not seg:
                continue
            if "".join(w["text"] for w in seg).strip() == "":
                continue
            p_tags.append(_render_p(seg))
    return p_tags


def _should_route_to_ocr(words, lines, force_ocr=False):
    """True when a page should use OCR instead of its embedded text layer."""
    return force_ocr or (
        len(lines) < MIN_TEXT_LINES_FOR_PAGE and len(words) < MIN_TEXT_WORDS_FOR_PAGE
    )


def _render_page(page, force_ocr=False, disable_ocr=False):
    # Fast path: read words + rule lines/rects directly off pdfminer's layout
    # tree (byte-identical to the pdfplumber calls below, ~1.8x faster on
    # text-dense pages).  Fall back to pdfplumber if anything unexpected arises
    # so a single odd page can never lose its text.
    try:
        chars, raw_lines, raw_rects = _fast_page_objects(page)
        words = _words_from_chars(chars)
        svg = _render_svg_from_paths(raw_lines, raw_rects, page.width, page.height)
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning("fast extraction failed (%s); using pdfplumber path", e)
        try:
            words = page.extract_words(
                x_tolerance=WORD_X_TOLERANCE,
                y_tolerance=WORD_Y_TOLERANCE,
                keep_blank_chars=False,
                use_text_flow=False,
                extra_attrs=["fontname", "size"],
            )
            svg = _render_svg(page)
        except Exception as e2:  # pragma: no cover - defensive fallback
            # The embedded text layer is unreadable (e.g. a malformed
            # ICC-colorspace stream that crashes pdfminer's layout analysis).
            # Rather than drop the whole page, treat it like a scanned page:
            # leave it text-less so the sparse-page detector below routes it to
            # OCR (which reads the rendered image and recovers the text).
            logger.warning("text extraction failed (%s); routing page to OCR", e2)
            words = []
            svg = ""
    lines = _group_words_into_lines(words)

    # Confident multi-column gutters (empty on single-column / table / OCR pages).
    # Computed once here so the same result drives both the column-aware line
    # grouping and the ``data-columns`` reading-order signal emitted on the page
    # div (which tells the downstream engine's order-fixer NOT to re-interleave
    # the already-column-ordered <p> stream).
    gutters = None

    # Route scanned / sparse pages to OCR when possible.  A page with a dense
    # embedded text layer keeps it even if its coordinates collapse to a few
    # visual lines; pages with little/no extractable text are OCR'd.
    # ``disable_ocr`` is the per-request counterpart of ``WARP_DISABLE_OCR``:
    # sparse pages keep their thin text layer, same as an absent OCR backend.
    use_ocr = not disable_ocr and _should_route_to_ocr(
        words, lines, force_ocr=force_ocr
    )
    if use_ocr:
        from warp_ingest.file_parser import ocr_parser

        if ocr_parser.ocr_available():
            ocr_lines = ocr_parser.ocr_page_lines(page)
            if ocr_lines:
                # Scanned signature/approval grids weld the same way as text
                # pages, so the grid-cell split applies here too (OCR word
                # boxes are origin-(0,0) PDF points).
                lines = _apply_grid_regions(
                    ocr_lines,
                    _detect_grid_regions(
                        ocr_lines, (0.0, 0.0, page.width, page.height)
                    ),
                )
        elif force_ocr:
            logger.warning(
                "OCR requested but rapidocr-onnxruntime is not installed; "
                "falling back to text layer."
            )
    else:
        # Column-aware grouping for genuine multi-column prose; byte-identical to
        # the plain grouping above for single-column / table / infographic pages.
        gutters = _detect_column_gutters(words, page.bbox)
        lines = _group_words_into_lines_columns(words, page.bbox, gutters=gutters)
        if not gutters:
            # Table-safe grid-cell split (two-column signature/approval/form
            # grids the prose gates exclude): route each grid region's cells
            # column-major so left/right cells stop welding into one block.
            # No-op (stream unchanged) on pages with no grid region.
            lines = _apply_grid_regions(lines, _detect_grid_regions(lines, page.bbox))

    p_tags = _lines_to_p_tags(lines)
    w = _round(page.width)
    h = _round(page.height)
    style = f"height:{h}px; width:{w}px; position: relative;"
    # Reading-order signal: only present on confident multi-column pages (empty
    # on single-column / table / OCR pages -> XHTML byte-identical there).  Each
    # gutter band is "xa,xb"; bands are pipe-separated.
    col_attr = ""
    if gutters:
        bands = "|".join(f"{_round(xa,2)},{_round(xb,2)}" for xa, xb in gutters)
        col_attr = f' data-columns="{bands}"'
    return f'<div class="page"{col_attr} style="{style}">{svg}{"".join(p_tags)}</div>'


# ---------------------------------------------------------------------------
# Parallel front-end.
#
# Page rendering is embarrassingly parallel: :func:`_render_page` reads only its
# own page's objects and emits one self-contained ``<div>`` string, so a
# document's pages can be striped across a small process pool with the parent
# reassembling the divs in page order -- the XHTML is byte-identical to the
# serial loop (same code, same per-page inputs), only the wall-clock changes.
# Each worker opens the PDF independently (pdfminer objects don't pickle);
# opening is ~1ms/page, tiny next to layout analysis.  The pool uses the
# "spawn" start method -- safe under the threaded daemon workers and never
# inherits parent state (e.g. an onnxruntime OCR session) via fork -- and is
# kept alive across documents so its startup cost is paid once per process.
#
# ``WARP_FE_WORKERS`` overrides the worker count (<=1 disables parallelism);
# ``WARP_FE_PARALLEL_MIN_PAGES`` (default 8) keeps short documents on the
# serial path, where pool startup would dominate any gain.
# ---------------------------------------------------------------------------
_FE_POOL = None
_FE_POOL_WORKERS = 0
_FE_POOL_PID = None
# Guards pool creation/teardown: concurrent daemon threads must not race two
# pools into existence (the loser's spawned workers would leak).
_FE_POOL_LOCK = threading.Lock()


def _fe_worker_count():
    env = os.environ.get("WARP_FE_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            logger.warning("invalid WARP_FE_WORKERS=%r; using default", env)
    return min(8, os.cpu_count() or 1)


def _fe_parallel_min_pages():
    env = os.environ.get("WARP_FE_PARALLEL_MIN_PAGES")
    if env is not None:
        try:
            return max(2, int(env))
        except ValueError:
            logger.warning("invalid WARP_FE_PARALLEL_MIN_PAGES=%r; using default", env)
    return 8


def _drop_fe_pool_locked():
    """Discard the cached pool (it broke, or the worker count changed).
    Caller must hold ``_FE_POOL_LOCK``."""
    global _FE_POOL, _FE_POOL_WORKERS, _FE_POOL_PID
    if _FE_POOL is not None and _FE_POOL_PID == os.getpid():
        _FE_POOL.shutdown(wait=False, cancel_futures=True)
    _FE_POOL = None
    _FE_POOL_WORKERS = 0
    _FE_POOL_PID = None


def _drop_fe_pool():
    with _FE_POOL_LOCK:
        _drop_fe_pool_locked()


def _get_fe_pool(workers):
    """Lazily create (and cache) the front-end worker pool.

    Recreated if the requested worker count changes or the process forked (a
    pool's worker handles are not valid across a fork)."""
    global _FE_POOL, _FE_POOL_WORKERS, _FE_POOL_PID
    with _FE_POOL_LOCK:
        if (
            _FE_POOL is None
            or _FE_POOL_WORKERS != workers
            or _FE_POOL_PID != os.getpid()
        ):
            _drop_fe_pool_locked()
            _FE_POOL = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            _FE_POOL_WORKERS = workers
            _FE_POOL_PID = os.getpid()
        return _FE_POOL


def _render_page_range(filepath, password, page_indices, force_ocr, disable_ocr=False):
    """Worker: render *page_indices* of *filepath* to ``(index, div)`` pairs.

    Runs in a spawned worker process.  Mirrors the serial loop in
    :func:`pdf_to_xhtml` exactly -- including the per-page empty-div fallback --
    so the assembled document is byte-identical to a serial parse."""
    divs = []
    with pdfplumber.open(filepath, password=password or "") as pdf:
        for i in page_indices:
            page = pdf.pages[i]
            try:
                div = _render_page(page, force_ocr=force_ocr, disable_ocr=disable_ocr)
            except Exception as e:  # never let one bad page kill the document
                logger.warning("pdfplumber failed on a page: %s", e)
                w = _round(page.width)
                h = _round(page.height)
                div = (
                    f'<div class="page" style="height:{h}px; width:{w}px; '
                    f'position: relative;"></div>'
                )
            divs.append((i, div))
            # Drop this page's cached chars/layout so a worker's peak memory is
            # one page, not its whole stripe (large S-1 bodies run to GBs).
            page.flush_cache()
    return divs


def _pdf_to_page_divs_parallel(
    filepath, password, force_ocr, n_pages, workers, disable_ocr=False
):
    """Render all pages via the worker pool; returns divs in page order.

    Pages are striped (worker *k* takes pages ``k, k+W, k+2W, ...``) so dense
    and sparse pages spread evenly across workers."""
    stripes = [list(range(k, n_pages, workers)) for k in range(workers)]
    pool = _get_fe_pool(workers)
    futures = [
        pool.submit(
            _render_page_range, filepath, password, stripe, force_ocr, disable_ocr
        )
        for stripe in stripes
        if stripe
    ]
    page_divs = [None] * n_pages
    for fut in futures:
        for i, div in fut.result():
            page_divs[i] = div
    return page_divs


def pdf_to_xhtml(filepath, password=None, force_ocr=False, disable_ocr=False):
    """Convert a PDF to Tika-compatible XHTML.

    Pages with a usable text layer are parsed directly; scanned/sparse pages are
    transparently routed to OCR (when the optional OCR engine is available).  Set
    ``force_ocr=True`` to OCR every page, or ``disable_ocr=True`` to keep every
    page on its embedded text layer (a per-request ``WARP_DISABLE_OCR``; it is
    passed to the pool workers as an argument because the persistent spawn pool
    inherits the environment only once, at spawn).
    """
    page_divs = None
    workers = _fe_worker_count()
    # Never nest pools: inside a multiprocessing child (our own front-end
    # worker, or a caller's job process without a __main__ guard) stay serial.
    if workers > 1 and multiprocessing.parent_process() is None:
        try:
            with pdfplumber.open(filepath, password=password or "") as pdf:
                n_pages = len(pdf.pages)
            if n_pages >= _fe_parallel_min_pages():
                page_divs = _pdf_to_page_divs_parallel(
                    filepath, password, force_ocr, n_pages, workers, disable_ocr
                )
        except Exception as e:
            # Any pool-level failure (spawn, pickling, a worker dying) falls
            # back to the serial path, which reproduces today's behavior --
            # including re-raising document-level errors from pdfplumber.open.
            # Drop the cached pool: a BrokenProcessPool is permanently unusable,
            # and the next document deserves a fresh one.
            logger.warning("parallel front-end failed (%s); using serial path", e)
            _drop_fe_pool()
            page_divs = None
    if page_divs is None:
        page_divs = []
        with pdfplumber.open(filepath, password=password or "") as pdf:
            for page in pdf.pages:
                try:
                    page_divs.append(
                        _render_page(page, force_ocr=force_ocr, disable_ocr=disable_ocr)
                    )
                except Exception as e:  # never let one bad page kill the document
                    logger.warning("pdfplumber failed on a page: %s", e)
                    w = _round(page.width)
                    h = _round(page.height)
                    page_divs.append(
                        f'<div class="page" style="height:{h}px; width:{w}px; '
                        f'position: relative;"></div>'
                    )
    body = "".join(page_divs)
    return (
        '<?xml version="1.1" encoding="UTF-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head>"
        '<meta name="Content-Type" content="application/pdf"/>'
        "</head>"
        f"<body>{body}</body>"
        "</html>"
    )


def count_text_pages(filepath, password=None):
    """Return (n_pages, n_sparse_pages) where a sparse page has < 4 text lines.
    Used to decide whether OCR is needed."""
    n_pages = 0
    n_sparse = 0
    with pdfplumber.open(filepath, password=password or "") as pdf:
        for page in pdf.pages:
            n_pages += 1
            try:
                words = page.extract_words(
                    x_tolerance=WORD_X_TOLERANCE, y_tolerance=WORD_Y_TOLERANCE
                )
                lines = _group_words_into_lines(words)
            except Exception:
                lines = []
            if _should_route_to_ocr(words, lines):
                n_sparse += 1
    return n_pages, n_sparse


class PdfPlumberFileParser(FileParser):
    """Pure-Python replacement for ``TikaFileParser`` (PDF only)."""

    def parse_to_html(self, filepath, do_ocr=False, disable_ocr=False):
        content = pdf_to_xhtml(filepath, force_ocr=do_ocr, disable_ocr=disable_ocr)
        return {
            "content": content,
            "metadata": {"Content-Type": "application/pdf"},
            "status": 200,
        }

    def parse_to_clean_html(self, filepath):
        with open(filepath, "rb") as fh:
            head = fh.read(5)
        if head[:4] == b"%PDF":
            return self.parse_to_html(filepath)
        with open(filepath, errors="ignore") as fh:
            data = fh.read()
        return {"metadata": {}, "content": data, "status": 200}
