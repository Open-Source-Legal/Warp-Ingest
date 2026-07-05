"""Front-end grid-cell splitter (two-column form/signature grids).

Pure-unit tests on synthetic pdfplumber-style word dicts pinning the two new
front-end primitives (the table-safe counterpart of the PR #17 prose XY-cut):

  * ``_detect_grid_regions(lines, bbox)`` — banded, geometry-only detection of
    two-column *grid* regions (signature/approval blocks, address|quote-field
    stacks): a run of rows sharing exactly ONE wide interior gutter, with both
    sides populated and (almost) no row crossing it.  Prose pages, data tables
    (2+ real gutters), TOC-like wide|narrow layouts, and label|value tables
    (one-sided or narrow columns) never qualify.
  * ``_apply_grid_regions(lines, regions)`` — routes each region's rows
    column-major (all left cells top-to-bottom, then all right cells) so the
    downstream engine, which welds same-top stream-adjacent segments into one
    block, forms per-cell/per-stack blocks instead ("By:  By:" fusion fix).
    Rows outside a region — and crossing rows inside one (which flush the
    buffered columns first) — pass through unchanged.

The invariant protecting the Tika XHTML contract: when no grid region is
detected the line stream is returned unchanged (byte-identical pages).
"""

from warp_ingest.file_parser import pdf_plumber_parser as P

BBOX = (0.0, 0.0, 612.0, 792.0)  # x0, top, x1, bottom
PAGE_W = 612.0


def W(text, x0, top, size=10.0, font="Times", x1=None):
    """A minimal pdfplumber-style word dict (fields the parser uses)."""
    return {
        "text": text,
        "x0": float(x0),
        "x1": float(x1 if x1 is not None else x0 + len(text) * size * 0.5),
        "top": float(top),
        "bottom": float(top + size),
        "size": float(size),
        "fontname": font,
    }


def row(tokens, x_start, top, size=10.0, gap=4.0):
    out = []
    x = x_start
    for t in tokens:
        w = W(t, x, top, size)
        out.append(w)
        x = w["x1"] + gap
    return out


def _texts(lines):
    return [" ".join(w["text"] for w in ln) for ln in lines]


def sig_grid_lines(n_rows=6, y0=100.0, dy=18.0, left_x=80.0, right_x=320.0):
    """A synthetic two-party signature grid: mirrored short label cells at two
    x-positions (left cells span ~[80,240], right ~[320,480]; gutter ~[240,320]).
    Modeled on fw_wert p1 / fw_vertigis p2 geometry (612pt page)."""
    labels = [
        ["CITYPARTY:"],
        ["By:", "someone", "signing"],
        ["Namefield:", "Person", "Name"],
        ["Titlefield:", "Assistant", "Manager"],
        ["Datefield:", "9/15/2025"],
        ["Extrafield:", "value", "here"],
    ]
    lines = []
    for i in range(n_rows):
        toks = labels[i % len(labels)]
        top = y0 + i * dy
        lines.append(row(toks, left_x, top) + row(toks, right_x, top))
    return lines


# --------------------------------------------------------------------------
# _detect_grid_regions
# --------------------------------------------------------------------------


def test_detect_grid_on_signature_grid():
    lines = sig_grid_lines()
    regions = P._detect_grid_regions(lines, BBOX)
    assert len(regions) == 1
    start, end, (xa, xb) = regions[0]
    assert (start, end) == (0, len(lines))
    # gutter sits in the empty band between the two cell stacks
    assert 200 <= xa < xb <= 321


def test_grid_region_scoped_to_band_after_prose():
    """A full-width prose paragraph, a big vertical gap, then the signature
    grid: only the grid band is returned as a region."""
    prose = []
    for i in range(4):
        prose.append(row(["word"] * 18, 60, 60.0 + i * 12))
    grid = sig_grid_lines(y0=200.0)
    regions = P._detect_grid_regions(prose + grid, BBOX)
    assert len(regions) == 1
    start, end, _ = regions[0]
    assert (start, end) == (len(prose), len(prose) + len(grid))


def test_no_grid_on_single_column_prose():
    lines = [row(["word"] * 18, 60, 100.0 + i * 12) for i in range(10)]
    assert P._detect_grid_regions(lines, BBOX) == []


def test_no_grid_on_three_column_table():
    """A 3-column data table (two real gutters, both sides populated) is
    row-associative — never split (the fw_vertigis renewal-table case)."""
    lines = []
    for i in range(8):
        top = 100.0 + i * 16
        lines.append(
            row(["RowLabel", "Number", str(i)], 80, top)
            + row(["October", "2,", "2023"], 290, top)
            + row(["October", "1,", "2024"], 470, top)
        )
    assert P._detect_grid_regions(lines, BBOX) == []


def test_no_grid_on_label_value_table():
    """cerebras-style definition table: a narrow label column with only a
    couple of labelled rows beside a wide prose value column -> one-sided
    (min-side-rows) and column-width gates reject it."""
    lines = []
    top = 100.0
    for label, n_body in (("Services", 4), ("Location", 4)):
        lines.append(row([label], 66, top) + row(["value"] * 8, 182, top))
        for _ in range(n_body):
            top += 16
            lines.append(row(["value"] * 8, 182, top))
        top += 16
    assert P._detect_grid_regions(lines, BBOX) == []


def test_no_grid_on_toc_wide_narrow():
    """TOC-like layout: wide title column | narrow page-number column.  The
    narrow side fails the minimum column width -> never split (page numbers
    must stay row-associated with their titles)."""
    lines = []
    for i in range(10):
        top = 100.0 + i * 14
        lines.append(
            row(["Section", "title", "words", "here", "for", "row"], 54, top)
            + row([str(100 + i)], 520, top)
        )
    assert P._detect_grid_regions(lines, BBOX) == []


def test_no_grid_when_too_many_crossing_rows():
    """A band where >10% of rows cross the candidate gutter (a data table
    bleeding into the band) is not a grid."""
    lines = sig_grid_lines(n_rows=6)
    # two full-width rows inside the band (tops interleaved, small gaps)
    for top in (109.0, 145.0):
        lines.append([W("x" * 60, 80, top, x1=500.0)])
    lines.sort(key=lambda ln: ln[0]["top"])
    assert P._detect_grid_regions(lines, BBOX) == []


def test_no_grid_on_short_band():
    """Fewer than GRID_MIN_ROWS rows never form a region."""
    lines = sig_grid_lines(n_rows=3)
    assert P._detect_grid_regions(lines, BBOX) == []


# --------------------------------------------------------------------------
# _apply_grid_regions
# --------------------------------------------------------------------------


def test_apply_routes_cells_column_major():
    lines = sig_grid_lines(n_rows=4)
    regions = [(0, 4, (250.0, 315.0))]
    out = P._apply_grid_regions(lines, regions)
    # every output line is entirely one cell (no cross-gutter fusion) ...
    for ln in out:
        assert max(w["x1"] for w in ln) <= 250 or min(w["x0"] for w in ln) >= 315
    # ... all left cells precede all right cells, tops ascending per column
    col_of = [0 if ln[0]["x0"] < 250 else 1 for ln in out]
    assert col_of == sorted(col_of)
    assert col_of.count(0) == 4 and col_of.count(1) == 4
    tops = [ln[0]["top"] for ln in out]
    assert tops[:4] == sorted(tops[:4]) and tops[4:] == sorted(tops[4:])


def test_apply_flushes_on_crossing_row():
    """A crossing row inside the region flushes the buffered columns and is
    emitted full-width in place."""
    lines = sig_grid_lines(n_rows=4)
    crossing = [W("FULLWIDTHHEADER", 80, 127.0, x1=500.0)]
    lines = lines[:2] + [crossing] + lines[2:]
    out = P._apply_grid_regions(lines, [(0, 5, (250.0, 315.0))])
    texts = _texts(out)
    k = texts.index("FULLWIDTHHEADER")
    # the two rows above the crossing line: left cells then right cells
    pre = out[:k]
    assert [ln[0]["x0"] < 250 for ln in pre] == [True, True, False, False]
    post = out[k + 1 :]
    assert [ln[0]["x0"] < 250 for ln in post] == [True, True, False, False]


def test_apply_leaves_out_of_region_lines_unchanged():
    prose = [row(["word"] * 18, 60, 40.0 + i * 12) for i in range(3)]
    grid = sig_grid_lines(n_rows=4, y0=200.0)
    lines = prose + grid
    out = P._apply_grid_regions(lines, [(3, 7, (250.0, 315.0))])
    assert out[:3] == prose
    assert len(out) == 3 + 8


def test_no_regions_returns_stream_unchanged():
    lines = [row(["word"] * 18, 60, 100.0 + i * 12) for i in range(5)]
    assert P._apply_grid_regions(lines, []) == lines


# --------------------------------------------------------------------------
# end-to-end: the fw_wert signature page un-welds
# --------------------------------------------------------------------------


def test_fw_wert_signature_cells_not_fused():
    """Parsing the FortWorth wert renewal (2 pages): the p1 signature grid's
    left/right cells must no longer weld into single cross-gutter blocks."""
    from warp_ingest.ingestor.pdf_ingestor import parse_blocks, parse_pdf

    xhtml = parse_pdf(
        "tests/fixtures/contracts/fw_wert_bookbinding.pdf", parse_options={}
    )
    blocks = parse_blocks(xhtml, render_format="all")[0]
    texts = [b["block_text"] for b in blocks if b["page_idx"] == 1]
    joined = "\n".join(texts)
    # the five welded blocks from the audit (goal statement) must be gone
    assert "By: By:" not in joined
    assert not any("Dana Burghdoff Gary L. Wert" in t for t in texts)
    assert not any("Assistant City Manager President" in t for t in texts)
    assert not any("CITY: VENDOR" in t for t in texts)
    assert not any("Date: Date:" in t for t in texts)
    # content is preserved (both parties' names still present)
    assert "Gary L. Wert" in joined and "Dana Burghdoff" in joined
