"""Unit tests for the license-clean native table engine
(``warp_ingest.ingestor.table_engine``).

The engine's core is pure functions over plain word/edge dicts, so most tests
need no PDF at all.  The ruled-path tests build a minimal one-page PDF (a 3x3
stroked grid with cell text) from scratch — no extra dependencies.
"""

from __future__ import annotations

import io
import zlib

import pytest

from warp_ingest.ingestor.table_engine import (
    Cell,
    PageTable,
    _cluster_rects,
    _extend_header_up,
    _looks_prose_grid,
    _looks_undersegmented,
    _merge_symbol_columns,
    _n_header_rows,
    extract_page_tables,
    infer_grid,
    infer_tables,
    propose_regions,
    render_table_html,
)

# ---------------------------------------------------------------------------
# Minimal PDF builder (valid xref) — enough for pdfplumber to see lines + text.
# ---------------------------------------------------------------------------


def _build_pdf(content: str, width: float = 612, height: float = 792) -> bytes:
    """Assemble a one-page PDF with a single content stream and Helvetica."""
    stream = zlib.compress(content.encode("latin-1"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".encode(),
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        ),
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(f"{i} 0 obj\n".encode())
        buf.write(obj)
        buf.write(b"\nendobj\n")
    xref_at = buf.tell()
    buf.write(f"xref\n0 {len(objects) + 1}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets:
        buf.write(f"{off:010d} 00000 n \n".encode())
    buf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF".encode()
    )
    return buf.getvalue()


@pytest.fixture(scope="module")
def ruled_grid_pdf(tmp_path_factory):
    """3 cols x 3 rows stroked grid; row 1 = header labels, rows 2-3 = data."""
    lines = []
    # verticals at x = 100, 170, 240, 310 spanning y 640..700
    for x in (100, 170, 240, 310):
        lines.append(f"{x} 640 m {x} 700 l S")
    # horizontals at y = 700, 680, 660, 640 spanning x 100..310
    for y in (700, 680, 660, 640):
        lines.append(f"100 {y} m 310 {y} l S")
    texts = [
        (105, 686, "Name"),
        (175, 686, "Qty"),
        (245, 686, "Price"),
        (105, 666, "foo"),
        (175, 666, "1"),
        (245, 666, "$2.00"),
        (105, 646, "bar"),
        (175, 646, "23"),
        (245, 646, "$4.50"),
    ]
    text_ops = [f"BT /F1 10 Tf {x} {y} Td ({s}) Tj ET" for x, y, s in texts]
    content = "0.5 w\n" + "\n".join(lines) + "\n" + "\n".join(text_ops)
    path = tmp_path_factory.mktemp("tblpdf") / "ruled.pdf"
    path.write_bytes(_build_pdf(content))
    return str(path)


# ---------------------------------------------------------------------------
# Helpers to make synthetic words
# ---------------------------------------------------------------------------


def w(x0, top, x1, bottom, text):
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom, "text": text}


def _rowline(top, cols, h=10.0):
    """cols = [(x0, x1, text), ...] all on one visual line."""
    return [w(x0, top, x1, top + h, t) for x0, x1, t in cols]


# ---------------------------------------------------------------------------
# render_table_html
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_header_and_body_tags(self):
        t = PageTable(
            bbox=(0, 0, 100, 100),
            grid=[
                [Cell("Name"), Cell("Qty")],
                [Cell("foo"), Cell("1")],
            ],
            n_header_rows=1,
        )
        html = render_table_html(t)
        assert "<th>Name</th><th>Qty</th>" in html
        assert "<td>foo</td><td>1</td>" in html
        assert html.startswith("<table>") and html.endswith("</table>")

    def test_escaping_and_colspan_and_empty(self):
        t = PageTable(
            bbox=(0, 0, 100, 100),
            grid=[
                [Cell("A & B", colspan=2)],
                [Cell(""), Cell("<x>")],
            ],
            n_header_rows=1,
        )
        html = render_table_html(t)
        assert '<th colspan="2">A &amp; B</th>' in html
        assert "<td></td><td>&lt;x&gt;</td>" in html

    def test_whitespace_collapsed(self):
        t = PageTable(
            bbox=(0, 0, 10, 10),
            grid=[[Cell("a\nb   c")], [Cell("d")]],
            n_header_rows=1,
        )
        assert "<th>a b c</th>" in render_table_html(t)


# ---------------------------------------------------------------------------
# _cluster_rects (envelope augmentation)
# ---------------------------------------------------------------------------


class TestClusterRects:
    def test_neighbors_merge(self):
        # two stacked row-band rects 2pt apart -> one cluster
        boxes = [(100, 100, 300, 110), (100, 112, 300, 122)]
        merged = _cluster_rects(boxes, tol=3.0)
        assert len(merged) == 1
        assert merged[0] == (100, 100, 300, 122)

    def test_far_apart_stay_separate(self):
        boxes = [(100, 100, 300, 110), (100, 400, 300, 410)]
        assert len(_cluster_rects(boxes, tol=3.0)) == 2


# ---------------------------------------------------------------------------
# infer_grid — region-local grid inference
# ---------------------------------------------------------------------------


class TestInferGrid:
    def _simple_table_words(self):
        words = []
        # header
        words += _rowline(
            100, [(50, 90, "Name"), (150, 175, "Qty"), (250, 290, "Price")]
        )
        # data rows, one per 20pt
        words += _rowline(120, [(50, 75, "foo"), (150, 160, "1"), (250, 285, "$2.00")])
        words += _rowline(140, [(50, 74, "bar"), (150, 165, "23"), (250, 285, "$4.50")])
        words += _rowline(
            160, [(50, 78, "baz"), (150, 168, "456"), (250, 285, "$9.99")]
        )
        return words

    def test_three_column_grid(self):
        words = self._simple_table_words()
        t = infer_grid(words, (45, 95, 295, 175))
        assert t is not None
        assert len(t.grid) == 4
        assert all(len(r) == 3 for r in t.grid)
        assert t.grid[0][0].text == "Name"
        assert t.grid[2][1].text == "23"
        assert t.grid[3][2].text == "$9.99"

    def test_wrapped_continuation_merges_into_row_above(self):
        words = self._simple_table_words()
        # a wrapped continuation of the col-0 description right under "bar"
        # (tight gap, only col 0 occupied)
        words += _rowline(150.5, [(50, 88, "(cont)")], h=8.0)
        t = infer_grid(words, (45, 95, 295, 175))
        assert t is not None
        assert len(t.grid) == 4  # no extra row
        assert "bar" in t.grid[2][0].text and "(cont)" in t.grid[2][0].text

    def test_fullwidth_caption_row_spans(self):
        words = self._simple_table_words()
        # a full-width caption line above the header
        words.insert(0, w(46, 80, 292, 90, "Consolidated results of operations"))
        t = infer_grid(words, (45, 75, 295, 175))
        assert t is not None
        cap = t.grid[0]
        assert len(cap) == 1 and cap[0].colspan == 3
        assert "Consolidated" in cap[0].text

    def test_prose_returns_none(self):
        # left-aligned ragged prose: no persistent whitespace channel
        words = []
        sentences = [
            "The quick brown fox jumps over the dog",
            "A completely different line of text here",
            "Short one",
            "And another line without any columns at",
        ]
        for i, s in enumerate(sentences):
            x = 50.0
            top = 100 + i * 14
            for tok in s.split():
                width = 6.0 * len(tok)
                words.append(w(x, top, x + width, top + 10, tok))
                x += width + 4.0
        assert infer_grid(words, (45, 95, 350, 165)) is None

    def test_too_few_rows_returns_none(self):
        words = _rowline(100, [(50, 90, "Name"), (150, 175, "Qty")])
        assert infer_grid(words, (45, 95, 200, 115)) is None

    def test_stacked_multiline_header_folds_into_one_row(self):
        words = []
        # "Exchange" / "Trading | on which" / "Title... | Symbol | registered"
        words += _rowline(100, [(300, 350, "Exchange")], h=9.0)
        words += _rowline(111, [(200, 240, "Trading"), (300, 345, "on which")], h=9.0)
        words += _rowline(
            122,
            [
                (50, 145, "Title of each class"),
                (200, 235, "Symbol"),
                (300, 350, "registered"),
            ],
            h=9.0,
        )
        words += _rowline(
            140, [(50, 120, "Common stock"), (200, 215, "GS"), (300, 330, "NYSE")]
        )
        words += _rowline(
            155, [(50, 110, "Notes 2029"), (200, 225, "GS29"), (300, 330, "NYSE")]
        )
        t = infer_grid(words, (45, 95, 355, 170))
        assert t is not None
        assert len(t.grid) == 3
        assert t.grid[0][1].text == "Trading Symbol"
        assert t.grid[0][2].text == "Exchange on which registered"
        assert t.n_header_rows == 1

    def test_wrapped_numeric_cell_still_merges(self):
        # LDF-style multi-line numeric cell: continuation is numeric but must
        # still fold into the row above (regression guard for SERFF pages)
        words = []
        words += _rowline(
            100, [(50, 100, "Scenario"), (150, 200, "LDF"), (250, 300, "Range")]
        )
        words += _rowline(
            115,
            [
                (50, 95, "1. As Filed"),
                (150, 205, "1.227, 1.042,"),
                (250, 310, "-32.4% to -12.4%"),
            ],
        )
        words += _rowline(126, [(150, 200, "1.000, 1.000,")], h=8.0)
        words += _rowline(
            140,
            [
                (50, 90, "2. Other"),
                (150, 205, "1.100, 1.050,"),
                (250, 300, "-10% to 5%"),
            ],
        )
        t = infer_grid(words, (45, 95, 315, 155))
        assert t is not None
        assert len(t.grid) == 3
        assert "1.000, 1.000," in t.grid[1][1].text

    def test_v_rule_forces_separator(self):
        # two columns with a narrow 5pt gap but an explicit vertical rule
        words = []
        words += _rowline(100, [(50, 100, "Alpha"), (105, 150, "Beta")])
        words += _rowline(120, [(50, 100, "a1a1a"), (105, 150, "b1b1b")])
        words += _rowline(140, [(50, 100, "a2a2a"), (105, 150, "b2b2b")])
        t = infer_grid(
            words, (45, 95, 155, 155), v_rules=[{"x": 102.5, "top": 95, "bottom": 155}]
        )
        assert t is not None
        assert all(len(r) == 2 for r in t.grid)
        assert t.grid[1][0].text == "a1a1a" and t.grid[1][1].text == "b1b1b"


# ---------------------------------------------------------------------------
# standalone region proposal
# ---------------------------------------------------------------------------


class TestProposeRegions:
    def _prose_lines(self, top0, n=3):
        words = []
        for i in range(n):
            x = 50.0
            top = top0 + i * 14
            for tok in ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur"]:
                width = 6.0 * len(tok)
                words.append(w(x, top, x + width, top + 10, tok))
                x += width + 4.0
        return words

    def test_table_band_inside_prose(self):
        words = self._prose_lines(50)
        # a 3-column aligned band at y 120..180 (4 lines)
        for i in range(4):
            top = 120 + i * 15
            words += _rowline(
                top, [(60, 100, f"r{i}"), (200, 240, f"{i}00"), (350, 400, f"{i}.5%")]
            )
        words += self._prose_lines(220)
        regions = propose_regions(words)
        assert len(regions) == 1
        x0, top, x1, bottom = regions[0]
        assert 115 <= top <= 125 and 175 <= bottom <= 190
        assert x0 <= 62 and x1 >= 395

    def test_two_column_prose_not_proposed(self):
        # two prose columns split by a page gutter: both sides wordy text,
        # no numeric side -> must NOT become a table region
        words = []
        for i in range(5):
            top = 100 + i * 14
            words += _rowline(
                top,
                [
                    (50, 200, "leftish prose words here"),
                    (260, 410, "rightish prose words too"),
                ],
            )
        assert propose_regions(words) == []

    def test_label_value_band_is_proposed(self):
        # 2 columns but the right side is numeric -> classic financial table
        words = []
        for i in range(4):
            top = 100 + i * 15
            words += _rowline(
                top, [(60, 150, f"Line item {i}"), (300, 340, f"{i},{i}00")]
            )
        assert len(propose_regions(words)) == 1


# ---------------------------------------------------------------------------
# under-segmentation detection (fragmentary-rule "ruled" tables)
# ---------------------------------------------------------------------------


class TestUnderseg:
    def test_fused_numeric_cells_detected(self):
        grid = [
            [Cell("U.S."), Cell("$ 40,274"), Cell("$ 35,664")],
            [Cell("China (1) 3,617 4,797", colspan=3)],
            [Cell("Other countries 5,943 5,219", colspan=3)],
        ]
        assert _looks_undersegmented(grid)

    def test_clean_grid_not_flagged(self):
        grid = [
            [Cell("Name"), Cell("Qty"), Cell("Price")],
            [Cell("foo"), Cell("1"), Cell("$2.00")],
            [Cell("bar"), Cell("23"), Cell("$4.50")],
        ]
        assert not _looks_undersegmented(grid)


# ---------------------------------------------------------------------------
# stacked-table splitting (infer_tables returns multiple tables)
# ---------------------------------------------------------------------------


class TestInferTablesSplitting:
    def test_header_restart_splits(self):
        words = []
        # table 1: header + 2 numeric rows
        words += _rowline(
            100, [(50, 90, "Region"), (200, 230, "2024"), (300, 330, "2023")]
        )
        words += _rowline(115, [(50, 80, "US"), (200, 230, "100"), (300, 330, "90")])
        words += _rowline(130, [(50, 85, "EU"), (200, 230, "50"), (300, 330, "45")])
        # table 2 restarts with a header-ish row after data
        words += _rowline(
            160, [(50, 95, "Segment"), (200, 235, "Units"), (300, 340, "Share")]
        )
        words += _rowline(175, [(50, 80, "Mac"), (200, 230, "12"), (300, 330, "3%")])
        words += _rowline(190, [(50, 84, "iPad"), (200, 230, "18"), (300, 330, "5%")])
        tables = infer_tables(words, (45, 95, 345, 205))
        assert len(tables) == 2
        assert tables[0].grid[0][0].text == "Region"
        assert tables[1].grid[0][0].text == "Segment"

    def test_big_gap_splits(self):
        # a large inter-row gap (after wrapped-cell merging) is real table
        # separation; intra-table group gaps disappear once wraps merge, so
        # this stays safe for rate-table scenario blocks
        words = []
        words += _rowline(100, [(50, 90, "Aaa"), (200, 240, "111")])
        words += _rowline(115, [(50, 90, "Bbb"), (200, 240, "222")])
        words += _rowline(130, [(50, 90, "Ccc"), (200, 240, "333")])
        # 60pt gap (~4x pitch)
        words += _rowline(190, [(50, 90, "Ddd"), (200, 240, "444")])
        words += _rowline(205, [(50, 90, "Eee"), (200, 240, "555")])
        words += _rowline(220, [(50, 90, "Fff"), (200, 240, "666")])
        tables = infer_tables(words, (45, 95, 245, 235))
        assert len(tables) == 2

    def test_single_table_not_split(self):
        words = []
        words += _rowline(100, [(50, 90, "Name"), (200, 240, "Qty")])
        for i in range(4):
            words += _rowline(115 + i * 15, [(50, 90, f"it{i}"), (200, 240, str(i))])
        tables = infer_tables(words, (45, 95, 245, 185))
        assert len(tables) == 1
        assert len(tables[0].grid) == 5


# ---------------------------------------------------------------------------
# currency-symbol column merging
# ---------------------------------------------------------------------------


class TestMergeSymbolColumns:
    def test_dollar_column_merges_right(self):
        grid = [
            [Cell(""), Cell("2024"), Cell(""), Cell("2023")],
            [Cell("U.S."), Cell("100"), Cell("$"), Cell("90")],
            [Cell("China"), Cell("50"), Cell("$"), Cell("45")],
        ]
        out = _merge_symbol_columns(grid)
        assert all(len(r) == 3 for r in out)
        assert out[1] == [Cell("U.S."), Cell("100"), Cell("$ 90")]
        assert out[0] == [Cell(""), Cell("2024"), Cell("2023")]

    def test_no_symbol_columns_untouched(self):
        grid = [
            [Cell("a"), Cell("b")],
            [Cell("1"), Cell("2")],
        ]
        assert _merge_symbol_columns(grid) == grid

    def test_caption_row_colspan_shrinks(self):
        grid = [
            [Cell("Section", colspan=3)],
            [Cell("x"), Cell("$"), Cell("10")],
            [Cell("y"), Cell("$"), Cell("20")],
        ]
        out = _merge_symbol_columns(grid)
        assert out[0][0].colspan == 2
        assert out[1] == [Cell("x"), Cell("$ 10")]


# ---------------------------------------------------------------------------
# external header extension
# ---------------------------------------------------------------------------


class TestExtendHeaderUp:
    def _numeric_table(self):
        words = []
        words += _rowline(120, [(50, 80, "U.S."), (200, 240, "100"), (300, 340, "90")])
        words += _rowline(135, [(50, 85, "China"), (200, 240, "50"), (300, 340, "45")])
        words += _rowline(150, [(50, 88, "Other"), (200, 240, "25"), (300, 340, "20")])
        t = infer_grid(words, (45, 115, 345, 165))
        assert t is not None
        return t

    def test_year_labels_above_attach_as_header(self):
        t = self._numeric_table()
        page_words = [
            w(205, 105, 235, 113, "2024"),
            w(305, 105, 335, 113, "2023"),
        ]
        _extend_header_up(t, page_words, [])
        assert t.grid[0][1].text == "2024" and t.grid[0][2].text == "2023"
        assert t.n_header_rows == 1
        assert len(t.grid) == 4
        assert t.bbox[1] <= 105.5

    def test_prose_above_not_attached(self):
        t = self._numeric_table()
        # a long prose line straddling the column cuts
        page_words = [
            w(50 + i * 34, 105, 50 + i * 34 + 30, 113, tok)
            for i, tok in enumerate(
                [
                    "Securities",
                    "registered",
                    "pursuant",
                    "to",
                    "Section",
                    "12(b)",
                    "hereof",
                    "now",
                ]
            )
        ]
        before = len(t.grid)
        _extend_header_up(t, page_words, [])
        assert len(t.grid) == before

    def test_label_header_table_not_extended(self):
        words = []
        words += _rowline(120, [(50, 90, "Name"), (200, 240, "Qty")])
        words += _rowline(135, [(50, 80, "foo"), (200, 240, "1")])
        words += _rowline(150, [(50, 80, "bar"), (200, 240, "2")])
        t = infer_grid(words, (45, 115, 245, 165))
        page_words = [w(60, 105, 200, 113, "Some caption text above")]
        before = len(t.grid)
        _extend_header_up(t, page_words, [])
        assert len(t.grid) == before


# ---------------------------------------------------------------------------
# header detection
# ---------------------------------------------------------------------------


class TestHeaderRows:
    def test_label_row_then_numeric(self):
        grid = [
            [Cell("Name"), Cell("Qty"), Cell("Price")],
            [Cell("foo"), Cell("1"), Cell("$2.00")],
        ]
        assert _n_header_rows(grid) == 1

    def test_two_label_rows(self):
        grid = [
            [Cell("Group A", colspan=2), Cell("Group B")],
            [Cell("2019"), Cell("2020"), Cell("2021")],
            [Cell("1"), Cell("2"), Cell("3")],
        ]
        assert _n_header_rows(grid) == 2

    def test_numeric_first_row_still_gets_one_header(self):
        # always at least one <th> row: TRM demotes harmlessly when GT is
        # headerless, and it is the only shot at real keys otherwise
        grid = [
            [Cell("1.0"), Cell("2.0")],
            [Cell("3.0"), Cell("4.0")],
        ]
        assert _n_header_rows(grid) == 1

    def test_all_text_table_headers_only_first_row(self):
        # a deep leading-non-numeric run must NOT th every row (poisoned keys)
        grid = [[Cell("a"), Cell("b")]] * 5 + [[Cell("1"), Cell("2")]]
        assert _n_header_rows(grid) == 1


# ---------------------------------------------------------------------------
# ruled path end-to-end on the synthetic PDF
# ---------------------------------------------------------------------------


class TestRuledPdf:
    def test_grid_detected(self, ruled_grid_pdf):
        import pdfplumber

        with pdfplumber.open(ruled_grid_pdf) as pdf:
            tables = extract_page_tables(pdf.pages[0])
        assert len(tables) == 1
        t = tables[0]
        assert t.ruled
        assert len(t.grid) == 3 and all(len(r) == 3 for r in t.grid)
        assert t.grid[0][0].text == "Name"
        assert t.grid[1][2].text == "$2.00"
        assert t.grid[2][1].text == "23"
        assert t.n_header_rows == 1
        html = render_table_html(t)
        assert "<th>Name</th>" in html and "<td>$4.50</td>" in html

    def test_bbox_sane(self, ruled_grid_pdf):
        import pdfplumber

        with pdfplumber.open(ruled_grid_pdf) as pdf:
            page = pdf.pages[0]
            tables = extract_page_tables(page)
        x0, top, x1, bottom = tables[0].bbox
        # PDF y=700 -> top = 792-700 = 92 ; y=640 -> bottom = 152
        assert 95 <= x0 <= 105 and 305 <= x1 <= 315
        assert 88 <= top <= 96 and 148 <= bottom <= 156


# ---------------------------------------------------------------------------
# prose-grid rejection (multi-column prose must never become a fake table)
# ---------------------------------------------------------------------------


class TestLooksProseGrid:
    def test_two_column_prose_rows_rejected(self):
        grid = [
            [
                Cell("Several foreign governments subscribing to the treaty"),
                Cell("The examined ways to comply with the panel ruling"),
            ],
            [
                Cell("alleged that the program was an illegal export subsidy"),
                Cell("repealed the rules and created an exemption for income"),
            ],
            [
                Cell("governments complained that entities were able to"),
                Cell("act provided for a transition period by allowing use"),
            ],
        ]
        assert _looks_prose_grid(grid)

    def test_numeric_data_grid_kept(self):
        grid = [
            [Cell("Name"), Cell("Qty"), Cell("Price")],
            [Cell("Widget"), Cell("12"), Cell("$4.50")],
            [Cell("Gadget"), Cell("23"), Cell("$2.00")],
        ]
        assert not _looks_prose_grid(grid)

    def test_long_description_column_kept(self):
        # one sentence-length description column beside short data columns
        grid = [
            [
                Cell("Item"),
                Cell("Description of the requirement in a sentence"),
                Cell("2024"),
            ],
            [
                Cell("A-1"),
                Cell("The vendor shall provide onsite support each week"),
                Cell("12"),
            ],
            [
                Cell("B-2"),
                Cell("All deliverables must be submitted by the deadline"),
                Cell("7"),
            ],
        ]
        assert not _looks_prose_grid(grid)

    def test_empty_grid_rejected(self):
        assert _looks_prose_grid([[Cell(""), Cell("  ")]])

    def test_digit_dense_toc_grid_kept(self):
        # a merged-TOC grid: long cells but digit-dense (page numbers) -> table
        grid = [
            [
                Cell("03 OUR WORLD 3.1 Our company 11 3.2 Story of Group 14"),
                Cell("04 APPROACH TO ESG 4.1 Our commitment 46"),
            ],
            [
                Cell("6.1 Pillars of the people Strategy 98 6.2 Employment 99"),
                Cell("7.1 Good governance 152 7.2 Our structure 154"),
            ],
        ]
        assert not _looks_prose_grid(grid)
