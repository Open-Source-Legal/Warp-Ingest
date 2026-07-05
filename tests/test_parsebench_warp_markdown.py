"""Tests for the Warp-Ingest -> Markdown renderer used by the ParseBench provider.

These exercise the dependency-free renderer (``benchmarks/parsebench/warp_markdown.py``)
with synthetic blocks plus one end-to-end parse of a real fixture, so they run
without the ParseBench framework installed.
"""

from pathlib import Path

import pytest

from benchmarks.parsebench import warp_markdown as wm
from benchmarks.parsebench.warp_markdown import (
    _bold_decorate,
    _box_xywh,
    _page_columns_from_xhtml,
    _reorder_page_columns,
    _strip_cjk,
    _strip_page_chrome,
    _table_html,
    extract_warp_blocks,
    render_markdown,
    render_pages,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_bold_decorate_wraps_runin_label():
    # "NOTE:" bold, the rest not -> a run-in label.
    block = {"block_text": "NOTE: keep it dry", "bold_mask": "1000"}
    assert _bold_decorate("NOTE: keep it dry", block) == "**NOTE:** keep it dry"


def test_bold_decorate_wraps_whole_and_inline():
    assert (
        _bold_decorate("All Bold Heading", {"bold_mask": "111"})
        == "**All Bold Heading**"
    )
    # An inline emphasised run in the middle.
    block = {"bold_mask": "0110"}
    assert _bold_decorate("a b c d", block) == "a **b c** d"


def test_bold_decorate_no_bold_is_passthrough():
    assert _bold_decorate("plain text here", {"bold_mask": "000"}) == "plain text here"
    assert _bold_decorate("no mask", {}) == "no mask"


def test_bold_decorate_wraps_italic_runs():
    # italic-only run from the engine's real font-style mask
    assert _bold_decorate("Roe v. Wade", {"italic_mask": "111"}) == "*Roe v. Wade*"
    # an inline italic run in the middle
    assert (
        _bold_decorate("see Roe v. Wade now", {"italic_mask": "01110"})
        == "see *Roe v. Wade* now"
    )


def test_bold_decorate_combines_bold_and_italic():
    # adjacent bold then italic words are wrapped separately
    assert (
        _bold_decorate("BOLD ital", {"bold_mask": "10", "italic_mask": "01"})
        == "**BOLD** *ital*"
    )
    # a word that is both bold and italic -> ***...***
    assert (
        _bold_decorate("both words", {"bold_mask": "11", "italic_mask": "11"})
        == "***both words***"
    )
    # italic mask present but bold mask misaligned -> italic still applies
    assert (
        _bold_decorate("a b c", {"bold_mask": "11", "italic_mask": "010"}) == "a *b* c"
    )


def test_bold_decorate_misaligned_mask_falls_back_safely():
    # Mask length != word count -> only wrap if essentially all-bold, else plain.
    assert _bold_decorate("a b c", {"bold_mask": "11", "bold_ratio": 0.5}) == "a b c"
    assert (
        _bold_decorate("a b c", {"bold_mask": "11", "bold_ratio": 0.95}) == "**a b c**"
    )


def test_table_html_marks_header_and_body():
    html = _table_html(["A", "B"], [["1", "2"], ["3", "4"]])
    assert "<th>A</th><th>B</th>" in html
    assert "<td>1</td><td>2</td>" in html
    assert html.startswith("<table>") and html.endswith("</table>")


def test_table_html_escapes_cells():
    html = _table_html(None, [["<x> & y"]])
    assert "&lt;x&gt; &amp; y" in html
    assert "<x>" not in html.replace("<td>", "").replace("</td>", "")


def test_box_xywh_from_indexable_and_attrs():
    class _Box:
        # [top, left, right, width, height]
        top, left, right, width, height = 10.0, 20.0, 120.0, 100.0, 12.0

        def __getitem__(self, i):
            return [self.top, self.left, self.right, self.width, self.height][i]

    assert _box_xywh(_Box()) == [20.0, 10.0, 100.0, 12.0]  # [left, top, w, h]
    assert _box_xywh(None) is None


def test_render_pages_groups_tables_and_dedups_header():
    raw = {
        "num_pages": 1,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "header",
                "block_text": "Title",
                "level": 1,
                "box": [0, 0, 10, 10],
            },
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Body text.",
                "level": None,
                "box": [0, 20, 10, 10],
            },
            {
                "page_idx": 0,
                "block_type": "list_item",
                "block_text": "First",
                "level": None,
                "box": [0, 40, 10, 10],
            },
            # a table: the physical header row repeats header_cell_values
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "h1 h2",
                "cell_values": ["h1", "h2"],
                "header_cell_values": ["h1", "h2"],
                "table_idx": 0,
                "box": [0, 60, 10, 10],
            },
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "a b",
                "cell_values": ["a", "b"],
                "header_cell_values": ["h1", "h2"],
                "table_idx": 0,
                "box": [0, 70, 10, 10],
            },
        ],
    }
    pages = render_pages(raw)
    assert len(pages) == 1
    md = pages[0][1]
    assert "# Title" in md
    assert "Body text." in md
    assert "- First" in md
    # table rendered once, header as <th>, repeated header row dropped from body
    assert md.count("<table>") == 1
    assert "<th>h1</th><th>h2</th>" in md
    assert "<td>a</td><td>b</td>" in md
    assert "<td>h1</td><td>h2</td>" not in md


def test_render_pages_region_aware_ext_tables_mixed_page():
    """A bbox'd provider table replaces only the warp table it overlaps; a
    second warp table on the same page still renders natively, and a provider
    table matching no warp block is appended at page end."""
    raw = {
        "num_pages": 1,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Intro prose.",
                "level": None,
                "box": [50, 10, 200, 10],
            },
            # warp table A (covered by the provider bbox below)
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "a1 a2",
                "cell_values": ["a1", "a2"],
                "table_idx": 0,
                "box": [50, 100, 200, 12],
            },
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "a3 a4",
                "cell_values": ["a3", "a4"],
                "table_idx": 0,
                "box": [50, 114, 200, 12],
            },
            # warp table B (NOT covered — provider abstained here)
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "b1 b2",
                "cell_values": ["b1", "b2"],
                "table_idx": 1,
                "box": [50, 400, 200, 12],
            },
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "b3 b4",
                "cell_values": ["b3", "b4"],
                "table_idx": 1,
                "box": [50, 414, 200, 12],
            },
        ],
        "ext_tables": {
            0: [
                [[45, 95, 255, 130], "<table>\n<tr><th>A</th></tr>\n</table>"],
                [[300, 600, 400, 640], "<table>\n<tr><th>ORPHAN</th></tr>\n</table>"],
            ]
        },
    }
    md = render_pages(raw)[0][1]
    # provider table A replaces warp table A (whose cells are gone)
    assert "<th>A</th>" in md
    assert "a1" not in md and "a3" not in md
    # warp table B still rendered natively
    assert "<td>b1</td><td>b2</td>" in md
    # prose intact, orphan provider table appended
    assert "Intro prose." in md
    assert "<th>ORPHAN</th>" in md
    # provider A appears before warp B (emitted at covered position)
    assert md.index("<th>A</th>") < md.index("<td>b1</td>")


def test_render_pages_legacy_string_ext_tables_page_granular():
    """Bare-string provider tables keep the legacy page-level replacement."""
    raw = {
        "num_pages": 1,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Prose stays.",
                "level": None,
                "box": [50, 10, 200, 10],
            },
            {
                "page_idx": 0,
                "block_type": "table_row",
                "block_text": "x y",
                "cell_values": ["x", "y"],
                "table_idx": 0,
                "box": [50, 100, 200, 12],
            },
        ],
        "ext_tables": {0: ["<table>\n<tr><th>L</th></tr>\n</table>"]},
    }
    md = render_pages(raw)[0][1]
    assert "<th>L</th>" in md
    assert "<td>x</td>" not in md
    assert "Prose stays." in md

    # idempotent: the already-normalized [None, html] form (what
    # extract_warp_blocks stores) renders identically
    raw["ext_tables"] = {0: [[None, "<table>\n<tr><th>L</th></tr>\n</table>"]]}
    md2 = render_pages(raw)[0][1]
    assert "<th>L</th>" in md2 and "<td>x</td>" not in md2


def test_render_pages_ranks_heading_levels():
    raw = {
        "num_pages": 1,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "header",
                "block_text": "Top",
                "level": 2,
                "box": [0, 0, 1, 1],
            },
            {
                "page_idx": 0,
                "block_type": "header",
                "block_text": "Sub",
                "level": 7,
                "box": [0, 10, 1, 1],
            },
        ],
    }
    md = render_pages(raw)[0][1]
    # shallowest distinct depth -> h1, next -> h2 (not clamped-to-6 noise)
    assert "# Top" in md and "## Sub" in md


@pytest.mark.parametrize("fixture", ["USC Title 1 - CHAPTER 1.pdf"])
def test_extract_and_render_real_fixture(fixture):
    path = FIXTURES / fixture
    if not path.exists():
        pytest.skip(f"fixture missing: {fixture}")
    raw = extract_warp_blocks(str(path))
    assert raw["num_pages"] >= 1
    assert raw["blocks"], "expected blocks"
    assert all("box" in b for b in raw["blocks"])
    assert raw["page_dim"] and len(raw["page_dim"]) == 2
    md = render_markdown(raw)
    assert len(md) > 1000
    assert "<table>" in md  # USC chapter table
    assert "# TITLE 1" in md or "#TITLE 1" in md  # the document title as a heading


# --- olmOCR-bench renderer levers (chrome strip / CJK / column signal) --------


def test_strip_cjk_removes_cjk_and_emoji_but_keeps_latin(monkeypatch):
    monkeypatch.setattr(wm, "_CJK_STRIP", True)
    assert _strip_cjk("Hello 三号 world \U0001f600 ok") == "Hello world ok"


def test_strip_cjk_reverts_when_output_would_be_empty(monkeypatch):
    monkeypatch.setattr(wm, "_CJK_STRIP", True)
    # A pure-CJK string would blank out; keep the original (no worse for baseline).
    s = "三号登对"
    assert _strip_cjk(s) == s


def test_strip_cjk_passthrough_plain_text():
    assert _strip_cjk("The quick brown fox 123.") == "The quick brown fox 123."


def test_page_columns_from_xhtml_parses_only_signalled_pages():
    html = (
        '<div class="page" data-columns="285.6,303.4" style="height:800px">x</div>'
        '<div class="page" style="height:800px">y</div>'
        '<div class="page" data-columns="200.0,215.0|400.0,412.0">z</div>'
    )
    cols = _page_columns_from_xhtml(html)
    assert cols == {0: [(285.6, 303.4)], 2: [(200.0, 215.0), (400.0, 412.0)]}


def test_page_columns_from_xhtml_ignores_malformed():
    assert (
        _page_columns_from_xhtml('<div class="page" data-columns="oops">x</div>') == {}
    )
    assert _page_columns_from_xhtml("") == {}


def _blk(text, left, top, w, h, btype="para"):
    return {"block_text": text, "block_type": btype, "box": [left, top, w, h]}


def test_strip_page_chrome_removes_margin_running_head_and_footer():
    page_h = 800.0
    body = [_blk(f"body line {i}", 60, 120 + i * 20, 400, 14) for i in range(6)]
    head = _blk("Running Head 1116", 60, 30, 200, 12)  # compact, isolated, top margin
    foot = _blk("Journal of Things, 21: 1-9 (2020)", 60, 772, 300, 12)  # bottom margin
    kept = _strip_page_chrome([head] + body + [foot], page_h)
    texts = [b["block_text"] for b in kept]
    assert "Running Head 1116" not in texts
    assert "Journal of Things, 21: 1-9 (2020)" not in texts
    assert all(f"body line {i}" in texts for i in range(6))  # body untouched


def test_strip_page_chrome_keeps_large_font_title():
    # A page whose title sits in the top-margin band but is LARGER than the body
    # font is a heading, not running chrome -> must be kept (title-protection).
    page_h = 800.0
    body = [_blk(f"body line {i}", 60, 120 + i * 20, 400, 12) for i in range(6)]
    title = _blk("CORPORATE OBJECTIVES", 60, 40, 300, 22)  # top margin, bigger font
    small_folio = _blk("iv", 60, 30, 20, 10)  # a tiny folio next to it -> chrome
    kept_texts = [
        b["block_text"] for b in _strip_page_chrome([title, small_folio] + body, page_h)
    ]
    assert "CORPORATE OBJECTIVES" in kept_texts  # larger-than-body title spared
    assert "iv" not in kept_texts  # small folio still stripped


def test_strip_page_chrome_keeps_tall_block_that_reaches_into_body():
    page_h = 800.0
    # A tall block whose top is in the margin but which extends into the body
    # (fused running-head + body) must NOT be dropped -- it holds real content.
    body = [_blk(f"line {i}", 60, 120 + i * 20, 400, 14) for i in range(5)]
    fused = _blk("Header text plus a whole paragraph of body ...", 60, 40, 400, 220)
    kept = _strip_page_chrome([fused] + body, page_h)
    assert any(b["block_text"].startswith("Header text") for b in kept)


def test_reorder_page_columns_is_column_major():
    gutters = [(290.0, 310.0)]
    l1 = _blk("L1", 50, 10, 200, 12)
    r1 = _blk("R1", 350, 10, 200, 12)
    l2 = _blk("L2", 50, 40, 200, 12)
    r2 = _blk("R2", 350, 40, 200, 12)
    # Input is top-interleaved (what the engine's top-sort produces).
    out = _reorder_page_columns([l1, r1, l2, r2], gutters)
    assert [b["block_text"] for b in out] == ["L1", "L2", "R1", "R2"]


def test_reorder_page_columns_full_width_block_flushes():
    gutters = [(290.0, 310.0)]
    l1 = _blk("L1", 50, 10, 200, 12)
    banner = _blk("BANNER", 20, 30, 560, 16)  # spans the gutter -> flush point
    r1 = _blk("R1", 350, 50, 200, 12)
    out = _reorder_page_columns([l1, banner, r1], gutters)
    assert [b["block_text"] for b in out] == ["L1", "BANNER", "R1"]


def test_reorder_identity_without_gutters():
    blocks = [_blk("a", 50, 10, 200, 12), _blk("b", 50, 30, 200, 12)]
    assert _reorder_page_columns(blocks, []) is blocks
