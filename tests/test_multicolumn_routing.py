"""Unit tests for the multi-column routing / extraction-fidelity fixes:

* ``_strip_cid_words`` -- unmapped-glyph ("(cid:NN)") markers are dropped so a
  garbled text layer routes to OCR instead of emitting garbage tokens.
* ``_group_words_into_lines_columns`` -- a word merely *overhanging* a gutter
  must not flush the column buffers (only a genuine bridge does), so
  narrow-gutter multi-column pages keep column-major reading order.
* ``_line_effective_word_count`` / ``_count_prose_columns`` -- CJK-aware word
  density and the relaxed 3+-column words-per-line floor.
* ``ocr_parser._trim_det_box`` -- horizontal compensation of the OCR
  detector's box dilation.
"""

from warp_ingest.file_parser import ocr_parser
from warp_ingest.file_parser.pdf_plumber_parser import (
    COL_MIN_MEDIAN_WORDS_3COL,
    _count_prose_columns,
    _group_words_into_lines_columns,
    _line_effective_word_count,
    _strip_cid_words,
)


def w(text, x0, x1, top=0.0, size=10.0):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": top + size,
        "size": size,
        "fontname": "F",
    }


# ---------------------------------------------------------------------------
# _strip_cid_words
# ---------------------------------------------------------------------------


def test_strip_cid_words_noop_on_clean_words():
    words = [w("hello", 0, 20), w("world", 25, 45)]
    assert _strip_cid_words(words) is words  # identity: no copy on clean pages


def test_strip_cid_words_drops_fully_unmapped_words():
    words = [w("(cid:12)(cid:13)", 0, 20), w("real", 25, 45)]
    out = _strip_cid_words(words)
    assert [x["text"] for x in out] == ["real"]


def test_strip_cid_words_cleans_partially_unmapped_words():
    words = [w("foo(cid:561)bar", 0, 30)]
    out = _strip_cid_words(words)
    assert [x["text"] for x in out] == ["foobar"]
    # original list untouched (front-end may reuse it)
    assert words[0]["text"] == "foo(cid:561)bar"


# ---------------------------------------------------------------------------
# column grouping: overhang vs bridge
# ---------------------------------------------------------------------------

GUTTERS = [(100.0, 112.0)]
BBOX = (0.0, 0.0, 220.0, 300.0)


def _texts(lines):
    return [" ".join(x["text"] for x in ln) for ln in lines]


def _mk_two_col_page(overhang=False):
    """Three rows of 2-column text; row 1's left word optionally pokes 4pt
    into the gutter (a hyphenated overhang), which must NOT flush columns."""
    left_x1 = 104.0 if overhang else 98.0
    rows = []
    for i, top in enumerate((10.0, 24.0, 38.0)):
        rows.append(w(f"L{i}", 10, left_x1 if i == 1 else 98.0, top=top))
        rows.append(w(f"R{i}", 120, 210, top=top))
    return rows


def test_column_grouping_routes_column_major():
    lines = _group_words_into_lines_columns(_mk_two_col_page(), BBOX, GUTTERS)
    assert _texts(lines) == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_gutter_overhang_does_not_flush_columns():
    # The overhanging L1 (x1=104 pokes into the 100-112 band) must still be
    # routed to the left column, keeping full column-major order.
    lines = _group_words_into_lines_columns(
        _mk_two_col_page(overhang=True), BBOX, GUTTERS
    )
    assert _texts(lines) == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_bridging_row_flushes_columns():
    words = _mk_two_col_page()
    # a full-width title fully crossing the gutter band, above the columns
    words.append(w("TITLE", 40, 180, top=1.0))
    lines = _group_words_into_lines_columns(words, BBOX, GUTTERS)
    texts = _texts(lines)
    assert texts[0] == "TITLE"
    assert texts[1:] == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_word_inside_gutter_counts_as_bridge():
    # A word sitting (mostly) inside the gutter band is full-width text
    # evidence -- the row is emitted as a spanning row in reading order.
    words = _mk_two_col_page()
    words.append(w("of", 102.0, 110.0, top=1.0))
    lines = _group_words_into_lines_columns(words, BBOX, GUTTERS)
    assert _texts(lines)[0] == "of"


# ---------------------------------------------------------------------------
# CJK-aware word density + 3-column floor
# ---------------------------------------------------------------------------


def test_effective_word_count_cjk():
    # ~2 CJK chars per word: a 16-char prose line reads as 8 words ...
    ln = [w("中文分栏排版测试中文分栏排版测试", 0, 160)]
    assert _line_effective_word_count(ln) == 8
    # ... while a short glued table cell stays below the prose floors
    assert _line_effective_word_count([w("四十二億円", 0, 50)]) == 2
    ln2 = [w("two", 0, 20), w("words", 25, 60)]
    assert _line_effective_word_count(ln2) == 2


def _prose_column(x0, x1, n_lines=6, words_per_line=4):
    out = []
    step = (x1 - x0) / words_per_line
    for i in range(n_lines):
        for k in range(words_per_line):
            out.append(
                w(
                    f"w{i}{k}",
                    x0 + k * step,
                    x0 + (k + 1) * step - 2.0,
                    top=10.0 + 14.0 * i,
                )
            )
    return out


def test_three_column_layout_uses_relaxed_word_floor():
    assert COL_MIN_MEDIAN_WORDS_3COL < 6
    words = (
        _prose_column(0, 95)  # 4 words/line -- fails the strict 2-col floor
        + _prose_column(112, 207)
        + _prose_column(224, 320)
    )
    two_gutters = [(95.0, 112.0), (207.0, 224.0)]
    assert _count_prose_columns(words, two_gutters) == 3
    # the same density with a single gutter (2 columns) keeps the strict floor
    one_gutter = [(95.0, 112.0)]
    assert _count_prose_columns(words[: 2 * 24], one_gutter) == 0


# ---------------------------------------------------------------------------
# OCR det-box trim
# ---------------------------------------------------------------------------


def test_trim_det_box_symmetric_and_capped():
    # 10px-high line: trim = (1-0.72)/2 * 10 = 1.4px per side
    x0, x1 = ocr_parser._trim_det_box(100.0, 200.0, 0.0, 10.0)
    assert abs(x0 - 101.4) < 1e-6 and abs(x1 - 198.6) < 1e-6
    # pathologically narrow box: trim is capped at 25% of width, never inverts
    x0, x1 = ocr_parser._trim_det_box(100.0, 104.0, 0.0, 40.0)
    assert x0 < x1
    assert abs((x1 - x0) - 2.0) < 1e-6
