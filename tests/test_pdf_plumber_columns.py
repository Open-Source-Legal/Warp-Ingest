"""Front-end column-aware line grouping (multi-column reading-order fix).

These are pure-unit tests on synthetic pdfplumber-style word dicts — no PDFs, no
engine. They pin the two new front-end primitives:

  * ``_detect_column_gutters(words, bbox)`` — geometry-only gutter detection that
    fires ONLY on genuine multi-column *prose* (not single column, not data
    tables, not borderless label/value tables).
  * ``_group_words_into_lines_columns(words, bbox)`` — column-partitioned line
    grouping that returns the SAME shape as ``_group_words_into_lines`` and is
    byte-identical to it for single-column / table pages.

The invariant that protects the Tika XHTML contract: when no confident
multi-column layout is detected, output is identical to ``_group_words_into_lines``.
"""

from warp_ingest.file_parser import pdf_plumber_parser as P

BBOX = (0.0, 0.0, 600.0, 800.0)  # x0, top, x1, bottom


def W(text, x0, top, size=10.0, font="Times"):
    """A minimal pdfplumber-style word dict (fields the parser uses)."""
    return {
        "text": text,
        "x0": float(x0),
        "x1": float(x0 + len(text) * size * 0.5),
        "top": float(top),
        "bottom": float(top + size),
        "size": float(size),
        "fontname": font,
    }


def row(tokens, x_start, top, size=10.0, gap=4.0):
    """A horizontal run of word dicts starting at x_start."""
    out = []
    x = x_start
    for t in tokens:
        w = W(t, x, top, size)
        out.append(w)
        x = w["x1"] + gap
    return out


# A flowing-prose line: 10 short words so it clears COL_MIN_MEDIAN_WORDS_PER_LINE
# while keeping the column narrow enough to leave a clean gutter on a 600pt page.
PROSE = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"]


def full_width_row(top, size=10.0):
    """A continuous full-width line (e.g. a running header/footer/title) that
    crosses the column gutter — words abut with small real-space gaps."""
    out = []
    x = 40.0
    while x < 460:
        w = W("word", x, top, size)
        out.append(w)
        x = w["x1"] + 3
    return out


def two_col_prose_words(n_lines=12, aligned=True):
    """Genuine 2-column prose: left [40,~176], right [320,~456], clean gutter
    [~176,320]. 10 words/line (clears the prose gate)."""
    words = []
    for i in range(n_lines):
        top = 100.0 + i * 16
        words += row(PROSE, 40, top)
        words += row(PROSE, 320, top if aligned else top + 7)
    return words


def single_col_words(n_lines=12):
    wide = PROSE + ["kk", "ll", "mm", "nn", "oo", "pp"]  # 16-word full-width line
    words = []
    for i in range(n_lines):
        words += row(wide, 40, 100.0 + i * 16)
    return words


# --------------------------------------------------------------------------
# _detect_column_gutters
# --------------------------------------------------------------------------


def test_detect_gutter_on_two_column_prose():
    guts = P._detect_column_gutters(two_col_prose_words(), BBOX)
    assert len(guts) == 1
    xa, xb = guts[0]
    # gutter sits in the clean band between the columns (left text ends ~176,
    # right text starts at 320)
    assert 170 <= xa < xb <= 321


def test_no_gutter_on_single_column():
    assert P._detect_column_gutters(single_col_words(), BBOX) == []


def test_no_gutter_on_data_table_narrow_cells():
    """4-column data table: gaps between cells are empty (would look like
    gutters) but each column is <20% of page width -> rejected."""
    words = []
    for i in range(10):
        top = 100.0 + i * 16
        words += row(["12.3"], 40, top)
        words += row(["45.6"], 180, top)
        words += row(["78.9"], 320, top)
        words += row(["10.1"], 460, top)
    assert P._detect_column_gutters(words, BBOX) == []


def test_no_gutter_on_imbalanced_columns_infographic():
    """A narrow prose column + a much wider region (e.g. an infographic matrix):
    even when both read prose-like (>=10 words/line) and each is >=20% wide, a
    ~1.7:1 width imbalance is not a clean text-column layout -> not split (drops
    AXP-style false positives)."""
    words = []
    for i in range(14):
        top = 100.0 + i * 16
        words += row(PROSE, 40, top, size=9.0)  # narrow left column (~126px)
        words += row(PROSE, 300, top, size=18.0)  # wide right column (~216px)
    guts = P._detect_column_gutters(words, BBOX)
    # left ~126px vs right ~216px -> ~1.7:1 imbalance > 1.6 -> not a clean 2-col
    assert guts == []


def test_tabular_row_fraction():
    """Prose lines are single-segment (0.0); rows that split into 3+ segments
    (multiple column gaps) are tabular (1.0)."""
    prose_lines = [row(PROSE, 40, 100.0 + i * 16) for i in range(5)]
    assert P._tabular_row_fraction(prose_lines) == 0.0
    # each row = 3 cells separated by wide gaps -> 3 segments -> tabular
    table_lines = [
        [W("aa", 40, t), W("bb", 200, t), W("cc", 360, t)]
        for t in (100.0, 116.0, 132.0, 148.0, 164.0)
    ]
    assert P._tabular_row_fraction(table_lines) == 1.0


def test_prose_count_rejects_table_column():
    """A column whose rows are tabular (3+ segments) is not counted as prose even
    when it clears the words-per-line bar -- so a 2-col page where one side is a
    ragged table won't be split (the c826 patent case)."""
    gutters = [(180.0, 320.0)]
    words = []
    for i in range(14):
        top = 100.0 + i * 16
        words += row(PROSE, 40, top)  # left: flowing prose -> prose column
        # right: 3 cells of 2 words each (6 words/line clears the wpl bar) but
        # each row splits into 3 segments -> tabular -> NOT prose
        for cx in (330.0, 430.0, 530.0):
            words += [W("aa", cx, top), W("bb", cx + 14, top)]
    assert P._count_prose_columns(words, gutters) == 1  # only the left column


def test_no_gutter_on_borderless_label_value_table():
    """2 wide columns with a clean gutter, but cells are 1-2 words (med wpl 2)
    -> the per-column prose gate rejects it (would be scrambled if split)."""
    words = []
    for i in range(10):
        top = 100.0 + i * 16
        words += row(["Label", str(i)], 40, top)  # 2 words, left
        words += row(["Value", "here"], 320, top)  # 2 words, right
    assert P._detect_column_gutters(words, BBOX) == []


# --------------------------------------------------------------------------
# _group_words_into_lines_columns
# --------------------------------------------------------------------------


def _texts(lines):
    return [" ".join(w["text"] for w in ln) for ln in lines]


def test_single_column_identical_to_plain_grouping():
    words = single_col_words()
    assert P._group_words_into_lines_columns(words, BBOX) == P._group_words_into_lines(
        words
    )


def test_data_table_identical_to_plain_grouping():
    words = []
    for i in range(10):
        top = 100.0 + i * 16
        words += row(["12.3"], 40, top)
        words += row(["45.6"], 180, top)
        words += row(["78.9"], 320, top)
        words += row(["10.1"], 460, top)
    assert P._group_words_into_lines_columns(words, BBOX) == P._group_words_into_lines(
        words
    )


def test_two_column_prose_reading_order_left_then_right():
    """All left-column lines come before all right-column lines, and no output
    line mixes both columns (the fusion bug)."""
    words = two_col_prose_words(n_lines=6)
    lines = P._group_words_into_lines_columns(words, BBOX)
    # every output line is entirely within one column (no cross-column fusion)
    for ln in lines:
        xs = [w["x0"] for w in ln]
        assert max(xs) < 300 or min(xs) >= 300, _texts([ln])
    # left column fully precedes right column
    col_of = [0 if max(w["x0"] for w in ln) < 300 else 1 for ln in lines]
    assert col_of == sorted(col_of), col_of
    assert col_of.count(0) == 6 and col_of.count(1) == 6


def test_full_width_header_and_footer_flush_columns():
    """A full-width header (text across the gutter) precedes the columns; a
    full-width footer follows them (spanning-row flush)."""
    header = full_width_row(60)
    footer = full_width_row(560)
    # tag header/footer first words so we can find them in the output
    header[0]["text"] = "HEADERSTART"
    footer[0]["text"] = "FOOTERSTART"
    words = header + two_col_prose_words(n_lines=24) + footer
    lines = P._group_words_into_lines_columns(words, BBOX)
    texts = _texts(lines)
    assert texts[0].startswith("HEADERSTART")  # full-width header emitted first
    assert texts[-1].startswith("FOOTERSTART")  # full-width footer emitted last
    # between header and footer: all left-column lines, then all right-column
    body = lines[1:-1]
    col_of = [0 if max(w["x0"] for w in ln) < 300 else 1 for ln in body]
    assert col_of == sorted(col_of)
    assert col_of.count(0) == 24 and col_of.count(1) == 24
