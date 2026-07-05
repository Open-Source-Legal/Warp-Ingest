"""Unit tests for the pure-Python pdfplumber front-end word grouping.

The most important contract these guard is the *word-fragment de-fragmentation*:
``page.extract_words(..., extra_attrs=["fontname", "size"])`` starts a new word
on every font/size change, so a single display word whose glyphs carry slightly
different subset font names (or sub-pixel sizes) is shattered into touching
fragments (gap <= the word x-tolerance).  Rendering those fragments joined by
spaces produced garbled text like ``"A bou t 1 ou t o f 10"`` for the words
"About 1 out of 10".  ``_render_p`` must re-join fragments that pdfplumber only
split because of an attribute change (gap <= ``WORD_X_TOLERANCE``) while keeping
genuine inter-word spaces.
"""

import re

from warp_ingest.file_parser import pdf_plumber_parser as P


def _w(text, x0, x1, top=100.0, size=14.0, fontname="ABCDEF+Times"):
    return {
        "text": text,
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": top + size,
        "size": size,
        "fontname": fontname,
    }


def _build_attr_split_segment():
    """Replicate pdfplumber's attr-split fragmentation of "About 1 out of 10".

    Within-word fragments touch (gap ~= -0.1); real spaces are ~3pt (0.21*size).
    """
    cursor = 0.0
    seg = []
    # (fragment_text, leading_gap)  -- gap before this fragment
    plan = [
        ("A", 0.0),
        ("bou", -0.1),
        ("t", -0.1),
        ("1", 2.98),
        ("ou", 2.98),
        ("t", -0.1),
        ("o", 2.98),
        ("f", -0.1),
        ("10", 2.98),
    ]
    for text, gap in plan:
        cursor += gap
        x0 = cursor
        x1 = cursor + 7.0 * len(text)
        seg.append(_w(text, x0, x1))
        cursor = x1
    return seg


def _p_text(p_tag):
    m = re.search(r">([^<]*)</p>", p_tag)
    return m.group(1) if m else ""


def _word_fonts(p_tag):
    m = re.search(r"word-fonts:\[(.*?)\]", p_tag)
    if not m or not m.group(1):
        return []
    return re.findall(r"\([^)]*\)", m.group(1))


def test_attr_split_fragments_are_rejoined():
    seg = _build_attr_split_segment()
    p_tag = P._render_p(seg)
    text = _p_text(p_tag)
    assert text == "About 1 out of 10", text


def test_word_tuples_match_token_count_after_merge():
    """The XHTML contract: len(text.split()) == number of word-* tuples."""
    seg = _build_attr_split_segment()
    p_tag = P._render_p(seg)
    text = _p_text(p_tag)
    n_tokens = len(text.split())
    assert len(_word_fonts(p_tag)) == n_tokens
    starts = re.search(r"word-start-positions:\[(.*?)\]", p_tag).group(1)
    ends = re.search(r"word-end-positions:\[(.*?)\]", p_tag).group(1)
    assert len(re.findall(r"\([^)]*\)", starts)) == n_tokens
    assert len(re.findall(r"\([^)]*\)", ends)) == n_tokens


def test_genuine_spaces_are_preserved():
    """Two real words separated by a normal space stay separate words."""
    seg = [_w("hello", 0.0, 35.0), _w("world", 40.0, 75.0)]  # gap 5 > tolerance
    text = _p_text(P._render_p(seg))
    assert text == "hello world"


def test_single_word_unchanged():
    seg = [_w("Marijuana", 0.0, 70.0)]
    assert _p_text(P._render_p(seg)) == "Marijuana"


def test_dense_text_layer_with_few_lines_does_not_route_to_ocr():
    words = [_w(f"word{i}", float(i * 10), float(i * 10 + 5)) for i in range(30)]
    lines = [words]
    assert not P._should_route_to_ocr(words, lines)


def test_sparse_text_layer_routes_to_ocr():
    words = [_w("tiny", 0.0, 10.0)]
    lines = [words]
    assert P._should_route_to_ocr(words, lines)


def test_forced_ocr_overrides_dense_text_layer():
    words = [_w(f"word{i}", float(i * 10), float(i * 10 + 5)) for i in range(30)]
    lines = [words]
    assert P._should_route_to_ocr(words, lines, force_ocr=True)
