"""Unit + invariant tests for the OpenContractDocExport exporter.

The unit tests drive the converter against a tiny hand-built XHTML + blocks so
geometry, hierarchy, labels, and validation are checked in isolation. The
fixture-driven regression tests (real sample PDFs, metric floors) live at the
bottom and are added once the unit layer is green.
"""

import html
import pathlib

import pytest

from warp_ingest.ingestor import opencontracts_exporter as ex
from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.opencontracts_exporter import (
    ExportValidationError,
    _is_nonheading_furniture,
    _reconcile_runin_overflow,
    to_opencontracts_export,
    validate_export,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _token_text_recall(export):
    """Fraction of `content` word occurrences recoverable from PAWLS tokens."""
    from collections import Counter

    have = Counter()
    for page in export["pawls_file_content"]:
        have.update(t["text"] for t in page["tokens"])
    want = Counter(export["content"].split())
    if not want:
        return 1.0
    covered = sum(min(c, have.get(w, 0)) for w, c in want.items())
    return covered / sum(want.values())


# --------------------------------------------------------------------------- #
# synthetic XHTML helpers (reproduce the pdf_plumber_parser <p> contract)
# --------------------------------------------------------------------------- #
def _p(words, *, top, height=12.0, size=10.0):
    """words: list of (text, x0, x1). Emits one Tika-format <p>."""
    starts = ", ".join(f"({x0},{top})" for _, x0, _ in words)
    ends = ", ".join(f"({x1},{top})" for _, _, x1 in words)
    fonts = ", ".join(
        f"(Times,400,normal,{size},{size},{round(size/4,2)})" for _ in words
    )
    text = html.escape(" ".join(t for t, _, _ in words))
    style = (
        f"height:{height};margin-top: 0px;font-size:{size}px;font-family:Times;"
        f"font-style:normal;font-weight:400;top:{top}px;position:absolute;"
        f"text-indent:{words[0][1]}px;"
        f"word-start-positions:[{starts}];"
        f"word-end-positions:[{ends}];"
        f"word-fonts:[{fonts}]"
    )
    return f'<p style="{style}">{text}</p>'


def _page(p_tags, *, width=612.0, height=792.0):
    return (
        f'<div class="page" style="height:{height}px; width:{width}px; '
        f'position: relative;">{"".join(p_tags)}</div>'
    )


def _doc(pages_html, *, title="Synthetic Doc"):
    return (
        "<html><head>"
        f'<meta name="dc:title" content="{title}"/>'
        f"</head><body>{''.join(pages_html)}</body></html>"
    )


def _block(idx, btype, text, box_style, *, page_idx=0, level_chain=None):
    return {
        "block_idx": idx,
        "block_type": btype,
        "block_text": text,
        "box_style": box_style,  # (top, left, right, width, height)
        "page_idx": page_idx,
        "level_chain": level_chain or [],
    }


@pytest.fixture
def simple_export():
    """One page: a header 'Hello World' and a child paragraph 'Foo bar baz'."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Hello", 72, 110), ("World", 114, 160)], top=100),
                    _p(
                        [("Foo", 72, 96), ("bar", 100, 122), ("baz", 126, 150)], top=120
                    ),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "header", "Hello World", (100.0, 72.0, 160.0, 88.0, 12.0)),
        _block(
            1,
            "para",
            "Foo bar baz",
            (120.0, 72.0, 150.0, 78.0, 12.0),
            level_chain=[{"block_idx": 0, "block_text": "Hello World"}],
        ),
    ]
    return to_opencontracts_export(xhtml, blocks, title="Synthetic Doc")


# --------------------------------------------------------------------------- #
# top-level shape
# --------------------------------------------------------------------------- #
def test_required_top_level_keys_present(simple_export):
    for key in (
        "title",
        "content",
        "description",
        "pawls_file_content",
        "page_count",
        "doc_labels",
        "labelled_text",
        "relationships",
    ):
        assert key in simple_export
    assert simple_export["title"] == "Synthetic Doc"
    assert isinstance(simple_export["content"], str)
    assert simple_export["description"] is None
    assert simple_export["doc_labels"] == []
    assert isinstance(simple_export["relationships"], list)


def test_page_count_is_real_page_count_not_off_by_one(simple_export):
    assert simple_export["page_count"] == 1
    assert len(simple_export["pawls_file_content"]) == 1


# --------------------------------------------------------------------------- #
# PAWLS token layer
# --------------------------------------------------------------------------- #
def test_page_boundary_index_matches_position(simple_export):
    page = simple_export["pawls_file_content"][0]["page"]
    assert page == {"width": 612.0, "height": 792.0, "index": 0}


def test_tokens_are_word_level_with_faithful_geometry(simple_export):
    tokens = simple_export["pawls_file_content"][0]["tokens"]
    assert [t["text"] for t in tokens] == ["Hello", "World", "Foo", "bar", "baz"]
    # x = x0, y = top, width = x1 - x0, height = line height
    assert tokens[0] == {
        "x": 72.0,
        "y": 100.0,
        "width": 38.0,
        "height": 12.0,
        "text": "Hello",
    }
    assert tokens[1]["x"] == 114.0 and tokens[1]["width"] == 46.0
    # no image fields when absent
    assert "is_image" not in tokens[0]


def test_token_text_split_count_matches_word_boxes(simple_export):
    # contract: text.split() length == number of word tuples (no dropped words)
    tokens = simple_export["pawls_file_content"][0]["tokens"]
    assert len(tokens) == 5


# --------------------------------------------------------------------------- #
# annotation layer
# --------------------------------------------------------------------------- #
def test_one_annotation_per_block_with_token_label_structural(simple_export):
    anns = simple_export["labelled_text"]
    assert len(anns) == 2
    for a in anns:
        assert a["annotation_type"] == "TOKEN_LABEL"
        assert a["structural"] is True
        assert a["content_modalities"] == ["TEXT"]


def test_annotation_label_mapping(simple_export):
    by_id = {a["id"]: a for a in simple_export["labelled_text"]}
    assert by_id["0"]["annotationLabel"] == "Section Header"
    assert by_id["1"]["annotationLabel"] == "Paragraph"


def _single_header_block_export(n_words, btype="header"):
    """One page, one header-typed block whose text is `n_words` words long."""
    words = [(f"w{i}", 72 + i * 12, 100 + i * 12) for i in range(n_words)]
    xhtml = _doc([_page([_p(words, top=100)])])
    text = " ".join(w for w, _, _ in words)
    blocks = [_block(0, btype, text, (100.0, 72.0, 600.0, 528.0, 12.0))]
    return to_opencontracts_export(xhtml, blocks, title="t")


def test_overlong_header_block_relabeled_paragraph():
    """A header-typed block with a long body run (a run-in heading that absorbed
    the section body) is not a real section header. Docling reads such spans as
    body text; the exporter must label them 'Paragraph', not 'Section Header'."""
    export = _single_header_block_export(13)  # > _HEADER_MAX_WORDS
    assert export["labelled_text"][0]["annotationLabel"] == "Paragraph"


def test_header_block_at_threshold_stays_section_header():
    """A short header (<= _HEADER_MAX_WORDS words) is still a Section Header."""
    export = _single_header_block_export(12)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


def test_overlong_inline_header_block_relabeled_paragraph():
    """The relabel covers every header-ish block_type, not just plain 'header'."""
    export = _single_header_block_export(30, btype="inline_header")
    assert export["labelled_text"][0]["annotationLabel"] == "Paragraph"


def test_annotation_json_shape_a_with_bounds_and_tokens(simple_export):
    header = simple_export["labelled_text"][0]
    aj = header["annotation_json"]
    assert set(aj.keys()) == {"0"}  # stringified page index
    page = aj["0"]
    assert page["rawText"] == "Hello World"
    assert page["bounds"] == {
        "top": 100.0,
        "left": 72.0,
        "right": 160.0,
        "bottom": 112.0,
    }
    assert page["tokensJsons"] == [
        {"pageIndex": 0, "tokenIndex": 0},
        {"pageIndex": 0, "tokenIndex": 1},
    ]


def test_child_paragraph_gets_its_own_tokens(simple_export):
    para = simple_export["labelled_text"][1]
    refs = para["annotation_json"]["0"]["tokensJsons"]
    assert [r["tokenIndex"] for r in refs] == [2, 3, 4]


def test_parent_id_from_level_chain(simple_export):
    by_id = {a["id"]: a for a in simple_export["labelled_text"]}
    assert by_id["0"]["parent_id"] is None  # root
    assert by_id["1"]["parent_id"] == "0"  # child of header


# --------------------------------------------------------------------------- #
# OC_PARENT_CHILD relationships
# --------------------------------------------------------------------------- #
def test_parent_child_relationships_emitted(simple_export):
    """The parent_id tree is also emitted as explicit OC_PARENT_CHILD edges."""
    rels = simple_export["relationships"]
    assert len(rels) == 1
    r = rels[0]
    assert r["relationshipLabel"] == "OC_PARENT_CHILD"
    assert r["source_annotation_ids"] == ["0"]
    assert r["target_annotation_ids"] == ["1"]
    assert r["structural"] is True


def test_no_relationships_when_flat(simple_export):
    # sanity: a doc with no parent-child edges has no relationships
    flat = to_opencontracts_export(
        _doc([_page([_p([("Solo", 72, 110)], top=100)])]),
        [_block(0, "para", "Solo", (100.0, 72.0, 110.0, 38.0, 12.0))],
    )
    assert flat["relationships"] == []


def test_validate_rejects_dangling_relationship_target(simple_export):
    simple_export["relationships"] = [
        {
            "id": "r0",
            "relationshipLabel": "OC_PARENT_CHILD",
            "source_annotation_ids": ["0"],
            "target_annotation_ids": ["999"],
            "structural": True,
        }
    ]
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


def test_validate_rejects_empty_relationship_endpoints(simple_export):
    simple_export["relationships"] = [
        {
            "id": "r0",
            "relationshipLabel": "OC_PARENT_CHILD",
            "source_annotation_ids": [],
            "target_annotation_ids": ["1"],
            "structural": True,
        }
    ]
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


# --------------------------------------------------------------------------- #
# list-item run parented to its introducing lead-in paragraph
# --------------------------------------------------------------------------- #
def test_list_items_parented_to_introducing_colon_paragraph():
    """A colon-terminated lead-in ('… agree as follows:') is the parent of the
    list-item run it introduces — expressed via parent_id and an OC_PARENT_CHILD
    relationship."""
    xhtml = _doc(
        [
            _page(
                [
                    _p(
                        [("Intro", 72, 100), ("follows", 104, 150), (":", 152, 156)],
                        top=100,
                    ),
                    _p(
                        [("A.", 72, 90), ("first", 94, 130), ("item", 134, 170)],
                        top=120,
                    ),
                    _p(
                        [("B.", 72, 90), ("second", 94, 140), ("item", 144, 180)],
                        top=140,
                    ),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "para", "Intro follows:", (100.0, 72.0, 156.0, 84.0, 12.0)),
        _block(1, "list_item", "A. first item", (120.0, 72.0, 170.0, 98.0, 12.0)),
        _block(2, "list_item", "B. second item", (140.0, 72.0, 180.0, 108.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["0"]["annotationLabel"] == "Paragraph"
    assert by["1"]["parent_id"] == "0"  # list items nest under the lead-in
    assert by["2"]["parent_id"] == "0"
    rels = [r for r in export["relationships"] if r["source_annotation_ids"] == ["0"]]
    assert rels and set(rels[0]["target_annotation_ids"]) == {"1", "2"}
    validate_export(export)


def test_list_items_without_colon_leadin_not_reparented():
    """A plain (non-colon) paragraph does not adopt a following list run."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Plain", 72, 100), ("body", 104, 140)], top=100),
                    _p([("A.", 72, 90), ("item", 94, 130)], top=120),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "para", "Plain body", (100.0, 72.0, 140.0, 68.0, 12.0)),
        _block(1, "list_item", "A. item", (120.0, 72.0, 130.0, 58.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["1"]["parent_id"] is None  # not adopted by a non-colon paragraph


# --------------------------------------------------------------------------- #
# tree consistency: a demoted (non-header) block must not parent children
# --------------------------------------------------------------------------- #
def test_demoted_runin_header_is_spliced_out_of_parent_chain():
    """A header block demoted to Paragraph (run-in heading) must not be a parent;
    its children attach to the nearest still-heading ancestor instead."""
    long_runin = "1.1 " + " ".join(["body"] * 20)  # >12 words -> demoted to Paragraph
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Article", 72, 110), ("One", 114, 140)], top=100),
                    _p([(w, 72, 90) for w in long_runin.split()], top=120),
                    _p([("child", 72, 110), ("clause", 114, 160)], top=140),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "header", "Article One", (100.0, 72.0, 140.0, 68.0, 12.0)),
        _block(
            1,
            "header",
            long_runin,
            (120.0, 72.0, 560.0, 488.0, 12.0),
            level_chain=[{"block_idx": 0, "block_text": "Article One"}],
        ),
        _block(
            2,
            "para",
            "child clause",
            (140.0, 72.0, 160.0, 88.0, 12.0),
            level_chain=[
                {"block_idx": 1, "block_text": long_runin},
                {"block_idx": 0, "block_text": "Article One"},
            ],
        ),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by_id = {a["id"]: a for a in export["labelled_text"]}
    assert by_id["1"]["annotationLabel"] == "Paragraph"  # demoted
    # the demoted block must NOT be anyone's parent
    assert by_id["2"]["parent_id"] == "0"
    assert by_id["1"]["parent_id"] == "0"
    parents = {a["parent_id"] for a in export["labelled_text"]}
    assert "1" not in parents
    validate_export(export)


def test_non_header_block_never_has_children():
    """General invariant: no Paragraph/List Item/Table Row annotation is a parent."""
    export = pdf_ingestor.parse_to_opencontracts(
        str(FIXTURES / "USC Title 1 - CHAPTER 1.pdf")
    )
    by_id = {a["id"]: a for a in export["labelled_text"]}
    parent_ids = {a["parent_id"] for a in export["labelled_text"] if a["parent_id"]}
    for pid in parent_ids:
        assert by_id[pid]["annotationLabel"] in ("Section Header", "Title")


# --------------------------------------------------------------------------- #
# text-anchored token assignment (coverage + untokened recovery)
# --------------------------------------------------------------------------- #
def test_runin_header_demoted_by_assigned_token_count():
    """A run-in heading ('6.7 Audits.') has a tiny block_text but its box absorbs
    the whole clause body, so its assigned token count is large. The word-count
    rule can't see it; the token-count rule demotes it to Paragraph."""
    words = [(f"w{i}", 72 + i * 8, 78 + i * 8) for i in range(30)]
    xhtml = _doc([_page([_p(words, top=100)])])
    line_box = (100.0, 72.0, 320.0, 248.0, 12.0)  # box spans the whole 30-token line
    blocks = [_block(0, "header", "6.7 Audits", line_box)]  # only 2 words of text
    export = to_opencontracts_export(xhtml, blocks)
    a = export["labelled_text"][0]
    assert a["annotation_json"]["0"]["tokensJsons"]  # it really absorbed the tokens
    assert a["annotationLabel"] == "Paragraph"  # ntok > _HEADER_MAX_TOKENS


def test_short_header_with_few_tokens_stays_section_header():
    xhtml = _doc([_page([_p([("Definitions", 72, 140)], top=100)])])
    blocks = [_block(0, "header", "Definitions", (100.0, 72.0, 140.0, 68.0, 12.0))]
    export = to_opencontracts_export(xhtml, blocks)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


# --------------------------------------------------------------------------- #
# furniture / metadata header demotion (60-page audit, class C1)
# --------------------------------------------------------------------------- #
def _header_text_export(text):
    """One page, one ``header`` block whose block_text is exactly ``text`` (its box
    tightly wraps the line, so its assigned token count stays == word count)."""
    words = text.split()
    placed = [(w, 72 + i * 40, 96 + i * 40) for i, w in enumerate(words)]
    xhtml = _doc([_page([_p(placed, top=100)])])
    right = 96.0 + (len(words) - 1) * 40 + 6
    blocks = [_block(0, "header", text, (100.0, 72.0, right, right - 72.0, 12.0))]
    return to_opencontracts_export(xhtml, blocks, title="t")


@pytest.mark.parametrize(
    "text",
    [
        "105",  # bare arabic page folio
        "- 33 -",  # centered dashed page folio
        "iii",  # lowercase roman page folio
        "ii",
        "F-18",  # letter-dash financial-statement folio
        "C-17",
        "April.Jacquez@fortworthtexas.gov",  # bare email (format token)
        "https://fortworthtexas.gov/purchasing/",  # bare URL (format token)
    ],
)
def test_furniture_header_demoted_to_paragraph(text):
    """A header block whose *entire text is a single non-prose structural token*
    (page folio / bare email / bare URL) is not a section heading; the exporter
    relabels it Paragraph (which also splices it out of the parent chain so it
    can't adopt body as children)."""
    export = _header_text_export(text)
    assert export["labelled_text"][0]["annotationLabel"] == "Paragraph", text


@pytest.mark.parametrize(
    "text",
    [
        "ARTICLE 9",  # real headings that look superficially similar
        "Section 9.01 Events of Default",
        "WITNESSETH",
        "RECITALS",
        "Indemnification",
        "Definitions",
        "Date of Report",
        "Notice to Prospective Investors in Canada",
    ],
)
def test_real_heading_not_demoted_as_furniture(text):
    """Genuine section headings must survive the furniture filter."""
    export = _header_text_export(text)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header", text


def test_page_n_of_m_demotion_deferred_for_oracle_neutrality():
    """The 'Page N of M' footer form is *intentionally* left as-is: it is not among
    the 60-page audit's defects and demoting it perturbs the Docling head-ancestor
    oracle (cerebras_ex1013e) beyond its margin. Locking the deferral so a future
    change is a conscious one (would require a Docling baseline regeneration)."""
    export = _header_text_export("Page 4 of 12")
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


@pytest.mark.parametrize(
    "text",
    [
        # --- content/semantic phrases: avoided on principle (match document
        #     *words/meaning*, not structure). Mislabels of these are deferred to a
        #     future geometric (margin-furniture) rule or to the layout engine. ---
        "WHEREAS, the City and Garver entered into",  # recital opener
        "NOW, THEREFORE, the parties agree",
        "Name: Prineha Narang",  # signature-block field
        "Title: City Secretary",
        "By: /s/ Sarah-Marie Martin",
        "INVESTOR:",  # signature-block party label
        "TABLE OF CONTENTS",  # running link (a phrase)
        "Docusign Envelope ID: 2439509D-2B97-43F2",  # e-signature watermark
        "9975 Allentown Boulevard Grantville, PA 17028",  # postal address
        # --- jurisdiction-specific clerk/filing stamps: doubly avoided (content
        #     AND corpus-specific) ---
        "CSC No. 65007",  # municipal City-Secretary-Contract filing stamp
        "OFFICIAL RECORD CITY SECRETARY",  # clerk record stamp
        "FT. WORTH, TX",  # a specific locale
    ],
)
def test_content_based_furniture_deliberately_not_demoted(text):
    """The furniture *text* filter is structural/format only. It deliberately does
    NOT match document content — recital words, signature-field words, specific
    phrases ("TABLE OF CONTENTS", e-signature watermarks), addresses, or
    corpus-specific clerk stamps. (The geometric corner rule below handles the
    corner stamps by *position* instead — never by these words.) Locks the
    structural-vs-content boundary of the text predicate."""
    assert _is_nonheading_furniture(text) is False


# --------------------------------------------------------------------------- #
# geometric corner-furniture demotion (margin band + right-offset; structural)
# --------------------------------------------------------------------------- #
def _positioned_header_export(text, *, top, left, page_w=612.0, page_h=792.0):
    """One page, one ``header`` block whose single line of tokens starts at (left,
    top). Lets a test control the block's *geometry* (corner vs column) so the
    structural corner rule can be exercised independent of the text."""
    words = text.split()
    placed = [(w, left + i * 40, left + i * 40 + 30) for i, w in enumerate(words)]
    right = left + (len(words) - 1) * 40 + 30
    xhtml = _doc([_page([_p(placed, top=top)], width=page_w, height=page_h)])
    blocks = [
        _block(
            0,
            "header",
            text,
            (float(top), float(left), float(right), float(right - left), 12.0),
        )
    ]
    return to_opencontracts_export(xhtml, blocks)


def test_top_right_corner_header_demoted_by_geometry():
    """A header pinned in the top-right page corner (margin band + right-offset) is
    a corner stamp/exhibit-id/running-header, not a section heading — demoted to
    Paragraph by *position*, regardless of its text."""
    # neutral text (not furniture-text, not over-long) so only geometry can demote
    export = _positioned_header_export("Alpha Beta", top=20, left=460)
    assert export["labelled_text"][0]["annotationLabel"] == "Paragraph"


def test_bottom_right_corner_header_demoted_by_geometry():
    export = _positioned_header_export("Alpha Beta", top=760, left=460)
    assert export["labelled_text"][0]["annotationLabel"] == "Paragraph"


def test_single_page_bottom_right_header_above_band_not_demoted():
    """A bottom-right header sitting just *above* the strict deep band (bottom-frac
    ~0.87) on a single page is NOT demoted geometrically — pure position cannot
    tell a one-off real heading ("Integrated Reports Inquiries", caught by the
    hetero-100 eval) from a clerk stamp. Those stamps are handled by cross-page
    *position* repetition instead (see below)."""
    export = _positioned_header_export("Alpha Beta", top=672, left=430)  # bot~0.87
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


# --------------------------------------------------------------------------- #
# cross-page repeated-POSITION furniture (OCR-robust clerk stamps / running marks)
# --------------------------------------------------------------------------- #
def _corner_stamp_doc(stamp_pages, *, total=3, top=667, left=430):
    """`total` pages; on the first `stamp_pages` a header-ish block sits in the
    bottom-right corner with OCR-varying text but the SAME position (a clerk
    stamp). Returns (xhtml, blocks)."""
    ocr_variants = ["OFFICIAL RECORD", "FFICIAL RECORD", "OFFICIAL RECORD CITY"]
    pages, blocks = [], []
    for pg in range(total):
        elems = [_p([("Body", 72, 110), ("content", 114, 200)], top=120)]
        blocks.append(
            _block(
                pg * 2,
                "para",
                "Body content",
                (120.0, 72.0, 200.0, 128.0, 12.0),
                page_idx=pg,
            )
        )
        if pg < stamp_pages:
            t = ocr_variants[pg % len(ocr_variants)]
            placed = [
                (w, left + i * 70, left + i * 70 + 60) for i, w in enumerate(t.split())
            ]
            right = placed[-1][2]
            elems.append(_p(placed, top=top))
            blocks.append(
                _block(
                    pg * 2 + 1,
                    "header",
                    t,
                    (float(top), float(left), float(right), float(right - left), 14.0),
                    page_idx=pg,
                )
            )
        pages.append(_page(elems))
    return _doc(pages), blocks


def test_repeated_corner_stamp_demoted_across_pages():
    """A bottom-right header repeating at the same position on >=2 pages (OCR text
    varying) is a clerk stamp / running mark — demoted, even though no single page
    and no verbatim text would catch it. (legal-100 FortWorth finding.)"""
    xhtml, blocks = _corner_stamp_doc(stamp_pages=2, total=3)
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    # the two stamp headers are block_idx 1 and 3
    assert by["1"]["annotationLabel"] == "Paragraph"
    assert by["3"]["annotationLabel"] == "Paragraph"


def test_one_off_corner_header_not_demoted_by_position():
    """The same corner header on only ONE page is not repeated furniture — kept."""
    xhtml, blocks = _corner_stamp_doc(stamp_pages=1, total=3)
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["1"]["annotationLabel"] == "Section Header"


def test_top_left_header_not_demoted_by_geometry():
    """A header at the left text margin (where real section headings start) stays a
    Section Header even in the top margin band."""
    export = _positioned_header_export("Alpha Beta", top=20, left=72)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


def test_top_center_header_not_demoted_by_geometry():
    """A centered title (left-offset < page center) is not corner furniture."""
    export = _positioned_header_export("Alpha Beta", top=20, left=250)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


def test_mid_page_right_header_not_demoted_by_geometry():
    """Right-offset but *not* in a margin band (mid-page) — not corner furniture."""
    export = _positioned_header_export("Alpha Beta", top=400, left=460)
    assert export["labelled_text"][0]["annotationLabel"] == "Section Header"


def test_furniture_header_does_not_bear_children():
    """Generic furniture the engine made a header (here a bare email) must not
    parent the document body once demoted: the body re-attaches to the real
    heading instead."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("clerk@city.gov", 60, 160)], top=40),
                    _p([("AGREEMENT", 250, 360)], top=100),
                    _p([("This", 72, 96), ("Agreement", 100, 170)], top=140),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "header", "clerk@city.gov", (40.0, 60.0, 160.0, 100.0, 12.0)),
        _block(
            1,
            "header",
            "AGREEMENT",
            (100.0, 250.0, 360.0, 110.0, 12.0),
            level_chain=[{"block_idx": 0, "block_text": "clerk@city.gov"}],
        ),
        _block(
            2,
            "para",
            "This Agreement",
            (140.0, 72.0, 170.0, 98.0, 12.0),
            level_chain=[
                {"block_idx": 1, "block_text": "AGREEMENT"},
                {"block_idx": 0, "block_text": "clerk@city.gov"},
            ],
        ),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by_id = {a["id"]: a for a in export["labelled_text"]}
    assert by_id["0"]["annotationLabel"] == "Paragraph"  # email demoted
    assert by_id["1"]["annotationLabel"] == "Section Header"  # real heading
    # the demoted furniture parents nobody; body nests under the real heading
    assert "0" not in {a["parent_id"] for a in export["labelled_text"]}
    assert by_id["2"]["parent_id"] == "1"
    validate_export(export)


# --------------------------------------------------------------------------- #
# block re-segmentation: a fused centered heading + body para is split
# --------------------------------------------------------------------------- #
def _fused_heading_body_export(*, heading_centered=True, page_w=612.0):
    """One 'para' block fusing a (centered) heading line with a left body block."""
    if heading_centered:
        head = _p([("RECITALS", 281, 331)], top=100)  # narrow, symmetric margins
    else:
        head = _p([("Heading", 50, 110)], top=100)  # left-aligned, not a heading
    body1 = _p(
        [
            ("WHEREAS", 72, 120),
            ("the", 124, 150),
            ("body", 154, 200),
            ("runs", 204, 250),
            ("wide", 254, 300),
            ("across", 304, 380),
            ("the", 384, 420),
            ("page", 424, 540),
        ],
        top=114,
    )
    body2 = _p(
        [("here", 72, 110), ("and", 114, 150), ("continues", 154, 250)],
        top=128,
    )
    xhtml = _doc([_page([head, body1, body2], width=page_w)])
    text = "RECITALS WHEREAS the body runs wide across the page here and continues"
    blocks = [_block(0, "para", text, (100.0, 50.0, 560.0, 510.0, 40.0))]
    return to_opencontracts_export(xhtml, blocks)


def test_fused_centered_heading_is_split_from_body():
    export = _fused_heading_body_export(heading_centered=True)
    anns = export["labelled_text"]
    assert len(anns) == 2  # heading + body, not one fused block
    heading = next(a for a in anns if a["annotationLabel"] == "Section Header")
    body = next(a for a in anns if a["annotationLabel"] == "Paragraph")
    assert heading["rawText"].split() == ["RECITALS"]
    assert "WHEREAS" in body["rawText"] and "continues" in body["rawText"]
    assert body["parent_id"] == heading["id"]  # the heading parents its body
    # tokens partitioned, not duplicated
    htok = [r["tokenIndex"] for r in heading["annotation_json"]["0"]["tokensJsons"]]
    btok = [r["tokenIndex"] for r in body["annotation_json"]["0"]["tokensJsons"]]
    assert set(htok).isdisjoint(btok)
    assert htok == [0]
    validate_export(export)


def test_left_aligned_lead_line_is_not_split():
    """A short left-aligned first line is ordinary body, not a centered heading."""
    export = _fused_heading_body_export(heading_centered=False)
    assert len(export["labelled_text"]) == 1  # no split


def test_centered_furniture_line_not_promoted_to_heading():
    """Re-segmentation must not promote a *furniture* leading line (email, recital,
    address, folio) to Section Header even when it is geometrically centered — the
    furniture filter applies on the split path too, not just to header blocks."""
    head = _p([("a@b.com", 281, 331)], top=100)  # centered, narrow, symmetric
    body1 = _p(
        [
            ("The", 72, 120),
            ("body", 124, 200),
            ("runs", 204, 260),
            ("wide", 264, 340),
            ("across", 344, 420),
            ("the", 424, 460),
            ("page", 464, 540),
        ],
        top=114,
    )
    body2 = _p([("here", 72, 110), ("and", 114, 150), ("on", 154, 200)], top=128)
    xhtml = _doc([_page([head, body1, body2], width=612.0)])
    text = "a@b.com The body runs wide across the page here and on"
    blocks = [_block(0, "para", text, (100.0, 72.0, 560.0, 488.0, 40.0))]
    export = to_opencontracts_export(xhtml, blocks)
    bad = [
        a
        for a in export["labelled_text"]
        if a["annotationLabel"] == "Section Header"
        and _is_nonheading_furniture(a["rawText"])
    ]
    assert not bad, [a["rawText"] for a in bad]
    validate_export(export)


def test_plain_body_block_is_not_split():
    """A block with no centered heading line stays a single annotation."""
    body = _p(
        [
            (w, 72 + i * 30, 96 + i * 30)
            for i, w in enumerate("a b c d e f g h".split())
        ],
        top=100,
    )
    xhtml = _doc([_page([body])])
    blocks = [_block(0, "para", "a b c d e f g h", (100.0, 72.0, 320.0, 248.0, 12.0))]
    export = to_opencontracts_export(xhtml, blocks)
    assert len(export["labelled_text"]) == 1


def test_block_rect_undercovering_text_still_gets_all_tokens():
    """A block whose box_style rect undercovers its line (a table cell, etc.) must
    still claim all of its words via text alignment, not drop them (coverage)."""
    words = [(f"w{i}", 72 + i * 30, 96 + i * 30) for i in range(8)]
    xhtml = _doc([_page([_p(words, top=100)])])
    text = " ".join(w for w, _, _ in words)
    # rect only covers the first ~2 words (right=140) though the line runs to ~330
    blocks = [_block(0, "table_row", text, (100.0, 72.0, 140.0, 68.0, 12.0))]
    export = to_opencontracts_export(xhtml, blocks)
    refs = export["labelled_text"][0]["annotation_json"]["0"]["tokensJsons"]
    assert [r["tokenIndex"] for r in refs] == list(range(8))  # all 8 recovered
    validate_export(export)


# --------------------------------------------------------------------------- #
# validation (spec §6)
# --------------------------------------------------------------------------- #
def test_validate_export_accepts_well_formed(simple_export):
    validate_export(simple_export)  # must not raise


def test_validate_rejects_page_count_mismatch(simple_export):
    simple_export["page_count"] = 2
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


def test_validate_rejects_dangling_parent_id(simple_export):
    simple_export["labelled_text"][1]["parent_id"] = "999"
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


def test_validate_rejects_out_of_range_token_ref(simple_export):
    simple_export["labelled_text"][0]["annotation_json"]["0"]["tokensJsons"] = [
        {"pageIndex": 0, "tokenIndex": 9999}
    ]
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


def test_validate_rejects_cycle(simple_export):
    # make 0 -> 1 and 1 -> 0
    by_id = {a["id"]: a for a in simple_export["labelled_text"]}
    by_id["0"]["parent_id"] = "1"
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


# --------------------------------------------------------------------------- #
# end-to-end integration on real sample PDFs (no content loss)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["sample.pdf", "USC Title 1 - CHAPTER 1.pdf"])
def test_real_pdf_export_is_valid(name):
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / name))
    validate_export(export)  # all spec §6 invariants hold
    assert export["page_count"] == len(export["pawls_file_content"]) >= 1
    assert export["file_type"] == "application/pdf"


@pytest.mark.parametrize("name", ["sample.pdf", "USC Title 1 - CHAPTER 1.pdf"])
def test_real_pdf_has_tokens_and_annotations(name):
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / name))
    total_tokens = sum(len(p["tokens"]) for p in export["pawls_file_content"])
    assert total_tokens > 100
    assert len(export["labelled_text"]) > 5
    # most annotations are anchored to at least one token
    anchored = sum(
        1
        for a in export["labelled_text"]
        if any(pg["tokensJsons"] for pg in a["annotation_json"].values())
    )
    assert anchored / len(export["labelled_text"]) >= 0.9


@pytest.mark.parametrize("name", ["sample.pdf", "USC Title 1 - CHAPTER 1.pdf"])
def test_real_pdf_no_content_loss(name):
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / name))
    assert _token_text_recall(export) >= 0.95


@pytest.mark.parametrize("name", ["sample.pdf", "USC Title 1 - CHAPTER 1.pdf"])
def test_real_pdf_hierarchy_has_roots_and_children(name):
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / name))
    parents = [a["parent_id"] for a in export["labelled_text"]]
    assert any(p is None for p in parents)  # at least one root
    assert any(p is not None for p in parents)  # at least one child


# --------------------------------------------------------------------------- #
# box-only annotations + optional-field hygiene
# --------------------------------------------------------------------------- #
def test_box_only_annotation_falls_back_to_box_style_bounds():
    # a block positioned in an empty region (no tokens) -> tokensJsons == []
    xhtml = _doc([_page([_p([("Hello", 72, 110)], top=100)])])
    blocks = [
        _block(0, "para", "Hello", (100.0, 72.0, 110.0, 38.0, 12.0)),
        _block(1, "para", "Orphan", (400.0, 300.0, 360.0, 60.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    orphan = export["labelled_text"][1]["annotation_json"]["0"]
    assert orphan["tokensJsons"] == []
    assert orphan["bounds"] == {
        "top": 400.0,
        "left": 300.0,
        "right": 360.0,
        "bottom": 412.0,
    }
    validate_export(export)


def test_validate_rejects_null_optional_token_field(simple_export):
    simple_export["pawls_file_content"][0]["tokens"][0]["is_image"] = None
    with pytest.raises(ExportValidationError):
        validate_export(simple_export)


# --------------------------------------------------------------------------- #
# daemon / ingestor_api path (renderFormat=opencontracts)
# --------------------------------------------------------------------------- #
def test_ingestor_api_render_format_opencontracts(tmp_path):
    import shutil

    from warp_ingest.ingestor import ingestor_api

    src = FIXTURES / "sample.pdf"
    work = tmp_path / "sample.pdf"  # ingest_document unlinks its input
    shutil.copyfile(src, work)
    return_dict, _ = ingestor_api.ingest_document(
        "sample.pdf",
        str(work),
        "application/pdf",
        parse_options={"render_format": "opencontracts"},
    )
    export = return_dict["result"]
    validate_export(export)
    assert export["page_count"] >= 1
    assert export["labelled_text"]


# --------------------------------------------------------------------------- #
# structural corrections: cross-page repeated furniture
# --------------------------------------------------------------------------- #
def _running_header_doc(pages, *, text="Quarterly Report 2026"):
    """N pages, each: a left-margin running header `text` at the top + a body line."""
    page_html = []
    for _ in range(pages):
        page_html.append(
            _page(
                [
                    _p(
                        [
                            (w, 72 + 40 * i, 108 + 40 * i)
                            for i, w in enumerate(text.split())
                        ],
                        top=20,
                    ),
                    _p([("Body", 72, 110), ("content", 114, 170)], top=400),
                ]
            )
        )
    return _doc(page_html)


def test_repeated_top_margin_header_demoted_as_furniture():
    xhtml = _running_header_doc(3)
    blocks = []
    for pg in range(3):
        base = pg * 2
        blocks.append(
            _block(
                base,
                "header",
                "Quarterly Report 2026",
                (20.0, 72.0, 230.0, 158.0, 12.0),
                page_idx=pg,
            )
        )
        blocks.append(
            _block(
                base + 1,
                "para",
                "Body content",
                (400.0, 72.0, 170.0, 98.0, 12.0),
                page_idx=pg,
            )
        )
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    # every repeated banner is demoted; no banner remains a Section Header
    for pg in range(3):
        assert by[str(pg * 2)]["annotationLabel"] == "Paragraph"


def test_one_off_top_margin_header_kept():
    """A header with the same text on only ONE page is not furniture (repetition gate)."""
    xhtml = _running_header_doc(1)
    blocks = [
        _block(0, "header", "Quarterly Report 2026", (20.0, 72.0, 230.0, 158.0, 12.0)),
        _block(1, "para", "Body content", (400.0, 72.0, 170.0, 98.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["0"]["annotationLabel"] == "Section Header"


# --------------------------------------------------------------------------- #
# structural corrections: sparse-cover front-matter (keep one Title)
# --------------------------------------------------------------------------- #
def _centered_p(words_text, *, top, size, x_center=306.0, char_w=9.0):
    """One centered <p>: tokens centered on a 612pt page."""
    half = (len(words_text) * char_w) / 2.0
    x0 = x_center - half
    words = []
    cur = x0
    for w in words_text.split():
        w_px = len(w) * char_w
        words.append((w, round(cur, 1), round(cur + w_px, 1)))
        cur += w_px + char_w
    return _p(words, top=top, height=size, size=size)


def test_sparse_cover_keeps_one_title_demotes_rest():
    """A sparse page-0 cover stack: largest centered line is Title, others Paragraph."""
    xhtml = _doc(
        [
            _page(
                [
                    _centered_p("ACME CORPORATION", top=120, size=20),
                    _centered_p("Form Registration Statement", top=200, size=12),
                    _centered_p("Dated June 2026", top=260, size=10),
                ]
            ),
            _page([_p([("Real", 72, 110), ("body", 114, 160)], top=100)]),
        ]
    )
    blocks = [
        _block(0, "header", "ACME CORPORATION", (120.0, 216.0, 396.0, 180.0, 20.0)),
        _block(
            1,
            "header",
            "Form Registration Statement",
            (200.0, 180.0, 432.0, 252.0, 12.0),
        ),
        _block(2, "header", "Dated June 2026", (260.0, 250.0, 362.0, 112.0, 10.0)),
        _block(3, "para", "Real body", (100.0, 72.0, 160.0, 88.0, 12.0), page_idx=1),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["0"]["annotationLabel"] == "Title"  # most prominent (size 20)
    assert by["1"]["annotationLabel"] == "Paragraph"
    assert by["2"]["annotationLabel"] == "Paragraph"


def test_left_aligned_heading_on_cover_kept():
    """A left-aligned heading on a sparse page 0 is not a centered cover line; kept."""
    xhtml = _doc(
        [
            _page(
                [
                    _centered_p("ACME CORPORATION", top=120, size=20),
                    _centered_p("Dated June 2026", top=180, size=10),
                    _p(
                        [
                            ("ARTICLE", 72, 130),
                            ("1.", 134, 150),
                            ("Definitions", 154, 250),
                        ],
                        top=300,
                    ),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "header", "ACME CORPORATION", (120.0, 216.0, 396.0, 180.0, 20.0)),
        _block(1, "header", "Dated June 2026", (180.0, 250.0, 362.0, 112.0, 10.0)),
        _block(
            2, "header", "ARTICLE 1. Definitions", (300.0, 72.0, 250.0, 178.0, 12.0)
        ),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["2"]["annotationLabel"] == "Section Header"  # left-aligned, untouched


def test_dense_first_page_not_treated_as_cover():
    """A page-0 with many blocks is body, not a cover; centered headings keep label."""
    p_tags = [
        _centered_p("Centered Heading Here", top=40 + 30 * i, size=12)
        for i in range(14)
    ]
    xhtml = _doc([_page(p_tags)])
    blocks = [
        _block(
            i,
            "header",
            "Centered Heading Here",
            (40.0 + 30 * i, 240.0, 372.0, 132.0, 12.0),
        )
        for i in range(14)
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    # 14 blocks > _COVER_MAX_BLOCKS: not a cover. (Also not repeated furniture —
    # they share text but sit mid-page, not in a margin band.)
    assert all(by[str(i)]["annotationLabel"] == "Section Header" for i in range(14))


# --------------------------------------------------------------------------- #
# structural corrections: embedded-in-table fragment demotion
# --------------------------------------------------------------------------- #
def test_header_interleaved_in_table_run_demoted():
    """A header interleaved in a Table Row run, WITHIN a section (so it has a heading
    ancestor its children can climb to), is a mis-promoted table fragment — demoted."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Fee", 72, 100), ("Schedule", 104, 170)], top=100),
                    _p([("Tier", 72, 110), ("Amount", 200, 260)], top=120),
                    _p([("Subtotal", 72, 140)], top=140),
                    _p([("Total", 72, 110), ("999", 200, 240)], top=160),
                ]
            )
        ]
    )
    cap = [{"block_idx": 0, "block_text": "Fee Schedule"}]
    blocks = [
        _block(0, "header", "Fee Schedule", (100.0, 72.0, 170.0, 98.0, 12.0)),
        _block(
            1,
            "table_row",
            "Tier Amount",
            (120.0, 72.0, 260.0, 188.0, 12.0),
            level_chain=cap,
        ),
        _block(
            2, "header", "Subtotal", (140.0, 72.0, 140.0, 68.0, 12.0), level_chain=cap
        ),
        _block(
            3,
            "table_row",
            "Total 999",
            (160.0, 72.0, 240.0, 168.0, 12.0),
            level_chain=cap,
        ),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["0"]["annotationLabel"] == "Section Header"  # the caption stays
    assert by["2"]["annotationLabel"] == "Paragraph"  # fragment demoted (inside table)


def test_embedded_fragment_without_heading_ancestor_kept():
    """A header interleaved in a table run but with NO heading ancestor (empty
    level_chain) is NOT demoted: demoting it would orphan its subtree into
    non-heading roots (the fw_vertigis quote-page regression). Guards that."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Tier", 72, 110), ("Amount", 200, 260)], top=100),
                    _p([("Canada", 72, 140)], top=120),
                    _p([("Total", 72, 110), ("999", 200, 240)], top=140),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "table_row", "Tier Amount", (100.0, 72.0, 260.0, 188.0, 12.0)),
        _block(
            1, "header", "Canada", (120.0, 72.0, 140.0, 68.0, 12.0)
        ),  # level_chain=[]
        _block(2, "table_row", "Total 999", (140.0, 72.0, 240.0, 168.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert (
        by["1"]["annotationLabel"] == "Section Header"
    )  # kept (no ancestor to catch children)


def test_heading_above_table_kept():
    """A heading directly ABOVE a table (table rows only after it) is not embedded."""
    xhtml = _doc(
        [
            _page(
                [
                    _p([("Fee", 72, 100), ("Schedule", 104, 170)], top=100),
                    _p([("Tier", 72, 110), ("Amount", 200, 260)], top=120),
                    _p([("Total", 72, 110), ("999", 200, 240)], top=140),
                ]
            )
        ]
    )
    blocks = [
        _block(0, "header", "Fee Schedule", (100.0, 72.0, 170.0, 98.0, 12.0)),
        _block(1, "table_row", "Tier Amount", (120.0, 72.0, 260.0, 188.0, 12.0)),
        _block(2, "table_row", "Total 999", (140.0, 72.0, 240.0, 168.0, 12.0)),
    ]
    export = to_opencontracts_export(xhtml, blocks)
    by = {a["id"]: a for a in export["labelled_text"]}
    assert by["0"]["annotationLabel"] == "Section Header"  # above, not inside


# --------------------------------------------------------------------------- #
# _is_prose_header: sentence/clause tells (Task 1)
# --------------------------------------------------------------------------- #
def test_is_prose_header_structural_tells():
    P = ex._is_prose_header
    # demote: clause continuations / sentences / mid-sentence starts
    assert P("WHEREAS, the current renewal term expires September 30,")  # ends comma
    assert P("herein, shall remain in full force and effect.")  # lowercase start
    assert P(
        "WHEREAS, the CONTRACT involves engineering services for"
    )  # ends conjunction "for"
    assert P(
        "The parties agree to the following terms, conditions."
    )  # period + interior comma
    assert P("now therefore the parties agree")  # recital trigger (two-word)
    # keep: real labels
    assert not P("AMENDMENTS")
    assert not P("ARTICLE 9")
    assert not P("WITNESSETH")
    assert not P("STATUTORY NOTES AND RELATED SUBSIDIARIES")
    assert not P(
        "Indemnification, Insurance and Liability"
    )  # heading w/ comma, no end punct, not sentence
    # Title-Case name with an "Inc."-style trailing period + interior comma is a
    # title, not prose: its only lowercase word ("of") is a stopword, so the
    # sentence rule must not fire (was wrongly demoting this real exhibit title).
    assert not P("Subsidiaries of Exyn Technologies, Inc.")


def test_is_prose_header_trigger_list_tunable():
    assert not ex._is_prose_header(
        "RECITALS", triggers=frozenset()
    )  # default would not match anyway
    assert ex._is_prose_header(
        "RECITALS follow below", triggers=frozenset({"recitals"})
    )


def test_cross_reference_header_demoted():
    # A numbered cross-reference to another instrument ("Section N of …") is body
    # text, not a section heading. This only matters once a run-in citation's box
    # is correct (its token count no longer trips the absorbed-body demotion) —
    # e.g. forbright's "Section 201 of the … Act", which the buggy shared box used
    # to demote by accident.
    X = ex._is_cross_reference_header
    assert X(
        "Section 201 of the Economic Growth, Regulatory Relief and Consumer "
        "Protection Act"
    )
    assert X("Article 4 of the Agreement")
    assert X("Section 10.2 of this Agreement")
    assert X("Sec. 3 of the Code")
    # genuine headings: a title follows the enumerator, never "of" straight after
    for t in (
        "6. Representations and Warranties",
        "1.1 Place of Meetings",
        "SECTION 13. WRITTEN ACTION BY DIRECTORS",
        "Section 5. Effectiveness",
        "12.3 Survival of Representations and Warranties",
        "ARTICLE II. TERM OF AGREEMENT",
        "Section 10.2 Governing Law",
        "Notice of Meetings",
        "Powers of the Board",
    ):
        assert not X(t), t


def test_cross_reference_header_demoted_via_resolve_label():
    block = _block(
        0, "header", "Section 201 of the Companies Act", (0, 0, 100, 100, 12)
    )
    assert ex._resolve_label(block, ntok=6) == "Paragraph"
    genuine = _block(0, "header", "6.6 Taxes", (0, 0, 100, 100, 12))
    assert ex._resolve_label(genuine, ntok=2) == "Section Header"


def test_contract_recital_not_section_header():
    exp = pdf_ingestor.parse_to_opencontracts("tests/fixtures/contracts/fw_garver.pdf")
    page0 = [a for a in exp["labelled_text"] if a["page"] == 0]
    checked = 0
    for a in page0:
        rt = a["rawText"].strip()
        if rt.lower().startswith("whereas") or rt.startswith("herein, shall remain"):
            assert a["annotationLabel"] != "Section Header", rt
            checked += 1
    assert checked >= 1, "selector matched nothing — guard is vacuous"
    # guard: a real label-like heading is NOT demoted by the prose rule
    assert not ex._is_prose_header("AMENDMENTS")


# --------------------------------------------------------------------------- #
# _is_corner_furniture: deep far-corner footer (Task 2)
# --------------------------------------------------------------------------- #
def test_deep_corner_footer_demoted_but_headings_spared():
    f = ex._is_corner_furniture
    # bottom-right clerk stamp just inside the strict band (fw_vertigis id22/23)
    assert f((0.886, 0.902, 0.79))  # deep far-corner footer
    assert f((0.863, 0.886, 0.778))  # same stamp, slightly higher
    # spared: centered heading low on page, left-margin footer
    assert not f((0.95, 0.97, 0.42))  # centered (left_frac < 0.70)
    assert not f((0.95, 0.97, 0.10))  # left-aligned footer
    assert not f((0.50, 0.52, 0.80))  # right-offset but mid-page (not bottom band)


def test_fw_vertigis_clerk_stamp_not_header():
    exp = pdf_ingestor.parse_to_opencontracts(
        "tests/fixtures/contracts/fw_vertigis.pdf"
    )
    checked = 0
    for a in exp["labelled_text"]:
        if a["page"] == 0 and a["rawText"].strip().upper() in (
            "OFFICIAL RECORD CITY SECRETARY",
            "FT. WORTH, TX",
        ):
            assert a["annotationLabel"] != "Section Header", a["rawText"]
            checked += 1
    assert checked >= 1, "selector matched nothing — guard is vacuous"


def test_exyn_exhibit_title_preserved_as_heading():
    # The Title-Case exhibit title "Subsidiaries of Exyn Technologies, Inc." must
    # NOT be demoted by the prose rule — it stays a heading (Section Header/Title).
    exp = pdf_ingestor.parse_to_opencontracts(
        "tests/fixtures/s1/exyn_s1__ex211_812970.pdf"
    )
    headings = [
        a
        for a in exp["labelled_text"]
        if a["annotationLabel"] in ("Section Header", "Title")
        and "Subsidiaries of Exyn" in a["rawText"]
    ]
    assert headings, "exhibit title was demoted away from a heading label"


def test_fw_vertigis_recital_not_re_promoted_by_segmentation():
    # The centered WHEREAS recital line that _resolve_label demotes to Paragraph
    # must NOT be re-promoted to Section Header by the re-segmentation split path.
    exp = pdf_ingestor.parse_to_opencontracts(
        "tests/fixtures/contracts/fw_vertigis.pdf"
    )
    bad = [
        a
        for a in exp["labelled_text"]
        if a["page"] == 0
        and a["annotationLabel"] == "Section Header"
        and "renewal term expires" in a["rawText"]
    ]
    assert not bad, [a["rawText"] for a in bad]
    # Non-vacuity: the recital text must appear on page 0 under some label.
    present = [
        a
        for a in exp["labelled_text"]
        if a["page"] == 0 and "renewal term expires" in a["rawText"]
    ]
    assert len(present) >= 1, "selector matched nothing — guard is vacuous"


# --------------------------------------------------------------------------- #
# engine: do not font-promote justified recital prose to header (Task 3)
# --------------------------------------------------------------------------- #
def _blocks(path):
    html_doc = pdf_ingestor.parse_pdf(path, {})
    blocks, *_ = pdf_ingestor.parse_blocks(html_doc, render_format="all")
    return blocks


def test_engine_does_not_promote_recital_to_header():
    blocks = _blocks("tests/fixtures/contracts/fw_garver.pdf")
    checked = 0
    for b in blocks:
        if b["page_idx"] == 0 and b["block_text"].strip().lower().startswith("whereas"):
            assert b["block_type"] not in (
                "header",
                "header_modified",
                "inline_header",
            ), b["block_text"][:60]
            checked += 1
    assert checked >= 1, "selector matched nothing — guard is vacuous"


# --------------------------------------------------------------------------- #
# recital_triggers passthrough: parse_options key threads end-to-end
# --------------------------------------------------------------------------- #
def test_recital_triggers_passthrough():
    """A custom recital_triggers list replaces the default and demotes matching
    headers to Paragraph through the full parse_to_opencontracts public API."""
    fw = "tests/fixtures/contracts/fw_vertigis.pdf"

    # Default: "AMENDMENTS" stays a Section Header.
    exp_default = pdf_ingestor.parse_to_opencontracts(fw)
    amendments_default = [
        a
        for a in exp_default["labelled_text"]
        if a["rawText"].strip() == "AMENDMENTS"
        and a["annotationLabel"] == "Section Header"
    ]
    assert amendments_default, (
        "AMENDMENTS is not a Section Header in the default export — "
        "test fixture assumption has changed"
    )

    # Custom: adding "amendments" to triggers must demote it to Paragraph.
    exp_custom = pdf_ingestor.parse_to_opencontracts(
        fw,
        parse_options={"recital_triggers": ["whereas", "now therefore", "amendments"]},
    )
    amendments_custom = [
        a
        for a in exp_custom["labelled_text"]
        if a["rawText"].strip() == "AMENDMENTS"
        and a["annotationLabel"] == "Section Header"
    ]
    assert (
        not amendments_custom
    ), "AMENDMENTS was not demoted when 'amendments' was in custom recital_triggers"


# --------------------------------------------------------------------------- #
# run-in header token collision: a heading that shares a physical line with the
# body that follows it steals the body's tokens (geometry can't tell them apart).
# _reconcile_runin_overflow re-partitions by text so each block's tokens are
# faithful to its own rawText — the reconstructed text layer matches a flat
# extract. These exercise the pure re-partition and the end-to-end guarantees.
# --------------------------------------------------------------------------- #
def test_reconcile_donates_runin_body_to_starved_next_block():
    # header block 0 == "1. Term"; its line-box also covered the body, so geometry
    # gave it every token and left the paragraph (block 1) with none.
    token_texts = ["1.", "Term", "The", "parties", "agree"]
    owner = [0, 0, 0, 0, 0]
    block_words = {0: ["1.", "Term"], 1: ["The", "parties", "agree"]}
    out = _reconcile_runin_overflow(list(owner), token_texts, [0, 1], block_words, {0})
    assert out == [0, 0, 1, 1, 1]


def test_reconcile_is_noop_when_already_faithful():
    token_texts = ["1.", "Term", "The", "parties", "agree"]
    owner = [0, 0, 1, 1, 1]
    block_words = {0: ["1.", "Term"], 1: ["The", "parties", "agree"]}
    out = _reconcile_runin_overflow(list(owner), token_texts, [0, 1], block_words, {0})
    assert out == [0, 0, 1, 1, 1]


def test_reconcile_only_headers_may_donate():
    # identical collision, but block 0 is not a header donor -> nothing moves.
    token_texts = ["1.", "Term", "The", "parties", "agree"]
    owner = [0, 0, 0, 0, 0]
    block_words = {0: ["1.", "Term"], 1: ["The", "parties", "agree"]}
    out = _reconcile_runin_overflow(
        list(owner), token_texts, [0, 1], block_words, frozenset()
    )
    assert out == [0, 0, 0, 0, 0]


def test_reconcile_tolerates_boundary_punctuation():
    # the boundary token carries the heading's terminal period ("Term.") — it must
    # still be recognised as the heading's own word, not overflow.
    token_texts = ["1.", "Term.", "The", "parties"]
    owner = [0, 0, 0, 0]
    block_words = {0: ["1.", "Term"], 1: ["The", "parties"]}
    out = _reconcile_runin_overflow(list(owner), token_texts, [0, 1], block_words, {0})
    assert out == [0, 0, 1, 1]


def test_reconcile_fills_only_the_missing_opening():
    # block 1 already holds its wrapped tail whose first token ("(the") merely
    # *looks* like its opening word ("The"); the donor must still fill the true
    # opening ("The cat") without duplicating anything.
    token_texts = ["Sec", "The", "cat", "(the", "dog)"]
    owner = [0, 0, 0, 1, 1]
    block_words = {0: ["Sec"], 1: ["The", "cat", "(the", "dog)"]}
    out = _reconcile_runin_overflow(list(owner), token_texts, [0, 1], block_words, {0})
    assert out == [0, 1, 1, 1, 1]


def test_reconcile_never_drops_or_duplicates_tokens():
    token_texts = ["1.", "Term", "The", "parties", "agree", "here"]
    owner = [0, 0, 0, 0, 0, 0]
    block_words = {0: ["1.", "Term"], 1: ["The", "parties", "agree", "here"]}
    out = _reconcile_runin_overflow(list(owner), token_texts, [0, 1], block_words, {0})
    # every originally-owned token still has exactly one owner; none lost
    assert all(o is not None for o in out)
    assert len(out) == len(owner)


def _runin_page_export():
    """A page whose single physical line is a run-in header: the heading
    '1. Term.' and the body 'The parties agree to this.' share one line, so the
    header block's box also spans the body."""
    line = [
        ("1.", 72, 82),
        ("Term.", 86, 118),
        ("The", 122, 140),
        ("parties", 144, 186),
        ("agree", 190, 218),
        ("to", 222, 234),
        ("this.", 238, 262),
    ]
    xhtml = _doc([_page([_p(line, top=100)])])
    # both blocks are (correctly) segmented in text but share the line's box.
    box = (100.0, 72.0, 262.0, 190.0, 12.0)
    blocks = [
        _block(0, "header", "1. Term.", box),
        _block(
            1,
            "para",
            "The parties agree to this.",
            box,
            level_chain=[{"block_idx": 0, "block_text": "1. Term."}],
        ),
    ]
    return to_opencontracts_export(xhtml, blocks)


def test_runin_paragraph_recovers_its_own_tokens():
    export = _runin_page_export()
    by_id = {a["id"]: a for a in export["labelled_text"]}
    header_tokens = by_id["0"]["annotation_json"]["0"]["tokensJsons"]
    para_tokens = by_id["1"]["annotation_json"]["0"]["tokensJsons"]
    tokens = export["pawls_file_content"][0]["tokens"]
    header_text = " ".join(tokens[t["tokenIndex"]]["text"] for t in header_tokens)
    para_text = " ".join(tokens[t["tokenIndex"]]["text"] for t in para_tokens)
    assert header_text == "1. Term."
    assert para_text == "The parties agree to this."


def test_runin_export_is_a_clean_token_partition():
    # validate_export enforces the fine-layer partition; assert it passes and that
    # the two blocks' token sets are disjoint and cover the whole line.
    export = _runin_page_export()
    validate_export(export)
    claimed = [
        ref["tokenIndex"]
        for a in export["labelled_text"]
        for ref in a["annotation_json"]["0"]["tokensJsons"]
    ]
    assert sorted(claimed) == [0, 1, 2, 3, 4, 5, 6]  # every token, exactly once


def test_validate_export_rejects_double_assigned_token(simple_export):
    # forge a duplicate: give annotation[1] a token already owned by annotation[0]
    stolen = dict(
        simple_export["labelled_text"][0]["annotation_json"]["0"]["tokensJsons"][0]
    )
    simple_export["labelled_text"][1]["annotation_json"]["0"]["tokensJsons"].append(
        stolen
    )
    with pytest.raises(ExportValidationError, match="claimed by two annotations"):
        validate_export(simple_export)


_ETON = (
    "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_"
    "EX-10.1_Development_Agreement_ZrZJLLv.pdf"
)


def test_real_runin_contract_blocks_are_faithful():
    # A real numbered-clause contract: every "N.M Heading" run-in used to hand its
    # whole clause body to the heading and starve the paragraph. Post-fix, no
    # annotation grossly overspans its own rawText and the export is a clean
    # partition (validate_export). Off-by-one tokenisation and the deferred
    # signature grid are tolerated.
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / _ETON))
    validate_export(export)
    overspanned = []
    for a in export["labelled_text"]:
        n_words = len(a["rawText"].split())
        n_tok = sum(
            len(v.get("tokensJsons", [])) for v in a["annotation_json"].values()
        )
        if n_words and n_tok > n_words * 3 and n_tok - n_words >= 15:
            overspanned.append((a["id"], a["annotationLabel"], n_words, n_tok))
    assert not overspanned, f"blocks absorbed neighbours' tokens: {overspanned}"


def test_real_runin_contract_paragraphs_are_not_starved():
    # every substantial paragraph/list block (>= 6 words) grounds to some tokens;
    # before the fix the run-in victims had zero.
    export = pdf_ingestor.parse_to_opencontracts(str(FIXTURES / _ETON))
    starved = []
    for a in export["labelled_text"]:
        if a["annotationLabel"] not in ("Paragraph", "List Item"):
            continue
        n_words = len(a["rawText"].split())
        n_tok = sum(
            len(v.get("tokensJsons", [])) for v in a["annotation_json"].values()
        )
        if n_words >= 6 and n_tok == 0:
            starved.append((a["id"], a["rawText"][:50]))
    assert not starved, f"paragraphs left with no tokens: {starved}"


# --------------------------------------------------------------------------- #
# image tokens (issue #1; design spec 2026-07-03)
# --------------------------------------------------------------------------- #
def _img_tok(x, y, w, h):
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "text": "",
        "is_image": True,
        "base64_data": "aGk=",
        "format": "png",
        "content_hash": "0" * 64,
        "original_width": 10,
        "original_height": 10,
        "image_type": "embedded",
    }


def _text_ann(aid, label, bounds, parent=None, tok_refs=()):
    return {
        "id": aid,
        "annotationLabel": label,
        "rawText": "x",
        "page": 0,
        "annotation_json": {
            "0": {
                "bounds": dict(bounds),
                "tokensJsons": list(tok_refs),
                "rawText": "x",
            }
        },
        "parent_id": parent,
        "annotation_type": "TOKEN_LABEL",
        "structural": True,
        "content_modalities": ["TEXT"],
    }


class TestAppendImageLayer:
    def _pawls(self):
        return [
            {
                "page": {"width": 612.0, "height": 792.0, "index": 0},
                "tokens": [
                    {"x": 72, "y": 100, "width": 40, "height": 12, "text": "Hello"}
                ],
            }
        ]

    def test_attach_inside_annotation(self):
        pawls = self._pawls()
        para = _text_ann(
            "1",
            "Paragraph",
            {"top": 130, "left": 72, "right": 540, "bottom": 400},
            parent="0",
            tok_refs=[{"pageIndex": 0, "tokenIndex": 0}],
        )
        anns = [para]
        ex._append_image_layer(pawls, anns, {0: [_img_tok(200, 200, 100, 80)]})
        assert len(anns) == 1  # attached, no standalone annotation
        pj = para["annotation_json"]["0"]
        assert {"pageIndex": 0, "tokenIndex": 1} in pj["tokensJsons"]
        assert para["content_modalities"] == ["IMAGE", "TEXT"]
        assert pawls[0]["tokens"][1]["is_image"] is True

    def test_standalone_parents_to_preceding_heading(self):
        pawls = self._pawls()
        heading = _text_ann(
            "0",
            "Section Header",
            {"top": 100, "left": 72, "right": 112, "bottom": 112},
        )
        anns = [heading]
        ex._append_image_layer(pawls, anns, {0: [_img_tok(200, 300, 100, 80)]})
        img = anns[-1]
        assert img["id"] == "img-0-0" and img["annotationLabel"] == "Image"
        assert img["parent_id"] == "0"
        assert img["content_modalities"] == ["IMAGE"]
        assert img["rawText"] == ""
        assert img["annotation_json"]["0"]["tokensJsons"] == [
            {"pageIndex": 0, "tokenIndex": 1}
        ]

    def test_standalone_inherits_parent_from_preceding_paragraph(self):
        pawls = self._pawls()
        heading = _text_ann(
            "0",
            "Section Header",
            {"top": 100, "left": 72, "right": 112, "bottom": 112},
        )
        para = _text_ann(
            "1",
            "Paragraph",
            {"top": 130, "left": 72, "right": 540, "bottom": 200},
            parent="0",
        )
        anns = [heading, para]
        # image below the paragraph, too far to attach
        ex._append_image_layer(pawls, anns, {0: [_img_tok(200, 500, 100, 80)]})
        assert anns[-1]["parent_id"] == "0"  # inherited from the paragraph

    def test_standalone_with_no_preceding_annotation_is_root(self):
        pawls = self._pawls()
        para = _text_ann(
            "1",
            "Paragraph",
            {"top": 400, "left": 72, "right": 540, "bottom": 500},
        )
        anns = [para]
        ex._append_image_layer(pawls, anns, {0: [_img_tok(500, 20, 40, 40)]})
        assert anns[-1]["parent_id"] is None

    def test_append_only_text_tokens_unshifted(self):
        pawls = self._pawls()
        before = list(pawls[0]["tokens"])
        ex._append_image_layer(pawls, [], {0: [_img_tok(200, 300, 100, 80)]})
        assert pawls[0]["tokens"][: len(before)] == before


class TestIncludeImagesEndToEnd:
    def test_flag_requires_pdf_bytes(self):
        with pytest.raises(ValueError, match="pdf_bytes"):
            to_opencontracts_export("<html></html>", [], include_images=True)

    def test_fixture_export_valid_with_images(self):
        export = pdf_ingestor.parse_to_opencontracts(
            str(FIXTURES / "with_images.pdf"),
            parse_options={"include_images": True},
        )
        validate_export(export)
        img_toks = [
            t
            for p in export["pawls_file_content"]
            for t in p["tokens"]
            if t.get("is_image")
        ]
        assert len(img_toks) == 2
        img_anns = [
            a for a in export["labelled_text"] if a["annotationLabel"] == "Image"
        ]
        assert img_anns  # at least the corner logo stands alone
        for a in img_anns:
            assert a["content_modalities"] == ["IMAGE"]

    def test_additivity_flag_off_unchanged(self):
        path = str(FIXTURES / "with_images.pdf")
        off = pdf_ingestor.parse_to_opencontracts(path)
        on = pdf_ingestor.parse_to_opencontracts(
            path, parse_options={"include_images": True}
        )
        # text-token layer identical; non-image annotations DEEP-equal (this
        # fixture produces no attach case, so nothing outside the image layer
        # may differ in any field — ids, bounds, refs, modalities, parents)
        for p_on, p_off in zip(on["pawls_file_content"], off["pawls_file_content"]):
            assert [t for t in p_on["tokens"] if not t.get("is_image")] == p_off[
                "tokens"
            ]
        assert [
            a for a in on["labelled_text"] if not str(a["id"]).startswith("img-")
        ] == off["labelled_text"]
        # relationships identical once image ids are stripped from edge targets
        rels_on = [
            {
                **r,
                "target_annotation_ids": [
                    t
                    for t in r["target_annotation_ids"]
                    if not str(t).startswith("img-")
                ],
            }
            for r in on["relationships"]
        ]
        assert rels_on == off["relationships"]
        assert not any(
            t.get("is_image") for p in off["pawls_file_content"] for t in p["tokens"]
        )


def _min_export(tokens, anns):
    return {
        "title": "x",
        "content": "hello",
        "description": None,
        "page_count": 1,
        "pawls_file_content": [
            {"page": {"width": 612.0, "height": 792.0, "index": 0}, "tokens": tokens}
        ],
        "doc_labels": [],
        "labelled_text": anns,
        "relationships": [],
    }


class TestValidateImageTokens:
    def _tok(self, **kw):
        base = _img_tok(10, 10, 50, 40)
        base.update(kw)
        return base

    def test_valid_image_token_passes(self):
        validate_export(_min_export([self._tok()], []))

    def test_image_token_with_text_fails(self):
        with pytest.raises(ExportValidationError, match="non-empty text"):
            validate_export(_min_export([self._tok(text="oops")], []))

    def test_image_token_without_data_fails(self):
        tok = self._tok()
        del tok["base64_data"]
        with pytest.raises(ExportValidationError, match="image_path or base64"):
            validate_export(_min_export([tok], []))

    def test_bad_format_fails(self):
        with pytest.raises(ExportValidationError, match="format"):
            validate_export(_min_export([self._tok(format="webp")], []))

    def test_bad_image_type_fails(self):
        with pytest.raises(ExportValidationError, match="image_type"):
            validate_export(_min_export([self._tok(image_type="inline")], []))

    def test_bad_original_dims_fail(self):
        with pytest.raises(ExportValidationError, match="original_width"):
            validate_export(_min_export([self._tok(original_width=0)], []))

    def test_bad_content_hash_fails(self):
        with pytest.raises(ExportValidationError, match="content_hash"):
            validate_export(_min_export([self._tok(content_hash="zz")], []))

    def test_image_field_on_text_token_fails(self):
        tok = {
            "x": 1,
            "y": 1,
            "width": 5,
            "height": 5,
            "text": "hi",
            "base64_data": "aGk=",
        }
        with pytest.raises(ExportValidationError, match="image-only field"):
            validate_export(_min_export([tok], []))


class TestValidateModalities:
    def test_image_ref_requires_image_modality(self):
        ann = _text_ann(
            "0",
            "Image",
            {"top": 10, "left": 10, "right": 60, "bottom": 50},
            tok_refs=[{"pageIndex": 0, "tokenIndex": 0}],
        )
        ann["content_modalities"] = ["TEXT"]  # wrong: ref points at an image
        with pytest.raises(ExportValidationError, match="IMAGE modality"):
            validate_export(_min_export([_img_tok(10, 10, 50, 40)], [ann]))

    def test_text_ref_requires_text_modality(self):
        ann = _text_ann(
            "0",
            "Paragraph",
            {"top": 10, "left": 10, "right": 60, "bottom": 50},
            tok_refs=[{"pageIndex": 0, "tokenIndex": 0}],
        )
        ann["content_modalities"] = ["IMAGE"]
        tok = {"x": 10, "y": 10, "width": 40, "height": 12, "text": "hi"}
        with pytest.raises(ExportValidationError, match="TEXT modality"):
            validate_export(_min_export([tok], [ann]))

    def test_bad_modality_value_fails(self):
        ann = _text_ann(
            "0",
            "Paragraph",
            {"top": 10, "left": 10, "right": 60, "bottom": 50},
        )
        ann["content_modalities"] = ["AUDIO"]
        with pytest.raises(ExportValidationError, match="content_modalities"):
            validate_export(_min_export([], [ann]))


class TestImagesWithSemanticUnits:
    def test_both_flags_valid_and_units_carry_image_modality(self):
        export = pdf_ingestor.parse_to_opencontracts(
            str(FIXTURES / "with_images.pdf"),
            parse_options={"include_images": True, "semantic_units": True},
        )
        validate_export(export)
        by_id = {a["id"]: a for a in export["labelled_text"]}
        member_of = {
            rel["source_annotation_ids"][0]: rel["target_annotation_ids"]
            for rel in export["relationships"]
            if rel["relationshipLabel"] == "OC_SEMANTIC_UNIT"
        }
        assert member_of  # units exist for this fixture
        for su_id, members in member_of.items():
            want = sorted(
                {
                    md
                    for m in members
                    for md in (by_id[m].get("content_modalities") or ("TEXT",))
                }
            ) or ["TEXT"]
            assert by_id[su_id]["content_modalities"] == want

    def test_unit_union_includes_image_modality_from_attach_member(self):
        # The attach case: a text annotation carrying an image token ref
        # (["IMAGE","TEXT"]) becomes a unit member; the unit must union it.
        from warp_ingest.ingestor.semantic_units import append_semantic_units

        heading = _text_ann(
            "0",
            "Section Header",
            {"top": 72, "left": 72, "right": 200, "bottom": 84},
        )
        heading["rawText"] = "ARTICLE 1"
        heading["annotation_json"]["0"]["rawText"] = "ARTICLE 1"
        body = _text_ann(
            "1",
            "Paragraph",
            {"top": 100, "left": 72, "right": 540, "bottom": 300},
            parent="0",
            tok_refs=[
                {"pageIndex": 0, "tokenIndex": 0},
                {"pageIndex": 0, "tokenIndex": 1},
            ],
        )
        body["rawText"] = "The parties agree to the delivery schedule shown below."
        body["annotation_json"]["0"]["rawText"] = body["rawText"]
        body["content_modalities"] = ["IMAGE", "TEXT"]
        tokens = [
            {"x": 72, "y": 100, "width": 60, "height": 12, "text": "The"},
            _img_tok(200, 150, 100, 80),
        ]
        export = _min_export(tokens, [heading, body])
        append_semantic_units(export)
        validate_export(export)
        units = [
            a
            for a in export["labelled_text"]
            if a["annotationLabel"] == "Semantic Unit"
        ]
        assert units
        member_of = {
            rel["source_annotation_ids"][0]: rel["target_annotation_ids"]
            for rel in export["relationships"]
            if rel["relationshipLabel"] == "OC_SEMANTIC_UNIT"
        }
        with_body = [u for u in units if "1" in member_of.get(u["id"], ())]
        assert with_body, "the attach-case paragraph never became a unit member"
        assert with_body[0]["content_modalities"] == ["IMAGE", "TEXT"]
