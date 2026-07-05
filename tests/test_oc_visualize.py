"""Tests for the OpenContractDocExport visualization / diagnostics utilities.

The pure functions (``build_tree``, ``render_tree_outline``, ``diagnostics``) are
exercised on synthetic exports; ``project_page`` gets a smoke test on a real
fixture, guarded on pypdfium2 being importable (same gating style as the OCR
tests).
"""

import pytest

from warp_ingest.ingestor import oc_visualize


# --------------------------------------------------------------------------- #
# synthetic export builders
# --------------------------------------------------------------------------- #
def _ann(aid, label, page, parent, text, *, top=0.0, left=0.0, tokens=None):
    """One labelled_text annotation; tokens is a list of (pageIndex, tokenIndex)."""
    tokens = tokens or []
    return {
        "id": aid,
        "annotationLabel": label,
        "rawText": text,
        "page": page,
        "annotation_json": {
            str(page): {
                "bounds": {
                    "top": top,
                    "left": left,
                    "right": left + 100,
                    "bottom": top + 10,
                },
                "tokensJsons": [{"pageIndex": p, "tokenIndex": t} for p, t in tokens],
                "rawText": text,
            }
        },
        "parent_id": parent,
        "annotation_type": "TOKEN_LABEL",
        "structural": True,
        "content_modalities": ["TEXT"],
    }


def _export(annotations, *, n_pages=1, tokens_per_page=10, title="Doc"):
    pawls = [
        {
            "page": {"width": 612.0, "height": 792.0, "index": i},
            "tokens": [
                {
                    "x": 1.0 * j,
                    "y": 1.0 * i,
                    "width": 5.0,
                    "height": 8.0,
                    "text": f"w{i}_{j}",
                }
                for j in range(tokens_per_page)
            ],
        }
        for i in range(n_pages)
    ]
    return {
        "title": title,
        "content": "x",
        "description": None,
        "pawls_file_content": pawls,
        "page_count": n_pages,
        "doc_labels": [],
        "labelled_text": annotations,
        "relationships": [],
        "file_type": "application/pdf",
    }


# --------------------------------------------------------------------------- #
# build_tree
# --------------------------------------------------------------------------- #
def test_build_tree_nesting_depth_and_subtree_root():
    export = _export(
        [
            _ann("h1", "Section Header", 0, None, "Article 1", top=10),
            _ann("h2", "Section Header", 0, "h1", "1.1 Defs", top=20),
            _ann("p1", "Paragraph", 0, "h2", "body text", top=30),
            _ann("h3", "Section Header", 0, None, "Article 2", top=40),
        ]
    )
    tree = oc_visualize.build_tree(export)

    assert tree.roots == ["h1", "h3"]  # reading order, parent None
    assert tree.children["h1"] == ["h2"]
    assert tree.children["h2"] == ["p1"]
    assert tree.depth["h1"] == 0
    assert tree.depth["h2"] == 1
    assert tree.depth["p1"] == 2
    assert tree.subtree_root["p1"] == "h1"
    assert tree.subtree_root["h3"] == "h3"


def test_build_tree_missing_parent_is_treated_as_root():
    export = _export(
        [
            _ann("p1", "Paragraph", 0, "ghost", "orphaned body", top=10),
        ]
    )
    tree = oc_visualize.build_tree(export)
    assert tree.roots == ["p1"]
    assert tree.parent["p1"] is None
    assert tree.depth["p1"] == 0


def test_build_tree_is_cycle_safe():
    # validate_export forbids cycles, but the visualizer must never hang on one.
    export = _export(
        [
            _ann("a", "Paragraph", 0, "b", "a", top=10),
            _ann("b", "Paragraph", 0, "a", "b", top=20),
        ]
    )
    tree = oc_visualize.build_tree(export)  # must terminate
    for aid in ("a", "b"):
        assert tree.depth[aid] >= 0


def test_build_tree_children_in_reading_order():
    # children sorted by (page, top, left), regardless of list order.
    export = _export(
        [
            _ann("h", "Section Header", 0, None, "H", top=10),
            _ann("c2", "Paragraph", 0, "h", "second", top=50),
            _ann("c1", "Paragraph", 0, "h", "first", top=20),
        ]
    )
    tree = oc_visualize.build_tree(export)
    assert tree.children["h"] == ["c1", "c2"]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_flags_childbearing_non_header():
    # a Paragraph that has children is a structural smell (demoted run-in header).
    export = _export(
        [
            _ann("p", "Paragraph", 0, None, "run-in heading body ...", top=10),
            _ann("c", "Paragraph", 0, "p", "child clause", top=20),
        ]
    )
    diag = oc_visualize.diagnostics(export)
    assert diag["childbearing_non_headers"] == 1
    assert "p" in diag["childbearing_non_header_ids"]


def test_diagnostics_allows_paragraph_parenting_list_items():
    # a lead-in paragraph whose children are all List Items is intended structure,
    # not a childbearing-non-header smell.
    export = _export(
        [
            _ann("p", "Paragraph", 0, None, "intro as follows:", top=10),
            _ann("a", "List Item", 0, "p", "A. first", top=20),
            _ann("b", "List Item", 0, "p", "B. second", top=30),
        ]
    )
    diag = oc_visualize.diagnostics(export)
    assert diag["childbearing_non_headers"] == 0


def test_diagnostics_flags_non_heading_roots():
    export = _export(
        [
            _ann("h", "Section Header", 0, None, "Heading", top=10),
            _ann("p", "Paragraph", 0, None, "dangling body", top=20),
        ]
    )
    diag = oc_visualize.diagnostics(export)
    # the Paragraph root is a non-heading root; the heading root is fine.
    assert diag["non_heading_roots"] == 1


def test_diagnostics_overlong_heading_and_untokened():
    long_text = " ".join(["word"] * 30)
    export = _export(
        [
            _ann("h", "Section Header", 0, None, long_text, top=10),
            _ann("p", "Paragraph", 0, None, "no tokens here", top=20),
            _ann(
                "q", "Paragraph", 0, None, "has tokens", top=30, tokens=[(0, 0), (0, 1)]
            ),
        ]
    )
    diag = oc_visualize.diagnostics(export)
    assert diag["overlong_headings"] == 1
    assert diag["untokened"] == 2  # h and p have no tokensJsons
    # 2 of 10 page-0 tokens claimed
    assert diag["token_coverage"] == pytest.approx(0.2, abs=1e-6)


def test_diagnostics_label_and_depth_histograms():
    export = _export(
        [
            _ann("h", "Section Header", 0, None, "H", top=10),
            _ann("p1", "Paragraph", 0, "h", "a", top=20),
            _ann("p2", "Paragraph", 0, "h", "b", top=30),
        ]
    )
    diag = oc_visualize.diagnostics(export)
    assert diag["label_histogram"]["Paragraph"] == 2
    assert diag["label_histogram"]["Section Header"] == 1
    assert diag["max_depth"] == 1
    assert diag["depth_histogram"][0] == 1
    assert diag["depth_histogram"][1] == 2


# --------------------------------------------------------------------------- #
# page_summary
# --------------------------------------------------------------------------- #
def test_page_summary_lists_page_annotations_with_parent_and_depth():
    export = _export(
        [
            _ann("h1", "Section Header", 0, None, "Article 1", top=10),
            _ann("p1", "Paragraph", 0, "h1", "body on page 0", top=20),
            _ann("p2", "Paragraph", 1, "h1", "body on page 1", top=5),
        ],
        n_pages=2,
    )
    rows = oc_visualize.page_summary(export, 0)
    ids = [r["id"] for r in rows]
    assert ids == ["h1", "p1"]  # only page-0 annotations, reading order
    p1 = next(r for r in rows if r["id"] == "p1")
    assert p1["parent_id"] == "h1"
    assert p1["parent_label"] == "Section Header"
    assert p1["depth"] == 1
    # page 1 has exactly the one annotation that lives there
    assert [r["id"] for r in oc_visualize.page_summary(export, 1)] == ["p2"]


# --------------------------------------------------------------------------- #
# render_tree_outline
# --------------------------------------------------------------------------- #
def test_render_tree_outline_indents_by_depth_and_lists_nodes():
    export = _export(
        [
            _ann("h1", "Section Header", 0, None, "Article 1", top=10),
            _ann("p1", "Paragraph", 0, "h1", "the body text here", top=20),
        ]
    )
    out = oc_visualize.render_tree_outline(export)
    lines = out.splitlines()

    # the heading line appears before, and less indented than, its child
    h_line = next(l for l in lines if "[h1]" in l)
    p_line = next(l for l in lines if "[p1]" in l)
    assert "Section Header" in h_line
    assert "Paragraph" in p_line
    assert (len(p_line) - len(p_line.lstrip())) > (len(h_line) - len(h_line.lstrip()))
    # stats header mentions counts
    assert "roots=" in out
    assert "Article 1" in h_line


def test_render_tree_outline_truncates_long_text():
    long_text = "lorem ipsum dolor sit amet " * 20
    export = _export([_ann("p", "Paragraph", 0, None, long_text, top=10)])
    out = oc_visualize.render_tree_outline(export, max_text=40)
    p_line = next(l for l in out.splitlines() if "[p]" in l)
    assert "…" in p_line
    # the rendered text payload should be bounded near max_text
    assert len(p_line) < 120


# --------------------------------------------------------------------------- #
# project_page smoke test (needs pypdfium2 + PIL)
# --------------------------------------------------------------------------- #
def test_project_page_smoke():
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")
    from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

    path = "tests/fixtures/USC Title 1 - CHAPTER 1.pdf"
    export = parse_to_opencontracts(path)
    with open(path, "rb") as fh:
        pdf_bytes = fh.read()

    img = oc_visualize.project_page(export, pdf_bytes, 0, scale=1.5)
    # a real rendered page is comfortably larger than this
    assert img.width > 200 and img.height > 200
