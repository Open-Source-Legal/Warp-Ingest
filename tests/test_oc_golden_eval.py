"""Unit tests for the golden-agreement scoring core (``tests/oc_golden_eval.py``).

Synthetic exports + goldens exercise overlap assignment, one-vs-rest label F1,
coverage, spurious counting, table rollup, the body-as-table-row (multi-column)
signal, furniture-as-heading, head-ancestor / parent-class agreement, and
reading-order agreement — no PDFs needed.
"""

from tests import oc_golden_eval as G

W, H = 100.0, 100.0


def _ann(aid, label, frac, text, parent=None):
    l, t, r, b = frac
    return {
        "id": aid,
        "annotationLabel": label,
        "rawText": text,
        "page": 0,
        "parent_id": parent,
        "annotation_json": {
            "0": {
                "bounds": {
                    "left": l * W,
                    "top": t * H,
                    "right": r * W,
                    "bottom": b * H,
                },
                "tokensJsons": [],
                "rawText": text,
            }
        },
    }


def _export(anns):
    return {
        "pawls_file_content": [
            {"page": {"width": W, "height": H, "index": 0}, "tokens": []}
        ],
        "labelled_text": anns,
    }


def _region(rid, label, frac, text, parent=None, ro=None):
    return {
        "region_id": rid,
        "bbox_frac": list(frac),
        "text": text,
        "gold_label": label,
        "gold_parent_region_id": parent,
        "ro_index": ro,
    }


def _perfect_case():
    gold = [
        _region(0, "Section Header", (0.1, 0.1, 0.5, 0.15), "Intro", None, 0),
        _region(1, "Paragraph", (0.1, 0.2, 0.6, 0.3), "body text here", 0, 1),
        _region(2, "List Item", (0.12, 0.32, 0.6, 0.36), "first item", 0, 2),
        _region(3, "Furniture", (0.45, 0.95, 0.55, 0.98), "page 1", None, 3),
    ]
    anns = [
        _ann("0", "Section Header", (0.1, 0.1, 0.5, 0.15), "Intro", None),
        _ann("1", "Paragraph", (0.1, 0.2, 0.6, 0.3), "body text here", "0"),
        _ann("2", "List Item", (0.12, 0.32, 0.6, 0.36), "first item", "0"),
        _ann("3", "Paragraph", (0.45, 0.95, 0.55, 0.98), "page 1", None),
    ]
    return _export(anns), gold


def test_perfect_alignment_scores_one():
    export, gold = _perfect_case()
    m = G.score_page(export, 0, gold)
    assert m["struct_macro_f1"] == 1.0
    assert m["heading_f1"] == 1.0
    assert m["list_f1"] == 1.0
    assert m["paragraph_f1"] == 1.0
    assert m["gold_coverage"] == 1.0
    assert m["spurious_frac"] == 0.0
    assert m["body_as_tablerow_frac"] == 0.0
    assert m["furniture_as_heading"] == 0
    assert m["head_ancestor_agreement"] == 1.0
    assert m["parent_class_agreement"] == 1.0


def test_mislabeled_paragraph_as_heading_drops_f1():
    export, gold = _perfect_case()
    export["labelled_text"][1]["annotationLabel"] = "Section Header"
    m = G.score_page(export, 0, gold)
    assert m["paragraph_f1"] < 1.0  # recall miss
    assert m["heading_f1"] < 1.0  # spurious heading prediction hurts precision


def test_furniture_promoted_to_heading_flagged():
    export, gold = _perfect_case()
    export["labelled_text"][3]["annotationLabel"] = "Section Header"
    m = G.score_page(export, 0, gold)
    assert m["furniture_as_heading"] == 1


def test_spurious_annotation_counted():
    export, gold = _perfect_case()
    export["labelled_text"].append(
        _ann("9", "Paragraph", (0.7, 0.5, 0.95, 0.55), "unrelated junk", None)
    )
    m = G.score_page(export, 0, gold)
    assert m["spurious_frac"] > 0.0


def test_missed_gold_region_drops_coverage():
    export, gold = _perfect_case()
    export["labelled_text"] = [a for a in export["labelled_text"] if a["id"] != "2"]
    m = G.score_page(export, 0, gold)
    assert m["gold_coverage"] < 1.0
    assert m["list_f1"] < 1.0


def test_wrong_heading_parent_drops_head_ancestor_agreement():
    export, gold = _perfect_case()
    for a in export["labelled_text"]:
        if a["id"] == "1":
            a["parent_id"] = None
    m = G.score_page(export, 0, gold)
    assert m["head_ancestor_agreement"] < 1.0


def test_table_rows_roll_up_to_one_gold_region():
    gold = [
        _region(0, "Table Row", (0.1, 0.1, 0.9, 0.5), "Tier Amount Total 999", None, 0)
    ]
    anns = [
        _ann("0", "Table Row", (0.1, 0.1, 0.9, 0.2), "Tier Amount", None),
        _ann("1", "Table Row", (0.1, 0.25, 0.9, 0.35), "Subtotal 10", None),
        _ann("2", "Table Row", (0.1, 0.4, 0.9, 0.5), "Total 999", None),
    ]
    m = G.score_page(_export(anns), 0, gold)
    assert m["table_region_coverage"] == 1.0
    assert m["spurious_frac"] == 0.0  # extra rows absorbed into the gold table


def test_body_fused_into_table_rows_flagged_not_penalized():
    """A gold paragraph that Warp covered with Table Rows shows as the
    multi-column / table-over-fire signal, not as a paragraph-F1 collapse."""
    gold = [
        _region(0, "Paragraph", (0.1, 0.1, 0.9, 0.3), "a real prose paragraph", None, 0)
    ]
    anns = [
        _ann("0", "Table Row", (0.1, 0.1, 0.5, 0.2), "a real", None),
        _ann("1", "Table Row", (0.5, 0.1, 0.9, 0.2), "prose paragraph", None),
    ]
    m = G.score_page(_export(anns), 0, gold)
    assert m["body_as_tablerow_frac"] == 1.0
    assert m["paragraph_f1"] == 1.0  # pulled out of the label metric, not penalized


def test_reading_order_agreement_detects_inversion():
    gold = [
        _region(0, "Paragraph", (0.1, 0.1, 0.9, 0.2), "alpha", None, 0),
        _region(1, "Paragraph", (0.1, 0.3, 0.9, 0.4), "bravo", None, 1),
        _region(2, "Paragraph", (0.1, 0.5, 0.9, 0.6), "charlie", None, 2),
    ]
    anns = [
        _ann("0", "Paragraph", (0.1, 0.5, 0.9, 0.6), "charlie", None),
        _ann("1", "Paragraph", (0.1, 0.3, 0.9, 0.4), "bravo", None),
        _ann("2", "Paragraph", (0.1, 0.1, 0.9, 0.2), "alpha", None),
    ]
    m = G.score_page(_export(anns), 0, gold, have_ro=True)
    assert m["reading_order_agreement"] < 1.0
