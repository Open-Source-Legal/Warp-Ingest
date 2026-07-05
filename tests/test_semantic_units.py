from warp_ingest.ingestor.semantic_units import append_semantic_units


def _ann(id, label, parent, page, top, left, right, bottom, toks, text):
    return {
        "id": id,
        "annotationLabel": label,
        "rawText": text,
        "page": page,
        "parent_id": parent,
        "annotation_type": "TOKEN_LABEL",
        "structural": True,
        "annotation_json": {
            str(page): {
                "bounds": {"top": top, "left": left, "right": right, "bottom": bottom},
                "tokensJsons": [{"pageIndex": page, "tokenIndex": t} for t in toks],
                "rawText": text,
            }
        },
    }


def _export(anns):
    return {
        "labelled_text": list(anns),
        "relationships": [],
        "pawls_file_content": [
            {"page": {"width": 100, "height": 100, "index": 0}, "tokens": []}
        ],
        "page_count": 1,
    }


def _unit_parent(exp):
    """child unit id -> parent unit id, from OC_PARENT_CHILD edges (not parent_id)."""
    out = {}
    for r in exp["relationships"]:
        src = r["source_annotation_ids"][0]
        if r["relationshipLabel"] == "OC_PARENT_CHILD" and str(src).startswith("su-"):
            for t in r["target_annotation_ids"]:
                out[t] = src
    return out


def test_heading_and_body_form_one_unit():
    # heading "H" (id 0) with two paragraph children (1, 2) -> one unit, 3 members
    exp = _export(
        [
            _ann("0", "Section Header", None, 0, 0, 0, 50, 10, [0], "1. Taxes"),
            _ann("1", "Paragraph", "0", 0, 10, 0, 90, 20, [1, 2], "Each party ..."),
            _ann("2", "Paragraph", "0", 0, 20, 0, 90, 30, [3], "Provided that ..."),
        ]
    )
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    assert len(units) == 1
    u = units[0]
    assert u["id"].startswith("su-")
    assert u["parent_id"] is None
    # union bounds cover all three members
    b = u["annotation_json"]["0"]["bounds"]
    assert b["top"] == 0 and b["bottom"] == 30 and b["right"] == 90
    # full concatenated text for classification
    assert u["rawText"] == "1. Taxes Each party ... Provided that ..."
    # OC_SEMANTIC_UNIT edge lists the three member fine ids
    su_rel = [
        r for r in exp["relationships"] if r["relationshipLabel"] == "OC_SEMANTIC_UNIT"
    ]
    assert len(su_rel) == 1
    assert su_rel[0]["source_annotation_ids"] == [u["id"]]
    assert set(su_rel[0]["target_annotation_ids"]) == {"0", "1", "2"}


def test_nested_heading_makes_child_unit():
    # H0 -> H1 (nested heading) -> P2 ; expect 2 units, unit(H1).parent == unit(H0)
    exp = _export(
        [
            _ann("0", "Section Header", None, 0, 0, 0, 50, 5, [0], "Article I"),
            _ann("1", "Section Header", "0", 0, 10, 0, 50, 15, [1], "1.1 Scope"),
            _ann("2", "Paragraph", "1", 0, 20, 0, 90, 30, [2], "The scope is ..."),
        ]
    )
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    assert len(units) == 2
    parent = next(u for u in units if u["rawText"].startswith("Article I"))
    child = next(u for u in units if u["rawText"].startswith("1.1 Scope"))
    # units are flat (no parent_id); nesting lives in the relationship
    assert parent["parent_id"] is None and child["parent_id"] is None
    assert _unit_parent(exp).get(child["id"]) == parent["id"]
    pc = [
        r
        for r in exp["relationships"]
        if r["relationshipLabel"] == "OC_PARENT_CHILD"
        and r["source_annotation_ids"] == [parent["id"]]
    ]
    assert pc and child["id"] in pc[0]["target_annotation_ids"]


def test_orphan_body_is_singleton_unit():
    exp = _export([_ann("0", "Paragraph", None, 0, 0, 0, 90, 10, [0], "Loose text")])
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    assert len(units) == 1 and units[0]["parent_id"] is None


def test_numbered_run_splits_into_child_units():
    exp = _export(
        [
            _ann("0", "Section Header", None, 0, 0, 0, 50, 10, [0], "Agreement"),
            _ann("1", "List Item", "0", 0, 10, 0, 90, 20, [1], "1. First clause"),
            _ann("2", "List Item", "0", 0, 20, 0, 90, 30, [2], "2. Second clause"),
            _ann("3", "List Item", "0", 0, 30, 0, 90, 40, [3], "3. Third clause"),
        ]
    )
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    assert len(units) == 4  # 1 root (Agreement) + 3 clause children
    root = next(u for u in units if u["rawText"].startswith("Agreement"))
    parent_of = _unit_parent(exp)
    kids = [u for u in units if parent_of.get(u["id"]) == root["id"]]
    assert len(kids) == 3
    assert sorted(u["rawText"] for u in kids) == [
        "1. First clause",
        "2. Second clause",
        "3. Third clause",
    ]


def test_lettered_subclause_nests_under_number():
    exp = _export(
        [
            _ann("0", "Section Header", None, 0, 0, 0, 50, 10, [0], "Agreement"),
            _ann("1", "List Item", "0", 0, 10, 0, 90, 20, [1], "1. Parent clause"),
            _ann("2", "List Item", "0", 0, 20, 0, 90, 30, [2], "(a) sub one"),
            _ann("3", "List Item", "0", 0, 30, 0, 90, 40, [3], "(b) sub two"),
        ]
    )
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    one = next(u for u in units if u["rawText"].startswith("1."))
    a_ = next(u for u in units if u["rawText"].startswith("(a)"))
    b_ = next(u for u in units if u["rawText"].startswith("(b)"))
    parent_of = _unit_parent(exp)
    assert parent_of.get(a_["id"]) == one["id"]
    assert parent_of.get(b_["id"]) == one["id"]


def test_folio_heading_is_demoted():
    exp = _export(
        [
            _ann("0", "Section Header", None, 0, 0, 0, 50, 10, [0], "Real Section"),
            _ann("1", "Section Header", "0", 0, 10, 0, 90, 20, [1], "Page 2 of 9"),
            _ann("2", "Paragraph", "1", 0, 20, 0, 90, 30, [2], "Body text here"),
        ]
    )
    append_semantic_units(exp)
    units = [a for a in exp["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    roots = [u for u in units if u["parent_id"] is None]
    assert len(roots) == 1 and roots[0]["rawText"].startswith("Real Section")
    assert any("Body text here" in u["rawText"] for u in units)
