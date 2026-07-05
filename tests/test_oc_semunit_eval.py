import tests.oc_golden_eval as G


def _export_with_units():
    # two fine paragraphs on page 0 under one unit su-0
    return {
        "pawls_file_content": [
            {"page": {"width": 100, "height": 100, "index": 0}, "tokens": []}
        ],
        "labelled_text": [
            {
                "id": "0",
                "annotationLabel": "Paragraph",
                "parent_id": None,
                "page": 0,
                "rawText": "a",
                "annotation_json": {
                    "0": {
                        "bounds": {"top": 0, "left": 0, "right": 40, "bottom": 10},
                        "tokensJsons": [],
                        "rawText": "a",
                    }
                },
            },
            {
                "id": "1",
                "annotationLabel": "Paragraph",
                "parent_id": None,
                "page": 0,
                "rawText": "b",
                "annotation_json": {
                    "0": {
                        "bounds": {"top": 10, "left": 0, "right": 80, "bottom": 20},
                        "tokensJsons": [],
                        "rawText": "b",
                    }
                },
            },
            {
                "id": "su-0",
                "annotationLabel": "Semantic Unit",
                "parent_id": None,
                "page": 0,
                "rawText": "a b",
                "annotation_json": {
                    "0": {
                        "bounds": {"top": 0, "left": 0, "right": 80, "bottom": 20},
                        "tokensJsons": [],
                        "rawText": "a b",
                    }
                },
            },
        ],
        "relationships": [
            {
                "id": "surel-0",
                "relationshipLabel": "OC_SEMANTIC_UNIT",
                "source_annotation_ids": ["su-0"],
                "target_annotation_ids": ["0", "1"],
                "structural": True,
            }
        ],
    }


def test_warp_units_projection():
    units = G.warp_units(_export_with_units(), 0)
    assert len(units) == 1
    u = units[0]
    assert u["unit_id"] == "su-0"
    assert set(u["member_ann_ids"]) == {"0", "1"}
    assert u["box"] == [0.0, 0.0, 0.8, 0.2]  # normalized by 100x100
    assert u["order_start"] == 0 and u["order_end"] == 1  # member doc positions


def test_score_units_perfect_recovery():
    exp = _export_with_units()
    gold = [
        {
            "unit_id": "g0",
            "bbox_frac": [0.0, 0.0, 0.8, 0.2],
            "text": "a b",
            "member_order": [0, 1],
        }
    ]
    m = G.score_units(exp, 0, gold)
    assert m["unit_coverage"] == 1.0
    assert m["mean_unit_iou"] >= 0.99
    assert m["fragmentation_frac"] == 0.0
    assert m["merge_frac"] == 0.0
    assert m["spurious_unit_frac"] == 0.0
    assert m["n_warp_units"] == 1 and m["n_gold_units"] == 1


def test_unit_regressions_flags_drop():
    base = {
        "unit_coverage": 1.0,
        "mean_unit_iou": 0.9,
        "fragmentation_frac": 0.0,
        "merge_frac": 0.0,
        "spurious_unit_frac": 0.0,
        "windowdiff": 0.1,
        "pk": 0.1,
    }
    worse = dict(base, unit_coverage=0.5, merge_frac=0.4)
    problems = G.unit_regressions(worse, base)
    assert any("unit_coverage" in p for p in problems)
    assert any("merge_frac" in p for p in problems)
    assert G.unit_regressions(base, base) == []
