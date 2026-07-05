"""Tests for the global column grid in TableParser (`_global_col_spans`).

The grid replaces the legacy row-by-row column accretion when it recovers more
columns; these tests pin the geometric invariants the downstream cell-assignment
depends on (sorted, non-overlapping spans) and the conservative gates.
"""

from warp_ingest.ingestor.visual_ingestor.table_parser import TableParser


class _FakeDoc:
    page_width = 612.0


def _vl(left, right, top=0.0, height=10.0):
    # box_style = [top, left, right, width, height]
    return {"box_style": [top, left, right, right - left, height], "text": "x"}


def _row(cells, top):
    return {
        "block_type": "table_row",
        "visual_lines": [_vl(l, r, top) for (l, r) in cells],
    }


def _make(rows):
    blocks = [_row(cells, top=i * 12.0) for i, cells in enumerate(rows)]
    return TableParser(_FakeDoc(), blocks)


def _assert_sorted_disjoint(spans):
    assert spans is not None
    for a, b in zip(spans, spans[1:]):
        assert a[0] < a[1], f"degenerate band {a}"
        assert a[1] <= b[0], f"overlapping/non-monotonic spans: {a} then {b}"


def test_global_grid_clean_three_columns():
    # Three clean columns, separators clearly voted by every row.
    rows = [[(50, 150), (250, 350), (450, 550)] for _ in range(4)]
    spans = _make(rows)._global_col_spans(0)
    _assert_sorted_disjoint(spans)
    assert len(spans) == 3


def test_global_grid_clamps_straddling_cell_to_disjoint():
    # A column-0 cell on one row straddles the ~200 separator (right edge 260 >
    # next column's left 250). The tight band would overlap; the clamp must make
    # the returned spans disjoint so downstream align_* invariants hold.
    rows = [
        [(50, 150), (250, 350)],
        [(50, 150), (250, 350)],
        [(50, 150), (250, 350)],
        [(50, 260), (250, 350)],  # crosses the separator
    ]
    spans = _make(rows)._global_col_spans(0)
    _assert_sorted_disjoint(spans)
    assert len(spans) == 2


def test_global_grid_returns_none_for_single_column():
    # No interior separator with majority support -> no grid (legacy spans kept).
    rows = [[(50, 550)] for _ in range(4)]
    assert _make(rows)._global_col_spans(0) is None


def test_global_grid_returns_none_too_few_rows():
    rows = [[(50, 150), (250, 350)]]
    assert _make(rows)._global_col_spans(0) is None
