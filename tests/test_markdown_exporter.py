"""Tests for the generic Markdown exporter."""

import os

from warp_ingest.ingestor import pdf_ingestor
from warp_ingest.ingestor.markdown_exporter import (
    box_xywh,
    parse_to_markdown,
    render_layout_predictions,
    render_pages,
)

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_box_xywh_from_indexable_and_attrs():
    class _Box:
        top, left, right, width, height = 10.0, 20.0, 120.0, 100.0, 12.0

        def __getitem__(self, index):
            return [self.top, self.left, self.right, self.width, self.height][index]

    assert box_xywh(_Box()) == [20.0, 10.0, 100.0, 12.0]
    assert box_xywh(None) is None


def test_box_xywh_from_attrs_only_and_dict():
    class _AttrOnlyBox:
        top, left, width, height = 10.0, 20.0, 100.0, 12.0

    assert box_xywh(_AttrOnlyBox()) == [20.0, 10.0, 100.0, 12.0]
    assert box_xywh({"top": 1, "left": 2, "width": 3, "height": 4}) == [
        2.0,
        1.0,
        3.0,
        4.0,
    ]


def test_render_pages_groups_tables_and_emphasis():
    payload = {
        "num_pages": 1,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "header",
                "block_text": "Title",
                "level": 2,
                "box": [0, 0, 10, 10],
            },
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "NOTE: keep it dry",
                "bold_mask": "1000",
                "box": [0, 20, 10, 10],
            },
            {
                "page_idx": 0,
                "block_type": "list_item",
                "block_text": "First",
                "box": [0, 40, 10, 10],
            },
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

    pages = render_pages(payload)

    assert len(pages) == 1
    markdown = pages[0][1]
    assert "# Title" in markdown
    assert "**NOTE:** keep it dry" in markdown
    assert "- First" in markdown
    assert "<th>h1</th><th>h2</th>" in markdown
    assert "<td>a</td><td>b</td>" in markdown
    assert "<td>h1</td><td>h2</td>" not in markdown


def test_render_pages_uses_native_table_html_when_supplied():
    payload = {
        "num_pages": 1,
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Intro prose.",
                "box": [50, 10, 200, 10],
            },
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
        ],
        "tables_by_page": {
            0: [[[45, 95, 255, 130], "<table>\n<tr><th>A</th></tr>\n</table>"]]
        },
    }

    markdown = render_pages(payload)[0][1]

    assert "Intro prose." in markdown
    assert "<th>A</th>" in markdown
    assert "a1" not in markdown
    assert "a3" not in markdown


def test_render_pages_appends_unplaced_table_html():
    payload = {
        "num_pages": 1,
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Keep this prose.",
                "box": [50, 10, 200, 10],
            }
        ],
        "tables_by_page": {0: [[None, "<table><tr><td>Loose</td></tr></table>"]]},
    }

    markdown = render_pages(payload)[0][1]

    assert "Keep this prose." in markdown
    assert "<td>Loose</td>" in markdown


def test_render_pages_accepts_ext_tables_alias():
    payload = {
        "num_pages": 1,
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Keep this prose.",
                "box": [50, 10, 200, 10],
            }
        ],
        "ext_tables": {0: [[None, "<table><tr><td>Alias</td></tr></table>"]]},
    }

    markdown = render_pages(payload)[0][1]

    assert "Keep this prose." in markdown
    assert "<td>Alias</td>" in markdown


def test_render_layout_predictions_merges_tables_per_page_and_table_idx():
    payload = {
        "num_pages": 2,
        "page_dim": [612.0, 792.0],
        "blocks": [
            {
                "page_idx": 0,
                "block_type": "para",
                "block_text": "Intro",
                "box": [10, 20, 100, 10],
            },
            {
                "page_idx": 0,
                "block_type": "table_row",
                "cell_values": ["a"],
                "table_idx": 0,
                "box": [50, 100, 200, 12],
            },
            {
                "page_idx": 1,
                "block_type": "table_row",
                "cell_values": ["b"],
                "table_idx": 0,
                "box": [60, 110, 180, 12],
            },
        ],
    }

    rendered = render_layout_predictions(payload)
    predictions = rendered["predictions"]

    assert rendered["image_width"] == 612
    assert [prediction["page"] for prediction in predictions] == [1, 1, 2]
    tables = [
        prediction
        for prediction in predictions
        if prediction["content"]["type"] == "table"
    ]
    assert len(tables) == 2
    assert tables[0]["bbox"] == [50.0, 100.0, 250.0, 112.0]
    assert tables[1]["bbox"] == [60.0, 110.0, 240.0, 122.0]


def test_parse_to_markdown_public_api():
    path = os.path.join(FIX_DIR, "sample.pdf")

    markdown = parse_to_markdown(path, include_native_tables=False)
    wrapper_markdown = pdf_ingestor.parse_to_markdown(
        path,
        include_native_tables=False,
    )
    payload = pdf_ingestor.parse_to_markdown_payload(
        path,
        include_native_tables=False,
    )
    layout = pdf_ingestor.parse_to_layout_predictions(
        path,
        include_native_tables=False,
    )

    assert len(markdown) > 100
    assert markdown == wrapper_markdown
    assert payload["blocks"]
    assert layout["predictions"]
