"""ParseBench layout (visual-grounding) adapter for Warp-Ingest.

The parse-track visual-grounding metric ("Element Pass Rate") attributes each
ground-truth layout element to a predicted block by text overlap + bbox IoA +
reading order.  Warp-Ingest carries real per-block geometry (``box_style`` in
PDF points) and reading order, so — unlike the text-only baselines that score
near the floor — warp can be scored honestly here.

This adapter converts warp's serialized blocks (from ``warp_markdown`` raw
output, which now includes per-block ``box`` + page ``page_dim``) into a
``LayoutOutput`` of ``LayoutPrediction``s.  Consecutive ``table_row`` blocks
sharing a ``table_idx`` are merged into a single ``Table`` element with the
rendered HTML so the table is attributed as one region (matching how the GT
annotates tables).

Importing this module registers the adapter for provider key ``warp_ingest``.
"""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.layout_adapters.base import LayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import register_layout_adapter
from parse_bench.evaluation.layout_label_mappers.base import (
    LayoutLabelMapper,
    MappingContext,
)
from parse_bench.evaluation.layout_label_mappers.registry import (
    register_layout_label_mapper,
)
from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline_io import InferenceResult

from benchmarks.parsebench.warp_markdown import _table_html

# Warp emits already-canonical labels; a non-LlamaParse carrier model + a
# warp-keyed passthrough mapper keep label resolution off any provider-specific
# (e.g. LlamaParse v2) mapper that would reject them.
_CARRIER_MODEL = LayoutDetectionModel.UNSTRUCTURED_LAYOUT


@register_layout_label_mapper("warp_ingest", priority=200)
class WarpIngestPassthroughLabelMapper(LayoutLabelMapper):
    """Warp already emits Canonical17 label strings — pass them through."""

    def to_canonical(
        self, label, prediction, context: MappingContext
    ) -> CanonicalLabel:
        del prediction, context
        return CanonicalLabel(label)


# Warp block_type -> ParseBench Canonical17 label.
_LABEL_MAP = {
    "header": CanonicalLabel.SECTION_HEADER.value,
    "para": CanonicalLabel.TEXT.value,
    "list_item": CanonicalLabel.LIST_ITEM.value,
    "table_row": CanonicalLabel.TABLE.value,
}


def _union_box(boxes: list[list[float]]) -> list[float] | None:
    """Union of ``[left, top, w, h]`` boxes -> ``[x1, y1, x2, y2]``."""
    rects = [b for b in boxes if b]
    if not rects:
        return None
    x1 = min(b[0] for b in rects)
    y1 = min(b[1] for b in rects)
    x2 = max(b[0] + b[2] for b in rects)
    y2 = max(b[1] + b[3] for b in rects)
    return [x1, y1, x2, y2]


@register_layout_adapter("warp_ingest", priority=100)
class WarpIngestLayoutAdapter(LayoutAdapter):
    """Build a ``LayoutOutput`` from Warp-Ingest's geometric block stream."""

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        raw = (
            inference_result.raw_output
            if isinstance(inference_result.raw_output, dict)
            else {}
        )
        page_dim = raw.get("page_dim") or [612.0, 792.0]
        img_w = max(1, int(round(float(page_dim[0]))))
        img_h = max(1, int(round(float(page_dim[1]))))
        blocks = raw.get("blocks", []) or []

        predictions: list[LayoutPrediction] = []
        order = 0

        # Table accumulator (merge consecutive table_row blocks per table_idx).
        tbl_idx: Any = None
        tbl_boxes: list[list[float]] = []
        tbl_header: list[str] | None = None
        tbl_rows: list[list[str]] = []
        tbl_page = 1

        def flush_table() -> None:
            nonlocal order, tbl_idx, tbl_boxes, tbl_header, tbl_rows
            if not (tbl_rows or tbl_header):
                tbl_idx = None
                return
            box = _union_box(tbl_boxes)
            if box is not None:
                header = tbl_header
                body = [r for r in tbl_rows if not (header and r == header)]
                predictions.append(
                    LayoutPrediction(
                        bbox=box,
                        score=1.0,
                        label=CanonicalLabel.TABLE.value,
                        page=tbl_page,
                        content=LayoutTableContent(html=_table_html(header, body)),
                        provider_metadata={"order_index": order},
                    )
                )
                order += 1
            tbl_idx = None
            tbl_boxes = []
            tbl_header = None
            tbl_rows = []

        for b in blocks:
            page = int(b.get("page_idx", 0) or 0) + 1
            btype = b.get("block_type")
            box = b.get("box")
            text = (b.get("block_text") or "").strip()

            if btype == "table_row":
                tidx = b.get("table_idx")
                if (tbl_rows or tbl_header) and tidx != tbl_idx:
                    flush_table()
                tbl_idx = tidx
                tbl_page = page
                if box:
                    tbl_boxes.append(box)
                hdr = b.get("header_cell_values")
                if tbl_header is None and hdr:
                    tbl_header = [str(c) for c in hdr]
                cells = b.get("cell_values") or ([text] if text else [])
                tbl_rows.append([str(c) for c in cells])
                continue

            if tbl_rows or tbl_header:
                flush_table()

            if not text or box is None:
                continue
            xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            predictions.append(
                LayoutPrediction(
                    bbox=xyxy,
                    score=1.0,
                    label=_LABEL_MAP.get(btype, CanonicalLabel.TEXT.value),
                    page=page,
                    content=LayoutTextContent(text=text),
                    provider_metadata={"order_index": order},
                )
            )
            order += 1

        if tbl_rows or tbl_header:
            flush_table()

        if page_filter is not None:
            predictions = [p for p in predictions if p.page == page_filter]

        return LayoutOutput(
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=_CARRIER_MODEL,  # neutral carrier; warp-keyed passthrough mapper handles labels
            image_width=img_w,
            image_height=img_h,
            predictions=predictions,
        )
