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

from warp_ingest.ingestor.markdown_exporter import render_layout_predictions

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
        rendered = render_layout_predictions(raw, page_filter=page_filter)
        img_w = max(1, int(rendered.get("image_width", 612)))
        img_h = max(1, int(rendered.get("image_height", 792)))
        predictions: list[LayoutPrediction] = []
        for prediction in rendered["predictions"]:
            content_payload = prediction.get("content") or {}
            if content_payload.get("type") == "table":
                content = LayoutTableContent(html=content_payload.get("html", ""))
            else:
                content = LayoutTextContent(text=content_payload.get("text", ""))
            predictions.append(
                LayoutPrediction(
                    bbox=prediction["bbox"],
                    score=float(prediction.get("score", 1.0)),
                    label=str(prediction.get("label") or CanonicalLabel.TEXT.value),
                    page=int(prediction.get("page", 1)),
                    content=content,
                    provider_metadata={
                        "order_index": int(prediction.get("order_index", 0))
                    },
                )
            )

        return LayoutOutput(
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=_CARRIER_MODEL,  # neutral carrier; warp-keyed passthrough mapper handles labels
            image_width=img_w,
            image_height=img_h,
            predictions=predictions,
        )
