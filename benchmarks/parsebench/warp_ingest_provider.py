"""ParseBench PARSE provider for Warp-Ingest.

Follows the provider contract documented in ParseBench's
``.claude/commands/integrate-pipeline.md`` and mirrors the local-library
template (``providers/parse/pymupdf.py`` / ``markitdown.py``):

* ``run_inference`` runs Warp-Ingest on the source PDF and stashes the
  serialized block stream as ``raw_output``.
* ``normalize`` renders that block stream to per-page Markdown
  (``PageIR``) + a whole-document Markdown string (``ParseOutput``).

Importing this module registers the ``warp_ingest`` provider with ParseBench's
provider registry.  The actual Markdown rendering lives in the dependency-free
``warp_markdown`` module so it can be unit-tested without ParseBench installed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_pages


@register_provider("warp_ingest")
class WarpIngestProvider(Provider):
    """Provider for Warp-Ingest (local, no API key). Apache-2.0."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def run_inference(
        self, pipeline: PipelineSpec, request: InferenceRequest
    ) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"WarpIngestProvider only supports PARSE, got {request.product_type}"
            )

        pdf_path = Path(request.source_file_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ProviderPermanentError(
                f"WarpIngestProvider only supports .pdf files, got {pdf_path.suffix}"
            )
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        try:
            from warp_ingest.ingestor.pdf_ingestor import PDFIngestor  # noqa: F401
        except ImportError as e:
            raise ProviderConfigError(
                "warp_ingest not importable. Install warp-ingest (pip install -e .) "
                "in the same environment."
            ) from e

        started_at = datetime.now()
        try:
            raw_output = extract_warp_blocks(str(pdf_path))
        except Exception as e:
            raise ProviderPermanentError(f"Warp-Ingest parse error: {e}") from e
        completed_at = datetime.now()

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"WarpIngestProvider only supports PARSE, got {raw_result.product_type}"
            )

        rendered = render_pages(raw_result.raw_output)
        pages = [PageIR(page_index=i, markdown=md) for i, md in rendered]
        full_text = "\n\n".join(md for _, md in rendered)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
