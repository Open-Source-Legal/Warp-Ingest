"""FastAPI ingestion service.

``POST /api/parse`` takes a multipart PDF plus snake_case query parameters for
the engine options (``render_format``, ``apply_ocr``, ``disable_ocr``,
``semantic_units``) and returns the parse result directly
(``{"page_dim": ..., "num_pages": ..., "result": ...}``).  Errors are
standard FastAPI ``{"detail": ...}`` bodies.  ``GET /`` and ``GET /healthz``
are unauthenticated health endpoints.  Launch via
``python -m warp_ingest.ingestion_daemon`` (the CPU-budgeting uvicorn launcher
in ``__main__``).
"""

import logging
import os
import shutil
import tempfile
import threading
import traceback
from importlib import metadata
from typing import Literal

import warp_ingest.ingestion_daemon.config as cfg
from warp_ingest.ingestion_daemon.service_dependencies import (
    require_any_service_dependency,
    require_service_dependency,
)

_fastapi = require_service_dependency("fastapi")
require_any_service_dependency(("python_multipart", "multipart"), "python-multipart")
Depends = _fastapi.Depends
FastAPI = _fastapi.FastAPI
File = _fastapi.File
HTTPException = _fastapi.HTTPException
Query = _fastapi.Query
UploadFile = _fastapi.UploadFile

from warp_ingest.ingestion_daemon.auth import require_api_key
from warp_ingest.ingestion_daemon.autotune import compute_settings
from warp_ingest.ingestor import ingestor_api
from warp_ingest.ingestor_utils import compat as file_utils

logger = logging.getLogger(__name__)
logger.setLevel(cfg.log_level())

SETTINGS = compute_settings()

# One CPU-bound parse per worker by default (WARP_WORKER_PARSE_SLOTS): a burst
# must not stack GIL-sharing parses inside this worker while sibling workers
# idle — excess requests wait here and the kernel steers new connections away.
_PARSE_SLOTS = threading.BoundedSemaphore(SETTINGS.parse_slots)


def _version():
    try:
        return metadata.version("warp-ingest")
    except metadata.PackageNotFoundError:  # editable/source checkout
        return "unknown"


app = FastAPI(
    title="Warp-Ingest",
    description="Deterministic, rule-based PDF parser.",
    version=_version(),
)


@app.get("/")
def root():
    return {"status": "ok", "service": "warp-ingest"}


@app.get("/healthz")
def healthz():
    from warp_ingest.file_parser import ocr_parser

    return {
        "status": "ok",
        "version": _version(),
        "ocr_available": ocr_parser.ocr_available(),
        "settings": SETTINGS.as_dict(),
    }


@app.post("/api/parse", dependencies=[Depends(require_api_key)])
def parse(
    file: UploadFile = File(...),
    render_format: Literal["all", "json", "html", "opencontracts"] = "all",
    apply_ocr: bool = Query(False, description="Force OCR on every page."),
    disable_ocr: bool = Query(
        False, description="Keep every page on its embedded text layer (no OCR)."
    ),
    semantic_units: bool = Query(
        False,
        description="Append the additive Semantic-Unit clause layer "
        "(render_format=opencontracts).",
    ),
    include_images: bool = Query(
        False,
        description="Emit embedded raster figures as is_image PAWLS tokens + "
        "Image annotations (render_format=opencontracts).",
    ),
):
    """Parse an uploaded PDF into layout-aware blocks.

    Scanned/sparse pages are OCR'd automatically when the OCR backend is
    available; ``apply_ocr`` forces OCR on every page and ``disable_ocr``
    suppresses it for this request — asking for both is a 422.  Non-PDF
    uploads are rejected with 415.
    """
    if apply_ocr and disable_ocr:
        raise HTTPException(
            status_code=422,
            detail="apply_ocr and disable_ocr are mutually exclusive",
        )
    parse_options = {
        "parse_and_render_only": True,
        "render_format": render_format,
        "parse_pages": (),
        "apply_ocr": apply_ocr,
        "disable_ocr": disable_ocr,
        "semantic_units": semantic_units,
        "include_images": include_images,
    }
    filename = os.path.basename(file.filename or "upload.pdf")
    _, file_extension = os.path.splitext(filename)
    tempfile_handler, tmp_file = tempfile.mkstemp(suffix=file_extension)
    try:
        with os.fdopen(tempfile_handler, "wb") as out:
            shutil.copyfileobj(file.file, out)
        props = file_utils.extract_file_properties(tmp_file)
        logger.info("Parsing document: %s", filename)
        with _PARSE_SLOTS:
            return_dict, _ = ingestor_api.ingest_document(
                filename,
                tmp_file,
                props["mimeType"],
                parse_options=parse_options,
            )
        return return_dict or {}
    except ingestor_api.UnsupportedMediaType as e:
        raise HTTPException(status_code=415, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.error("error parsing %s: %s", filename, traceback.format_exc())
        raise HTTPException(
            status_code=500, detail="internal error while parsing document"
        )
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
