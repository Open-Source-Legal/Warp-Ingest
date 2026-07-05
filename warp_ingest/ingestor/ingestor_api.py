import logging
import os

import warp_ingest.ingestion_daemon.config as cfg
from warp_ingest.ingestor import pdf_ingestor

# initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(cfg.log_level())


class UnsupportedMediaType(ValueError):
    """Raised for a non-PDF document — Warp-Ingest parses PDF only."""


def ingest_document(
    doc_name,
    doc_location,
    mime_type,
    parse_options: dict = None,
):
    """Parse a PDF into Warp-Ingest blocks / JSON / HTML / OpenContracts output.

    Warp-Ingest is a deterministic, PDF-only parser. A non-PDF ``mime_type`` raises
    :class:`UnsupportedMediaType` rather than attempting a best-effort parse.
    """
    print(f"Parsing {mime_type} at {doc_location} with name {doc_name}")
    if mime_type != "application/pdf":
        raise UnsupportedMediaType(
            f"unsupported media type {mime_type!r}: Warp-Ingest parses PDF only"
        )

    ingestor = pdf_ingestor.PDFIngestor(doc_location, parse_options)
    return_dict = ingestor.return_dict

    if doc_location and os.path.exists(doc_location):
        os.unlink(doc_location)
        print(f"File {doc_location} deleted")
    return return_dict, ingestor
