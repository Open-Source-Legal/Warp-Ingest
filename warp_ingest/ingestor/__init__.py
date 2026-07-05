import os

from warp_ingest.ingestor_utils.compat import generate_version

from .pdf_ingestor import *

VERSION = generate_version(
    [
        os.path.join(os.path.dirname(__file__), "../file_parser/"),
        os.path.join(os.path.dirname(__file__), "../ingestion_daemon/"),
        os.path.join(os.path.dirname(__file__), "../ingestor/"),
        os.path.join(os.path.dirname(__file__), "../ingestor_utils/"),
    ],
)

__all__ = ("VERSION",)
