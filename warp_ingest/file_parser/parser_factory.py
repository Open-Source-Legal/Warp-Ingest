import logging

import warp_ingest.ingestion_daemon.config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.log_level())


class FileParserFactory:
    """Factory for parser instances for various formats.

    The implementation is ``pdfplumber``, a pure-Python PDF parser that emits the
    same intermediate XHTML the legacy Java/Tika front-end produced.  No Java and
    no Apache Tika are required.
    """

    __instance = dict()
    supported_implementations = {
        "application/pdf": ["pdfplumber"],
    }

    @classmethod
    def instance(cls, format, impl):
        if impl not in FileParserFactory.supported_implementations.get(format, []):
            raise Exception(f"unknown implementation {impl} for file format {format}")
        key = (format, impl)
        if key not in FileParserFactory.__instance:
            if impl == "pdfplumber":
                from warp_ingest.file_parser.pdf_plumber_parser import (
                    PdfPlumberFileParser,
                )

                FileParserFactory.__instance[key] = PdfPlumberFileParser()
        return FileParserFactory.__instance[key]
