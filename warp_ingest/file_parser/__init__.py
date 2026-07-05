import warp_ingest.ingestion_daemon.config as cfg
from warp_ingest.file_parser.parser_factory import FileParserFactory

# The pure-Python pdfplumber front-end. Override with the PDF_PARSER env var.
pdf_file_parser = FileParserFactory.instance(
    "application/pdf",
    cfg.get_config("PDF_PARSER", "pdfplumber"),
)
