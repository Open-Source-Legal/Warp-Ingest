#!/bin/bash
# Pure-Python ingestor service (FastAPI/uvicorn). No Java / Tika required.
# A Python environment with the requirements installed is the only prerequisite.
# Concurrency is budgeted automatically from the CPUs available to the process;
# override via WARP_WEB_WORKERS / WARP_FE_WORKERS / WARP_OCR_THREADS.  Auth key
# comes from WARP_API_KEY (default abc123).
exec python -m warp_ingest.ingestion_daemon
