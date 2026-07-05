"""Service launcher: budget the CPUs, then run uvicorn with that budget.

``python -m warp_ingest.ingestion_daemon`` computes the concurrency settings
once from the CPUs actually available (affinity + cgroup quota — see
``autotune``), exports the per-worker knobs through the environment so the
uvicorn worker processes and their spawned front-end pools inherit them, and
serves the FastAPI app.  Every knob is overridable: WARP_WEB_WORKERS (or
WEB_CONCURRENCY), WARP_FE_WORKERS, WARP_OCR_THREADS, WARP_WORKER_PARSE_SLOTS,
WARP_HOST, WARP_PORT (or PORT), WARP_API_KEY, LOG_LEVEL.
"""

import os

import uvicorn

import warp_ingest.ingestion_daemon.config as cfg
from warp_ingest.ingestion_daemon.auth import DEFAULT_API_KEY, expected_api_key
from warp_ingest.ingestion_daemon.autotune import compute_settings


def main():
    settings = compute_settings()
    # Export the computed budget for the worker processes (a user-set variable
    # is already reflected in `settings` and setdefault leaves it alone).
    os.environ.setdefault("WARP_FE_WORKERS", str(settings.fe_workers))
    os.environ.setdefault("WARP_OCR_THREADS", str(settings.ocr_threads))
    os.environ.setdefault("WARP_WORKER_PARSE_SLOTS", str(settings.parse_slots))
    print(
        "Starting Warp-Ingest service: "
        f"cpus={settings.cpus} web_workers={settings.web_workers} "
        f"fe_workers/worker={settings.fe_workers} ocr_threads={settings.ocr_threads} "
        f"parse_slots/worker={settings.parse_slots} "
        f"on {settings.host}:{settings.port}"
    )
    if expected_api_key() == DEFAULT_API_KEY:
        print(
            "WARNING: serving with the default API key "
            f"({DEFAULT_API_KEY!r}); set WARP_API_KEY in production."
        )
    uvicorn.run(
        "warp_ingest.ingestion_daemon.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.web_workers,
        log_level=cfg.log_level().lower(),
    )


if __name__ == "__main__":
    main()
