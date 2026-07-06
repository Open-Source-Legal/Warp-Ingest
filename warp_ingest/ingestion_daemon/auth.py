"""API-key authentication for the ingestion service.

The expected key comes from ``WARP_API_KEY`` and falls back to ``abc123`` so a
bare container works out of the box (the launcher logs a loud warning when it
serves on the fallback).  The key is read per-request, so rotating it in tests
or via a live environment update needs no application reload.  Clients send it
as ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
"""

import os
import secrets

from warp_ingest.ingestion_daemon.service_dependencies import require_service_dependency

_fastapi = require_service_dependency("fastapi")
_fastapi_security = require_service_dependency("fastapi.security", "fastapi")
HTTPException = _fastapi.HTTPException
Security = _fastapi.Security
APIKeyHeader = _fastapi_security.APIKeyHeader

DEFAULT_API_KEY = "abc123"

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_authorization_header = APIKeyHeader(name="Authorization", auto_error=False)


def expected_api_key():
    return os.environ.get("WARP_API_KEY") or DEFAULT_API_KEY


def require_api_key(
    x_api_key: str = Security(_api_key_header),
    authorization: str = Security(_authorization_header),
):
    presented = x_api_key
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = token.strip()
    if presented and secrets.compare_digest(presented, expected_api_key()):
        return
    raise HTTPException(
        status_code=401,
        detail="invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
