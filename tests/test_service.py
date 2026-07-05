"""Tests for the FastAPI ingestion service.

Covers the CPU autotune math, API-key auth, the ``/api/parse`` contract
(option plumbing, render formats, the per-request ``disable_ocr``), and the
error paths (401/415/422/500).  Runs entirely in-process via TestClient — no
server, no docker.
"""

import os

import pytest
from fastapi.testclient import TestClient

from warp_ingest.ingestion_daemon import app as app_module
from warp_ingest.ingestion_daemon.app import app
from warp_ingest.ingestion_daemon.autotune import compute_settings, effective_cpu_count

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_PDF = os.path.join(FIXTURES, "sample.pdf")
NEEDS_OCR_PDF = os.path.join(FIXTURES, "needs_ocr.pdf")

DEFAULT_KEY = {"X-API-Key": "abc123"}

client = TestClient(app)


def _upload(path, name=None):
    with open(path, "rb") as fh:
        data = fh.read()
    return {"file": (name or os.path.basename(path), data, "application/pdf")}


# ---------------------------------------------------------------------------
# autotune
# ---------------------------------------------------------------------------


class TestAutotune:
    def test_defaults_saturate_cpus(self):
        s = compute_settings(cpus=8, env={})
        assert (s.web_workers, s.fe_workers, s.ocr_threads) == (8, 1, 1)
        assert s.parse_slots == 1
        assert (s.host, s.port) == ("0.0.0.0", 5001)

    def test_single_cpu(self):
        s = compute_settings(cpus=1, env={})
        assert (s.web_workers, s.fe_workers, s.ocr_threads) == (1, 1, 1)

    def test_pinned_workers_hand_slack_to_fe(self):
        s = compute_settings(cpus=8, env={"WARP_WEB_WORKERS": "2"})
        assert (s.web_workers, s.fe_workers, s.ocr_threads) == (2, 4, 1)

    def test_single_worker_gets_full_fe_stripe(self):
        s = compute_settings(cpus=8, env={"WARP_WEB_WORKERS": "1"})
        assert (s.web_workers, s.fe_workers, s.ocr_threads) == (1, 8, 1)

    def test_fe_stripe_capped_at_8(self):
        s = compute_settings(cpus=32, env={"WARP_WEB_WORKERS": "1"})
        assert s.fe_workers == 8
        assert s.ocr_threads == 4  # 32 // (1 * 8)

    def test_web_concurrency_fallback(self):
        s = compute_settings(cpus=8, env={"WEB_CONCURRENCY": "3"})
        assert s.web_workers == 3
        # WARP_WEB_WORKERS wins over WEB_CONCURRENCY
        s = compute_settings(
            cpus=8, env={"WEB_CONCURRENCY": "3", "WARP_WEB_WORKERS": "5"}
        )
        assert s.web_workers == 5

    def test_explicit_overrides_win(self):
        s = compute_settings(
            cpus=8,
            env={
                "WARP_FE_WORKERS": "3",
                "WARP_OCR_THREADS": "2",
                "WARP_WORKER_PARSE_SLOTS": "4",
                "WARP_HOST": "127.0.0.1",
                "WARP_PORT": "9000",
                "PORT": "7000",
            },
        )
        assert s.fe_workers == 3
        assert s.ocr_threads == 2
        assert s.parse_slots == 4
        assert (s.host, s.port) == ("127.0.0.1", 9000)

    def test_port_env_fallback(self):
        assert compute_settings(cpus=2, env={"PORT": "7000"}).port == 7000

    def test_invalid_values_fall_back(self):
        s = compute_settings(cpus=4, env={"WARP_WEB_WORKERS": "zero", "PORT": " "})
        assert s.web_workers == 4
        assert s.port == 5001

    def test_effective_cpu_count_positive(self):
        assert effective_cpu_count() >= 1


# ---------------------------------------------------------------------------
# auth + health
# ---------------------------------------------------------------------------


class TestAuth:
    def test_root_health_is_open(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_healthz_is_open(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["ocr_available"], bool)
        assert {"cpus", "web_workers", "fe_workers"} <= body["settings"].keys()

    def test_missing_key_401(self):
        r = client.post("/api/parse", files=_upload(SAMPLE_PDF))
        assert r.status_code == 401
        assert r.json() == {"detail": "invalid or missing API key"}

    def test_wrong_key_401(self):
        r = client.post(
            "/api/parse",
            files=_upload(SAMPLE_PDF),
            headers={"X-API-Key": "nope"},
        )
        assert r.status_code == 401

    def test_env_key_rotation(self, monkeypatch):
        monkeypatch.setenv("WARP_API_KEY", "s3cret")
        r = client.post("/api/parse", files=_upload(SAMPLE_PDF), headers=DEFAULT_KEY)
        assert r.status_code == 401  # fallback key no longer valid
        r = client.post(
            "/api/parse",
            params={"render_format": "json"},
            files=_upload(SAMPLE_PDF),
            headers={"X-API-Key": "s3cret"},
        )
        assert r.status_code == 200

    def test_bearer_header_accepted(self, monkeypatch):
        captured = {}

        def fake_ingest(doc_name, doc_location, mime_type, parse_options=None):
            captured["called"] = True
            return {}, None

        monkeypatch.setattr(app_module.ingestor_api, "ingest_document", fake_ingest)
        r = client.post(
            "/api/parse",
            files=_upload(SAMPLE_PDF),
            headers={"Authorization": "Bearer abc123"},
        )
        assert r.status_code == 200
        assert captured.get("called")


# ---------------------------------------------------------------------------
# option plumbing (no real parse — capture parse_options)
# ---------------------------------------------------------------------------


class TestOptionPlumbing:
    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}

        def fake_ingest(doc_name, doc_location, mime_type, parse_options=None):
            captured["doc_name"] = doc_name
            captured["parse_options"] = parse_options
            return {"ok": True}, None

        monkeypatch.setattr(app_module.ingestor_api, "ingest_document", fake_ingest)
        return captured

    def test_defaults(self, captured):
        r = client.post("/api/parse", files=_upload(SAMPLE_PDF), headers=DEFAULT_KEY)
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        opts = captured["parse_options"]
        assert opts["render_format"] == "all"
        assert opts["apply_ocr"] is False
        assert opts["disable_ocr"] is False
        assert opts["semantic_units"] is False
        assert opts["include_images"] is False

    @pytest.mark.parametrize("truthy", ["yes", "true", "1", "on"])
    def test_bool_spellings(self, captured, truthy):
        r = client.post(
            "/api/parse",
            params={
                "render_format": "opencontracts",
                "apply_ocr": truthy,
                "semantic_units": truthy,
                "include_images": truthy,
            },
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        opts = captured["parse_options"]
        assert opts["render_format"] == "opencontracts"
        assert opts["apply_ocr"] is True
        assert opts["semantic_units"] is True
        assert opts["include_images"] is True

    def test_disable_ocr_plumbed(self, captured):
        r = client.post(
            "/api/parse",
            params={"disable_ocr": "true"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        assert captured["parse_options"]["disable_ocr"] is True

    def test_force_and_disable_ocr_conflict_422(self, captured):
        r = client.post(
            "/api/parse",
            params={"apply_ocr": "true", "disable_ocr": "true"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 422
        assert "mutually exclusive" in r.json()["detail"]
        assert "parse_options" not in captured  # never reached the engine

    def test_bad_render_format_422(self, captured):
        r = client.post(
            "/api/parse",
            params={"render_format": "bogus"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 422

    def test_bad_bool_422(self, captured):
        r = client.post(
            "/api/parse",
            params={"apply_ocr": "maybe"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# end-to-end parses (real engine)
# ---------------------------------------------------------------------------


class TestParseEndToEnd:
    def test_parse_all(self):
        r = client.post("/api/parse", files=_upload(SAMPLE_PDF), headers=DEFAULT_KEY)
        assert r.status_code == 200
        body = r.json()
        assert body["num_pages"] >= 1
        assert body["result"]  # "all" render: document dict

    def test_parse_json(self):
        r = client.post(
            "/api/parse",
            params={"render_format": "json"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        assert r.json()["result"]

    def test_parse_opencontracts_with_semantic_units(self):
        r = client.post(
            "/api/parse",
            params={"render_format": "opencontracts", "semantic_units": "true"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        export = r.json()["result"]
        assert {"title", "content", "pawls_file_content", "labelled_text"} <= set(
            export
        )
        labels = {a["annotationLabel"] for a in export["labelled_text"]}
        assert "Semantic Unit" in labels
        rel_labels = {rel["relationshipLabel"] for rel in export["relationships"]}
        assert "OC_SEMANTIC_UNIT" in rel_labels

    def test_parse_opencontracts_without_semantic_units(self):
        r = client.post(
            "/api/parse",
            params={"render_format": "opencontracts"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        export = r.json()["result"]
        labels = {a["annotationLabel"] for a in export["labelled_text"]}
        assert "Semantic Unit" not in labels

    def test_parse_opencontracts_with_include_images(self):
        # SAMPLE_PDF has no embedded images: asserts the plumbing works and the
        # flag adds nothing on an image-less doc (the image-bearing end-to-end
        # lives in test_opencontracts_export.py).
        r = client.post(
            "/api/parse",
            params={"render_format": "opencontracts", "include_images": "true"},
            files=_upload(SAMPLE_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        export = r.json()["result"]
        assert {"title", "content", "pawls_file_content", "labelled_text"} <= set(
            export
        )
        assert not any(
            t.get("is_image") for p in export["pawls_file_content"] for t in p["tokens"]
        )

    def test_disable_ocr_degrades_scanned_doc(self):
        r = client.post(
            "/api/parse",
            params={"render_format": "json", "disable_ocr": "true"},
            files=_upload(NEEDS_OCR_PDF),
            headers=DEFAULT_KEY,
        )
        assert r.status_code == 200
        result = r.json()["result"]
        # No OCR: the scanned page keeps its (near-empty) text layer.  The
        # json render carries block text in "sentences" (an OCR'd parse of
        # this fixture recovers hundreds of characters there, plus tables).
        blocks = result.get("blocks") or []
        text = " ".join(s for b in blocks for s in b.get("sentences") or [])
        assert len(text) < 200

    def test_non_pdf_rejected_415(self):
        files = {"file": ("notes.txt", b"just some text", "text/plain")}
        r = client.post("/api/parse", files=files, headers=DEFAULT_KEY)
        assert r.status_code == 415
        assert "PDF only" in r.json()["detail"]

    def test_engine_error_returns_500(self, monkeypatch):
        def boom(doc_name, doc_location, mime_type, parse_options=None):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr(app_module.ingestor_api, "ingest_document", boom)
        r = client.post("/api/parse", files=_upload(SAMPLE_PDF), headers=DEFAULT_KEY)
        assert r.status_code == 500
        assert r.json() == {"detail": "internal error while parsing document"}
