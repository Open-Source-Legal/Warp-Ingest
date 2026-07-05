#!/usr/bin/env python
"""Regenerate the Docsling layout-oracle fixtures + regression baseline.

For each document in ``tests.docsling_compat.FIXTURE_DOCS`` this:

  1. POSTs the PDF to the Docling parser microservice and slims the returned
     ``OpenContractDocExport`` to the committed oracle shape (annotations only,
     no PAWLS grid)              -> ``tests/fixtures/docsling_targets/<slug>.json``
  2. Runs Warp-Ingest's ``parse_to_opencontracts`` on the same PDF,
  3. Computes the Warp-vs-Docling ``layout_metrics`` and records them in
     ``tests/fixtures/docsling_layout_baseline.json``.

``tests/test_layout_docsling_regression.py`` floors the live exporter against
that committed baseline (the test itself never touches the service).

The microservice is the ``jscrudato/docsling-local`` image. By default we read
its address from the running ``docling-parser`` container; override with
``--url http://host:8000/parse/``.

The endpoint is resolved as: ``--url`` > ``$DOCSLING_URL`` > the running
``docling-parser`` container's IP.

Usage::

    python scripts/build_docsling_fixtures.py
    DOCSLING_URL=http://127.0.0.1:8000/parse/ python scripts/build_docsling_fixtures.py
    python scripts/build_docsling_fixtures.py --url http://127.0.0.1:8000/parse/
"""

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.docsling_compat import (  # noqa: E402
    FIXTURE_DOCS,
    layout_metrics,
    slim_docling,
)
from warp_ingest.ingestor import pdf_ingestor  # noqa: E402
from warp_ingest.ingestor.opencontracts_exporter import validate_export  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
TARGETS = FIXTURES / "docsling_targets"
BASELINE = FIXTURES / "docsling_layout_baseline.json"


def _default_url():
    if os.environ.get("DOCSLING_URL"):
        return os.environ["DOCSLING_URL"]
    try:
        ip = (
            subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "docling-parser",
                    "--format",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(
            "could not locate the docsling service from the 'docling-parser' "
            "container. Start it, or pass the endpoint explicitly via "
            "--url http://HOST:8000/parse/ or $DOCSLING_URL.\n"
            f"(underlying error: {e})"
        )
    if not ip:
        raise SystemExit(
            "'docling-parser' container has no network IP (not running?). "
            "Pass --url http://HOST:8000/parse/ or set $DOCSLING_URL."
        )
    return f"http://{ip}:8000/parse/"


def _docsling(url, pdf_path):
    pdf_bytes = pdf_path.read_bytes()
    body = json.dumps(
        {
            "pdf_base64": base64.b64encode(pdf_bytes).decode(),
            "filename": pdf_path.name,
            "force_ocr": False,
            "roll_up_groups": False,
            "llm_enhanced_hierarchy": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url", help="docsling /parse/ endpoint (default: docling-parser container)"
    )
    args = ap.parse_args()
    url = args.url or _default_url()
    print(f"docsling endpoint: {url}")

    TARGETS.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for slug, rel, is_slow in FIXTURE_DOCS:
        pdf = FIXTURES / rel
        if not pdf.exists():
            print(f"  SKIP (missing PDF): {slug}")
            continue
        t0 = time.time()
        oracle = slim_docling(_docsling(url, pdf))
        (TARGETS / f"{slug}.json").write_text(json.dumps(oracle, indent=1) + "\n")

        warp = pdf_ingestor.parse_to_opencontracts(str(pdf))
        validate_export(warp)
        baseline[slug] = layout_metrics(warp, oracle)
        m = baseline[slug]
        print(
            f"  {slug:22s}{' [slow]' if is_slow else '':7s} {time.time()-t0:5.1f}s  "
            f"sim={m['word_seq_similarity']} label_agree={m['label_agree']} "
            f"head={m['head_ancestor_agree']} overlong_hdr={m['overlong_heading_count']}"
        )

    BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote oracle for {len(baseline)} docs -> {TARGETS}")
    print(f"wrote baseline -> {BASELINE}")


if __name__ == "__main__":
    main()
