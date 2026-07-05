#!/usr/bin/env python3
"""Regenerate the S-1 cross-engine regression fixtures.

The committed fixtures are:
  * ``tests/fixtures/s1/*.pdf``          - 50 real EDGAR S-1 PDFs (the corpus)
  * ``tests/fixtures/s1_targets/*.json`` - gold targets from the ORIGINAL engine
  * ``tests/fixtures/s1_baseline.json``  - current new-vs-gold compatibility scores
  * ``tests/fixtures/s1_manifest.json``  - provenance (source zip + member per PDF)

Two independent steps:

  --targets   Rebuild gold targets by POSTing every PDF to the *original*
              (Java/Tika) nlm-ingestor. Run the unmodified upstream image:
                  docker run -d -p 5011:5001 ghcr.io/nlmatics/nlm-ingestor:latest
              then: python scripts/build_s1_fixtures.py --targets \
                        --orig-url http://localhost:5011/api/parseDocument

  --baseline  Re-run the pure-Python engine on every PDF and recompute the
              new-vs-gold metrics that the regression test floors against.
              Run this after any *intended* engine change to re-bless the baseline.

With no flags, both steps run (``--targets`` requires the original engine up).
"""

import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tests import s1_compat as C  # noqa: E402

SLOW_PAGE_THRESHOLD = 40  # docs with >= this many pages are marked slow


def _post_pdf(url, path):
    import uuid

    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{os.path.basename(path)}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        url + "?renderFormat=all&useNewIndentParser=yes&applyOcr=no",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def build_targets(orig_url):
    os.makedirs(C.S1_TARGET_DIR, exist_ok=True)
    pdfs = sorted(f for f in os.listdir(C.S1_PDF_DIR) if f.endswith(".pdf"))
    for i, fn in enumerate(pdfs, 1):
        name = fn[:-4]
        d = _post_pdf(orig_url, os.path.join(C.S1_PDF_DIR, fn))
        rd = d.get("return_dict") or {}
        # The original engine returns {"status":"fail","reason":...} when it crashes
        # on a PDF (e.g. a KeyError in its visual_ingestor). Record that honestly so
        # the regression suite knows there is no gold to match for this doc.
        orig_failed = d.get("status") == "fail" or "return_dict" not in d
        blocks = C.result_to_blocks(rd.get("result") or {})
        target = {
            "name": name,
            "num_pages": rd.get("num_pages"),
            "n_blocks": len(blocks),
            "orig_failed": orig_failed,
            "orig_reason": d.get("reason") if orig_failed else None,
            "blocks": blocks,
        }
        with open(os.path.join(C.S1_TARGET_DIR, name + ".json"), "w") as f:
            json.dump(target, f, indent=0)
        print(f"[{i}/{len(pdfs)}] target {name} ({len(blocks)} blocks)")


def build_baseline():
    from warp_ingest.ingestor.pdf_ingestor import PDFIngestor

    opts = {
        "render_format": "all",
        "apply_ocr": False,
        "parse_pages": (),
    }
    baseline = {}
    names = C.target_names()
    for i, name in enumerate(names, 1):
        target = C.load_target(name)
        ing = PDFIngestor(os.path.join(C.S1_PDF_DIR, name + ".pdf"), dict(opts))
        new_blocks = C.result_to_blocks(ing.return_dict.get("result") or {})
        new_pages = ing.return_dict.get("num_pages")
        m = C.compare(target["blocks"], new_blocks)
        # ``slow`` uses the new engine's page count (the original may have failed and
        # have None pages); large multi-hundred-page bodies are the expensive tests.
        n_pages = (
            new_pages if isinstance(new_pages, int) else (target["num_pages"] or 0)
        )
        baseline[name] = {
            "token_recall": round(m["token_recall"], 4),
            "token_precision": round(m["token_precision"], 4),
            "seq_ratio": round(m["seq_ratio"], 4),
            "gold_pages": target["num_pages"],
            "new_pages": new_pages,
            "gold_blocks": m["gold_blocks"],
            "new_blocks": m["new_blocks"],
            "gold_tables": m["gold_tables"],
            "new_tables": m["new_tables"],
            "orig_failed": bool(target.get("orig_failed")),
            "slow": n_pages >= SLOW_PAGE_THRESHOLD,
        }
        print(f"[{i}/{len(names)}] baseline {name} recall={m['token_recall']:.3f}")
    with open(C.S1_BASELINE, "w") as f:
        json.dump(baseline, f, indent=1, sort_keys=True)
    print(f"wrote {C.S1_BASELINE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--orig-url", default="http://localhost:5011/api/parseDocument")
    args = ap.parse_args()
    do_all = not (args.targets or args.baseline)
    if args.targets or do_all:
        build_targets(args.orig_url)
    if args.baseline or do_all:
        build_baseline()


if __name__ == "__main__":
    main()
