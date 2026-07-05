#!/usr/bin/env python
"""CLI: project OpenContractDocExport annotations onto the PDF raster + dump tree.

Usage:
    python scripts/oc_visualize.py PDF [--pages 0,2,5] [--out DIR]
                                       [--scale 2.0] [--mode relationship|label]
    python scripts/oc_visualize.py EXPORT.json --pdf SOURCE.pdf [--pages ...]

Given a PDF it runs Warp (`parse_to_opencontracts`) to produce the export, then
writes one annotated PNG per selected page plus ``tree.txt`` (the sparse
annotation-tree outline) into the output directory.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warp_ingest.ingestor import oc_visualize as V  # noqa: E402


def _parse_pages(spec):
    if not spec:
        return None
    return [int(x) for x in spec.replace(" ", "").split(",") if x != ""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="a PDF, or an export .json (with --pdf)")
    ap.add_argument("--pdf", help="source PDF when `source` is an export .json")
    ap.add_argument("--pages", help="comma-separated 0-based page indices")
    ap.add_argument("--out", default="oc_viz_out", help="output directory")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--mode", choices=("relationship", "label"), default="relationship")
    args = ap.parse_args(argv)

    if args.source.lower().endswith(".json"):
        with open(args.source) as fh:
            export = json.load(fh)
        pdf_path = args.pdf
        if not pdf_path:
            ap.error("--pdf is required when source is an export .json")
    else:
        from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

        export = parse_to_opencontracts(args.source)
        pdf_path = args.source

    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    written = V.project_document(
        export,
        pdf_bytes,
        args.out,
        pages=_parse_pages(args.pages),
        scale=args.scale,
        mode=args.mode,
    )
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
