#!/usr/bin/env python
"""Run Warp over a diverse ~35-page PDF sample and emit projections + trees.

For each document in :data:`SAMPLE` this parses the PDF to an
``OpenContractDocExport`` once, then for every selected page writes an annotated
projection PNG and records the page's structural summary + doc-level diagnostics.
The roll-up (``INDEX.json`` / ``summary.json``) is what the heuristic-gap audit
(LLM-in-the-loop) consumes.

    python scripts/oc_sample_audit.py --out audit_out [--scale 2.0]
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warp_ingest.ingestor import oc_visualize as V  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

FX = "tests/fixtures"
PDF = "files/pdf"
S1 = "tests/fixtures/s1"

# (path, [0-based pages], doc_class, note) — chosen for *diversity* of layout, not depth.
SAMPLE = [
    # ---- negotiated agreements / contracts ----
    (
        f"{FX}/EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_"
        "EX-10.1_Development_Agreement_ZrZJLLv.pdf",
        [0, 1, 8, 12],
        "contract",
        "pharma dev agreement; defined terms + run-in headings + tables",
    ),
    (
        f"{PDF}/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf",
        [0, 2, 6],
        "contract",
        "affiliate agreement",
    ),
    (
        f"{PDF}/EcoScienceSolutionsInc_20180406_8-K_EX-10.1_11135398_EX-10.1_Sponsorship_Agreement.pdf",
        [0, 1],
        "contract",
        "sponsorship agreement",
    ),
    (
        f"{PDF}/SoupmanInc_20150814_8-K_EX-10.1_9230148_EX-10.1_Franchise Agreement2.pdf",
        [0, 1],
        "contract",
        "franchise agreement",
    ),
    (
        f"{PDF}/SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement.pdf",
        [0, 3],
        "contract",
        "affiliate agreement (older scan-like)",
    ),
    # ---- statute (deep numbering, centered note-headers) ----
    (
        f"{FX}/USC Title 1 - CHAPTER 1.pdf",
        [0, 2, 5, 8],
        "statute",
        "US Code Title 1; deep numbering + amendment notes",
    ),
    # ---- scanned / OCR ----
    (f"{FX}/needs_ocr.pdf", [0], "ocr", "scanned page through the OCR->XHTML path"),
    # ---- academic multi-column (equations, tables, references) ----
    (
        f"{PDF}/Beyond Examples- High-level Automated Reasoning Paradigm in In-Context Learning via MCTS.pdf",
        [0, 2, 5, 8],
        "academic",
        "two-column paper; abstract, body, tables/figures, references",
    ),
    # ---- long credit agreements (cover, TOC, defined terms, tables) ----
    (f"{PDF}/credit_aon.pdf", [0, 2, 40], "credit_agreement", "AON credit agreement"),
    (
        f"{PDF}/credit_citizens.pdf",
        [1, 30],
        "credit_agreement",
        "Citizens credit agreement",
    ),
    (
        f"{PDF}/credit_walt_disney.pdf",
        [0],
        "credit_agreement",
        "Disney credit agreement cover",
    ),
    # ---- SEC filings ----
    (f"{PDF}/8k.pdf", [0, 2], "sec_filing", "8-K"),
    (f"{PDF}/primary-document.pdf", [0], "sec_filing", "primary SEC document cover"),
    # ---- S-1 exhibits ----
    (
        f"{S1}/cerebras_s1__exhibit1013esx_f124a6.pdf",
        [0, 4],
        "s1_exhibit",
        "S-1 exhibit with tables",
    ),
    (
        f"{S1}/eikon_s1__ex1010_2e241e.pdf",
        [3],
        "s1_exhibit",
        "S-1 agreement exhibit body",
    ),
    (
        f"{S1}/spacex_s1__exhibit41sx1_331523.pdf",
        [0],
        "s1_exhibit",
        "specimen / certificate",
    ),
    (
        f"{S1}/cerebras_s1__exhibit231sx1_d5d6f3.pdf",
        [0],
        "s1_exhibit",
        "1-page auditor consent",
    ),
]


def _slug(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")[:50]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    index, doc_diags, total_pages, errors = [], [], 0, []
    for path, pages, doc_class, note in SAMPLE:
        slug = _slug(path)
        try:
            export = parse_to_opencontracts(path)
            with open(path, "rb") as fh:
                pdf_bytes = fh.read()
        except Exception:  # one bad doc shouldn't sink the sweep
            errors.append({"slug": slug, "error": traceback.format_exc()})
            print(f"!! {slug}: parse failed", file=sys.stderr)
            continue

        doc_dir = os.path.join(args.out, slug)
        os.makedirs(doc_dir, exist_ok=True)
        with open(os.path.join(doc_dir, "tree.txt"), "w") as fh:
            fh.write(V.render_tree_outline(export))
        diag = V.diagnostics(export)
        diag.update(
            {"slug": slug, "doc_class": doc_class, "page_count": export["page_count"]}
        )
        doc_diags.append(diag)

        n_pages = export["page_count"]
        for p in pages:
            if not (0 <= p < n_pages):
                continue
            png = os.path.join(doc_dir, f"page_{p:03d}.png")
            V.project_page(export, pdf_bytes, p, scale=args.scale).save(png)
            index.append(
                {
                    "job_id": f"{slug}__p{p}",
                    "slug": slug,
                    "doc_class": doc_class,
                    "note": note,
                    "pdf_path": path,
                    "page_index": p,
                    "png_path": png,
                    "tree_path": os.path.join(doc_dir, "tree.txt"),
                    "page_summary": V.page_summary(export, p),
                }
            )
            total_pages += 1
            print(f"  {slug} p{p} -> {png}")

    with open(os.path.join(args.out, "INDEX.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    summary = {
        "n_docs": len(doc_diags),
        "n_pages": total_pages,
        "doc_classes": sorted({d["doc_class"] for d in doc_diags}),
        "doc_diagnostics": doc_diags,
        "errors": errors,
        "aggregate": {
            "childbearing_non_headers": sum(
                d["childbearing_non_headers"] for d in doc_diags
            ),
            "non_heading_roots": sum(d["non_heading_roots"] for d in doc_diags),
            "overlong_headings": sum(d["overlong_headings"] for d in doc_diags),
            "untokened": sum(d["untokened"] for d in doc_diags),
            "min_token_coverage": min(
                (d["token_coverage"] for d in doc_diags), default=1.0
            ),
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{total_pages} pages across {len(doc_diags)} docs -> {args.out}")
    print("aggregate:", json.dumps(summary["aggregate"]))
    return summary


if __name__ == "__main__":
    main()
