#!/usr/bin/env python
"""Drive the 60-page structural-quality audit batch -> PNGs + trees + metrics.

Parses every document in :data:`tests.oc_batch_compat.BATCH_PAGES` **once**
(parallel across documents), then per selected page writes an annotated
projection PNG and records the page's structural summary + deterministic
``page_metrics``; per document it writes ``tree.txt`` and ``diagnostics.json``.

    python scripts/oc_audit60.py --out review_pngs/before [--scale 2.0] [--jobs 12]

Outputs under ``--out``:
  <slug>/page_NNN.png, <slug>/tree.txt, <slug>/diagnostics.json
  INDEX.json           per-page rows (png, summary, metrics, class, note)
  summary.json         aggregate health roll-up
  oc_batch_metrics.json baseline-shaped {page_key: metrics, _doc: {...}}

This is the *discovery* tool for the audit; the committed regression baseline is
produced by ``scripts/build_oc_batch_fixtures.py``.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tests.oc_batch_compat import BATCH_PAGES, page_key, page_metrics  # noqa: E402
from warp_ingest.ingestor import oc_visualize as V  # noqa: E402
from warp_ingest.ingestor.opencontracts_exporter import (  # noqa: E402
    ExportValidationError,
    validate_export,
)
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402


def _slug(relpath):
    base = os.path.splitext(os.path.basename(relpath))[0]
    return base.replace(" ", "_")[:50]


def _relationship_validity(export):
    try:
        validate_export(export)
    except ExportValidationError:
        return False
    return True


def _process(task):
    relpath, pages, is_slow, doc_class, note, out_dir, scale = task
    slug = _slug(relpath)
    abspath = os.path.join(REPO, relpath)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            export = parse_to_opencontracts(abspath)
        with open(abspath, "rb") as fh:
            pdf_bytes = fh.read()
    except Exception:
        return {"slug": slug, "relpath": relpath, "error": traceback.format_exc()}

    doc_dir = os.path.join(out_dir, slug)
    os.makedirs(doc_dir, exist_ok=True)
    with open(os.path.join(doc_dir, "tree.txt"), "w") as fh:
        fh.write(V.render_tree_outline(export))
    diag = V.diagnostics(export)
    with open(os.path.join(doc_dir, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2)

    n_pages = export["page_count"]
    rel_valid = _relationship_validity(export)
    rows, metrics = [], {}
    for p in pages:
        if not (0 <= p < n_pages):
            continue
        png = os.path.join(doc_dir, f"page_{p:03d}.png")
        try:
            V.project_page(export, pdf_bytes, p, scale=scale).save(png)
        except Exception:
            png = None
        pm = page_metrics(export, p)
        rows.append(
            {
                "job_id": f"{slug}__p{p}",
                "slug": slug,
                "doc_class": doc_class,
                "note": note,
                "relpath": relpath,
                "page_index": p,
                "png_path": png,
                "tree_path": os.path.join(doc_dir, "tree.txt"),
                "page_summary": V.page_summary(export, p),
                "metrics": pm,
            }
        )
        metrics[page_key(relpath, p)] = pm
    return {
        "slug": slug,
        "relpath": relpath,
        "doc_class": doc_class,
        "page_count": n_pages,
        "relationship_validity": rel_valid,
        "diagnostics": diag,
        "rows": rows,
        "metrics": metrics,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="review_pngs/before")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=min(12, (os.cpu_count() or 4)))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    tasks = [
        (relpath, pages, is_slow, doc_class, note, args.out, args.scale)
        for relpath, pages, is_slow, doc_class, note in BATCH_PAGES
    ]
    index, doc_diags, errors, batch_metrics = [], [], [], {}
    with Pool(processes=args.jobs) as pool:
        for res in pool.imap_unordered(_process, tasks):
            if "error" in res:
                errors.append({"slug": res["slug"], "error": res["error"]})
                print(f"!! {res['slug']}: parse failed", file=sys.stderr)
                continue
            index.extend(res["rows"])
            d = dict(res["diagnostics"])
            d.update(
                {
                    "slug": res["slug"],
                    "doc_class": res["doc_class"],
                    "page_count": res["page_count"],
                    "relationship_validity": res["relationship_validity"],
                }
            )
            doc_diags.append(d)
            for k, m in res["metrics"].items():
                batch_metrics[k] = m
            batch_metrics[f"{res['relpath']}::_doc"] = {
                "page_count": res["page_count"],
                "relationship_validity": res["relationship_validity"],
            }
            print(f"  {res['slug']}: {len(res['rows'])} pages")

    index.sort(key=lambda r: r["job_id"])
    with open(os.path.join(args.out, "INDEX.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    summary = {
        "n_docs": len(doc_diags),
        "n_pages": len(index),
        "doc_classes": sorted({d["doc_class"] for d in doc_diags}),
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
            "all_relationships_valid": all(
                d["relationship_validity"] for d in doc_diags
            ),
            "loose_boxes": sum(r["metrics"]["loose_boxes"] for r in index),
        },
        "doc_diagnostics": doc_diags,
        "errors": errors,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out, "oc_batch_metrics.json"), "w") as fh:
        json.dump(batch_metrics, fh, indent=2, sort_keys=True)
    print(f"\n{len(index)} pages across {len(doc_diags)} docs -> {args.out}")
    print("aggregate:", json.dumps(summary["aggregate"]))
    return summary


if __name__ == "__main__":
    main()
