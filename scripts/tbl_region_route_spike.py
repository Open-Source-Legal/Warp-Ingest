#!/usr/bin/env python3
"""Spike option B: warp's table-REGION detection + a dedicated cell extractor
cropped to each region. Warp finds table regions well (its weakness is the cell
grid), so for each warp table span (page_idx, table_idx) we union the block boxes
into a bbox, crop the PDF page to it, and re-extract the cell grid with pdfplumber
(text strategy) or PyMuPDF. Renders the resulting grids as HTML and scores
grits_trm_composite. This measures the integration that keeps warp's strengths and
swaps only the cell segmentation.

    python scripts/tbl_region_route_spike.py --engine pdfplumber_text --jobs 6
    python scripts/tbl_region_route_spike.py --engine pymupdf --jobs 6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "parsebench_work" / "data"

_PLUMBER_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "join_tolerance": 4,
}


def _load_table_cases():
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cases = _load_jsonl_dataset(DATA)
    return [c for c in cases if getattr(c, "group", None) == "table"]


def _clean_cell(c):
    if c is None:
        return ""
    return " ".join(str(c).split())  # collapse internal newlines/whitespace


def _html_table(rows, header=True):
    rows = [r for r in rows if r and any((c or "").strip() for c in r)]
    if not rows:
        return ""
    parts = ["<table>"]
    if header:
        cells = "".join(f"<th>{escape(_clean_cell(c))}</th>" for c in rows[0])
        parts.append(f"<tr>{cells}</tr>")
        body = rows[1:]
    else:
        body = rows
    for r in body:
        cells = "".join(f"<td>{escape(_clean_cell(c))}</td>" for c in r)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _warp_table_regions(pdf_path):
    """Return {page_idx: [ (x0, top, x1, bottom), ... ]} from warp's table spans."""
    from benchmarks.parsebench.warp_markdown import extract_warp_blocks

    raw = extract_warp_blocks(pdf_path)
    regions = {}
    cur = None  # (page, tidx, x0, top, x1, bottom)
    for b in raw["blocks"]:
        if b.get("block_type") == "table_row" and b.get("table_idx") is not None:
            box = b.get("box")  # [left, top, w, h]
            if not box:
                continue
            pg = int(b.get("page_idx", 0) or 0)
            tidx = b.get("table_idx")
            x0, top, x1, bot = box[0], box[1], box[0] + box[2], box[1] + box[3]
            if cur and cur[0] == pg and cur[1] == tidx:
                cur = (
                    pg,
                    tidx,
                    min(cur[2], x0),
                    min(cur[3], top),
                    max(cur[4], x1),
                    max(cur[5], bot),
                )
            else:
                if cur:
                    regions.setdefault(cur[0], []).append(cur[2:])
                cur = (pg, tidx, x0, top, x1, bot)
    if cur:
        regions.setdefault(cur[0], []).append(cur[2:])
    return regions


def _extract_region_plumber(page, bbox):
    pad = 2.0
    x0, top, x1, bot = bbox
    crop = page.crop(
        (
            max(0, x0 - pad),
            max(0, top - pad),
            min(page.width, x1 + pad),
            min(page.height, bot + pad),
        )
    )
    tables = crop.extract_tables(table_settings=_PLUMBER_TEXT) or []
    return [t for t in tables if t]


def _pages_plumber(pdf_path, regions):
    import pdfplumber

    pages = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pg_idx, bboxes in regions.items():
            if pg_idx >= len(pdf.pages):
                continue
            page = pdf.pages[pg_idx]
            mds = []
            for bbox in bboxes:
                for t in _extract_region_plumber(page, bbox):
                    mds.append(_html_table(t))
            pages[pg_idx] = "\n\n".join(mds)
    return pages


def _pages_pymupdf(pdf_path, regions):
    import fitz

    pages = {}
    doc = fitz.open(pdf_path)
    for pg_idx, bboxes in regions.items():
        if pg_idx >= doc.page_count:
            continue
        page = doc[pg_idx]
        mds = []
        for bbox in bboxes:
            try:
                clip = fitz.Rect(bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
                tabs = page.find_tables(clip=clip)
                for t in tabs.tables:
                    rows = t.extract()
                    if rows:
                        mds.append(_html_table(rows))
            except Exception:
                pass
        pages[pg_idx] = "\n\n".join(mds)
    doc.close()
    return pages


def _score_one(args):
    example_id, pdf_path, expected_markdown, trm_unsupported, engine = args
    from datetime import datetime as _dt

    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import ParseTestCase

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            regions = _warp_table_regions(pdf_path)
            if engine == "pymupdf":
                page_md = _pages_pymupdf(pdf_path, regions)
            else:
                page_md = _pages_plumber(pdf_path, regions)
        maxp = max(page_md.keys()) + 1 if page_md else 1
        rendered = [(i, page_md.get(i, "")) for i in range(maxp)]
        full_md = "\n\n".join(md for _, md in rendered)
        out = ParseOutput(
            task_type="parse",
            example_id=example_id,
            pipeline_name=engine,
            pages=[PageIR(page_index=i, markdown=md) for i, md in rendered],
            markdown=full_md,
        )
        req = InferenceRequest(
            example_id=example_id,
            source_file_path=pdf_path,
            product_type=ProductType.PARSE,
        )
        ir = InferenceResult(
            request=req,
            pipeline_name=engine,
            product_type=ProductType.PARSE,
            raw_output={},
            output=out,
            started_at=_dt.now(),
            completed_at=_dt.now(),
            latency_in_ms=0,
        )
        tc = ParseTestCase(
            test_id=example_id,
            group="table",
            file_path=pdf_path,
            expected_markdown=expected_markdown,
            test_rules=None,
            trm_unsupported=trm_unsupported,
        )
        ev = ParseEvaluator(
            enable_rule_based=False, enable_structural_consistency=False
        )
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            res = ev.evaluate(ir, tc)
        m = {x.metric_name: x.value for x in res.metrics}
        return example_id, m.get("grits_trm_composite"), None
    except Exception as e:
        return example_id, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine", choices=["pdfplumber_text", "pymupdf"], default="pdfplumber_text"
    )
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cases = _load_table_cases()
    cases.sort(key=lambda c: c.test_id)
    if args.limit:
        cases = cases[: args.limit]
    work = [
        (
            c.test_id,
            str(c.file_path),
            c.expected_markdown,
            getattr(c, "trm_unsupported", False),
            args.engine,
        )
        for c in cases
    ]
    results = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for eid, comp, err in pool.map(_score_one, work):
            if err:
                errors += 1
                continue
            results[eid] = comp if comp is not None else 0.0
    n = len(results)
    mean = sum(results.values()) / n * 100 if n else 0.0
    print(f"\nengine=warp-regions+{args.engine}  pages={n}  errors={errors}")
    print(f"mean grits_trm_composite = {mean:.2f}   (warp baseline = 35.67)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
