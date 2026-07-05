#!/usr/bin/env python3
"""Spike: how would a *dedicated* table extractor score on the ParseBench table
set vs warp's current engine (35.67)? Tests pdfplumber.extract_tables (MIT, already
a dep) and optionally PyMuPDF find_tables (AGPL). Pure table extraction: render
each page's detected tables as HTML and score grits_trm_composite with the
official evaluator. This quantifies the ceiling of a "detect table region ->
route to a real table engine" architecture before committing to it.

    python scripts/tbl_engine_spike.py --engine pdfplumber --jobs 8
    python scripts/tbl_engine_spike.py --engine pymupdf --jobs 8
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


def _load_table_cases():
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cases = _load_jsonl_dataset(DATA)
    return [c for c in cases if getattr(c, "group", None) == "table"]


def _clean(c):
    return " ".join(str(c).split()) if c is not None else ""


def _html_table(rows, header=True):
    rows = [r for r in rows if r and any((str(c) or "").strip() for c in r)]
    if not rows:
        return ""
    parts = ["<table>"]
    if header:
        parts.append(
            "<tr>" + "".join(f"<th>{escape(_clean(c))}</th>" for c in rows[0]) + "</tr>"
        )
        body = rows[1:]
    else:
        body = rows
    for r in body:
        parts.append(
            "<tr>" + "".join(f"<td>{escape(_clean(c))}</td>" for c in r) + "</tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


def _pdfplumber_pages(pdf_path, settings=None):
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            tables = pg.extract_tables(table_settings=settings) or []
            md = "\n\n".join(_html_table(t) for t in tables if t)
            pages.append(md)
    return pages


_PLUMBER_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "join_tolerance": 4,
}


def _pymupdf_pages(pdf_path, strategy=None):
    import fitz  # PyMuPDF

    pages = []
    doc = fitz.open(pdf_path)
    for pg in doc:
        try:
            tabs = pg.find_tables(strategy=strategy) if strategy else pg.find_tables()
            mds = []
            for t in tabs.tables:
                rows = t.extract()
                if rows:
                    mds.append(_html_table(rows))
            pages.append("\n\n".join(mds))
        except Exception:
            pages.append("")
    doc.close()
    return pages


def _md_tables_to_html(content):
    import markdown2

    lines = content.split("\n")
    out, tbl, in_t = [], [], False

    def flush():
        nonlocal tbl
        if len(tbl) >= 2:
            html = markdown2.markdown("\n".join(tbl), extras=["tables"]).strip()
            out.append(html if "<table>" in html.lower() else "\n".join(tbl))
        else:
            out.extend(tbl)
        tbl = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            in_t = True
            tbl.append(line)
        else:
            if in_t:
                flush()
                in_t = False
            out.append(line)
    if in_t:
        flush()
    return "\n".join(out)


def _pymupdf4llm_pages(pdf_path):
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        pdf_path, page_chunks=True, show_progress=False, use_ocr=False
    )
    return [_md_tables_to_html(ch.get("text", "")) for ch in chunks]


def _warp_regions(pdf_path):
    """{page_idx: [(x0, top, x1, bottom), ...]} from warp's table spans."""
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


def _native_pages(pdf_path, use_regions=True):
    import pdfplumber

    from warp_ingest.ingestor.table_engine import (
        extract_page_tables,
        render_table_html,
    )

    regions = _warp_regions(pdf_path) if use_regions else {}
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, pg in enumerate(pdf.pages):
            try:
                tables = extract_page_tables(pg, regions=regions.get(i))
                pages.append("\n\n".join(render_table_html(t) for t in tables))
            except Exception:
                pages.append("")
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
            if engine == "pdfplumber":
                pages = _pdfplumber_pages(pdf_path)
            elif engine == "pdfplumber_text":
                pages = _pdfplumber_pages(pdf_path, settings=_PLUMBER_TEXT)
            elif engine == "pymupdf_text":
                pages = _pymupdf_pages(pdf_path, strategy="text")
            elif engine == "pymupdf4llm":
                pages = _pymupdf4llm_pages(pdf_path)
            elif engine == "native":
                pages = _native_pages(pdf_path)
            elif engine == "native_ruled":
                pages = _native_pages(pdf_path, use_regions=False)
            else:
                pages = _pymupdf_pages(pdf_path)
        rendered = list(enumerate(pages))
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
        "--engine",
        choices=[
            "pdfplumber",
            "pdfplumber_text",
            "pymupdf",
            "pymupdf_text",
            "pymupdf4llm",
            "native",
            "native_ruled",
        ],
        default="pdfplumber",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--save", default=None, help="write {example_id: composite} JSON here"
    )
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
    print(f"\nengine={args.engine}  pages_scored={n}  errors={errors}")
    print(f"mean grits_trm_composite = {mean:.2f}   (warp baseline = 35.67)")
    if args.save:
        import json

        json.dump(results, open(args.save, "w"), indent=0)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
