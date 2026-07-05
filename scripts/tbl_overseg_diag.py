#!/usr/bin/env python3
"""Characterize table over-segmentation: for each ParseBench table page, count
how many <table>s warp's renderer emits vs GT, and categorize *why* a single
visual table gets split into several rendered tables.

Two split causes (mirroring warp_markdown._render_block_stream's flush logic):
  * interleave  -- a non-table_row block (header/para/list_item) sits between
                   two table_row runs, forcing a flush.
  * tidx_break  -- two adjacent table_row blocks carry different table_idx
                   (no interleaving block), so the renderer starts a new table.

Run: python scripts/tbl_overseg_diag.py --jobs 8 [--limit N]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "parsebench_work" / "data"


def _load_table_cases():
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cases = _load_jsonl_dataset(DATA)
    return [c for c in cases if getattr(c, "group", None) == "table"]


def _analyze_one(args):
    example_id, pdf_path, expected_markdown = args
    from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_pages

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            raw = extract_warp_blocks(pdf_path)
            rendered = render_pages(raw)
        full_md = "\n\n".join(md for _, md in rendered)
        warp_tables = full_md.count("<table>")
        gt_tables = expected_markdown.count("<table>")

        # Walk the block stream and reproduce the renderer's flush decisions.
        blocks = raw["blocks"]
        interleave = 0  # flushes caused by a non-table block between table runs
        tidx_break = 0  # flushes caused by table_idx change between adjacent rows
        interleave_types = Counter()
        in_table = False
        cur_tidx = None
        # group by page (renderer renders per page)
        by_page = {}
        for b in blocks:
            by_page.setdefault(b.get("page_idx", 0) or 0, []).append(b)
        for pg, pblocks in by_page.items():
            in_table = False
            cur_tidx = None
            for b in pblocks:
                bt = b.get("block_type")
                txt = (b.get("block_text") or "").strip()
                if bt == "table_row":
                    tidx = b.get("table_idx")
                    if in_table and tidx != cur_tidx:
                        tidx_break += 1
                    cur_tidx = tidx
                    in_table = True
                else:
                    if in_table and txt:
                        interleave += 1
                        interleave_types[bt] += 1
                        in_table = False
        return (
            example_id,
            gt_tables,
            warp_tables,
            interleave,
            tidx_break,
            dict(interleave_types),
            None,
        )
    except Exception as e:
        return example_id, None, None, None, None, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    cases = _load_table_cases()
    cases.sort(key=lambda c: c.test_id)
    if args.limit:
        cases = cases[: args.limit]
    work = [(c.test_id, str(c.file_path), c.expected_markdown) for c in cases]

    rows = []
    errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for r in pool.map(_analyze_one, work):
            if r[-1]:
                errors += 1
                continue
            rows.append(r)

    overseg = [r for r in rows if r[2] is not None and r[2] > r[1]]  # warp>gt
    underseg = [r for r in rows if r[2] is not None and r[2] < r[1]]
    exact = [r for r in rows if r[2] == r[1]]
    total_interleave = sum(r[3] for r in rows)
    total_tidx = sum(r[4] for r in rows)
    itypes = Counter()
    for r in rows:
        itypes.update(r[5] or {})

    print(f"pages={len(rows)} errors={errors}")
    print(f"  over-seg (warp>gt) : {len(overseg)}")
    print(f"  under-seg(warp<gt) : {len(underseg)}")
    print(f"  exact              : {len(exact)}")
    print(f"  extra <table>s on over-seg pages: " f"{sum(r[2]-r[1] for r in overseg)}")
    print(f"\nsplit-cause totals across all pages:")
    print(f"  interleave flushes : {total_interleave}")
    print(f"  tidx_break flushes : {total_tidx}")
    print(f"  interleave block types: {itypes.most_common()}")

    # worst over-seg pages
    overseg.sort(key=lambda r: r[2] - r[1], reverse=True)
    print("\nworst over-seg pages (gt -> warp | interleave/tidx):")
    for r in overseg[:25]:
        print(
            f"  {r[0][:50]:50s} {r[1]:>2} -> {r[2]:>2}  " f"il={r[3]} tb={r[4]} {r[5]}"
        )

    if args.save:
        json.dump(
            [
                {
                    "id": r[0],
                    "gt": r[1],
                    "warp": r[2],
                    "interleave": r[3],
                    "tidx_break": r[4],
                    "itypes": r[5],
                }
                for r in rows
            ],
            open(args.save, "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
