#!/usr/bin/env python3
"""Refine the over-seg diagnostic: for each interleave flush (non-table_row block
between two table_row runs), classify whether the interleaved block shares the
table_idx of the surrounding rows (engine already calls it ONE table -> safe to
keep the rendered table open) vs. sits outside any engine table span (None idx,
or different idx -> a real boundary the engine drew).
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
    example_id, pdf_path = args
    from benchmarks.parsebench.warp_markdown import extract_warp_blocks

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            raw = extract_warp_blocks(pdf_path)
        blocks = raw["blocks"]
        by_page = {}
        for b in blocks:
            by_page.setdefault(b.get("page_idx", 0) or 0, []).append(b)

        # classify interleave events
        same_idx = (
            0  # interleaved block.table_idx == prev row idx (and prev==next or no next)
        )
        none_idx = 0  # interleaved block has no table_idx (outside engine span)
        diff_idx = 0  # interleaved block has a different non-None table_idx
        same_types = Counter()
        for pg, pblocks in by_page.items():
            prev_row_idx = None
            in_table = False
            i = 0
            n = len(pblocks)
            for i, b in enumerate(pblocks):
                bt = b.get("block_type")
                txt = (b.get("block_text") or "").strip()
                if bt == "table_row":
                    prev_row_idx = b.get("table_idx")
                    in_table = True
                elif in_table and txt:
                    # interleave event: look at this block's idx + whether a table
                    # row follows on the page (so it's genuinely sandwiched)
                    blk_idx = b.get("table_idx")
                    nxt_is_row = any(
                        pblocks[j].get("block_type") == "table_row"
                        for j in range(i + 1, n)
                    )
                    if nxt_is_row:
                        if blk_idx is not None and blk_idx == prev_row_idx:
                            same_idx += 1
                            same_types[bt] += 1
                        elif blk_idx is None:
                            none_idx += 1
                        else:
                            diff_idx += 1
                    in_table = False
        return example_id, same_idx, none_idx, diff_idx, dict(same_types), None
    except Exception as e:
        return example_id, None, None, None, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = _load_table_cases()
    cases.sort(key=lambda c: c.test_id)
    if args.limit:
        cases = cases[: args.limit]
    work = [(c.test_id, str(c.file_path)) for c in cases]

    tot_same = tot_none = tot_diff = 0
    same_types = Counter()
    errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for eid, s, no, d, st, err in pool.map(_analyze_one, work):
            if err:
                errors += 1
                continue
            tot_same += s
            tot_none += no
            tot_diff += d
            same_types.update(st or {})

    print(f"interleave events (block sandwiched before another table row):")
    print(
        f"  SAME table_idx as surrounding rows : {tot_same}  <- safe renderer keep-open"
    )
    print(f"     by type: {same_types.most_common()}")
    print(f"  NONE table_idx (outside engine span): {tot_none}")
    print(f"  DIFF table_idx                      : {tot_diff}")
    print(f"  errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
