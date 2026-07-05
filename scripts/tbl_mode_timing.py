#!/usr/bin/env python3
"""Latency comparison: Warp-Fast (provider off) vs Warp-Quality (pymupdf4llm
table provider on), single-threaded per-document parse time — the service-latency
view for positioning the two modes.

For each sampled PDF we time extract_warp_blocks twice: once with the table
provider disabled (WARP_TABLE_PROVIDER=none) and once enabled (=auto). The
provider re-parses the whole document with pymupdf4llm, so the delta is its cost.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "parsebench_work" / "data"


def _load(group):
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cs = [c for c in _load_jsonl_dataset(DATA) if getattr(c, "group", None) == group]
    cs.sort(key=lambda c: c.test_id)
    return cs


def _time_parse(pdf_path, provider_mode):
    """Time a single extract_warp_blocks under the given provider mode (seconds),
    returning (seconds, num_pages)."""
    os.environ["WARP_TABLE_PROVIDER"] = provider_mode
    # warp_markdown caches get_table_provider's import, but reads the env each
    # call, so toggling the env between calls switches modes within one process.
    from benchmarks.parsebench import warp_markdown

    sink = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        raw = warp_markdown.extract_warp_blocks(pdf_path)
    dt = time.perf_counter() - t0
    return dt, int(raw.get("num_pages", 0) or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-table", type=int, default=20)
    ap.add_argument("--n-text", type=int, default=10)
    args = ap.parse_args()

    table = _load("table")[: args.n_table]
    text = _load("text_content")[: args.n_text]
    sample = [("table", c) for c in table] + [("text", c) for c in text]

    # Warm up imports + file cache (don't count the first parse).
    _time_parse(str(sample[0][1].file_path), "none")
    _time_parse(str(sample[0][1].file_path), "auto")

    rows = []
    for grp, c in sample:
        p = str(c.file_path)
        # quality first then fast, or vice versa; run fast then quality, both
        # after a warm file cache from the warmup of doc0 (other docs cold on
        # first read but both modes pay it).
        t_fast, npg = _time_parse(p, "none")
        t_qual, _ = _time_parse(p, "auto")
        rows.append((grp, c.test_id, npg, t_fast, t_qual))

    def _summ(label, idx):
        vals = sorted(r[idx] for r in rows)
        n = len(vals)
        total = sum(vals)
        mean = total / n
        med = vals[n // 2]
        return f"{label:16s} total={total:6.1f}s  mean={mean:5.2f}s  median={med:5.2f}s"

    print(f"\nsampled {len(rows)} docs ({len(table)} table + {len(text)} text)")
    print(_summ("Warp-Fast", 3))
    print(_summ("Warp-Quality", 4))
    fast_tot = sum(r[3] for r in rows)
    qual_tot = sum(r[4] for r in rows)
    print(
        f"\nWarp-Quality / Warp-Fast  = {qual_tot/fast_tot:.2f}x  (provider overhead)"
    )
    print(f"added latency per doc (mean) = {(qual_tot-fast_tot)/len(rows):.2f}s")

    rows.sort(key=lambda r: r[4] - r[3], reverse=True)
    print("\nslowest provider deltas (doc, pages, fast -> quality):")
    for grp, tid, npg, tf, tq in rows[:8]:
        print(f"  [{grp:5s}] {tid[:42]:42s} pg={npg:>3} {tf:6.2f}s -> {tq:6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
