#!/usr/bin/env python3
"""Fast in-process ParseBench *text_formatting* eval for iterating on inline
emphasis surfacing (bold / italic / superscript / ...).

Parses each text_formatting PDF fresh with warp, renders Markdown via the
benchmark renderer, and scores with the **official** ``ParseEvaluator`` (no
scoring re-implemented). Reports ``semantic_formatting`` (the leaderboard
Sem.Format headline) + ``normalized_text_styling`` (where bold/strikeout/sup/sub
live) + a raw is_sup pass-rate so superscript work can be measured directly.

    python scripts/parsebench_format_fasteval.py --jobs 8 [--baseline b.json --save b.json]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "parsebench_work" / "data"


def _load_format_cases():
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cases = _load_jsonl_dataset(DATA)
    return [c for c in cases if getattr(c, "group", None) == "text_formatting"]


def _score_one(args):
    test_id, idx = args
    from datetime import datetime as _dt

    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType

    from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_pages

    # Re-load the case inside the worker (ParseTestCase isn't trivially picklable
    # across the process boundary with all rule subclasses, so reload by index).
    cases = _load_format_cases_cached()
    tc = cases[idx]
    pdf_path = str(tc.file_path)
    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            raw = extract_warp_blocks(pdf_path)
            rendered = render_pages(raw)
        full_md = "\n\n".join(md for _, md in rendered)
        out = ParseOutput(
            task_type="parse",
            example_id=test_id,
            pipeline_name="warp_ingest",
            pages=[PageIR(page_index=i, markdown=md) for i, md in rendered],
            markdown=full_md,
        )
        req = InferenceRequest(
            example_id=test_id,
            source_file_path=pdf_path,
            product_type=ProductType.PARSE,
        )
        ir = InferenceResult(
            request=req,
            pipeline_name="warp_ingest",
            product_type=ProductType.PARSE,
            raw_output=raw,
            output=out,
            started_at=_dt.now(),
            completed_at=_dt.now(),
            latency_in_ms=0,
        )
        ev = ParseEvaluator(enable_rule_based=True, enable_structural_consistency=False)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            res = ev.evaluate(ir, tc)
        m = {x.metric_name: x.value for x in res.metrics}
        # per-rule is_sup pass-rate from rule_results metadata if present
        sup_pass = sup_tot = 0
        for x in res.metrics:
            md = getattr(x, "metadata", None) or {}
            rr = md.get("rule_results")
            if rr:
                for r in rr:
                    if r.get("type") == "is_sup":
                        sup_tot += 1
                        if r.get("passed"):
                            sup_pass += 1
        return (
            test_id,
            m.get("semantic_formatting"),
            m.get("normalized_text_styling"),
            sup_pass,
            sup_tot,
            None,
        )
    except Exception as e:
        return test_id, None, None, 0, 0, f"{type(e).__name__}: {e}"


_CASES = None


def _load_format_cases_cached():
    global _CASES
    if _CASES is None:
        _CASES = _load_format_cases()
    return _CASES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    cases = _load_format_cases()
    work = [(c.test_id, i) for i, c in enumerate(cases)]
    if args.limit:
        work = work[: args.limit]

    results = {}
    styling = {}
    sup_pass = sup_tot = errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tid, sf, ts, sp, st, err in pool.map(_score_one, work):
            if err:
                errors += 1
                continue
            results[tid] = sf if sf is not None else 0.0
            if ts is not None:
                styling[tid] = ts
            sup_pass += sp
            sup_tot += st

    n = len(results)
    mean_sf = sum(results.values()) / n * 100 if n else 0.0
    mean_ts = sum(styling.values()) / len(styling) * 100 if styling else 0.0
    print(f"\ncases={n} errors={errors}")
    print(f"mean semantic_formatting = {mean_sf:.2f}")
    print(f"mean normalized_text_styling = {mean_ts:.2f}")
    print(f"is_sup rules passed = {sup_pass}/{sup_tot}")

    if args.baseline and Path(args.baseline).exists():
        base = json.load(open(args.baseline))
        common = [i for i in results if i in base]
        if common:
            bmean = sum(base[i] for i in common) / len(common) * 100
            nmean = sum(results[i] for i in common) / len(common) * 100
            print(
                f"vs baseline ({len(common)}): {bmean:.2f} -> {nmean:.2f} ({nmean-bmean:+.2f})"
            )
    if args.save:
        json.dump(results, open(args.save, "w"), indent=0)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
