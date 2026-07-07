#!/usr/bin/env python3
"""Fast in-process ParseBench *text_content* (Content Faithfulness) eval.

Parses each text_content PDF fresh with warp, renders Markdown via the
benchmark renderer, and scores with the **official** ``ParseEvaluator`` (no
scoring re-implemented).  Reports ``content_faithfulness`` (the leaderboard
Content Faith. headline) plus its two components
(``normalized_text_correctness``, ``normalized_order``) and, with ``--diag``,
dumps per-rule results (type / passed / score / failure message) to a JSONL so
the loss can be root-caused per page.

    python scripts/parsebench_content_fasteval.py --jobs 8 \
        [--baseline b.json --save b.json] [--diag diag.jsonl]
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

_CASES = None


def _load_content_cases():
    global _CASES
    if _CASES is None:
        from parse_bench.test_cases.loader import _load_jsonl_dataset

        cases = _load_jsonl_dataset(DATA)
        _CASES = [c for c in cases if getattr(c, "group", None) == "text_content"]
    return _CASES


def _score_one(args):
    test_id, idx = args
    from datetime import datetime as _dt

    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType

    from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_pages

    cases = _load_content_cases()
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
        rule_results = None
        for x in res.metrics:
            md = getattr(x, "metadata", None) or {}
            if md.get("rule_results"):
                rule_results = [
                    {
                        "type": r.get("type"),
                        "passed": bool(r.get("passed")),
                        "score": r.get("score"),
                        "explanation": (r.get("explanation") or "")[:4000],
                    }
                    for r in md["rule_results"]
                ]
                break
        return (
            test_id,
            {
                "content_faithfulness": m.get("content_faithfulness"),
                "normalized_text_correctness": m.get("normalized_text_correctness"),
                "normalized_order": m.get("normalized_order"),
            },
            {"pdf": pdf_path, "rules": rule_results, "markdown": full_md},
            None,
        )
    except Exception as e:
        return test_id, None, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--diag", default=None, help="write per-rule diagnostics JSONL")
    ap.add_argument("--ids", default=None, help="comma-separated test_ids to run")
    args = ap.parse_args()

    cases = _load_content_cases()
    work = [(c.test_id, i) for i, c in enumerate(cases)]
    if args.ids:
        wanted = {t.strip() for t in args.ids.split(",")}
        work = [w for w in work if w[0] in wanted]
    if args.limit:
        work = work[: args.limit]

    results: dict[str, dict] = {}
    errors = []
    diag_fh = open(args.diag, "w") if args.diag else None
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for tid, metrics, diag, err in pool.map(_score_one, work):
            if err:
                errors.append((tid, err))
                continue
            results[tid] = metrics
            if diag_fh is not None:
                diag_fh.write(json.dumps({"test_id": tid, **metrics, **diag}) + "\n")
                diag_fh.flush()
    if diag_fh is not None:
        diag_fh.close()

    def _mean(key: str) -> float:
        vals = [v[key] for v in results.values() if v.get(key) is not None]
        return sum(vals) / len(vals) * 100 if vals else 0.0

    n = len(results)
    print(f"\ncases={n} errors={len(errors)}")
    for tid, err in errors[:10]:
        print(f"  ERROR {tid}: {err}")
    print(f"mean content_faithfulness       = {_mean('content_faithfulness'):.2f}")
    print(
        f"mean normalized_text_correctness = {_mean('normalized_text_correctness'):.2f}"
    )
    print(f"mean normalized_order            = {_mean('normalized_order'):.2f}")

    if args.baseline and Path(args.baseline).exists():
        base = json.load(open(args.baseline))
        common = [
            i
            for i in results
            if i in base and results[i].get("content_faithfulness") is not None
        ]
        if common:
            bvals = [
                (
                    base[i]["content_faithfulness"]
                    if isinstance(base[i], dict)
                    else base[i]
                )
                for i in common
            ]
            nvals = [results[i]["content_faithfulness"] for i in common]
            bmean = sum(bvals) / len(common) * 100
            nmean = sum(nvals) / len(common) * 100
            print(
                f"vs baseline ({len(common)}): {bmean:.2f} -> {nmean:.2f} ({nmean - bmean:+.2f})"
            )
            moved = sorted(
                (
                    (
                        results[i]["content_faithfulness"]
                        - (
                            base[i]["content_faithfulness"]
                            if isinstance(base[i], dict)
                            else base[i]
                        ),
                        i,
                    )
                    for i in common
                ),
            )
            regressions = [(d, i) for d, i in moved if d < -1e-9]
            wins = [(d, i) for d, i in reversed(moved) if d > 1e-9]
            print(f"pages regressed={len(regressions)} improved={len(wins)}")
            for d, i in regressions[:10]:
                print(f"  REG  {i}: {d*100:+.2f}")
            for d, i in wins[:10]:
                print(f"  WIN  {i}: {d*100:+.2f}")
    if args.save:
        json.dump(results, open(args.save, "w"), indent=0)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
