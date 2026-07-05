#!/usr/bin/env python3
"""Fast in-process ParseBench *table* eval for iterating on the engine.

Parses each table PDF fresh with the current warp engine, renders Markdown via
the benchmark renderer, and scores ``grits_trm_composite`` with the **official**
``ParseEvaluator`` (no scoring re-implemented). Runs a representative subset in
~1 minute so engine changes can be measured without a full 10-minute pipeline.

    python scripts/parsebench_table_fasteval.py --subset isolation   # 50 worst pages
    python scripts/parsebench_table_fasteval.py --subset all --jobs 8 # full 503
    python scripts/parsebench_table_fasteval.py --subset isolation --baseline base.json --save base.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA = REPO / "parsebench_work" / "data"
# Optional path to the diagnostic "50 worst pages" CSV (scripts/parsebench_summarize
# + diag); when present, ``--subset isolation`` scores just those, else all.
ISO_CSV = Path(
    "/tmp/claude-1000/-home-jman-Code-nlm-ingestor/"
    "aa433fef-f093-4ef8-8106-c4f5d6e3e32e/scratchpad/parsebench_diag/isolation_50.csv"
)


def _load_table_cases():
    from parse_bench.test_cases.loader import _load_jsonl_dataset

    cases = _load_jsonl_dataset(DATA)
    return [c for c in cases if getattr(c, "group", None) == "table"]


def _score_one(args):
    """Parse one PDF + score grits_trm_composite. Runs in a worker process."""
    example_id, pdf_path, expected_markdown, trm_unsupported = args
    # Imports inside the worker (ProcessPoolExecutor).
    from datetime import datetime as _dt

    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import ParseTestCase

    from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_pages

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            raw = extract_warp_blocks(pdf_path)
            rendered = render_pages(raw)
        full_md = "\n\n".join(md for _, md in rendered)
        out = ParseOutput(
            task_type="parse",
            example_id=example_id,
            pipeline_name="warp_ingest",
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
            pipeline_name="warp_ingest",
            product_type=ProductType.PARSE,
            raw_output=raw,
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
        return (
            example_id,
            m.get("grits_trm_composite"),
            m.get("tables_actual"),
            m.get("tables_expected"),
            None,
        )
    except Exception as e:  # keep going
        return example_id, None, None, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subset", default="isolation", help="isolation | all | <csv of example_ids>"
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument(
        "--baseline",
        default=None,
        help="JSON of {example_id: composite} to diff against",
    )
    ap.add_argument(
        "--save", default=None, help="write {example_id: composite} JSON here"
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = _load_table_cases()
    by_id = {c.test_id: c for c in cases}

    if args.subset == "all":
        ids = sorted(by_id)
    elif args.subset == "isolation":
        if ISO_CSV.exists():
            ids = [
                r["example_id"]
                for r in csv.DictReader(open(ISO_CSV))
                if r["example_id"] in by_id
            ]
        else:
            ids = sorted(by_id)
    else:
        want = {s.strip() for s in args.subset.split(",")}
        ids = [i for i in sorted(by_id) if i in want]
    if args.limit:
        ids = ids[: args.limit]

    work = []
    for i in ids:
        c = by_id[i]
        work.append(
            (
                i,
                str(c.file_path),
                c.expected_markdown,
                getattr(c, "trm_unsupported", False),
            )
        )

    results = {}
    actual_tbls = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for eid, comp, na, ne, err in pool.map(_score_one, work):
            if err:
                errors += 1
                continue
            results[eid] = comp if comp is not None else 0.0
            actual_tbls[eid] = na

    n = len(results)
    mean = sum(results.values()) / n * 100 if n else 0.0
    print(f"\nsubset={args.subset}  pages_scored={n}  errors={errors}")
    print(f"mean grits_trm_composite = {mean:.2f}")

    if args.baseline and Path(args.baseline).exists():
        base = json.load(open(args.baseline))
        common = [i for i in results if i in base]
        if common:
            bmean = sum(base[i] for i in common) / len(common) * 100
            nmean = sum(results[i] for i in common) / len(common) * 100
            print(
                f"vs baseline ({len(common)} common pages): {bmean:.2f} -> {nmean:.2f}  ({nmean-bmean:+.2f})"
            )
            deltas = sorted(common, key=lambda i: results[i] - base[i])
            print("  biggest regressions:")
            for i in deltas[:6]:
                if results[i] < base[i] - 1e-4:
                    print(f"    {i[:48]:48s} {base[i]:.3f} -> {results[i]:.3f}")
            print("  biggest gains:")
            for i in deltas[::-1][:6]:
                if results[i] > base[i] + 1e-4:
                    print(f"    {i[:48]:48s} {base[i]:.3f} -> {results[i]:.3f}")

    if args.save:
        json.dump(results, open(args.save, "w"), indent=0)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
