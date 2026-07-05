#!/usr/bin/env python3
"""Generate a candidate's Markdown for every olmOCR-bench PDF.

olmOCR-bench (AI2) scores one Markdown file per PDF, named
``<cat>/<base>_pg{page}_repeat{n}.md`` under a ``<bench_data>/<candidate>/``
folder. The PDFs are single-page, so every test is ``page=1`` and we emit
``<base>_pg1_repeat1.md`` (deterministic parsers → 1 repeat).

Two parsers are supported, each run through its *real* rendering path (nothing
tuned for this benchmark):

* ``warp``      — nlm-ingestor via ``benchmarks/parsebench/warp_markdown.py``
                  (the same renderer used for ParseBench), with the
                  olmOCR-bench render-boundary transforms explicitly enabled.
                  Process pool + SIGALRM timeout.
* ``liteparse`` — the LiteParse ``lit`` CLI exactly as ParseBench invokes it
                  (``lit parse --format markdown --quiet --no-links``, OCR on by
                  default). Raw markdown is emitted — olmOCR-bench reads markdown
                  pipe tables natively, so no HTML-table conversion is applied.

The working directory (dataset + candidate outputs) defaults to
``$OLMOCR_BENCH_DIR`` or ``$HOME/Code/olmocr-bench`` and is **never** inside the
repo — the dataset, generated markdown and scoring venv are not committed.

    python -m benchmarks.olmocr_bench.generate --parser warp
    python -m benchmarks.olmocr_bench.generate --parser liteparse --jobs 6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def bench_data_dir() -> Path:
    root = Path(
        os.environ.get("OLMOCR_BENCH_DIR", Path.home() / "Code" / "olmocr-bench")
    )
    return root / "olmOCR-bench" / "bench_data"


def out_path(bench_data: Path, candidate: str, pdf: Path) -> Path:
    rel = pdf.relative_to(bench_data / "pdfs")
    return bench_data / candidate / f"{rel.with_suffix('')}_pg1_repeat1.md"


# --------------------------------------------------------------------------- #
# Warp worker (process pool; SIGALRM soft-timeout guards a hung parse)
# --------------------------------------------------------------------------- #
class _Timeout(Exception):
    pass


def _warp_worker(pdf: str, out: str, timeout: float) -> dict:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    # These are olmOCR-bench-specific render-boundary choices. ParseBench's
    # faithful runner leaves them disabled.
    os.environ.setdefault("WARP_TABLE_PROVIDER", "auto")
    os.environ.setdefault("WARP_HF_STRIP", "1")
    os.environ.setdefault("WARP_CJK_STRIP", "1")
    from benchmarks.parsebench.warp_markdown import extract_warp_blocks, render_markdown

    rec = {"pdf": pdf, "ok": False, "chars": 0, "secs": 0.0, "err": None}
    t0 = time.time()

    def _alarm(signum, frame):
        raise _Timeout()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout))
    try:
        # Warp prints noisy per-page progress; keep the log clean.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            md = render_markdown(extract_warp_blocks(pdf))
        Path(out).write_text(md, encoding="utf-8")
        rec.update(ok=True, chars=len(md))
    except _Timeout:
        Path(out).write_text("", encoding="utf-8")
        rec["err"] = "timeout"
    except Exception as e:  # noqa: BLE001 - record parse failure, keep going
        Path(out).write_text("", encoding="utf-8")
        rec["err"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        if has_alarm:
            signal.alarm(0)
    rec["secs"] = round(time.time() - t0, 2)
    return rec


# --------------------------------------------------------------------------- #
# LiteParse worker (thread pool; the `lit` CLI is a subprocess)
# --------------------------------------------------------------------------- #
def _lite_worker(pdf: str, out: str, timeout: float) -> dict:
    rec = {"pdf": pdf, "ok": False, "chars": 0, "secs": 0.0, "err": None}
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["lit", "parse", pdf, "--format", "markdown", "--quiet", "--no-links"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            Path(out).write_text("", encoding="utf-8")
            rec["err"] = f"exit{proc.returncode}: {(proc.stderr or '')[:160]}"
        else:
            Path(out).write_text(proc.stdout, encoding="utf-8")
            rec.update(ok=True, chars=len(proc.stdout))
    except subprocess.TimeoutExpired:
        Path(out).write_text("", encoding="utf-8")
        rec["err"] = "timeout"
    except Exception as e:  # noqa: BLE001
        Path(out).write_text("", encoding="utf-8")
        rec["err"] = f"{type(e).__name__}: {e}"[:200]
    rec["secs"] = round(time.time() - t0, 2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parser", required=True, choices=["warp", "liteparse"])
    ap.add_argument(
        "--candidate", default=None, help="output subfolder (default: --parser)"
    )
    ap.add_argument(
        "--jobs", type=int, default=None, help="workers (default 5 warp / 6 lite)"
    )
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    candidate = args.candidate or args.parser
    jobs = args.jobs or (5 if args.parser == "warp" else 6)
    bench_data = bench_data_dir()
    pdf_root = bench_data / "pdfs"
    if not pdf_root.is_dir():
        sys.exit(f"dataset not found at {pdf_root} — run setup_olmocr_bench.sh first")
    log = Path(args.log or (bench_data.parent.parent / f"gen_{candidate}_log.jsonl"))

    pdfs = sorted(pdf_root.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    todo = []
    for p in pdfs:
        o = out_path(bench_data, candidate, p)
        if o.exists() and o.stat().st_size > 0:
            continue  # resume
        o.parent.mkdir(parents=True, exist_ok=True)
        todo.append((str(p), str(o)))
    print(
        f"parser={args.parser} candidate={candidate} total={len(pdfs)} "
        f"todo={len(todo)} jobs={jobs}",
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        return

    logf = open(log, "a", encoding="utf-8")
    t0 = time.time()
    done = 0
    fails = 0

    def _emit(rec):
        nonlocal done, fails
        done += 1
        if not rec["ok"]:
            fails += 1
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if done % 50 == 0 or not rec["ok"]:
            rate = done / (time.time() - t0)
            print(
                f"[{done}/{len(todo)}] fails={fails} rate={rate:.1f}/s "
                f"eta={(len(todo)-done)/rate/60:.1f}m {Path(rec['pdf']).name} "
                f"{rec['err'] or ''}",
                flush=True,
            )

    if args.parser == "warp":
        with mp.Pool(jobs, maxtasksperchild=15) as pool:
            for r in [
                pool.apply_async(_warp_worker, (pdf, out, args.timeout))
                for pdf, out in todo
            ]:
                _emit(r.get())
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(_lite_worker, pdf, out, args.timeout) for pdf, out in todo
            ]
            for f in as_completed(futs):
                _emit(f.result())

    logf.close()
    print(
        f"DONE candidate={candidate} done={done} fails={fails} "
        f"elapsed={(time.time()-t0)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
