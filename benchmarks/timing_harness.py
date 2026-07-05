"""Warp-Ingest parse-throughput timing harness (sequential, warm, same-corpus).

Separates the two halves of the PDF pipeline so their cost can be attributed
independently:

* **front-end** -- ``pdf_file_parser.parse_to_html`` (the pure-Python
  pdfplumber/pdfminer word/char/font extraction that emits Tika XHTML).
* **engine**    -- ``pdf_ingestor.parse_blocks`` (the ~6,300-line
  ``visual_ingestor`` rule engine + indent parser + sentence/block derivation).

For each document it does one warm-up parse (discarded) then ``--runs`` timed
parses, reporting the *median* wall time per stage and pages/second.  Mirrors the
ParseBench warp_ingest provider path exactly: ``apply_ocr=False``,
``render_format="all"``.

Optionally times the ``lit`` (LiteParse) binary on the same files for context
(``--with-lit``) -- sequential, warm, same corpus, so the comparison is fair.

Usage:
    python -m benchmarks.timing_harness --set dense
    python -m benchmarks.timing_harness --set dense --with-lit --runs 5
    python -m benchmarks.timing_harness --files a.pdf b.pdf --runs 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Document sets keyed by name.  Paths are relative to the repo root.
_F = "tests/fixtures"
DENSE_SET = [
    f"{_F}/s1/spacex_s1__exhibit41sx1_331523.pdf",  # 3pp  -- named 45x case
    f"{_F}/contracts/fw_vertigis.pdf",  # 5pp
    f"{_F}/contracts/fw_garver.pdf",  # 6pp
    f"{_F}/USC Title 1 - CHAPTER 1.pdf",  # 9pp
    f"{_F}/sample.pdf",  # 9pp
    f"{_F}/s1/cerebras_s1__exhibit1013esx_f124a6.pdf",  # 11pp
    f"{_F}/s1/fervo_s1__exhibit993sx1_6f30d7.pdf",  # 16pp
    f"{_F}/s1/forbright_s1__exhibit1012sx1_11ea38.pdf",  # 17pp
    f"{_F}/s1/generate_bio_s1__ex33_a3ab0f.pdf",  # 21pp
    f"{_F}/EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_EX-10.1_Development_Agreement_ZrZJLLv.pdf",  # 23pp -- named 28x case
    f"{_F}/s1/quantinuum_s1__exhibit33sx1_eefad8.pdf",  # 28pp
    f"{_F}/s1/eikon_s1__ex1010_2e241e.pdf",  # 36pp
    f"{_F}/s1/swarmer_s1__ex1018_765a25.pdf",  # 46pp
    f"{_F}/s1/exyn_s1__ex11_a8500b.pdf",  # 57pp
]
# A couple of larger bodies to show the scaling curve (slow).
BIG_SET = [
    f"{_F}/s1/eikon_s1__ex1011_d4f7e0.pdf",  # 86pp
    f"{_F}/s1/parabilis_s1__ex1011_424531.pdf",  # 96pp
    f"{_F}/s1/hawkeye360_s1__exhibit21sx1_fa98cc.pdf",  # 112pp
]
SETS = {
    "dense": DENSE_SET,
    "big": BIG_SET,
    "all": DENSE_SET + BIG_SET,
    "smoke": DENSE_SET[:5],
}


def _page_count(path: str) -> int:
    import pdfplumber

    try:
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def _disable_ocr() -> None:
    """Force the sparse-page detector to never route to OCR.

    Isolates the *text-extraction* front-end cost (the optimizable lever) from
    neural OCR cost, which is a separate concern the goal explicitly excludes
    ("text-extractable docs, no OCR"). Sparse pages then degrade to little/no
    text instead of running rapidocr.
    """
    from warp_ingest.file_parser import ocr_parser

    ocr_parser.ocr_available = lambda: False
    # Also set the env-var form so the disable reaches spawned front-end
    # workers (they re-import ocr_parser and would not see the patch above).
    os.environ["WARP_DISABLE_OCR"] = "1"


def _force_pdfplumber_path() -> None:
    """Force the original pdfplumber extraction path (disable Lever A).

    Makes the fast pdfminer-direct extractor raise so ``_render_page`` falls back
    to ``page.extract_words`` / ``_render_svg`` -- i.e. the pre-optimization
    front-end. Lets the harness measure the *before* number in the same process.
    """
    from warp_ingest.file_parser import pdf_plumber_parser as _p

    def _raise(_page):
        raise RuntimeError("force pdfplumber path")

    _p._fast_page_objects = _raise
    # The patch above only exists in this process, so also force the serial
    # (in-process) front-end -- "before" means pre-optimization on both axes.
    os.environ["WARP_FE_WORKERS"] = "1"


def _time_warp(path: str, runs: int) -> dict:
    """Time the front-end and engine stages of the warp pipeline (warm median)."""
    from warp_ingest.file_parser import pdf_file_parser
    from warp_ingest.ingestor import line_parser
    from warp_ingest.ingestor.pdf_ingestor import parse_blocks

    sink = io.StringIO()

    def _front():
        return pdf_file_parser.parse_to_html(path, do_ocr=False)

    def _engine(html):
        # Clear the engine's Line/Word LRU caches so every run measures a
        # from-scratch parse of this document (within-document reuse still
        # counts — that's real).  Without this, run N is timed against caches
        # populated by run N-1 of the same doc, which inflates the number
        # beyond what a single real-world parse sees.
        line_parser._cached_bare_line.cache_clear()
        line_parser._cached_word.cache_clear()
        return parse_blocks(html, render_format="all")

    fe_times: list[float] = []
    en_times: list[float] = []
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        html = _front()  # warm-up (also reused below per-run)
        _engine(html)  # warm-up
        for _ in range(runs):
            t0 = time.perf_counter()
            html = _front()
            t1 = time.perf_counter()
            _engine(html)
            t2 = time.perf_counter()
            fe_times.append(t1 - t0)
            en_times.append(t2 - t1)
    return {
        "front_end_s": statistics.median(fe_times),
        "engine_s": statistics.median(en_times),
        "total_s": statistics.median(fe_times) + statistics.median(en_times),
        "fe_runs": fe_times,
        "en_runs": en_times,
    }


def _time_lit(path: str, runs: int) -> float | None:
    import shutil

    lit = shutil.which("lit")
    if not lit:
        return None
    times: list[float] = []
    # warm-up
    subprocess.run([lit, "parse", path], capture_output=True, check=False)
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run([lit, "parse", path], capture_output=True, check=False)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="dense", choices=sorted(SETS))
    ap.add_argument("--files", nargs="*", default=None, help="Explicit file list.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--with-lit", action="store_true")
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR routing to isolate pure text-extraction cost.",
    )
    ap.add_argument(
        "--force-fallback",
        action="store_true",
        help="Force the original pdfplumber path (Lever A off) for before/after.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Front-end worker processes (sets WARP_FE_WORKERS; 1 = serial).",
    )
    ap.add_argument("--json-out", default=None, help="Write results JSON here.")
    args = ap.parse_args()

    if args.workers is not None:
        os.environ["WARP_FE_WORKERS"] = str(args.workers)
    if args.no_ocr:
        _disable_ocr()
    if args.force_fallback:
        _force_pdfplumber_path()

    files = args.files if args.files else SETS[args.set]
    files = [str((REPO / f) if not os.path.isabs(f) else Path(f)) for f in files]
    files = [f for f in files if Path(f).exists()]

    rows = []
    print(
        f"{'doc':<46}{'pp':>4}{'FE ms':>9}{'ENG ms':>9}{'TOT ms':>9}"
        f"{'pg/s':>8}{'FE%':>6}" + ("  lit ms   warp/lit" if args.with_lit else "")
    )
    print("-" * (95 if args.with_lit else 91))
    tot_pp = tot_fe = tot_en = tot_lit = 0.0
    for f in files:
        pp = _page_count(f)
        w = _time_warp(f, args.runs)
        litt = _time_lit(f, args.runs) if args.with_lit else None
        pgs = pp / w["total_s"] if w["total_s"] else 0
        fe_pct = 100 * w["front_end_s"] / w["total_s"] if w["total_s"] else 0
        name = Path(f).name
        if len(name) > 45:
            name = name[:42] + "..."
        line = (
            f"{name:<46}{pp:>4}{w['front_end_s']*1000:>9.1f}"
            f"{w['engine_s']*1000:>9.1f}{w['total_s']*1000:>9.1f}"
            f"{pgs:>8.1f}{fe_pct:>6.0f}"
        )
        if args.with_lit and litt:
            line += f"{litt*1000:>9.1f}{w['total_s']/litt:>10.1f}x"
        print(line)
        rows.append(
            {
                "doc": name,
                "pages": pp,
                "front_end_ms": w["front_end_s"] * 1000,
                "engine_ms": w["engine_s"] * 1000,
                "total_ms": w["total_s"] * 1000,
                "pages_per_s": pgs,
                "fe_pct": fe_pct,
                "lit_ms": (litt * 1000) if litt else None,
            }
        )
        tot_pp += pp
        tot_fe += w["front_end_s"]
        tot_en += w["engine_s"]
        if litt:
            tot_lit += litt

    print("-" * (95 if args.with_lit else 91))
    tot = tot_fe + tot_en
    agg = (
        f"{'TOTAL':<46}{int(tot_pp):>4}{tot_fe*1000:>9.1f}{tot_en*1000:>9.1f}"
        f"{tot*1000:>9.1f}{tot_pp/tot:>8.1f}{100*tot_fe/tot:>6.0f}"
    )
    if args.with_lit and tot_lit:
        agg += f"{tot_lit*1000:>9.1f}{tot/tot_lit:>10.1f}x"
    print(agg)
    print(
        f"\nAggregate: warp {tot_pp/tot:.1f} pg/s "
        f"(front-end {100*tot_fe/tot:.0f}% / engine {100*tot_en/tot:.0f}% of time)"
    )
    if args.with_lit and tot_lit:
        print(
            f"           lit  {tot_pp/tot_lit:.1f} pg/s   warp is {tot/tot_lit:.1f}x slower"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
