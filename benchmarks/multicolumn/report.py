"""Run the multi-column gate probe over proxy-labeled olmOCR-bench sets.

Positives (should split) = olmOCR-bench ``multi_column`` pages; negatives (must
NOT split) = ``tables``; single-column control = ``arxiv_math``. Reports recall,
false-fire, and — the deliverable — a histogram of *why* true multi-column pages
were rejected (which gate stage lost the recall), plus a per-page CSV.

    python -m benchmarks.multicolumn.report                 # full sets
    python -m benchmarks.multicolumn.report --limit 60      # quick sample
    python -m benchmarks.multicolumn.report --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .probe import REASONS, probe_pdf

# olmOCR-bench subdir -> (role, should_fire)
SETS = {
    "multi_column": ("positive", True),
    "tables": ("negative", False),
    "arxiv_math": ("control", False),
}


def bench_pdfs_dir() -> Path:
    root = Path(
        os.environ.get("OLMOCR_BENCH_DIR", Path.home() / "Code" / "olmocr-bench")
    )
    return root / "olmOCR-bench" / "bench_data" / "pdfs"


def _probe_one(pdf: str) -> dict:
    try:
        return probe_pdf(pdf)
    except Exception as e:  # noqa: BLE001 - record, keep going
        return {
            "pdf": pdf,
            "fired": False,
            "n_cols": 0,
            "reason": f"ERR:{type(e).__name__}",
            "n_words": 0,
            "n_rows": 0,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap PDFs per set (sampling)"
    )
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--csv", default=None, help="write per-page rows here")
    args = ap.parse_args()

    pdfs_root = bench_pdfs_dir()
    if not pdfs_root.is_dir():
        raise SystemExit(f"dataset not found at {pdfs_root} (set OLMOCR_BENCH_DIR)")

    tasks: list[tuple[str, str, bool, str]] = []  # (subdir, role, should_fire, pdf)
    for sub, (role, should_fire) in SETS.items():
        fs = sorted((pdfs_root / sub).glob("*.pdf"))
        if args.limit:
            fs = fs[: args.limit]
        for f in fs:
            tasks.append((sub, role, should_fire, str(f)))

    print(
        f"probing {len(tasks)} pages across {len(SETS)} sets (jobs={args.jobs})...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        recs = list(ex.map(_probe_one, [t[3] for t in tasks], chunksize=8))
    for (sub, role, should_fire, _), rec in zip(tasks, recs):
        rec.update(set=sub, role=role, should_fire=should_fire)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "set",
                    "role",
                    "should_fire",
                    "fired",
                    "n_cols",
                    "reason",
                    "n_words",
                    "n_rows",
                    "pdf",
                ],
            )
            w.writeheader()
            for rec in recs:
                w.writerow({k: rec.get(k) for k in w.fieldnames})
        print(f"per-page CSV -> {args.csv}")

    # ---- aggregate report -------------------------------------------------- #
    print("\n=== fire-rate by set ===")
    by_set: dict[str, list] = {}
    for rec in recs:
        by_set.setdefault(rec["set"], []).append(rec)
    for sub, (role, _) in SETS.items():
        rs = by_set.get(sub, [])
        fired = sum(r["fired"] for r in rs)
        n = len(rs)
        tag = {
            "positive": "recall (want HIGH)",
            "negative": "false-fire (want ~0)",
            "control": "fire-rate",
        }[role]
        print(
            f"  {sub:14s} [{role:8s}] {fired:3d}/{n:<3d} = {100*fired/max(n,1):5.1f}%  {tag}"
        )

    print("\n=== why true multi-column pages were REJECTED (recall loss) ===")
    missed = [r for r in by_set.get("multi_column", []) if not r["fired"]]
    hist = Counter(r["reason"] for r in missed)
    total_pos = len(by_set.get("multi_column", []))
    for reason in list(REASONS) + [r for r in hist if r not in REASONS]:
        c = hist.get(reason, 0)
        if c:
            print(
                f"  {reason:18s}: {c:3d}  ({100*c/max(len(missed),1):4.1f}% of misses, "
                f"{100*c/max(total_pos,1):4.1f}% of all positives)"
            )
    print(f"  {'(missed total)':18s}: {len(missed):3d}  of {total_pos} positives")

    print("\n=== false-fires on negatives (must NOT split) ===")
    for sub in ("tables",):
        ff = [r for r in by_set.get(sub, []) if r["fired"]]
        print(f"  {sub}: {len(ff)} false-fire(s)")
        for r in ff[:10]:
            print(
                f"    {Path(r['pdf']).name}  n_cols={r['n_cols']} n_words={r['n_words']}"
            )


if __name__ == "__main__":
    main()
