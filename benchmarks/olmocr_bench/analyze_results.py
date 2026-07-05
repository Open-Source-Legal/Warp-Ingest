#!/usr/bin/env python3
"""Turn the official olmOCR-bench stdout (+ generation logs) into a comparison table.

Reads a benchmark results text file (the captured stdout of
``python -m olmocr.bench.benchmark``) and prints per-category pass rates for
every candidate alongside the published olmOCR-bench leaderboard numbers.

    python -m benchmarks.olmocr_bench.analyze_results results_both.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# olmOCR-bench category -> the JSONL file it is scored from (Base = auto baseline)
CAT_JSONL = [
    ("ArXiv math", "arxiv_math.jsonl"),
    ("Old scans math", "old_scans_math.jsonl"),
    ("Tables", "table_tests.jsonl"),
    ("Old scans", "old_scans.jsonl"),
    ("Headers & footers", "headers_footers.jsonl"),
    ("Multi column", "multi_column.jsonl"),
    ("Long tiny text", "long_tiny_text.jsonl"),
    ("Base", "baseline"),
]

# Reproduced-in-house numbers from the olmocr repo README (same harness family).
REF = {
    #                 AR    OSM   TA    OS    HF    MC    LTT   Base  Overall
    "olmOCR v0.4.0": [83.0, 82.3, 84.9, 47.7, 96.1, 83.7, 81.9, 99.7, 82.4],
    "Marker 1.10.1": [83.8, 66.8, 72.9, 33.5, 86.6, 80.0, 85.7, 99.3, 76.1],
    "MinerU 2.5.4": [76.6, 54.6, 84.9, 33.7, 96.6, 78.2, 83.5, 93.7, 75.2],
}


def parse_candidates(text: str) -> dict[str, dict[str, float]]:
    """Return {candidate: {jsonl: rate}} from the benchmark's Final Summary."""
    out: dict[str, dict[str, float]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^(\w[\w.-]*)\s+:\s+Average Score:\s+([\d.]+)%", line)
        if m:
            cur = m.group(1)
            out[cur] = {"__overall__": float(m.group(2))}
            continue
        m = re.match(
            r"^\s+([\w.]+\.jsonl|baseline)\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)", line
        )
        if m and cur:
            out[cur][m.group(1)] = float(m.group(2))
    return out


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results_both.txt")
    cands = parse_candidates(results.read_text())
    if not cands:
        sys.exit(f"no candidates parsed from {results}")

    names = list(cands)
    header = (
        f"| {'Category':18s} |"
        + "".join(f" {n:>10s} |" for n in names)
        + " olmOCR v0.4.0 | Marker | MinerU |"
    )
    print(header)
    print(
        "|"
        + "-" * 20
        + "|"
        + "|".join("-" * 12 for _ in names)
        + "|---------------|--------|--------|"
    )
    for i, (cat, jl) in enumerate(CAT_JSONL):
        row = f"| {cat:18s} |"
        for n in names:
            v = cands[n].get(jl)
            row += f" {v:9.1f}% |" if v is not None else "     n/a   |"
        row += f" {REF['olmOCR v0.4.0'][i]:13.1f} |{REF['Marker 1.10.1'][i]:7.1f} |{REF['MinerU 2.5.4'][i]:7.1f} |"
        print(row)
    row = f"| {'OVERALL':18s} |"
    for n in names:
        row += f" {cands[n]['__overall__']:9.1f}% |"
    row += f" {82.4:13.1f} |{76.1:7.1f} |{75.2:7.1f} |"
    print(row)


if __name__ == "__main__":
    main()
