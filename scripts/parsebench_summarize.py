#!/usr/bin/env python3
"""Summarize a ParseBench run into a comprehensive, reconciliation-focused table.

Reads each pipeline's official per-category ``_evaluation_report.json`` from a
ParseBench output dir and prints:

* an 8-parser x 5-dimension table of the official headline metrics (x100),
  with the number of pages scored per dimension (transparency),
* the Overall column (arithmetic mean of dimensions present in this run),
* a faithfulness cross-check: every local-baseline row's in-run score minus the
  *published* ``leaderboard.csv`` value, so the harness can be shown to
  reproduce the published deterministic numbers within noise,
* per-pipeline empty-output / error counts (OCR + coverage transparency).

Nothing here re-implements scoring; it only reads the framework's own report
JSON. Usage:

    python scripts/parsebench_summarize.py \
        --output-dir parsebench_work/output \
        --leaderboard ~/Code/ParseBench/leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# category -> (display label, aggregate_metrics headline key without the avg_ prefix)
DIMS = [
    ("table", "Tables", "grits_trm_composite"),
    ("chart", "Charts", "rule_pass_rate"),
    ("text_content", "Content Faith.", "content_faithfulness"),
    ("text_formatting", "Sem. Format.", "semantic_formatting"),
    ("layout", "Visual Ground.", "layout_element_rule_pass_rate"),
]

# our pipeline name -> published leaderboard.csv "Provider" row (None = no row, e.g. ours)
PUBLISHED = {
    "warp_ingest": None,
    "warp_ingest_faithful": None,
    "warp_ingest_quality": None,
    "liteparse_markdown": "LiteParse (no OCR)",
    "pymupdf4llm_markdown": "PyMuPDF4LLM",
    "opendataloader_markdown": "OpenDataLoader",
    "markitdown": "MarkItDown",
    "pymupdf_html": "PyMuPDF (HTML)",
    "pymupdf_text": "PyMuPDF (Text)",
    "pypdf_baseline": "pypdf",
}

# leaderboard.csv column order for the five dims
LB_COLS = [
    "Tables",
    "Charts",
    "Content_Faithfulness",
    "Semantic_Formatting",
    "Visual_Grounding",
]


def read_dim(output_dir: Path, pipeline: str, cat: str, key: str):
    """Return (score_x100_or_None, n_pages_or_None) for one pipeline+dimension."""
    report = output_dir / pipeline / cat / "_evaluation_report.json"
    if not report.exists():
        return None, None
    try:
        data = json.loads(report.read_text())
    except Exception:
        return None, None
    agg = data.get("aggregate_metrics", {})
    raw = agg.get(f"avg_{key}")
    n = data.get("total_examples")
    val = round(float(raw) * 100, 2) if raw is not None else None
    return val, n


def count_empties(output_dir: Path, pipeline: str):
    """Count error entries across this pipeline's category _errors.json files."""
    total = 0
    for cat, _l, _k in DIMS:
        ef = output_dir / pipeline / cat / "_errors.json"
        if ef.exists():
            try:
                errs = json.loads(ef.read_text())
                total += (
                    len(errs) if isinstance(errs, list) else len(errs.get("errors", []))
                )
            except Exception:
                pass
    return total


def load_published(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path or not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            prov = row.get("Provider", "").strip()
            vals = {}
            for c in LB_COLS:
                try:
                    vals[c] = float(row[c]) if row.get(c) not in (None, "") else None
                except (ValueError, KeyError):
                    vals[c] = None
            out[prov] = vals
    return out


def fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="parsebench_work/output")
    ap.add_argument(
        "--leaderboard",
        default=str(Path.home() / "Code" / "ParseBench" / "leaderboard.csv"),
    )
    ap.add_argument(
        "--pipelines",
        default=",".join(PUBLISHED.keys()),
        help="Comma-separated pipeline order.",
    )
    ap.add_argument(
        "--borrow-published-layout",
        action="store_true",
        help=(
            "For text-only baselines with no in-run layout score, fill Visual "
            "Grounding from leaderboard.csv. Off by default so the summary is "
            "strictly this-run evidence."
        ),
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    published = load_published(Path(args.leaderboard))

    rows: dict[str, dict] = {}
    for p in pipelines:
        dims = {}
        npages = {}
        vg_borrowed = False
        for cat, _label, key in DIMS:
            v, n = read_dim(output_dir, p, cat, key)
            # Optional context mode only. The default is strict this-run evidence:
            # missing layout remains missing and Overall is mean-of-present.
            if cat == "layout" and v is None and args.borrow_published_layout:
                prov = PUBLISHED.get(p)
                pub_vg = (
                    published.get(prov, {}).get("Visual_Grounding") if prov else None
                )
                if pub_vg is not None:
                    v = round(float(pub_vg), 2)
                    vg_borrowed = True
            dims[cat] = v
            npages[cat] = n
        present = [dims[c] for c, _l, _k in DIMS if dims[c] is not None]
        # Overall = arithmetic mean of all 5 dims (the official leaderboard
        # aggregation) whenever all 5 are available; else mean of present.
        overall = round(sum(present) / len(present), 2) if present else None
        rows[p] = {
            "dims": dims,
            "npages": npages,
            "overall": overall,
            "errors": count_empties(output_dir, p),
            "vg_borrowed": vg_borrowed,
        }

    name_w = max([len("Pipeline")] + [len(p) for p in pipelines]) + 2
    hdr = (
        f"{'Pipeline':<{name_w}}"
        + "".join(f"{l:>15}" for _c, l, _k in DIMS)
        + f"{'Overall':>10}"
    )
    print("=" * len(hdr))
    print(
        "  ParseBench — this run (official deterministic metrics, x100; Overall = mean of present dims)"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for p in pipelines:
        r = rows[p]
        cells = ""
        for c, _l, _k in DIMS:
            s = fmt(r["dims"][c])
            if c == "layout" and r["vg_borrowed"] and s != "—":
                s = s + "†"
            cells += f"{s:>15}"
        print(f"{p:<{name_w}}{cells}{fmt(r['overall']):>10}")
    print("-" * len(hdr))
    if args.borrow_published_layout:
        print(
            "  † Visual Grounding for text-only baselines is the published leaderboard value"
        )
        print(
            "    (they emit no geometry → not scorable in-run); all other cells are from THIS run."
        )
    else:
        print("  Missing cells are not filled from published scores.")

    # pages scored per dimension (from warp, representative; all share the GT set)
    print("\n  Pages scored per dimension (total_examples in each category report):")
    for p in pipelines:
        r = rows[p]
        ns = "  ".join(f"{l}={r['npages'][c]}" for c, l, _k in DIMS)
        print(f"    {p:<24} {ns}  | empties/errors={r['errors']}")

    # Faithfulness cross-check vs published leaderboard.csv
    print(
        "\n  Faithfulness cross-check — in-run minus published leaderboard.csv (deterministic baselines):"
    )
    print(
        f"    {'Pipeline':<24}{'(published row)':<22}"
        + "".join(f"{l:>15}" for _c, l, _k in DIMS)
    )
    for p in pipelines:
        prov = PUBLISHED.get(p)
        if not prov or prov not in published:
            continue
        pub = published[prov]
        r = rows[p]
        deltas = []
        for (cat, _l, _k), col in zip(DIMS, LB_COLS):
            iv = r["dims"][cat]
            pv = pub.get(col)
            if iv is None or pv is None:
                deltas.append("—")
            else:
                d = iv - pv
                deltas.append(f"{d:+.1f}")
        cells = "".join(f"{d:>15}" for d in deltas)
        print(f"    {p:<24}{prov:<22}{cells}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
