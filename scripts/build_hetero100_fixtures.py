#!/usr/bin/env python
"""Build the heterogeneous-100 structural-eval golden set + page fixtures.

Source of truth is ParseBench's **human-verified** ``layout.jsonl`` (downloaded
under ``parsebench_work/data``; HuggingFace ``llamaindex/ParseBench``). Each
record's ``rule`` describes one layout element with a ``canonical_class``, a
fractional ``bbox``, a reading-order index, and a ``title_level``. We:

1. select 100 diverse, text-structural pages (biased small; capped per source
   document; guaranteed coverage of lists / tables / furniture / hierarchy),
2. map every element onto Warp's OC label vocabulary and reconstruct a
   ``parent`` hierarchy from reading order + heading depth, and
3. copy the 100 single-page PDFs into ``tests/fixtures/hetero100/pages/`` and
   write the golden set to ``tests/fixtures/hetero100_golden.json``.

The golden is **parser-independent**: regions carry stable ``region_id``s, a
normalized ``bbox_frac``, ``text``, a mapped ``gold_label`` and
``gold_parent_region_id``. Scoring (``tests/oc_golden_eval.py``) aligns Warp's
live annotations to these regions geometrically — annotation IDs are never used.

    python scripts/build_hetero100_fixtures.py            # default 100 pages
    python scripts/build_hetero100_fixtures.py --n 100 --max-per-doc 3

This is deterministic: same dataset -> same selection -> same golden.
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "parsebench_work" / "data"
OUT_PAGES = REPO / "tests" / "fixtures" / "hetero100" / "pages"
OUT_GOLDEN = REPO / "tests" / "fixtures" / "hetero100_golden.json"

# ParseBench canonical_class / source_label / title_level -> Warp OC label.
# Heading depth: smaller = more prominent (Title=0). Body elements get depth 99.
_TITLE_LEVELS = {"title", "document"}
_SECTION_LEVELS = {"section-header", "heading", "paragraph"}


def _norm(s):
    return (s or "").strip().lower()


def map_label(canonical_class, source_label, title_level):
    """(gold_label, heading_depth) for one ParseBench element."""
    cc = _norm(canonical_class)
    sl = _norm(source_label).replace("_", "-")
    tl = _norm(title_level)
    if "list-item" in sl:
        return "List Item", 99
    if cc == "table" or sl in {"table"}:
        return "Table Row", 99
    if cc in {"page-header", "page-footer"} or sl in {
        "page-header",
        "page-footer",
        "header",
        "footer",
    }:
        return "Furniture", -1
    if cc == "picture" or sl in {"image", "picture", "chart"}:
        return "Image", -1
    if cc == "section" or sl in {"paragraph-title", "section-header", "caption"}:
        # caption acts as a small heading over a figure/table in ParseBench
        if sl == "caption":
            return "Paragraph", 99
        if tl in _TITLE_LEVELS:
            return "Title", 0
        if tl == "paragraph":
            return "Section Header", 2
        return "Section Header", 1
    # Text, key-value-region, form, footnote, formula, None -> body paragraph
    return "Paragraph", 99


def _bbox_frac(bbox):
    """ParseBench [x, y, w, h] fractional -> [left, top, right, bottom]."""
    x, y, w, h = bbox
    return [round(x, 5), round(y, 5), round(x + w, 5), round(y + h, 5)]


def _build_regions(rules):
    """Map elements -> golden regions with reconstructed parent hierarchy."""
    elems = []
    for r in rules:
        bbox = r.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        label, depth = map_label(
            r.get("canonical_class"),
            r.get("source_label"),
            (r.get("attributes") or {}).get("title_level"),
        )
        content = r.get("content") or {}
        text = content.get("text", "") if isinstance(content, dict) else ""
        ro = r.get("ro_index")
        bf = _bbox_frac(bbox)
        elems.append(
            {
                "label": label,
                "depth": depth,
                "text": (text or "").strip(),
                "ro_index": ro,
                "bbox_frac": bf,
                "_sort": (ro if ro is not None else 10_000, bf[1], bf[0]),
            }
        )
    elems.sort(key=lambda e: e["_sort"])

    regions = []
    stack = []  # (region_id, depth) of open headings, increasing depth
    for i, e in enumerate(elems):
        is_heading = e["label"] in ("Title", "Section Header")
        is_furniture = e["label"] in ("Furniture", "Image")
        if is_furniture:
            parent = None
        else:
            if is_heading:
                while stack and stack[-1][1] >= e["depth"]:
                    stack.pop()
            parent = stack[-1][0] if stack else None
        regions.append(
            {
                "region_id": i,
                "bbox_frac": e["bbox_frac"],
                "text": e["text"][:200],
                "gold_label": e["label"],
                "gold_parent_region_id": parent,
                "ro_index": e["ro_index"],
            }
        )
        if is_heading:
            stack.append((i, e["depth"]))
    return regions


def _page_features(regions):
    labels = [r["gold_label"] for r in regions]
    return {
        "n": len(regions),
        "has_list": "List Item" in labels,
        "has_table": "Table Row" in labels,
        "has_furniture": "Furniture" in labels,
        "n_headings": sum(1 for x in labels if x in ("Title", "Section Header")),
        "pic_frac": (labels.count("Image") / len(labels)) if labels else 0.0,
    }


def _select(pages, n, max_per_doc):
    """Deterministic, stratified, biased-small selection of n pages."""

    def doc_key(pdf):
        base = Path(pdf).stem
        return re.sub(r"[_-]?p(age)?\d+$", "", base, flags=re.I) or base

    usable = []
    for pdf, info in pages.items():
        f = info["features"]
        if f["n"] < 3 or f["pic_frac"] >= 0.6:
            continue
        usable.append((pdf, info))
    # stable sort by (file_size, pdf) so "biased small" is deterministic
    usable.sort(key=lambda kv: (kv[1]["size"], kv[0]))

    per_doc = defaultdict(int)
    chosen = {}

    def take(pred, limit):
        for pdf, info in usable:
            if len(chosen) >= n:
                return
            if pdf in chosen:
                continue
            if per_doc[doc_key(pdf)] >= max_per_doc:
                continue
            if pred(info["features"]):
                chosen[pdf] = info
                per_doc[doc_key(pdf)] += 1
                if limit is not None and sum(1 for _ in chosen) >= limit:
                    return

    # guarantee coverage of the rarer/important structures first, then fill
    take(lambda f: f["has_list"], None)
    take(lambda f: f["has_table"], None)
    take(lambda f: f["has_furniture"] and f["n_headings"] >= 1, None)
    take(lambda f: f["n_headings"] >= 3, None)
    take(lambda f: True, None)  # fill remainder, smallest-first
    return [(pdf, chosen[pdf]) for pdf in sorted(chosen)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-per-doc", type=int, default=3)
    args = ap.parse_args(argv)

    data = Path(args.data)
    by_pdf = defaultdict(list)
    with open(data / "layout.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            rule = r["rule"]
            rule = json.loads(rule) if isinstance(rule, str) else rule
            by_pdf[r["pdf"]].append(rule)

    pages = {}
    for pdf, rules in by_pdf.items():
        src = data / pdf
        if not src.exists():
            continue
        regions = _build_regions(rules)
        if not regions:
            continue
        pages[pdf] = {
            "regions": regions,
            "features": _page_features(regions),
            "size": src.stat().st_size,
            "src": src,
        }

    selected = _select(pages, args.n, args.max_per_doc)
    print(f"selected {len(selected)} / {len(pages)} usable pages")

    OUT_PAGES.mkdir(parents=True, exist_ok=True)
    # clear stale fixtures so the dir matches the manifest exactly
    for old in OUT_PAGES.glob("*.pdf"):
        old.unlink()

    golden = {}
    feat_tally = defaultdict(int)
    for pdf, info in selected:
        fixture_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(pdf).name)
        dst = OUT_PAGES / fixture_name
        shutil.copyfile(info["src"], dst)
        relpath = os.path.relpath(dst, REPO)
        golden[relpath] = {
            "source_pdf": pdf,
            "page_index": 0,
            "regions": info["regions"],
        }
        f = info["features"]
        for k in ("has_list", "has_table", "has_furniture"):
            feat_tally[k] += int(f[k])

    OUT_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GOLDEN, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)

    n_regions = sum(len(v["regions"]) for v in golden.values())
    print(
        f"wrote {OUT_GOLDEN.relative_to(REPO)}: {len(golden)} pages, "
        f"{n_regions} golden regions"
    )
    print("coverage:", dict(feat_tally))
    return golden


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
