#!/usr/bin/env python
"""Prepare per-page vision-adjudication tasks for the legal-100 golden.

The legal batch has no external structural truth, so its golden is built by
**vision adjudication of Warp's own output** (the user's chosen method): for each
manifest page we render a numbered-box overlay of Warp's annotations and emit a
compact annotation list; a vision agent then judges, per numbered box, the
*correct* structural label and *correct* parent box. This script produces the
inputs; ``scripts/adjudicate_legal_golden`` (a Workflow) consumes them.

    python scripts/build_legal_vision_tasks.py            # all cached pages
    python scripts/build_legal_vision_tasks.py --limit 4  # smoke test

Reads cached exports from ``audit_out/legal/exports`` (produced by
``run_structural_eval.py --set legal``). Writes:
  audit_out/legal/vision/<key>.png          numbered-box overlay per page
  audit_out/legal/vision_tasks.json         [{key, png, annotations:[...]}]
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from warp_ingest.ingestor import oc_visualize as V  # noqa: E402

EXPORTS = REPO / "audit_out" / "legal" / "exports"
OUT_DIR = REPO / "audit_out" / "legal" / "vision"
TASKS = REPO / "audit_out" / "legal" / "vision_tasks.json"
MANIFEST = REPO / "tests" / "fixtures" / "legal100_manifest.json"


def _slug(relpath):
    return relpath.replace("/", "__").replace(" ", "_")


def _page_annotations(export, page):
    """On-page Warp annotations in document order with a stable 1-based number."""
    pkey = str(page)
    rows = []
    id_to_n = {}
    for a in export.get("labelled_text", []):
        aj = (a.get("annotation_json") or {}).get(pkey)
        if aj is None:
            continue
        id_to_n[a["id"]] = len(rows) + 1
        rows.append(a)
    out = []
    w = h = None
    meta = export["pawls_file_content"][page]["page"]
    w, h = float(meta["width"]), float(meta["height"])
    for n, a in enumerate(rows, start=1):
        b = a["annotation_json"][pkey]["bounds"]
        out.append(
            {
                "n": n,
                "id": a["id"],
                "warp_label": a["annotationLabel"],
                "text": (
                    a["annotation_json"][pkey].get("rawText") or a.get("rawText") or ""
                )[:120],
                "bbox_frac": [
                    round(b["left"] / w, 4),
                    round(b["top"] / h, 4),
                    round(b["right"] / w, 4),
                    round(b["bottom"] / h, 4),
                ],
                "warp_parent_n": id_to_n.get(a.get("parent_id")),
            }
        )
    return out, rows


def _render_numbered(export, pdf_bytes, page, rows, scale=2.0):
    base = V._render_pdf_page(pdf_bytes, page, scale).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = V._load_font(max(13, int(8 * scale)))
    pkey = str(page)
    for n, a in enumerate(rows, start=1):
        b = a["annotation_json"][pkey]["bounds"]
        x0, y0 = b["left"] * scale, b["top"] * scale
        x1, y1 = b["right"] * scale, b["bottom"] * scale
        is_head = a["annotationLabel"] in ("Section Header", "Title")
        color = (200, 30, 30) if is_head else (30, 90, 200)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(scale)))
        tag = str(n)
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x0, y0, x0 + tw + 6, y0 + th + 6], fill=color)
        draw.text((x0 + 3, y0 + 1), tag, fill=(255, 255, 255), font=font)
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args(argv)

    manifest = json.load(open(MANIFEST))
    pages_by_doc = {}
    for r in manifest:
        pages_by_doc.setdefault(r["relpath"], []).append(r["page_index"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    docs = sorted(pages_by_doc)
    for rel in docs:
        cache = EXPORTS / f"{_slug(rel)}.json"
        if not cache.exists():
            print(f"  !! no export for {rel}", file=sys.stderr)
            continue
        export = json.loads(cache.read_text())
        pdf_bytes = (REPO / rel).read_bytes()
        for page in sorted(set(pages_by_doc[rel])):
            if not (0 <= page < export["page_count"]):
                continue
            anns, rows = _page_annotations(export, page)
            if not anns:
                continue
            key = f"{rel}::p{page}"
            png = OUT_DIR / f"{_slug(rel)}__p{page}.png"
            try:
                _render_numbered(export, pdf_bytes, page, rows, args.scale).save(png)
            except Exception as exc:  # pragma: no cover - render fallback
                print(f"  !! render failed {key}: {exc}", file=sys.stderr)
                continue
            tasks.append(
                {
                    "key": key,
                    "png": str(png),
                    "n_annotations": len(anns),
                    "annotations": anns,
                }
            )
            if args.limit and len(tasks) >= args.limit:
                break
        if args.limit and len(tasks) >= args.limit:
            break

    TASKS.write_text(json.dumps(tasks, indent=1))
    # per-page sidecars (read by the vision agents) + a lightweight index
    # (passed to the adjudication Workflow; the heavy annotation text stays on
    # disk so neither the Workflow args nor its return carry it).
    index = []
    for t in tasks:
        slug = _slug(t["key"].replace("::p", "__p"))
        sidecar = OUT_DIR / f"{slug}.task.json"
        sidecar.write_text(json.dumps(t))
        index.append(
            {
                "key": t["key"],
                "png": t["png"],
                "task_json": str(sidecar),
                "n_annotations": t["n_annotations"],
            }
        )
    (REPO / "audit_out" / "legal" / "vision_index.json").write_text(
        json.dumps(index, indent=1)
    )
    print(f"wrote {len(tasks)} page tasks -> {TASKS.relative_to(REPO)}")
    print(f"sidecars + vision_index.json -> {OUT_DIR.relative_to(REPO)}/")
    print(f"overlays -> {OUT_DIR.relative_to(REPO)}/")
    from collections import Counter

    print(
        "annotation counts:",
        dict(
            Counter(
                (
                    "0"
                    if t["n_annotations"] == 0
                    else (
                        "1-20"
                        if t["n_annotations"] <= 20
                        else "21-50" if t["n_annotations"] <= 50 else "50+"
                    )
                )
                for t in tasks
            )
        ),
    )
    return tasks


if __name__ == "__main__":
    main()
