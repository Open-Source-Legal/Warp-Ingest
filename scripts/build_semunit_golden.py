#!/usr/bin/env python
"""Derive the Semantic-Unit golden from Warp's fine-annotation *text numbering*.

Non-circular by construction: the golden groups are cut on the leading
enumerator of each fine annotation's text (``1.``/``ARTICLE II``/``(a)`` …),
which is independent of the ``parent_id`` tree the coarsener consumes. A new
top-level enumerator starts a new gold unit; blocks until the next such
enumerator are its members. Out-of-range manifest pages are skipped with a
warning. Writes ``tests/fixtures/semunit_golden.json``.

    python scripts/build_semunit_golden.py
"""

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests import oc_semunit_compat as C  # noqa: E402
from warp_ingest.ingestor.line_parser import Line  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

# A section-number-prefixed heading ("ARTICLE III", "Section 5", "Item 7.").
# Text-intrinsic, so it is independent of the coarsener's label decisions and,
# unlike matching every Section Header, does NOT fire on folio furniture
# ("Page 2 of 9") — exactly the boundary a numbering-derived truth should mark.
_HEADING_ORDINAL = re.compile(
    r"^(?:ARTICLE|Article|SECTION|Section|ITEM|Item)\s+"
    r"([0-9]+(?:\.[0-9]+)*|[IVXLC]+|[ivxlc]+|[A-Z])\b"
)


def _starts_unit(text):
    """True if the text opens a new clause: a leading enumerator, a section-
    number-prefixed heading, or a short ALL-CAPS heading (``REPEALS``,
    ``STATUTORY NOTES``). All text-intrinsic — independent of the engine's
    font-based Section-Header labels, so the golden stays a fair oracle."""
    t = (text or "").strip()
    if not t:
        return False
    # a leading enumerator with body text is a clause; a bare "2." is a folio
    if getattr(Line(t), "numbered_line", False) and len(t.split()) >= 2:
        return True
    if _HEADING_ORDINAL.match(t):
        return True
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    return (
        1 <= len(words) <= 6
        and bool(letters)
        and all(c.isupper() for c in letters)
        and not any(c.isdigit() for c in t)  # excludes ALL-CAPS statute citations
    )


def _gold_units_for_page(export, page):
    pkey = str(page)
    fine = [
        a
        for a in export["labelled_text"]
        if a["annotationLabel"] != "Semantic Unit"
        and pkey in (a.get("annotation_json") or {})
    ]
    pw = export["pawls_file_content"][page]["page"]["width"] or 1.0
    ph = export["pawls_file_content"][page]["page"]["height"] or 1.0
    units, cur = [], None
    for order, a in enumerate(fine):
        b = a["annotation_json"][pkey]["bounds"]
        box = [b["left"] / pw, b["top"] / ph, b["right"] / pw, b["bottom"] / ph]
        starts = _starts_unit(a.get("rawText"))
        if starts or cur is None:
            cur = {
                "unit_id": f"g{len(units)}",
                "bbox_frac": list(box),
                "text": a.get("rawText") or "",
                "member_order": [order],
            }
            units.append(cur)
        else:
            cur["member_order"].append(order)
            cur["text"] += " " + (a.get("rawText") or "")
            cur["bbox_frac"] = [
                min(cur["bbox_frac"][0], box[0]),
                min(cur["bbox_frac"][1], box[1]),
                max(cur["bbox_frac"][2], box[2]),
                max(cur["bbox_frac"][3], box[3]),
            ]
    return units


def main():
    golden, seen = {}, {}
    for relpath, page, _slow in C.manifest_pairs():
        if relpath not in seen:
            with contextlib.redirect_stdout(io.StringIO()):
                seen[relpath] = parse_to_opencontracts(
                    os.path.join(REPO, relpath),
                    parse_options={"semantic_units": True},
                )
        export = seen[relpath]
        if page >= export.get("page_count", 0):
            print(f"  SKIP (out of range): {relpath} p{page}")
            continue
        golden[C.page_key(relpath, page)] = {
            "units": _gold_units_for_page(export, page)
        }
    with open(C.GOLDEN, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)
    print("wrote", C.GOLDEN, len(golden), "pages")


if __name__ == "__main__":
    main()
