#!/usr/bin/env python
"""Build the legal-100 structural-eval manifest (+ copy new FortWorth fixtures).

The legal batch reuses the already-committed EDGAR S-1 fixtures
(``tests/fixtures/s1/*.pdf``) and FortWorth municipal contracts
(``tests/fixtures/contracts/fw_*.pdf``) by ``(relpath, page)`` reference, and
adds a handful of small new FortWorth contracts (copied into
``tests/fixtures/legal100/``) for vendor/type diversity. ~70 EDGAR pages + ~30
FortWorth pages, deterministically selected.

Unlike the heterogeneous batch (whose golden is programmatic), the legal golden
has no external truth and is produced separately by the vision-Workflow
adjudicator; this script only fixes the **page manifest** the golden is built
against.

    python scripts/build_legal100_fixtures.py

Writes ``tests/fixtures/legal100_manifest.json``: a list of
``{relpath, page_index, doc_class, source, is_slow}`` rows.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pypdfium2 as pdfium

REPO = Path(__file__).resolve().parent.parent
S1_DIR = REPO / "tests" / "fixtures" / "s1"
CONTRACTS_DIR = REPO / "tests" / "fixtures" / "contracts"
LEGAL_DIR = REPO / "tests" / "fixtures" / "legal100"
FW_SRC = Path("/home/jman/Code/FortWorthCrawler/fw_output/documents/Contracts")
OUT_MANIFEST = REPO / "tests" / "fixtures" / "legal100_manifest.json"

_SLOW_PAGES = 60  # mirror oc_batch: docs longer than this run only under --runslow

# New FortWorth contracts to copy in (chosen small + diverse vendor/type).
# (source basename stem fragment, destination fixture name, doc_class)
FW_NEW = [
    ("058017-R3", "fw_mentalix.pdf", "municipal_contract"),
    ("064028-R1", "fw_knight_office.pdf", "municipal_contract"),
    ("063013-R1", "fw_southern_computer.pdf", "municipal_contract"),
    ("061008-R2", "fw_coast_biomedical.pdf", "municipal_contract"),
    ("060001-A1", "fw_komatsu_rangel.pdf", "municipal_construction"),
    ("060005-R2", "fw_granicus.pdf", "municipal_contract"),
]


def _classify_s1(name):
    n = name.lower()
    if re.search(r"ex_?231|_ex231|exhibit231|consent", n):
        return "consent"
    if re.search(r"ex_?211|exhibit211|subsidiar", n):
        return "subsidiaries"
    if re.search(r"exhibit3|_ex3|charter|bylaw|articles|certificate", n):
        return "charter"
    if re.search(r"exhibit4|_ex4|warrant", n):
        return "securities"
    if re.search(r"filingfee|exfiling", n):
        return "fee"
    if re.search(r"exhibit1[01]|_ex10|_ex11|agreement", n):
        return "agreement"
    if re.search(r"_s1_|drsa|primary|iron|_sx1$", n):
        return "body"
    if re.search(r"exhibit|_ex", n):
        return "agreement"
    return "body"


def _page_count(path):
    try:
        d = pdfium.PdfDocument(str(path))
        n = len(d)
        d.close()
        return n
    except Exception:
        return 0


def _pick_pages(n, doc_class):
    """Representative 0-based pages for a doc of n pages."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n <= 4:
        return list(range(min(n, 3)))  # short scanned contract: first 3 pages
    if n <= 12:
        return sorted({0, n // 2, n - 1})
    if n <= _SLOW_PAGES:
        return sorted({0, n // 3, (2 * n) // 3})
    # large body / agreement: cover/TOC, summary, mid, deep
    return sorted({3, n // 4, n // 2, (3 * n) // 4})


def _select_edgar(slow_cap=32):
    """Deterministic EDGAR rows: all small docs + slow docs capped at slow_cap."""
    docs = []
    for p in sorted(S1_DIR.glob("*.pdf")):
        n = _page_count(p)
        docs.append((_classify_s1(p.name), n, p))

    # per-class caps on the near-identical 1-page exhibits so they don't flood
    small_caps = {"consent": 3, "subsidiaries": 3}
    small_counts = {c: 0 for c in small_caps}

    def row(c, n, p, pg):
        return {
            "relpath": os.path.relpath(p, REPO),
            "page_index": pg,
            "doc_class": f"edgar_{c}",
            "source": "edgar_s1",
            "is_slow": n > _SLOW_PAGES,
        }

    rows = []
    # 1) all small/medium (non-slow) docs, exhaustively (capped exhibits)
    for c, n, p in sorted(docs, key=lambda d: (d[0], d[2].name)):
        if n <= 0 or n > _SLOW_PAGES:
            continue
        if c in small_caps:
            if small_counts[c] >= small_caps[c]:
                continue
            small_counts[c] += 1
        for pg in _pick_pages(n, c):
            rows.append(row(c, n, p, pg))

    # 2) large (slow) docs round-robin across class, capped at slow_cap pages
    order = ["body", "agreement", "charter", "securities"]
    pools = {
        c: sorted(
            [(n, p) for cc, n, p in docs if cc == c and n > _SLOW_PAGES],
            key=lambda np: (-np[0], np[1].name),
        )
        for c in order
    }
    slow = []
    progress = True
    while len(slow) < slow_cap and progress:
        progress = False
        for c in order:
            if not pools[c]:
                continue
            n, p = pools[c].pop(0)
            for pg in _pick_pages(n, c):
                slow.append(row(c, n, p, pg))
            progress = True
            if len(slow) >= slow_cap:
                break
    return rows + slow[:slow_cap]


def _copy_new_fw():
    LEGAL_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for frag, dest_name, doc_class in FW_NEW:
        matches = sorted(FW_SRC.glob(f"**/{frag}*.pdf"))
        if not matches:
            print(f"  !! no FW source for {frag}", file=sys.stderr)
            continue
        dst = LEGAL_DIR / dest_name
        shutil.copyfile(matches[0], dst)
        copied.append((dst, doc_class))
    return copied


def _select_fortworth():
    rows = []
    # the 4 already-committed FW contracts (structural representatives)
    committed = [
        ("fw_wert_bookbinding.pdf", "municipal_contract"),
        ("fw_vertigis.pdf", "municipal_contract"),
        ("fw_nctcog.pdf", "municipal_contract"),
        ("fw_garver.pdf", "municipal_construction"),
    ]
    for name, doc_class in committed:
        p = CONTRACTS_DIR / name
        if not p.exists():
            continue
        n = _page_count(p)
        for pg in _pick_pages(n, doc_class):
            rows.append(
                {
                    "relpath": os.path.relpath(p, REPO),
                    "page_index": pg,
                    "doc_class": doc_class,
                    "source": "fortworth",
                    "is_slow": False,
                }
            )
    for dst, doc_class in _copy_new_fw():
        n = _page_count(dst)
        for pg in _pick_pages(n, doc_class):
            rows.append(
                {
                    "relpath": os.path.relpath(dst, REPO),
                    "page_index": pg,
                    "doc_class": doc_class,
                    "source": "fortworth",
                    "is_slow": False,
                }
            )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slow-cap", type=int, default=32)
    args = ap.parse_args(argv)

    edgar = _select_edgar(args.slow_cap)
    fortworth = _select_fortworth()
    manifest = edgar + fortworth

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1)

    from collections import Counter

    print(
        f"legal-100 manifest: {len(manifest)} pages "
        f"({len(edgar)} edgar + {len(fortworth)} fortworth)"
    )
    print("by doc_class:", dict(Counter(r["doc_class"] for r in manifest)))
    print("slow pages:", sum(1 for r in manifest if r["is_slow"]))
    print(f"wrote {OUT_MANIFEST.relative_to(REPO)}")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
