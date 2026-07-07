#!/usr/bin/env python
"""Exhaustive text-faithfulness evaluation for the Warp-Ingest pipeline.

Question answered: **does Warp ever LOSE text?** Structure (labels, hierarchy,
reading order) is out of scope — only content loss is scored.

Oracle: pypdfium2 (PDFium — Chrome's PDF text stack), a text extractor fully
independent of the pdfplumber/pdfminer front-end, plus optionally a blind
pdfplumber flat extract as a second opinion. For every page, each oracle's
token multiset is compared against Warp's exported text surfaces:

  * ``pawls``   — the OpenContracts PAWLS token stream (rebuilt from the
                  front-end XHTML; page-level comparison).
  * ``content`` — the OC export's ``content`` field == the engine's
                  ``block_text`` (the json/html render surface; doc-level
                  comparison, since blocks may move across page boundaries).

Scoring is multiset token recall with a *re-tokenization forgiveness* pass:
an oracle token missing from Warp's token multiset is forgiven when its
whitespace-free form still appears verbatim in Warp's whitespace-free page
character stream (covers different word segmentation, glued/split words and
soft-hyphen joins — the text is present, just tokenized differently). Missing
tokens containing at least one alphanumeric are *material*; punctuation-only
tokens are counted separately as trivial.

Usage::

    uv run python scripts/text_faithfulness_eval.py \
        --out eval_results.jsonl [--workers 3] [--docs substr ...] [--limit N]

Results are checkpointed one JSON line per document; a re-run skips documents
already present in the output file. Summarize with --summarize.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# text normalization
# ---------------------------------------------------------------------------

# characters different extractors render differently but that are the "same"
# text for faithfulness purposes
_CHAR_MAP = {
    0x00AD: None,  # soft hyphen
    0x200B: None,  # zero-width space
    0x200C: None,  # zero-width non-joiner
    0x200D: None,  # zero-width joiner
    0xFEFF: None,  # BOM / zero-width no-break
    0xFFFE: None,  # PDFium end-of-line marker artifacts
    0x2018: ord("'"),
    0x2019: ord("'"),
    0x201C: ord('"'),
    0x201D: ord('"'),
    0x2010: ord("-"),  # hyphen
    0x2011: ord("-"),  # non-breaking hyphen
    0x2012: ord("-"),  # figure dash
    0x2013: ord("-"),  # en dash
    0x2014: ord("-"),  # em dash
    0x2015: ord("-"),  # horizontal bar
    0x00A0: ord(" "),  # nbsp
    0x2022: None,  # bullet (glyph choice varies across extractors)
    0x00B7: None,  # middle dot
    0xF0B7: None,  # private-use bullet (Symbol font)
    0xF0A7: None,  # private-use bullet
    0x25CF: None,  # black circle
    0x25AA: None,  # small black square
    0x25A0: None,  # black square
    0x25E6: None,  # white bullet
    0x2219: None,  # bullet operator
    0x25BC: None,  # down triangle (often decorative)
    0x25B2: None,  # up triangle
}


def norm_text(s: str) -> str:
    """Normalize extractor output for comparison (case/ligature/dash folded)."""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_CHAR_MAP)
    # drop remaining control / format chars except whitespace
    s = "".join(
        c if not unicodedata.category(c).startswith(("Cc", "Cf")) else " " for c in s
    )
    return s.lower()


def toks(s: str) -> list[str]:
    return norm_text(s).split()


_ALNUM = re.compile(r"[a-z0-9]")


def material(token: str) -> bool:
    """A missing token matters if it carries any alphanumeric content."""
    return bool(_ALNUM.search(token))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score(oracle_tokens: list[str], warp_tokens: list[str], warp_streams: list[str]):
    """Multiset recall of oracle tokens in warp tokens, with substring forgiveness.

    ``warp_streams`` are whitespace-free character streams (page-level and/or
    doc-level) used to forgive re-tokenization: a deficit token is forgiven if
    it appears as a contiguous substring beyond what token matches account for.
    """
    want = Counter(oracle_tokens)
    have = Counter(warp_tokens)
    matched = sum(min(c, have.get(t, 0)) for t, c in want.items())
    missing: Counter[str] = Counter()
    forgiven = 0
    for t, c in want.items():
        deficit = c - min(c, have.get(t, 0))
        if not deficit:
            continue
        t_sub = t.strip("-") or t
        extra = 0
        for stream in warp_streams:
            occ = stream.count(t_sub) if t_sub else 0
            extra = max(extra, occ - have.get(t, 0))
            if extra >= deficit:
                break
        f = min(deficit, max(0, extra))
        forgiven += f
        if deficit - f:
            missing[t] += deficit - f
    n = sum(want.values())
    miss_mat = sum(c for t, c in missing.items() if material(t))
    miss_triv = sum(c for t, c in missing.items() if not material(t))
    return {
        "n_oracle": n,
        "matched": matched,
        "forgiven": forgiven,
        "missing_material": miss_mat,
        "missing_trivial": miss_triv,
        "recall_strict": round(matched / n, 6) if n else 1.0,
        "recall_effective": round((matched + forgiven) / n, 6) if n else 1.0,
        "missing_samples": [
            {"token": t, "count": c}
            for t, c in sorted(
                ((t, c) for t, c in missing.items() if material(t)),
                key=lambda x: -x[1],
            )[:25]
        ],
    }


# ---------------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------------


def pdfium_pages(path: str) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        out = []
        for page in pdf:
            tp = page.get_textpage()
            out.append(tp.get_text_bounded() or "")
            tp.close()
            page.close()
        return out
    finally:
        pdf.close()


def plumber_pages(path: str) -> list[str]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


# ---------------------------------------------------------------------------
# per-document evaluation (worker)
# ---------------------------------------------------------------------------


def eval_doc(path: str, oracles: list[str]) -> dict:
    os.environ.setdefault("WARP_FE_WORKERS", "1")  # no nested pools in workers

    from warp_ingest.ingestor.opencontracts_exporter import to_opencontracts_export
    from warp_ingest.ingestor.pdf_ingestor import parse_blocks, parse_pdf

    rec: dict = {"doc": os.path.relpath(path, REPO)}
    try:
        xhtml = parse_pdf(path, {"render_format": "all"})
        blocks, *_ = parse_blocks(xhtml, render_format="all", parse_pages=())
        export = to_opencontracts_export(
            xhtml, blocks, pdf_bytes=Path(path).read_bytes()
        )
    except Exception as e:  # noqa: BLE001 - a crash IS a total-text-loss finding
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    pawls = export["pawls_file_content"]
    warp_page_tokens = [toks(" ".join(t["text"] for t in pg["tokens"])) for pg in pawls]
    warp_page_streams = ["".join(ts) for ts in warp_page_tokens]
    warp_doc_stream = "".join(warp_page_streams)
    content_tokens = toks(export["content"])
    content_stream = "".join(content_tokens)

    rec["pages"] = len(pawls)
    rec["oracles"] = {}
    for name in oracles:
        try:
            pages = pdfium_pages(path) if name == "pdfium" else plumber_pages(path)
        except Exception as e:  # noqa: BLE001
            rec["oracles"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        o_page_tokens = [toks(t) for t in pages]
        o_all = [t for ts in o_page_tokens for t in ts]

        # pawls surface: page-level (fall back to doc stream for cross-page moves)
        page_rows = []
        agg = Counter()
        for i, ots in enumerate(o_page_tokens):
            wts = warp_page_tokens[i] if i < len(warp_page_tokens) else []
            ws = warp_page_streams[i] if i < len(warp_page_streams) else ""
            s = score(ots, wts, [ws, warp_doc_stream])
            for k in (
                "n_oracle",
                "matched",
                "forgiven",
                "missing_material",
                "missing_trivial",
            ):
                agg[k] += s[k]
            if s["missing_material"]:
                page_rows.append({"page": i, **s})
        n = agg["n_oracle"]
        rec["oracles"][name] = {
            "pawls": {
                **dict(agg),
                "recall_strict": round(agg["matched"] / n, 6) if n else 1.0,
                "recall_effective": (
                    round((agg["matched"] + agg["forgiven"]) / n, 6) if n else 1.0
                ),
                "flagged_pages": page_rows[:2000],
            },
            "content": score(o_all, content_tokens, [content_stream]),
        }
    return rec


def _worker(args):
    path, oracles = args
    try:
        return eval_doc(path, oracles)
    except Exception as e:  # noqa: BLE001
        return {
            "doc": os.path.relpath(path, REPO),
            "error": f"worker: {type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def find_pdfs() -> list[str]:
    roots = [REPO / "tests" / "fixtures", REPO / "files"]
    out: list[str] = []
    for r in roots:
        out.extend(str(p) for p in r.rglob("*.pdf"))
    return sorted(set(out))


def page_count(path: str) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def summarize(out_path: Path, flag_threshold: float) -> int:
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} documents evaluated")
    bad = 0
    for surface in ("pawls", "content"):
        for oracle in ("pdfium", "plumber"):
            tot = Counter()
            worst: list[tuple[float, str]] = []
            for r in rows:
                o = (r.get("oracles") or {}).get(oracle)
                if not o or "error" in o:
                    continue
                s = o[surface]
                for k in (
                    "n_oracle",
                    "matched",
                    "forgiven",
                    "missing_material",
                    "missing_trivial",
                ):
                    tot[k] += s[k]
                worst.append((s["recall_effective"], r["doc"]))
            if not tot["n_oracle"]:
                continue
            eff = (tot["matched"] + tot["forgiven"]) / tot["n_oracle"]
            print(
                f"\n== {surface} vs {oracle}: eff-recall {eff:.6f} "
                f"({tot['missing_material']} material / {tot['missing_trivial']} trivial "
                f"missing of {tot['n_oracle']})"
            )
            for rec_, doc in sorted(worst)[:12]:
                mark = " <-- FLAG" if rec_ < flag_threshold else ""
                if mark:
                    bad += 1
                print(f"  {rec_:.6f}  {doc}{mark}")
    errs = [r for r in rows if "error" in r]
    for r in errs:
        print(f"ERROR {r['doc']}: {r['error']}")
    return 1 if (bad or errs) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "eval_results.jsonl"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--oracles", default="pdfium,plumber")
    ap.add_argument("--docs", nargs="*", help="substring filters on doc path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--flag-threshold", type=float, default=0.995)
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.summarize:
        return summarize(out_path, args.flag_threshold)

    oracles = [o for o in args.oracles.split(",") if o]
    pdfs = find_pdfs()
    if args.docs:
        pdfs = [p for p in pdfs if any(d in p for d in args.docs)]
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["doc"])
    pdfs = [p for p in pdfs if os.path.relpath(p, REPO) not in done]
    pdfs.sort(key=page_count, reverse=True)  # big docs first for load balance
    if args.limit:
        pdfs = pdfs[: args.limit]
    print(f"{len(pdfs)} documents to evaluate ({len(done)} already done)")

    os.environ["WARP_FE_WORKERS"] = "1"
    with out_path.open("a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (p, oracles)): p for p in pdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            status = f"ERROR {rec.get('error')}" if "error" in rec else ""
            if not status:
                pw = rec["oracles"].get("pdfium", {}).get("pawls", {})
                status = f"pawls eff {pw.get('recall_effective')}"
            print(f"[{i}/{len(futs)}] {rec['doc']}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
