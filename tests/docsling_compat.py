"""Layout-comparison metrics for the Docsling cross-engine regression suite.

Docling (the ``jscrudato/docsling-local`` microservice) is used as a *structure
oracle*: it produces an ``OpenContractDocExport`` from the same PDF, and we floor
Warp-Ingest's agreement with it so layout regressions fail while improvements
pass — the same shape as ``tests/s1_compat.py`` / ``tests/oc_compat.py``.

Docling is an oracle, **not** ground truth. The cross-engine study (see
``docs/docsling_layout_oracle.md``) found real Docling defects: it drops tables
entirely, under-labels numbered legal headings as ``text``, and over-applies
``list_item`` to statutory note paragraphs. So the metrics here deliberately
floor agreement *with a margin* (don't-regress) rather than demand equality, and
add one hard ceiling — ``overlong_heading_count`` — that locks in the one change
the study justified: demoting run-in "header" blocks that absorbed a whole
section body (a 100+ word "Section Header") to paragraphs.

The Docling side is a committed, slimmed oracle (annotations only; no token
grid) under ``tests/fixtures/docsling_targets/``. Regenerate with
``scripts/build_docsling_fixtures.py`` (needs the microservice running). The test
runs only Warp live, so CI needs no service — exactly like the S-1 suite.
"""

import re
from difflib import SequenceMatcher

# (slug, relative PDF path under tests/fixtures roots, is_slow)
FIXTURE_DOCS = [
    ("cerebras_ex231", "s1/cerebras_s1__exhibit231sx1_d5d6f3.pdf", False),
    ("exyn_ex211", "s1/exyn_s1__ex211_812970.pdf", False),
    ("spacex_ex211", "s1/spacex_s1__exhibit211sx1_3ecd38.pdf", False),
    ("usc_title1", "USC Title 1 - CHAPTER 1.pdf", False),
    ("exyn_ex1010", "s1/exyn_s1__ex1010_de36c8.pdf", False),
    ("cerebras_ex1013e", "s1/cerebras_s1__exhibit1013esx_f124a6.pdf", False),
    ("forbright_ex1012", "s1/forbright_s1__exhibit1012sx1_11ea38.pdf", False),
    ("generate_bio_ex33", "s1/generate_bio_s1__ex33_a3ab0f.pdf", True),
    (
        "eton_dev_agreement",
        "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_"
        "EX-10.1_Development_Agreement_ZrZJLLv.pdf",
        True,
    ),
    ("spacex_ex41", "s1/spacex_s1__exhibit41sx1_331523.pdf", False),
]

# a real section header is short; longer "header" blocks are run-in headings
# (kept in sync with opencontracts_exporter._HEADER_MAX_WORDS)
HEADER_MAX_WORDS = 12

# regression tolerances (live must stay within these of the committed baseline)
_SIM_DROP = 0.02
_AGREE_DROP = 0.02
_HEAD_DROP = 0.03
# head_ancestor_agree is only floored when the oracle baseline is this high: on a
# few docs Docling's own hierarchy is so sparse/defective (e.g. it under-labels
# numbered legal headings) that baseline agreement is ~0, where an additive floor
# is unreachable and would guard nothing. We skip the floor there rather than
# present a vacuous pass; the high-baseline docs still catch a systemic regression.
_HEAD_MIN_BASELINE = 0.10

# --- canonical label vocabulary ----------------------------------------------
_DOCLING_CANON = {
    "title": "TITLE",
    "section_header": "HEADING",
    "paragraph": "BODY",
    "text": "BODY",
    "list_item": "LIST",
    "table": "TABLE",
    "caption": "CAPTION",
    "picture": "PICTURE",
    "chart": "PICTURE",
    "formula": "FORMULA",
    "code": "CODE",
    "footnote": "FOOTNOTE",
    "page_header": "PAGE_HEADER",
    "page_footer": "PAGE_FOOTER",
    "document_index": "TABLE",
    "reference": "BODY",
}
_WARP_CANON = {
    "Section Header": "HEADING",
    "Paragraph": "BODY",
    "List Item": "LIST",
    "Table Row": "TABLE",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _canon(raw, engine):
    if engine == "docling":
        return _DOCLING_CANON.get(str(raw).strip().lower(), "OTHER")
    return _WARP_CANON.get(str(raw).strip(), "OTHER")


def _words(text):
    return _WORD_RE.findall((text or "").lower())


# --- normalization (both engines -> common annotation list) ------------------
def _norm_warp(export):
    anns = []
    for a in export["labelled_text"]:
        page0 = int(a["page"])
        bounds = None
        for k, v in (a["annotation_json"] or {}).items():
            if int(k) == page0:
                bounds = v.get("bounds")
        anns.append(
            {
                "id": str(a["id"]),
                "canon": _canon(a["annotationLabel"], "warp"),
                "raw_label": a["annotationLabel"],
                "page": page0,
                "bounds": bounds,
                "words": _words(a.get("rawText", "")),
                "split_words": len((a.get("rawText", "") or "").split()),
                "parent_id": (
                    str(a["parent_id"]) if a.get("parent_id") is not None else None
                ),
            }
        )
    return {"page_count": export["page_count"], "anns": anns}


def _norm_oracle(oracle):
    anns = []
    for a in oracle["annotations"]:
        anns.append(
            {
                "id": str(a["id"]),
                "canon": _canon(a["label"], "docling"),
                "raw_label": a["label"],
                "page": int(a["page"]),
                "bounds": a.get("bounds"),
                "words": _words(a.get("rawText", "")),
                "parent_id": (
                    str(a["parent_id"]) if a.get("parent_id") is not None else None
                ),
            }
        )
    return {"page_count": oracle["page_count"], "anns": anns}


def _reading_order(anns):
    def key(a):
        b = a["bounds"] or {}
        return (a["page"], round(b.get("top", 0.0), 1), round(b.get("left", 0.0), 1))

    return sorted(anns, key=key)


def _stream(norm):
    out = []
    for a in _reading_order(norm["anns"]):
        for w in a["words"]:
            out.append((w, a["canon"], a["id"]))
    return out


def _heading_ancestor(ann, idmap):
    seen, node = set(), ann
    while node is not None:
        pid = node["parent_id"]
        if pid is None or pid in seen or pid not in idmap:
            return None
        seen.add(pid)
        parent = idmap[pid]
        if parent["canon"] in ("HEADING", "TITLE"):
            return " ".join(parent["words"])
        node = parent
    return None


# --- public API --------------------------------------------------------------
def slim_docling(export):
    """Reduce a full Docling OpenContractDocExport to the committed oracle shape
    (annotations + page boxes; no PAWLS token grid)."""
    pawls = export.get("pawlsFileContent", [])
    pages = [
        {"index": i, "width": p["page"]["width"], "height": p["page"]["height"]}
        for i, p in enumerate(pawls)
    ]
    anns = []
    for a in export["labelledText"]:
        page0 = int(a["page"])
        bounds = None
        for k, v in (a["annotation_json"] or {}).items():
            if int(k) == page0:
                bounds = v.get("bounds")
        anns.append(
            {
                "id": a["id"],
                "label": a["annotationLabel"],
                "page": page0,
                "bounds": bounds,
                "rawText": a.get("rawText", ""),
                "parent_id": a.get("parent_id"),
            }
        )
    return {"page_count": export["pageCount"], "pages": pages, "annotations": anns}


def layout_metrics(warp_export, oracle):
    """Compute Warp-vs-Docling layout-agreement metrics for one document."""
    wn, dn = _norm_warp(warp_export), _norm_oracle(oracle)
    ws, ds = _stream(wn), _stream(dn)
    w_by = {a["id"]: a for a in wn["anns"]}
    d_by = {a["id"]: a for a in dn["anns"]}

    sm = SequenceMatcher(a=[t[0] for t in ds], b=[t[0] for t in ws], autojunk=False)
    same = aligned = head_m = head_t = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            di, wi = i1 + k, j1 + k
            aligned += 1
            dl, wl = ds[di][1], ws[wi][1]
            if dl == wl:
                same += 1
            if dl in ("BODY", "LIST"):
                head_t += 1
                da = _heading_ancestor(d_by[ds[di][2]], d_by)
                wa = _heading_ancestor(w_by[ws[wi][2]], w_by)
                if da == wa or (da and wa and (da in wa or wa in da)):
                    head_m += 1

    overlong = sum(
        1
        for a in wn["anns"]
        if a["canon"] == "HEADING" and a["split_words"] > HEADER_MAX_WORDS
    )
    return {
        "page_count": wn["page_count"],
        "aligned_words": aligned,
        # symmetric difflib similarity 2*M/(len_d+len_w) of the two word streams —
        # a content-preservation proxy, not one-sided recall (hence the name).
        "word_seq_similarity": round(sm.ratio(), 4),
        "label_agree": round(same / aligned, 4) if aligned else 1.0,
        "head_ancestor_agree": round(head_m / head_t, 4) if head_t else 1.0,
        "overlong_heading_count": overlong,
    }


def regressions(live, base):
    """Human-readable regression messages (empty == passing)."""
    out = []
    if live["page_count"] != base["page_count"]:
        out.append(f"page_count {live['page_count']} != baseline {base['page_count']}")
    if live["word_seq_similarity"] < base["word_seq_similarity"] - _SIM_DROP:
        out.append(
            f"word_seq_similarity {live['word_seq_similarity']} < "
            f"{base['word_seq_similarity'] - _SIM_DROP:.4f} "
            f"(baseline {base['word_seq_similarity']})"
        )
    if live["label_agree"] < base["label_agree"] - _AGREE_DROP:
        out.append(
            f"label_agree {live['label_agree']} < "
            f"{base['label_agree'] - _AGREE_DROP:.4f} (baseline {base['label_agree']})"
        )
    if (
        base["head_ancestor_agree"] >= _HEAD_MIN_BASELINE
        and live["head_ancestor_agree"] < base["head_ancestor_agree"] - _HEAD_DROP
    ):
        out.append(
            f"head_ancestor_agree {live['head_ancestor_agree']} < "
            f"{base['head_ancestor_agree'] - _HEAD_DROP:.4f} "
            f"(baseline {base['head_ancestor_agree']})"
        )
    if live["overlong_heading_count"] > base["overlong_heading_count"]:
        out.append(
            f"overlong_heading_count {live['overlong_heading_count']} > "
            f"baseline {base['overlong_heading_count']} "
            "(run-in 'header' blocks absorbing section bodies)"
        )
    return out
