"""Manifest + deterministic per-page metrics for the 60-page structural batch.

This is the *second* OpenContractDocExport regression family (alongside
``oc_compat.py``). Where ``oc_compat`` floors a few whole-document metrics over 3
fixtures, this one floors **per-page structural-quality** signals over a diverse
batch of 60 pages chosen by the structural-quality audit
(``docs/runbooks/structural-quality-audit.md``): parent/child rationality, bbox
tightness, and label correctness.

Everything here is **deterministic** — the LLM audit only *discovers* defects; the
committed test gates these computed numbers (counts ceiled = must not grow,
fractions/tightness floored = must not drop, ``relationship_validity`` must stay
true, ``page_count`` exact).

The manifest is shared by the audit driver (``scripts/oc_audit60.py``) and the
baseline builder (``scripts/build_oc_batch_fixtures.py``). All paths are
repo-relative so the suite is reproducible on a clean checkout; the FortWorth
municipal contracts were copied into ``tests/fixtures/contracts/`` for this.
"""

# (relpath, [0-based pages], is_slow, doc_class, note). is_slow ~ page_count > 60
# (large bodies run only under --runslow), mirroring the other suites.
BATCH_PAGES = [
    # ---- FortWorth municipal contracts (NEW committed source) ----
    (
        "tests/fixtures/contracts/fw_wert_bookbinding.pdf",
        [0, 1],
        False,
        "municipal_contract",
        "M&C cover + signature block (sparse / part-scanned)",
    ),
    (
        "tests/fixtures/contracts/fw_vertigis.pdf",
        [0, 2, 4],
        False,
        "municipal_contract",
        "general services contract; defined terms + signature",
    ),
    (
        "tests/fixtures/contracts/fw_nctcog.pdf",
        [0, 2, 4],
        False,
        "municipal_contract",
        "interlocal agreement; recitals + numbered clauses",
    ),
    (
        "tests/fixtures/contracts/fw_garver.pdf",
        [0, 3, 5],
        False,
        "municipal_contract",
        "construction-related contract; scope/exhibits",
    ),
    # ---- S-1 prospectus bodies (deep pages: TOC, summary, risk, MD&A, business) ----
    (
        "tests/fixtures/s1/yesway_s1__s1_558e28.pdf",
        [3, 60, 160, 280],
        True,
        "s1_body",
        "prospectus body; cover/TOC, summary, risk factors, deep body",
    ),
    (
        "tests/fixtures/s1/solv_energy_s1__s1_becbd8.pdf",
        [4, 90, 200],
        True,
        "s1_body",
        "prospectus body; summary, MD&A tables, business",
    ),
    (
        "tests/fixtures/s1/xenergy_s1__drsa_1ae6c6.pdf",
        [7, 150, 300],
        True,
        "s1_body",
        "draft S-1 body; deep numbering + tables",
    ),
    (
        "tests/fixtures/s1/pershing_square_s1__s1_f498b3.pdf",
        [5, 120, 260],
        True,
        "s1_body",
        "prospectus body; summary + financial tables",
    ),
    (
        "tests/fixtures/s1/liftoff_s1__iron20260113_231d60.pdf",
        [5, 140, 240],
        True,
        "s1_body",
        "prospectus body; risk factors + MD&A",
    ),
    (
        "tests/fixtures/s1/infleqtion_s1__s1_60ea51.pdf",
        [6, 130],
        True,
        "s1_body",
        "prospectus body; summary + business",
    ),
    # ---- long S-1 exhibit agreements (defined terms, run-in headings, tables) ----
    (
        "tests/fixtures/s1/cerebras_s1__exhibit1018sx1_56fdc8.pdf",
        [5, 150, 260],
        True,
        "s1_exhibit",
        "long exhibit agreement; defined terms + schedules",
    ),
    (
        "tests/fixtures/s1/fervo_s1__exhibit1016sx1_9df974.pdf",
        [4, 180],
        True,
        "s1_exhibit",
        "long exhibit agreement body",
    ),
    (
        "tests/fixtures/s1/generate_bio_s1__ex1015_68a8fd.pdf",
        [5, 120],
        True,
        "s1_exhibit",
        "exhibit agreement; numbered sections + signatures",
    ),
    (
        "tests/fixtures/s1/neutron_lime_s1__exhibit45sx1_4208ac.pdf",
        [6, 200],
        True,
        "s1_exhibit",
        "legal-opinion / exhibit; centered headings",
    ),
    # ---- medium S-1 exhibits ----
    (
        "tests/fixtures/s1/kardigan_s1__ex1017_6628ff.pdf",
        [4, 90],
        True,
        "s1_exhibit",
        "exhibit agreement; defined terms",
    ),
    (
        "tests/fixtures/s1/hawkeye360_s1__exhibit21sx1_fa98cc.pdf",
        [6, 60, 100],
        True,
        "s1_exhibit",
        "exhibit agreement; clauses + signature",
    ),
    (
        "tests/fixtures/s1/parabilis_s1__ex1011_424531.pdf",
        [3, 55],
        True,
        "s1_exhibit",
        "exhibit agreement body",
    ),
    (
        "tests/fixtures/s1/eikon_s1__ex1011_d4f7e0.pdf",
        [3, 40],
        True,
        "s1_exhibit",
        "exhibit agreement body",
    ),
    (
        "tests/fixtures/s1/exyn_s1__ex11_a8500b.pdf",
        [2, 30],
        False,
        "s1_exhibit",
        "exhibit; numbered clauses",
    ),
    (
        "tests/fixtures/s1/pershing_square_s1__ex109_9133b3.pdf",
        [4, 120],
        True,
        "s1_exhibit",
        "exhibit agreement; tables + clauses",
    ),
    # ---- small / single-page structural exhibits (consents, articles, subsidiaries) ----
    (
        "tests/fixtures/s1/quantinuum_s1__exhibit33sx1_eefad8.pdf",
        [0, 14],
        False,
        "s1_charter",
        "certificate/articles; ARTICLE headings",
    ),
    (
        "tests/fixtures/s1/generate_bio_s1__ex33_a3ab0f.pdf",
        [0, 12],
        False,
        "s1_charter",
        "certificate of incorporation; sections",
    ),
    (
        "tests/fixtures/s1/forbright_s1__exhibit1012sx1_11ea38.pdf",
        [0, 8],
        False,
        "s1_exhibit",
        "short exhibit agreement",
    ),
    (
        "tests/fixtures/s1/eikon_s1__ex231_6fa7f0.pdf",
        [0],
        False,
        "s1_consent",
        "auditor consent (1 page)",
    ),
    (
        "tests/fixtures/s1/exyn_s1__ex211_812970.pdf",
        [0],
        False,
        "s1_subsidiaries",
        "subsidiaries list (1 page)",
    ),
    (
        "tests/fixtures/s1/quantinuum_s1__exhibit997sx1_bead0b.pdf",
        [0],
        False,
        "s1_misc",
        "1-page exhibit",
    ),
]


def manifest_pairs():
    """Flatten the manifest to ``(relpath, page, is_slow, doc_class, note)`` rows."""
    out = []
    for relpath, pages, is_slow, doc_class, note in BATCH_PAGES:
        for p in pages:
            out.append((relpath, p, is_slow, doc_class, note))
    return out


def page_key(relpath, page):
    """Baseline-JSON key for one audited page."""
    return f"{relpath}::p{page}"


# --------------------------------------------------------------------------- #
# deterministic per-page metrics
# --------------------------------------------------------------------------- #
_HEADER_LABELS = frozenset({"Section Header", "Title"})


def _anns_on_page(export, page):
    pkey = str(page)
    return [
        a
        for a in export.get("labelled_text", [])
        if pkey in (a.get("annotation_json") or {})
    ]


def _children_map(export):
    ids = {a["id"] for a in export.get("labelled_text", [])}
    kids = {}
    for a in export.get("labelled_text", []):
        pid = a.get("parent_id")
        if pid is not None and pid in ids and pid != a["id"]:
            kids.setdefault(pid, []).append(a["id"])
    return kids


def _label_of(export):
    return {a["id"]: a["annotationLabel"] for a in export.get("labelled_text", [])}


def _ann_page_tokens(a, page):
    return (a.get("annotation_json") or {}).get(str(page), {}).get(
        "tokensJsons", []
    ) or []


def page_metrics(export, page):
    """Deterministic structural-quality metrics for one (export, page)."""
    pawls = export.get("pawls_file_content", [])
    page_tokens = pawls[page]["tokens"] if 0 <= page < len(pawls) else []
    total = len(page_tokens)

    anns = _anns_on_page(export, page)
    n = len(anns)
    kids = _children_map(export)
    label = _label_of(export)

    # anchored: annotations on this page that have >= 1 token on the page
    anchored = sum(1 for a in anns if _ann_page_tokens(a, page))

    # token coverage on this page
    claimed = set()
    for a in anns:
        for r in _ann_page_tokens(a, page):
            if r["pageIndex"] == page:
                claimed.add(r["tokenIndex"])
    token_coverage = len(claimed) / total if total else 1.0

    # parent/child rationality smells, scoped to this page's annotations
    def _bad_parent(aid):
        if label.get(aid) in _HEADER_LABELS:
            return False
        ch = kids.get(aid) or []
        if not ch:
            return False
        # a lead-in paragraph whose children are all List Items is intended
        if label.get(aid) == "Paragraph" and all(
            label.get(c) == "List Item" for c in ch
        ):
            return False
        return True

    childbearing_non_headers = sum(1 for a in anns if _bad_parent(a["id"]))
    non_heading_roots = sum(
        1
        for a in anns
        if a.get("parent_id") is None and label.get(a["id"]) not in _HEADER_LABELS
    )
    overlong_headings = sum(
        1
        for a in anns
        if a["annotationLabel"] == "Section Header"
        and len((a.get("rawText") or "").split()) > 12
    )
    untokened = sum(1 for a in anns if not _ann_page_tokens(a, page))

    # bbox tightness: sum(token area)/bbox area per tokened annotation
    tights = []
    for a in anns:
        refs = _ann_page_tokens(a, page)
        if not refs:
            continue
        b = a["annotation_json"][str(page)]["bounds"]
        box_area = max(1e-6, (b["right"] - b["left"]) * (b["bottom"] - b["top"]))
        tok_area = 0.0
        for r in refs:
            if r["pageIndex"] != page:
                continue
            ti = r["tokenIndex"]
            if 0 <= ti < total:
                t = page_tokens[ti]
                tok_area += max(0.0, t["width"]) * max(0.0, t["height"])
        tights.append(min(1.0, tok_area / box_area))
    mean_tightness = sum(tights) / len(tights) if tights else 1.0
    loose_boxes = sum(1 for t in tights if t < 0.5)

    return {
        "n_annotations": n,
        "anchored_fraction": round(anchored / n, 4) if n else 1.0,
        "token_coverage": round(token_coverage, 4),
        "childbearing_non_headers": childbearing_non_headers,
        "non_heading_roots": non_heading_roots,
        "overlong_headings": overlong_headings,
        "untokened": untokened,
        "mean_tightness": round(mean_tightness, 4),
        "loose_boxes": loose_boxes,
    }


# tolerances: live must stay within these of the committed baseline
_FRACTION_DROP = 0.02  # anchored / coverage / tightness may dip at most 2pt


def page_regressions(live: dict, base: dict) -> list:
    """Human-readable regression messages for one page (empty == passing)."""
    out = []
    # counts are ceiled: deterministic, so must not grow
    for key in (
        "childbearing_non_headers",
        "non_heading_roots",
        "overlong_headings",
        "untokened",
        "loose_boxes",
    ):
        if live[key] > base[key]:
            out.append(f"{key} {live[key]} > baseline {base[key]}")
    # fractions/tightness are floored
    for key in ("anchored_fraction", "token_coverage", "mean_tightness"):
        if live[key] < base[key] - _FRACTION_DROP:
            out.append(
                f"{key} {live[key]} < {base[key] - _FRACTION_DROP:.4f} "
                f"(baseline {base[key]})"
            )
    return out
