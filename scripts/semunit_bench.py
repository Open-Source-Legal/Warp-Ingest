"""Semantic-Unit benchmark derived from the 106-page vision audit.

Content-based (id-independent) detectors reproduce the machine-checkable defect
classes the subagents flagged, so the score re-evaluates after any fix:

  clean_unit_fraction = 1 - (units hit by any detector) / total units
  token_coverage      = fraction of page tokens enclosed by some unit
                        (captures dropped-tail / MISSING / xpage-truncation)

Headline "unit accuracy" = clean_unit_fraction (primary) with token_coverage
reported alongside. Detectors: furniture, bare-marker, testimonium-merge,
signature-field-parent, marker-(i)-misnest, staircased-siblings.
"""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/jman/Code/tmp/Warp-Ingest")
sys.path.insert(0, str(REPO))
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

DOCS = [
    "tests/fixtures/s1/exyn_s1__ex1010_de36c8.pdf",
    "tests/fixtures/s1/cerebras_s1__exhibit1013esx_f124a6.pdf",
    "tests/fixtures/EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_EX-10.1_Development_Agreement_ZrZJLLv.pdf",
    "tests/fixtures/USC Title 1 - CHAPTER 1.pdf",
    "tests/fixtures/s1/forbright_s1__exhibit1012sx1_11ea38.pdf",
    "tests/fixtures/s1/generate_bio_s1__ex33_a3ab0f.pdf",
    "tests/fixtures/contracts/fw_nctcog.pdf",
    "tests/fixtures/contracts/fw_garver.pdf",
    "tests/fixtures/contracts/fw_vertigis.pdf",
    "tests/fixtures/contracts/fw_wert_bookbinding.pdf",
]

_FOLIO = re.compile(r"^(page\s+)?[0-9ivxlcdm]+(\s+of\s+[0-9ivxlcdm]+)?$", re.I)
_DOCNUM = re.compile(r"^[0-9]{3,}[-\s]?[a-z0-9]{0,6}$", re.I)
_EMAIL = re.compile(r"^\S+@\S+\.\S+$")
_SIGFIELD = re.compile(
    r"^(by:|name:|title:|attest\b|date:?\b|/s/|signed by|"
    r"assistant city (manager|attorney)\b|city secretary\b|agency\b)",
    re.I,
)


_PLACEHOLDER = re.compile(
    r"^\[?\s*(balance of page|signature page|remainder of (this|the) page)", re.I
)


def is_furniture(text):
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    core = t.strip("-—– ")
    if _FOLIO.match(core):
        return True
    if _DOCNUM.match(re.sub(r"\s+", "", t)):
        return True
    if _EMAIL.match(t):
        return True
    if _PLACEHOLDER.match(t):
        return True
    if "docusign envelope id" in low or "release point" in low:
        return True
    if low in ("fort worth.", "fort worth"):
        return True
    if low.startswith("csc no"):
        return True
    if len(t.split()) <= 8 and re.match(
        r"^(official record|city secretary|ft\.? worth)", low
    ):
        return True
    return False


def is_bare_marker(text):
    t = (text or "").strip()
    return bool(re.match(r"^\(?[a-z0-9]{1,4}[.)]?$", t, re.I)) and len(t.split()) == 1


def is_sigfield(text):
    t = (text or "").strip()
    return bool(_SIGFIELD.match(t)) and len(t.split()) <= 10


def _lead_marker(text):
    """Return ('num', '15') / ('num', '1.11') / ('let', 'i') / ('rom','i') / None."""
    t = (text or "").strip()
    m = re.match(r"^\(?([0-9]+(?:\.[0-9]+)*)[.)]?\s", t)
    if m:
        return ("num", m.group(1))
    m = re.match(r"^\(([a-hj-z])\)\s", t)  # single letter (exclude i handled below)
    if m:
        return ("let", m.group(1))
    m = re.match(r"^\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s", t)
    if m:
        return ("rom", m.group(1))
    m = re.match(r"^\(i\)\s", t)
    if m:
        return ("let_or_rom_i", "i")
    return None


def analyze(export):
    us = [a for a in export["labelled_text"] if a["annotationLabel"] == "Semantic Unit"]
    by_id = {u["id"]: u for u in us}
    # hierarchy from OC_PARENT_CHILD relationships among units, not parent_id
    parent_of = {}
    for r in export.get("relationships", []):
        if r.get("relationshipLabel") == "OC_PARENT_CHILD":
            src = r["source_annotation_ids"][0]
            if isinstance(src, str) and src.startswith("su-"):
                for tgt in r["target_annotation_ids"]:
                    parent_of[tgt] = src
    has_children = set(parent_of.values())
    defects = {}  # unit_id -> set(classes)

    def flag(uid, cls):
        defects.setdefault(uid, set()).add(cls)

    for u in us:
        t = u["rawText"] or ""
        has_kids = u["id"] in has_children
        if is_furniture(t):
            flag(u["id"], "furniture")
        if is_bare_marker(t):
            flag(u["id"], "bare_marker")
        if re.match(r"^\d+\.\s", t) and re.search(
            r"\bIN WITNESS WHEREOF\b|\bAS WITNESS\b", t
        ):
            flag(u["id"], "testimonium_merge")
        if has_kids and is_sigfield(t):
            flag(u["id"], "sigfield_parent")
        # marker (i) nested under (h)
        lm = _lead_marker(t)
        pid = parent_of.get(u["id"])
        if lm and lm[0] == "let_or_rom_i" and pid in by_id:
            plm = _lead_marker(by_id[pid]["rawText"] or "")
            if plm and plm == ("let", "h"):
                flag(u["id"], "marker_i_misnest")
        # staircase: numbered section nested under its immediate predecessor
        if lm and lm[0] == "num" and pid in by_id:
            plm = _lead_marker(by_id[pid]["rawText"] or "")
            if plm and plm[0] == "num" and _is_predecessor(plm[1], lm[1]):
                flag(u["id"], "staircase")

    # token coverage per page
    tot_tokens = covered = 0
    for p in export.get("pawls_file_content", []):
        tot_tokens += len(p["tokens"])
    covered_ids = set()
    for r in export.get("relationships", []):
        if r.get("relationshipLabel") == "OC_SEMANTIC_UNIT":
            for m in r["target_annotation_ids"]:
                covered_ids.add(m)
    fine = {a["id"]: a for a in export["labelled_text"]}
    for mid in covered_ids:
        for pj in (fine.get(mid, {}).get("annotation_json") or {}).values():
            covered += len(pj.get("tokensJsons") or [])
    return us, defects, covered, tot_tokens


def _is_predecessor(a, b):
    """True if section number a is the immediate predecessor sibling of b."""
    if "." in a or "." in b:
        pa, pb = a.split("."), b.split(".")
        if len(pa) != len(pb) or pa[:-1] != pb[:-1]:
            return False
        try:
            return int(pb[-1]) - int(pa[-1]) == 1
        except ValueError:
            return False
    try:
        return int(b) - int(a) == 1
    except ValueError:
        return False


def main():
    per_doc = []
    total_units = total_defective = 0
    all_tok = all_cov = 0
    cls_counts = {}
    for rp in DOCS:
        with contextlib.redirect_stdout(io.StringIO()):
            e = parse_to_opencontracts(
                str(REPO / rp), parse_options={"semantic_units": True}
            )
        us, defects, cov, tot = analyze(e)
        nd = len(defects)
        for cs in defects.values():
            for c in cs:
                cls_counts[c] = cls_counts.get(c, 0) + 1
        total_units += len(us)
        total_defective += nd
        all_tok += tot
        all_cov += cov
        per_doc.append(
            {
                "doc": Path(rp).stem[:34],
                "units": len(us),
                "defective": nd,
                "clean_frac": round(1 - nd / len(us), 3) if us else 1.0,
                "tok_cov": round(cov / tot, 3) if tot else 1.0,
            }
        )
    clean_frac = round(1 - total_defective / total_units, 4)
    tok_cov = round(all_cov / all_tok, 4)
    print(f"{'doc':<36}{'units':>6}{'defect':>7}{'clean':>7}{'tokcov':>8}")
    for d in per_doc:
        print(
            f"{d['doc']:<36}{d['units']:>6}{d['defective']:>7}"
            f"{d['clean_frac']:>7}{d['tok_cov']:>8}"
        )
    print("-" * 64)
    print(
        f"{'TOTAL':<36}{total_units:>6}{total_defective:>7}"
        f"{clean_frac:>7}{tok_cov:>8}"
    )
    print(f"\nCLEAN_UNIT_FRACTION = {clean_frac}  TOKEN_COVERAGE = {tok_cov}")
    print("defect classes:", dict(sorted(cls_counts.items(), key=lambda x: -x[1])))
    json.dump(
        {
            "clean_unit_fraction": clean_frac,
            "token_coverage": tok_cov,
            "total_units": total_units,
            "defective_units": total_defective,
            "class_counts": cls_counts,
            "per_doc": per_doc,
        },
        open(REPO / "audit_out" / "semunit_audit" / "bench_results.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
