"""Additive Semantic-Unit grouping layer.

Reads the finalized OpenContracts export and materializes a coarser, NESTED
"Semantic Unit" layer over the fine structural annotations. Purely additive:
existing annotations, their parent_id tree, and existing relationships are never
mutated. Each unit carries the full concatenated member text in ``rawText`` so a
downstream classifier can label a unit directly.

Algorithm (B-prime, deterministic):

1. **Coarsen** — group each fine annotation under its nearest heading-ancestor;
   a heading is a unit-root, a headingless block is its own singleton.
2. **Guard** — demote folio furniture ("Page 2 of 9") and prose-sentence
   pseudo-headings from being unit-roots (their content then climbs to the real
   heading above), using text-intrinsic tells only (never document vocabulary).
3. **Numbering split** — within a unit, split a run of enumerated clauses
   (``1.`` / ``(a)`` / ``(i)`` …) into nested per-clause sub-units by enumerator
   depth, so each discrete clause is its own addressable unit.

Unit ``annotation_json`` bounds are the union of a unit's *direct* members only
(not its whole subtree), so a parent clause's box is its lead-in line and each
leaf clause's box is exactly its own text — the geometry a classifier/retriever
wants, and what aligns to a per-clause golden. See
docs/superpowers/specs/2026-07-01-semantic-unit-grouping-design.md.
"""

import re

from warp_ingest.ingestor.line_parser import Line

_UNIT_LABEL = "Semantic Unit"
_HEADING_LABELS = frozenset({"Section Header", "Title"})
_MEMBER_REL = "OC_SEMANTIC_UNIT"
_PARENT_CHILD_REL = "OC_PARENT_CHILD"

# A page folio / running number: "Page 2 of 9", "12", "iii", "- 33 -". Whole
# text must be furniture — never fires on a titled heading ("ARTICLE I").
_FOLIO_RE = re.compile(r"^(?:page\s+)?[0-9ivxlcdm]+(?:\s+of\s+[0-9ivxlcdm]+)?$", re.I)
# A bare filing / contract-number stamp: "55011-R5", "52004-A6R7", "58012-A2".
_DOCNUM_RE = re.compile(r"^[0-9]{3,}[-\s]?[a-z0-9]{0,6}$", re.I)
_EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")
# A signature/execution field label, never a clause heading.
_SIGFIELD_RE = re.compile(
    r"^(by:|by\.|name:|title:|date:?\b|/s/|(docu)?signed by|witness(eth)?\s|"
    r"address( for notices)?:|attn:|email:|its:)",
    re.I,
)
# A bare form-label line of a label-below-value grid: the field word alone,
# optionally with a parenthetical instruction ("Name (please print)",
# "Title") — universal blank-form format.  Kept short (<=4 words) so a long
# caption starting with the word ("Signature(s) NOTICE: THE ...") never
# reads as a field.
_BARE_FORMLABEL_RE = re.compile(r"^(name|title|signature)s?\s*(\(|$)", re.I)
# Hard clause terminators: an execution/testimonium onset opens a fresh top-level
# unit, never extends the preceding clause.  "/s/" conformed-signature lines and
# universal letter closings ("Sincerely,") likewise open their own unit (the
# signature of a consent letter is not part of the consent paragraph).
_HARD_BOUNDARY_RE = re.compile(
    r"^(in witness whereof|as witness|executed and effective|/s/\s|"
    r"the undersigned (approves|agrees|acknowledges)|"
    r"sincerely[,.]?$|very truly yours|respectfully( submitted)?[,.]?$)",
    re.I,
)
# Front-matter boundaries: universal contract-boilerplate openers that start a
# fresh top-level unit even when unnumbered — a recital ("WHEREAS, ..."), the
# operative lead-in ("NOW, THEREFORE ..."), or an all-caps preamble sentence
# ("THIS EXCLUSIVE LICENSE AND ... AGREEMENT (this ...").  The same stance as
# the exporter's recital_triggers: canonical boilerplate format, not a corpus's
# vocabulary.  A recital under an explicit RECITALS/WITNESSETH heading stays in
# that heading's unit (the heading already delimits the clause).
_FRONT_BOUNDARY_RE = re.compile(r"^(whereas\b|now,?\s+therefore\b)", re.I)
_PREAMBLE_RE = re.compile(r"^THIS\s+[A-Z][A-Z0-9 ,\-]{8,}")
# An uppercase ARTICLE/SECTION heading rendered as a body line ("SECTION 1.
# ANNUAL MEETING.") — a real outline level even when the engine typed it as a
# paragraph.  Case-sensitive: prose references ("Section 409A of the Code")
# are mixed-case and never match.
_CAPS_SECTION_RE = re.compile(r"^(ARTICLE|SECTION)\s+([IVXLC]+|[0-9]+)\b")
# A table-of-contents row: an enumerated entry whose last token is a bare page
# number ("2.9 Resignation 6") — navigation, not a clause.  One trailing word
# is tolerated (a wrapped title whose page number landed mid-text: "District
# of Columbia Code and 201 Supplements").
_TOC_ROW_TAIL_RE = re.compile(r"\s[0-9]{1,4}(\s\S{1,20})?$")
# A row-label that opens a definition/field row ("Security Deposit: ...",
# "Permitted Use: ..."): a short, digit-free lead ending with a colon.  A data
# row like "Months 3-12: $ 26.00" has digits in the lead and stays a
# continuation of the open field.
_ROW_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z '’\-]{0,40}:\s")
# A statute-style row enumerator ("30a R.S., §908 113", "54c Mar. 2, 1929 ...").
_ROW_ENUM_RE = re.compile(r"^\(?[0-9]+[a-z]?[.)]?\s")
# Date/citation format inside a row ("Mar. 2, 1929," / "§908"): a run of such
# rows is a data/disposition table whose rows are parallel entries — never a
# table of contents (whose entries are prose titles + a page number).
_CITATION_ROW_RE = re.compile(r"§|\d,")
# A notes/footnotes heading — its unit is always top-level (never nested under
# the table or section it annotates).
_NOTES_RE = re.compile(r"^notes?\s*[:.]?$", re.I)
# A definition paragraph: a quoted defined term followed by "shall mean"/
# "means" — the universal legal definition format.  Each opens its own unit
# even when its outline number was lost to the block segmentation.
_DEF_START_RE = re.compile(r"^[\"“][A-Z][^\"”]{1,80}[\"”]\s+(shall\s+mean|means\b)")
# A parent heading a child heading may nest under: an ARTICLE-style container.
_CONTAINER_HEADING_RE = re.compile(r"^ARTICLE\b")
# A signature-region role seed: a label line whose lead is a run of
# capitalized words ending with ":" ("VENDOR:", "ACCEPTED AND AGREED:",
# "Contract Compliance Manager:", "LANDLORD: Velocity ...").  Field labels
# (By:/Name:/Title:/...) are members, never seeds.
_ROLE_SEED_RE = re.compile(
    r"^([A-Z][A-Za-z.,&/'’\-]*(?:\s+[A-Za-z.,&/'’\-]+){0,6}):(\s|$)"
)
# An e-signature stamp line ("Dana Burghdoff (Sep 15, 2025 11:56:00 CDT)").
_ESIG_RE = re.compile(r"\([A-Za-z]{3}\.?\s+\d{1,2},?\s*\d{2,4}|[CPME][SD]T\)")
# A run-in clause heading whose enumerator was lost to the segmentation:
# a short TitleCase lead phrase ending in ". " followed by a new sentence
# ("Improvement. If Eikon ...", "Progress Reports on Clinical Development
# Plan. Eikon shall ..." ) — the classic bold run-in heading format.
_RUNIN_RE = re.compile(
    r"^([A-Z][A-Za-z/'’\-]*"
    r"(?:\s+(?:[A-Z][A-Za-z/'’\-]*|[–—-]|of|on|and|the|to|in|for|with|or|a|an)){0,7})"
    r"\.\s+[\"“]?[A-Z]"
)


def _is_runin(text):
    """A run-in clause lead ("Improvement. If Eikon ...").  The lead's final
    word needs >=3 letters so an honorific/abbreviation period ("Dear Mr.
    Upreti:", "Amendment No. 2 ...") never reads as the clause boundary."""
    m = _RUNIN_RE.match((text or "").strip())
    if not m:
        return False
    last = re.sub(r"[^A-Za-z]", "", m.group(1).split()[-1])
    return len(last) >= 3


# Placeholder / page-transition furniture lines (never a clause).
_PLACEHOLDER_RE = re.compile(
    r"^\[?\s*(balance of page|signature page|remainder of (this|the) page)", re.I
)
# A form question row: a question mark followed by Yes/No checkbox tokens
# ("M&C Approved by the Council? * Yes No If so, ..."). Universal checkbox
# format — each question row is its own semantic field.
_FORM_QUESTION_RE = re.compile(r"\?\s*\*?\s*[D0O\[\]_ ]{0,3}Yes\b.{0,20}\bNo\b")


def _is_folio(text):
    t = (text or "").strip().strip("-—–").strip()
    return bool(t) and bool(_FOLIO_RE.match(t))


def _is_signature_field(text):
    t = (text or "").strip()
    if _BARE_FORMLABEL_RE.match(t) and len(t.split()) <= 4:
        return True
    return bool(_SIGFIELD_RE.match(t)) and len(t.split()) <= 10


def _is_bare_marker(text):
    t = (text or "").strip()
    return len(t.split()) == 1 and bool(re.match(r"^\(?[a-z0-9]{1,4}[.)]?$", t, re.I))


def _is_furniture(text):
    """A block that is page furniture / a filing stamp — never a semantic unit.

    Text-intrinsic and structural: page folios, corner filing/clerk stamps, bare
    doc-number stamps, running header/footer banners, version/processing marks,
    logo captions, bare emails. The whole (short) text must be furniture, so a
    clause that merely mentions "City Secretary" is never caught.
    """
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    core = t.strip("-—– ")
    nospace = re.sub(r"\s+", "", t)  # "55011- R5" -> "55011-R5"
    if _FOLIO_RE.match(core) or _DOCNUM_RE.match(nospace) or _EMAIL_RE.match(t):
        return True
    if _PLACEHOLDER_RE.match(t):
        return True
    if "docusign envelope id" in low:  # e-sign platform artifact (format)
        return True
    # a fully-parenthesized short line ("(Release Point 117-285)",
    # "(Signatures continue on the following page)") is a parenthetical
    # aside/banner, never a clause — format-only: the parens ARE the signal.
    # >=2 words inside, so a lone parenthesized table cell ("(State)",
    # "(MW)") stays table content.
    if 2 <= len(t.split()) <= 6 and re.fullmatch(r"\([^()]{2,60}\)", t):
        return True
    if _looks_ocr_junk(t):
        return True
    return False


def _junky_token(w):
    """An OCR-debris token: shape only (case scramble, digit-letter mash,
    lone letters, repeated-char runs, non-ascii squiggles)."""
    core = re.sub(r"[^\w°%§]", "", w, flags=re.UNICODE)
    if not core:
        return True
    if re.search(r"[^\x00-\x7f]", w):
        return True
    if re.search(r"\d", core) and re.search(r"[a-zA-Z]", core):
        return True
    if len(core) == 1 and core.isalpha():
        return True
    # a repeated-single-char token ("oo", "nn", "000") — seal/squiggle debris;
    # roman-numeral repeats ("ii", "iii") are real enumerators, never debris
    if (
        len(core) >= 2
        and len(set(core.lower())) == 1
        and core.lower()[0] not in "ivxcm"
    ):
        return True
    # a lowercase->uppercase transition mid-word ("aBn", "nBZP5") — OCR
    # case scramble, never a real word (Mc/Di names live in longer blocks)
    if re.search(r"[a-z][A-Z]", core):
        return True
    return False


def _looks_ocr_junk(text):
    """A short block whose tokens are mostly OCR debris — DocuSign id strings
    ("241A3FC61A444C4"), seal/squiggle fragments ("a°° n nezas4b", "Qa0"),
    lone letters — never a semantic unit.  Structural: token shape only.
    """
    text = text or ""
    toks = text.split()
    if not toks or len(toks) > 6:
        return False
    # a field-label row ("M& C Date: N/ A") is a form field, not seal debris,
    # even when OCR shattered its remaining tokens to single letters
    if re.search(r"[A-Za-z]:\s", text):
        return False
    nj = sum(1 for w in toks if _junky_token(w))
    return nj * 2 > len(toks)


def _garble_pages(anns):
    """Pages that are unreadable OCR'd table/spreadsheet debris — a swarm of
    tiny mis-promoted 'headings' over junk tokens, with no signature fields
    (a signature page also has many short headings but keeps its structure).
    The fine structure on such a page is noise; the whole page is ONE unit.
    Signals are shape-only: junk-token fraction, short-heading count.
    """
    by_page = {}
    for a in anns:
        by_page.setdefault(a.get("page"), []).append(a)
    out = set()
    for pg, page_anns in by_page.items():
        if len(page_anns) < 15:
            continue
        toks = [w for a in page_anns for w in (a.get("rawText") or "").split()]
        if not toks:
            continue
        short_hdrs = sum(
            1
            for a in page_anns
            if a.get("annotationLabel") in _HEADING_LABELS
            and len((a.get("rawText") or "").split()) <= 2
        )
        n_fields = sum(1 for a in page_anns if _is_signature_field(a.get("rawText")))
        junk_frac = sum(1 for w in toks if _junky_token(w)) / len(toks)
        if short_hdrs >= 10 and junk_frac >= 0.12 and n_fields == 0:
            out.add(pg)
    return out


def _is_recitals_heading(text):
    """A recitals / witnesseth section heading (handles spaced 'R E C I T A L S'
    and a trailing colon/period)."""
    t = re.sub(r"\s+", "", (text or "")).lower().rstrip(":.")
    return t in ("recitals", "witnesseth")


def _is_toc_heading(text):
    """A heading whose enumerated body is ONE block, not separate clauses: a table
    of contents / index (navigation entries), or a Notes / footnotes block. Its
    ``1. 2. 3.`` items look like clauses but form one logical unit."""
    t = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return (
        t.startswith("table of contents")
        or t in ("contents", "index")
        or re.match(r"^notes?\b[:.]?$", t) is not None
    )


def _is_weak_root(a, page_dims):
    """A heading that opens a letter or cover page rather than a clause: a
    mixed-case colon-ended salutation line ("Ladies and Gentlemen:") or a
    first-page centered mixed-case short title ("Statement of Work").  Such a
    root is its own unit but owns no body — each body paragraph under it is a
    separate top-level unit (a date line, a party clause, an interpretation
    note), and a numbered clause after it is top-level, never its child.
    Format-only: ALL-CAPS section headings and numbered headings never match.
    Returns "salutation" (splits body anywhere), "title" (splits only its own
    page's body — a cover title's fine-tree children can span the whole doc),
    or None.
    """
    t = (a.get("rawText") or "").strip()
    if not t or t.isupper() or len(t.split()) > 6:
        return None
    if _section_number(t) is not None or _is_recitals_heading(t):
        return None
    if _is_toc_heading(t) or _NOTES_RE.match(t):
        return None
    if t.endswith(":") and _SALUTATION_RE.match(t):
        # a colon-terminated heading is a weak root ONLY when it is an actual
        # letter salutation — "Intellectual Property:" clause headings own
        # their body (held-out finding)
        return "salutation"
    if a.get("page") == 0:
        fr = _page_frac(a, page_dims)
        if fr is not None:
            left, right = fr[2], fr[3]
            if left > 0.15 and abs(left - (1.0 - right)) < 0.05:
                return "title"
    return None


def _is_demoted_heading(ann):
    """A heading that is not a real clause root: furniture, signature field, or a
    sentence.

    Text-intrinsic tells only (folio/stamp patterns; a signature/execution field
    label; ``Line.is_prose_continuation`` — a mid-sentence continuation or recital
    opener). Never matches a corpus's specific words, so it cannot overfit.
    """
    t = (ann.get("rawText") or "").strip()
    if not t:
        return True
    if _is_folio(t) or _is_signature_field(t):
        return True
    if _is_idlead_data(t):
        return True
    return bool(getattr(Line(t), "is_prose_continuation", False))


def _is_idlead_data(text):
    """A mis-promoted data row: the lead token is an id string ("CUS000003210
    City of Fort Worth") — table content, never a heading."""
    t = (text or "").strip()
    if not t:
        return False
    core = re.sub(r"[^\w]", "", t.split()[0])
    return bool(
        len(core) >= 6 and re.search(r"\d", core) and re.search(r"[a-zA-Z]", core)
    )


def _colon_lead(text):
    """First word of a short colon-label lead ("Quote NO.: ..." -> "quote"),
    or None.  Tolerates dotted abbreviations the strict row-label regex
    rejects."""
    m = re.match(r"^([A-Za-z][\w.&'’\-]*)\b[^:]{0,30}:\s", (text or "").strip())
    return m.group(1).lower() if m else None


def _enum_level(text):
    """Nesting depth of a block's leading enumerator, or None if unnumbered.

    Integers (and dotted section numbers) are shallowest, then letters, then
    romans — the common legal nesting ``1. > (a) > (i)``. Dotted integers deepen
    by dot count (``1`` -> 1, ``6.6`` -> 2).  An uppercase ``ARTICLE``/
    ``SECTION`` heading rendered as a body line is a level-1 outline entry.
    """
    t = (text or "").strip()
    if _CAPS_SECTION_RE.match(t):
        return 1
    ln = Line(t)
    if not getattr(ln, "numbered_line", False):
        return None
    # A bare ordinal with no body ("2.", "(a)") is page-folio furniture, not a
    # clause — it must not open a unit or (worse) adopt following sub-clauses.
    if len(t.split()) < 2:
        return None
    if getattr(ln, "roman_numbered_line", False):
        return 3
    if getattr(ln, "letter_numbered_line", False):
        return 2
    sn = getattr(ln, "start_number", "") or ""
    return 1 + sn.count(".")


def _section_number(text):
    """Leading outline section number ("15", "1.11", "6.3.7") or None."""
    m = re.match(r"^\(?([0-9]+(?:\.[0-9]+)*)[.)]?\s", (text or "").strip())
    return m.group(1) if m else None


def _should_promote(parent_num, child_num):
    """A section-numbered child at the same-or-shallower outline depth than its
    section-numbered parent is a sibling, not a sub-clause (staircase fix):
    ``15`` under ``14``, ``1.11`` under ``1.10``, ``6.4`` under ``6.3.7``."""
    return (
        parent_num is not None
        and child_num is not None
        and child_num.count(".") <= parent_num.count(".")
    )


def _is_section_start(text):
    """Text that opens a real document section: an outline number, or an
    ``ARTICLE`` / ``SECTION`` heading. Used to stop a real section from ever
    being nested under a title / salutation / prose 'heading' (over-scoping)."""
    t = (text or "").strip()
    return _section_number(t) is not None or bool(
        re.match(r"^(ARTICLE|SECTION)\b", t, re.I)
    )


def _union_annotation_json(members):
    """Union member annotations' per-page bounds + tokensJsons into shape-A.

    A unit may span pages, so emit one page key per covered page.
    """
    pages = {}
    for m in members:
        for pkey, pj in (m.get("annotation_json") or {}).items():
            bounds = pj.get("bounds") or {}
            acc = pages.setdefault(
                pkey,
                {
                    "top": None,
                    "left": None,
                    "right": None,
                    "bottom": None,
                    "tokensJsons": [],
                    "texts": [],
                },
            )
            for edge, fn in (
                ("top", min),
                ("left", min),
                ("right", max),
                ("bottom", max),
            ):
                v = bounds.get(edge)
                if v is None:
                    continue
                acc[edge] = v if acc[edge] is None else fn(acc[edge], v)
            acc["tokensJsons"].extend(pj.get("tokensJsons") or [])
            if pj.get("rawText"):
                acc["texts"].append(pj["rawText"])
    out = {}
    for pkey, acc in pages.items():
        out[pkey] = {
            "bounds": {e: (acc[e] or 0.0) for e in ("top", "left", "right", "bottom")},
            "tokensJsons": acc["tokensJsons"],
            "rawText": " ".join(acc["texts"]),
        }
    return out


def _is_role_seed(text):
    t = (text or "").strip()
    m = _ROLE_SEED_RE.match(t)
    if not m or _is_signature_field(t):
        return False
    lead = m.group(1)
    # every word of the lead starts uppercase ("Contract Compliance Manager")
    return all(w[:1].isupper() for w in lead.split() if w[:1].isalpha())


def _is_sigish(text):
    t = (text or "").strip()
    if not t:
        return False
    return bool(
        _is_signature_field(t)
        or _ESIG_RE.search(t)
        or (t.isupper() and len(t.split()) <= 6)
        or _is_role_seed(t)
    )


def _collect_sig_stacks(anns, excluded, page_dims):
    """Geometric signature/approval-stack grouping.

    On a page with >=4 signature-field blocks, the region from the first
    sig-ish block down is a signature grid: role seeds ("VENDOR:", "APPROVAL
    RECOMMENDED:") open stacks, and every other narrow block joins the open
    stack it x-overlaps and vertically follows — reconstructing each party's/
    role's column stack regardless of the interleaved reading order.  Returns
    ``(assign, seeds)``: block id -> stack seed id, and the seed-id set.
    """
    assign, seeds = {}, set()
    by_page = {}
    for a in anns:
        if a["id"] in excluded:
            continue
        tb = _ann_top_bottom(a)
        if tb is None:
            continue
        pg, top, bottom = tb
        pj = next(iter((a.get("annotation_json") or {}).values()))
        b = pj.get("bounds") or {}
        by_page.setdefault(pg, []).append(
            (top, b.get("left", 0.0), b.get("right", 0.0), bottom, a)
        )
    for pg, rows in by_page.items():
        page_w = page_dims.get(pg, (612.0, 792.0))[0] or 612.0
        n_fields = sum(1 for *_x, a in rows if _is_signature_field(a.get("rawText")))
        if n_fields < 4:
            continue
        rows.sort(key=lambda r: (r[0], r[1]))
        sig_tops = [top for top, _l, _r, _bt, a in rows if _is_sigish(a.get("rawText"))]
        if not sig_tops:
            continue
        region_top = min(sig_tops)
        stacks = []  # {"seed", "left", "right", "bottom", "n"}

        def overlap(l1, r1, l2, r2):
            inter = min(r1, r2) - max(l1, l2)
            return inter / max(1.0, min(r1 - l1, r2 - l2))

        for top, left, right, bottom, a in rows:
            if top < region_top - 2:
                continue
            t = (a.get("rawText") or "").strip()
            wide = (right - left) > 0.62 * page_w
            if wide and not _is_sigish(t):
                continue  # full-width prose stays in its normal unit
            if wide and _is_role_seed(t) and not _is_signature_field(t):
                continue  # a full-width "Rent: ..." def-table row is no stack
            caps_role = (
                t.isupper()
                and 2 <= len(t.split()) <= 6
                and not _is_signature_field(t)
                and not re.search(r"\b(inc|llc|ltd|corp|l\.?p)\b\.?,?$", t, re.I)
                and not re.search(r"[0-9]", t)
            )
            if (_is_role_seed(t) or caps_role) and not wide:
                # a role label directly under another heading-only stack (no
                # field lines yet) continues that seed ("CONTRACT COMPLIANCE"
                # / "MANAGER:", "APPROVED AS TO FORM AND" / "LEGALITY:")
                # instead of opening a second stack
                merged = False
                for st in stacks:
                    if (
                        st["fields"] == 0
                        and top - st["bottom"] < 16.0
                        and overlap(left, right, st["left"], st["right"]) > 0.25
                    ):
                        assign[a["id"]] = st["seed"]
                        st.update(
                            left=min(st["left"], left),
                            right=max(st["right"], right),
                            bottom=max(st["bottom"], bottom),
                        )
                        merged = True
                        break
                if not merged:
                    stacks.append(
                        {
                            "seed": a["id"],
                            "seed_top": top,
                            "left": left,
                            "right": right,
                            "bottom": bottom,
                            "fields": 0,
                            # a long colon-TERMINATED heading ("FOR CITY OF
                            # FORT WORTH INTERNAL PROCESSES:", 7 words) is a
                            # page/section heading standing alone, never a
                            # member-accepting stack; a long role seed still
                            # owns its stack — value-carrying ("LANDLORD:
                            # Velocity Real Estate Holdings, LLC, ...") or a
                            # 6-word role banner ("APPROVED AS TO FORM AND
                            # LEGALITY:")
                            "standalone": len(t.split()) >= 7
                            and t.rstrip().endswith(":"),
                        }
                    )
                    assign[a["id"]] = a["id"]
                    seeds.add(a["id"])
                continue
            if len(t.split()) >= 6 and t.rstrip().endswith(":"):
                continue  # a long colon-terminated heading is no stack member
            best, best_key = None, None
            for st in stacks:
                if st.get("standalone"):
                    continue
                ov = overlap(left, right, st["left"], st["right"])
                # a narrow seed ("CITY:") barely spans its column: a member
                # starting just right of it (its e-sig / name lines) is still
                # the same column stack
                adjacent = 0.0 <= left - st["right"] < 16.0
                gap = top - st["bottom"]
                if (ov > 0.25 or adjacent) and -12.0 < gap < 70.0:
                    # shadowed: another seed opened between this stack's seed
                    # and the member, in the member's own column — the member
                    # belongs to that nearer role, however far this stack's
                    # box has crept down (junk fragments bridge OCR pages)
                    shadowed = any(
                        s2 is not st
                        and st["seed_top"] < s2["seed_top"] <= top
                        and (
                            overlap(left, right, s2["left"], s2["right"]) > 0.25
                            or 0.0 <= left - s2["right"] < 16.0
                        )
                        for s2 in stacks
                    )
                    if shadowed:
                        continue
                    key = (max(ov, 0.26) if adjacent else ov, -max(gap, 0.0))
                    if best_key is None or key > best_key:
                        best, best_key = st, key
            if best is not None:
                assign[a["id"]] = best["seed"]
                best.update(
                    left=min(best["left"], left),
                    right=max(best["right"], right),
                    bottom=max(best["bottom"], bottom),
                    fields=best["fields"] + 1,
                )
            # unmatched narrow blocks keep their normal unit
        if not stacks:
            # a label-below-value grid ("William Johnson" over "Name (please
            # print)"): signature fields exist but no role seeds — group the
            # whole region as one signature unit
            seed = None
            for top, left, right, bottom, a in rows:
                if top < region_top - 2:
                    continue
                t = (a.get("rawText") or "").strip()
                if (right - left) > 0.62 * page_w and not _is_sigish(t):
                    continue
                if seed is None:
                    seed = a["id"]
                    seeds.add(seed)
                assign[a["id"]] = seed
    return assign, seeds


def _page_dims(export):
    dims = {}
    for i, p in enumerate(export.get("pawls_file_content") or []):
        pg = p.get("page") or {}
        dims[i] = (pg.get("width") or 1.0, pg.get("height") or 1.0)
    return dims


def _is_corner_stamp(a, page_dims):
    """A short exhibit / classification stamp pinned in the page's top-right
    corner ("Exhibit 10.1", "CONFIDENTIAL") — filing furniture, never a unit.
    Geometry (top-right corner) + shape (an Exhibit stamp or a short all-caps
    mark); a centered exhibit-cover heading ("EXHIBIT A-5" mid-page) never
    matches the corner geometry.
    """
    t = (a.get("rawText") or "").strip()
    if not t or len(t.split()) > 4:
        return False
    # only a NUMBERED exhibit id is a stamp ("Exhibit 10.1", "Exhibit 99.7");
    # a lettered "Exhibit E" is a real internal exhibit heading, and an
    # uppercase "SECTION 9. COMPENSATION." at a page top never is
    exhibit = bool(re.match(r"^exhibits?\s+(no\.?\s*)?[0-9][a-z0-9.\-()]*$", t, re.I))
    stampish = exhibit or (
        t.isupper() and len(t.split()) <= 3 and not _CAPS_SECTION_RE.match(t)
    )
    if not stampish:
        return False
    aj = a.get("annotation_json") or {}
    if not aj:
        return False
    for pk, pj in aj.items():
        w, h = page_dims.get(int(pk), (1.0, 1.0))
        b = pj.get("bounds") or {}
        top_frac = b.get("top", h) / (h or 1.0)
        # true corner position: the block STARTS past page centre (a wide
        # heading whose box merely extends to the right margin never counts)
        right_ish = (b.get("left", 0.0) / (w or 1.0)) > 0.5
        # a right-aligned numbered exhibit stamp may sit as low as the top
        # third (a short consent letter); a caps mark must be in the TRUE
        # corner (a right-column role heading near the page top is not)
        if exhibit and right_ish:
            ok = top_frac < 0.35
        elif exhibit:
            ok = top_frac < 0.2
        else:
            ok = top_frac < 0.1 and (b.get("left", 0.0) / (w or 1.0)) > 0.6
        if not ok:
            return False
    return True


def _ann_top_bottom(a):
    """(page, top, bottom) of a single-page annotation, or None."""
    aj = a.get("annotation_json") or {}
    if len(aj) != 1:
        return None
    pk, pj = next(iter(aj.items()))
    b = pj.get("bounds") or {}
    return (int(pk), b.get("top", 0.0), b.get("bottom", 0.0))


def _page_frac(a, page_dims):
    """(top, bottom, left, right) page-fraction box of a single-page ann."""
    aj = a.get("annotation_json") or {}
    if len(aj) != 1:
        return None
    pk, pj = next(iter(aj.items()))
    b = pj.get("bounds") or {}
    w, h = page_dims.get(int(pk), (1.0, 1.0))
    return (
        b.get("top", 0.0) / (h or 1.0),
        b.get("bottom", 0.0) / (h or 1.0),
        b.get("left", 0.0) / (w or 1.0),
        b.get("right", 0.0) / (w or 1.0),
    )


def _margin_stamp_ids(anns, page_dims):
    """Ids of margin-stamp furniture the text filters can't safely name.

    (a) A clerk/filing stamp line ("OFFICIAL RECORD", "CITY SECRETARY",
    "FT. WORTH, TX") pinned in the page's extreme bottom or top margin — the
    geometry gate keeps a signer's mid-page "City Secretary" title alive.
    (b) A short line whose first-5-word key recurs in the bottom band across
    >=2 distinct pages: a running template footer, keyed on repetition +
    position (never this corpus's words); the loose key survives OCR variance.
    The 5-word key + no-enumerator gate keep real content safe: a TOC row that
    mirrors a body heading ("4.1 Issuance of Stock") is enumerated and short,
    and sibling headings sharing a lead ("REFERENCES IN PUB. L. 117-103" /
    "...116-6") differ within their first five words.
    """
    out = set()
    key_hits = {}
    pos_hits = {}
    for a in anns:
        t = (a.get("rawText") or "").strip()
        if not t:
            continue
        fr = _page_frac(a, page_dims)
        if fr is None:
            continue
        nw = len(t.split())
        if (
            nw <= 8
            and fr[1] > 0.88
            and _section_number(t) is None
            and not any(c.isdigit() for c in t)  # citation/reference rows live
            and t[:1].isupper()  # a wrapped continuation fragment lives
        ):
            # repeated-POSITION key (quantized corner) so an OCR-varying
            # clerk/filing stamp pinned at the same bottom-margin spot across
            # pages keys even when its text scrambles ("FT. WQRTH") — the
            # geometric replacement for the old corpus-worded clerk-stamp
            # regex.  Digit-free short lines only: a real bottom-band heading
            # ("REFERENCES IN PUB. L. 117–103") carries its citation digits.
            pos = (round(fr[2] * 20), round(fr[0] * 20))
            pos_hits.setdefault(pos, []).append((a.get("page"), a["id"]))
        if nw <= 10 and fr[1] > 0.88 and _section_number(t) is None:
            toks = tuple(w.lower() for w in re.findall(r"[A-Za-z]+", t)[:5])
            if len(toks) == 5:
                key_hits.setdefault(toks, []).append((a.get("page"), a["id"]))
    for hits in key_hits.values():
        if len({pg for pg, _i in hits}) >= 2:
            out.update(i for _pg, i in hits)
    for hits in pos_hits.values():
        if len({pg for pg, _i in hits}) >= 2:
            out.update(i for _pg, i in hits)
    return out


def _row_gap(prev_a, a):
    """Vertical gap (PDF points) between two same-page annotations, or None."""
    pb, cb = _ann_top_bottom(prev_a), _ann_top_bottom(a)
    if pb is None or cb is None or pb[0] != cb[0]:
        return None
    return cb[1] - pb[2]


def _compute_toc_ids(anns, excluded):
    """Ids of blocks that are table-of-contents rows: runs (>=3, or >=2 under a
    bare "Sec." column header) of short enumerated entries where at least half
    end in a bare page number (or a "Sec." trigger is active) — navigation
    entries that must never root or be numbering-split into clauses.  The
    half-with-tails rule lets a run carry entries whose page number was lost
    to the segmentation ("(a) Repair Estimate").
    """
    toc = set()
    run, trigger = [], False

    def flush():
        nonlocal run, trigger
        n_tail = sum(1 for x in run if _TOC_ROW_TAIL_RE.search(x["rawText"] or ""))
        n_cit = sum(1 for x in run if _CITATION_ROW_RE.search(x["rawText"] or ""))
        if n_cit * 2 > len(run):
            run, trigger = [], False  # citation/data rows, not navigation
            return
        if (len(run) >= 3 and n_tail * 2 >= len(run)) or (trigger and len(run) >= 2):
            toc.update(x["id"] for x in run)
        run, trigger = [], False

    for a in anns:
        t = (a.get("rawText") or "").strip()
        if a["id"] in excluded:
            if re.fullmatch(r"sec\.?", t, re.I):
                flush()
                trigger = True
            elif _is_folio(t):
                # a front-matter page folio ("-ii-") ends the TOC: the body
                # headings after it must not ride the run into toc-land
                flush()
            continue
        enum = _enum_level(t) is not None or bool(_ROW_ENUM_RE.match(t))
        nw = len(t.split())
        # a long entry still counts when it carries the page-number tail (a
        # wrapped TOC title); without the tail, long enum text is a clause
        if enum and (nw <= 14 or (nw <= 24 and _TOC_ROW_TAIL_RE.search(t))):
            run.append(a)
        elif run and nw <= 6 and _TOC_ROW_TAIL_RE.search(t):
            # a short tail-carrying section-title row mid-run ("GENERAL
            # PROVISIONS 12") is a TOC group header, not a run break
            run.append(a)
        else:
            flush()
    flush()
    return toc


def _repeated_prefix_rows(anns, excluded):
    """Ids in runs of >=3 consecutive Table Rows sharing the same first word
    ("Renewal Term Number 5 ..." x5): a data table whose rows are parallel
    semantic entries — each row becomes its own unit.  Runs carrying "$"
    amounts (a rent/fee schedule under one field label) stay together.
    """
    out = set()
    run, prev_tok = [], None

    def flush():
        nonlocal run
        if len(run) >= 3 and not any("$" in (x.get("rawText") or "") for x in run):
            out.update(x["id"] for x in run)
        run = []

    for a in anns:
        if a["id"] in excluded:
            continue
        t = (a.get("rawText") or "").strip()
        tok = t.split()[0].lower() if t.split() else ""
        if a.get("annotationLabel") == "Table Row" and len(tok) >= 3 and tok.isalpha():
            if tok != prev_tok:
                flush()
            run.append(a)
            prev_tok = tok
        else:
            flush()
            prev_tok = None
    flush()
    return out


def _repeated_line_ids(anns):
    """Ids of short all-alpha lines whose token key (trailing bare folio
    number stripped) recurs on >=40% of pages (>=3): a running header/footer
    stamp, wherever it lands on the page ("DeGolyer and MacNaughton" /
    "DeGolyer and MacNaughton 10" — the folio-welded variant escapes the text
    filters).  Format-only gates keep real content safe: a data row ("Low
    Estimate 7,571"), a numbered sibling heading ("REFERENCES IN PUB. L.
    117–229"), colon-ended labels ("Notes:", "By:"), and signature fields
    never key; the high page-fraction bar keeps a heading repeated a few
    times ("EDITORIAL NOTES") or a party name on cover + signature pages
    ("Lucid Capital Markets, LLC") alive.
    """
    pages = {a.get("page") for a in anns}
    need = max(3, -(-len(pages) * 2 // 5))  # ceil(40% of pages)
    key_hits = {}
    for a in anns:
        t = (a.get("rawText") or "").strip()
        toks = t.split()
        if not toks or len(toks) > 6:
            continue
        if t.endswith(":") or _is_signature_field(t):
            continue
        if _enum_level(t) is not None or _section_number(t) is not None:
            continue
        if toks[-1].strip(".-").isdigit() or _is_folio(toks[-1]):
            toks = toks[:-1]  # strip a welded trailing folio number
        if len(toks) < 2 or not all(
            re.fullmatch(r"[A-Za-z][A-Za-z.,'’&-]*", w) for w in toks
        ):
            continue  # a data row / numbered sibling heading never keys
        key = tuple(w.lower().strip(".,") for w in toks)
        key_hits.setdefault(key, []).append((a.get("page"), a["id"]))
    out = set()
    for hits in key_hits.values():
        if len({pg for pg, _i in hits}) >= need:
            out.update(i for _pg, i in hits)
    return out


# A letter salutation lead — the universal letter-opening forms (same
# sanctioned-boilerplate class as recital triggers).  A colon-terminated
# CLAUSE heading ("Intellectual Property:") must never match: held-out
# contracts showed colon-endedness alone shatters ordinary clause bodies.
_SALUTATION_RE = re.compile(
    r"^(dear\b|ladies\s+and\s+gentlemen|gentlemen\b|to\s+whom\b)", re.I
)

_URL_RE = re.compile(r"www\.\S+|https?://|\S+\.(com|org|net)\b", re.I)
_PHONE_RE = re.compile(r"\d{3}[.\s-]+\d{3,4}[.\s-]+\d{4}")


def _letterhead_ids(anns, by_id, excluded, page_dims):
    """Ids of letterhead address runs: consecutive short non-heading lines in
    a page's top band, all parented under an excluded block (the letterhead
    logo), where at least one line carries a URL or phone-number token —
    address/contact furniture, never a semantic unit.
    """
    out = set()
    run = []

    def flush():
        nonlocal run
        if any(
            _URL_RE.search(x.get("rawText") or "")
            or _PHONE_RE.search(x.get("rawText") or "")
            for x in run
        ):
            out.update(x["id"] for x in run)
        run = []

    for a in anns:
        t = (a.get("rawText") or "").strip()
        fr = _page_frac(a, page_dims)
        if (
            a["id"] not in excluded
            and a.get("annotationLabel") not in _HEADING_LABELS
            and a.get("parent_id") in excluded
            and fr is not None
            and fr[0] < 0.28
            and len(t.split()) <= 8
            and (not run or run[-1].get("parent_id") == a.get("parent_id"))
        ):
            run.append(a)
        else:
            flush()
    flush()
    return out


def _is_welded_twin(text):
    """A two-column signature row welded into one line: its leading token
    sequence repeats mid-row ("Signed by: Signed by:", "For and on behalf of
    X ... For and on behalf of Y")."""
    toks = (text or "").split()
    for k in (5, 4, 3, 2):
        if len(toks) < 2 * k:
            continue
        lead = toks[:k]
        for j in range(k, len(toks) - k + 1):
            if toks[j : j + k] == lead:
                return True
        break  # only test the longest feasible lead
    return False


def _repeated_prefix_headings(anns, excluded):
    """Ids in runs of >=3 consecutive headings sharing the same first word
    ("Exhibit Outline of Premises A" / "Exhibit Building Rules..." /
    "Exhibit Basic Costs C"): a mis-promoted index/list, not real outline
    levels — demoted, they join the unit of the true heading above.  Real
    same-word sibling headings always have body between them.
    """
    out = set()
    run, prev_tok = [], None

    def flush():
        nonlocal run
        if len(run) >= 3:
            out.update(x["id"] for x in run)
        run = []

    for a in anns:
        if a["id"] in excluded:
            continue
        t = (a.get("rawText") or "").strip()
        tok = t.split()[0].lower() if t.split() else ""
        if (
            a.get("annotationLabel") in _HEADING_LABELS
            and len(tok) >= 3
            and tok.isalpha()
        ):
            if tok != prev_tok:
                flush()
            run.append(a)
            prev_tok = tok
        else:
            flush()
            prev_tok = None
    flush()
    return out


def append_semantic_units(export, *, options=None):
    """Append the guarded, numbering-split Semantic-Unit layer to ``export``."""
    anns = export.get("labelled_text") or []
    by_id = {a["id"]: a for a in anns}
    page_dims = _page_dims(export)

    # --- guard: furniture / bare markers never seed a unit or pollute one ---
    stamp_ids = {a["id"] for a in anns if _is_corner_stamp(a, page_dims)}
    excluded = stamp_ids | _margin_stamp_ids(anns, page_dims)
    excluded |= _repeated_line_ids(anns)
    excluded |= {
        a["id"]
        for a in anns
        if _is_furniture(a.get("rawText")) or _is_bare_marker(a.get("rawText"))
    }
    excluded |= _letterhead_ids(anns, by_id, excluded, page_dims)
    # A bare ordinal directly above a heading is that heading's outline number
    # ("2." centered over "RENEWALS"): attach it to the heading's unit (its
    # number then drives the sibling staircase) instead of dropping it.
    marker_attach = {}
    for i, a in enumerate(anns[:-1]):
        t = (a.get("rawText") or "").strip()
        nxt = anns[i + 1]
        if (
            a["id"] in excluded
            and re.fullmatch(r"\(?[0-9]{1,3}[.)]?", t)
            and nxt["annotationLabel"] in _HEADING_LABELS
            and nxt["id"] not in excluded
        ):
            marker_attach[a["id"]] = nxt["id"]
            excluded.discard(a["id"])
    # --- guard: which heading roots are demoted (not real clause roots) ---
    demoted = {
        a["id"]
        for a in anns
        if a["annotationLabel"] in _HEADING_LABELS and _is_demoted_heading(a)
    }
    demoted |= _repeated_prefix_headings(anns, excluded)
    # id-lead pseudo-headings ("CUS000003210 City of Fort Worth") are table
    # data: they continue the table unit above rather than standing alone
    data_headings = {
        a["id"]
        for a in anns
        if a["annotationLabel"] in _HEADING_LABELS and _is_idlead_data(a.get("rawText"))
    }
    # a colon-label VALUE row mis-promoted to heading ("Client ID:
    # CUS000003210") inside a tight run of colon-label rows is a metadata
    # field, not a clause root: it joins the field group instead of rooting
    prev_nx = None
    for a in anns:
        if a["id"] in excluded:
            continue
        t = (a.get("rawText") or "").strip()
        if (
            a["annotationLabel"] in _HEADING_LABELS
            and _colon_lead(t) is not None
            and not t.rstrip().endswith(":")
            and len(t.split()) <= 7
            and prev_nx is not None
            and _colon_lead((prev_nx.get("rawText") or "").strip()) is not None
            and _row_gap(prev_nx, a) is not None
            and _row_gap(prev_nx, a) < 14.0
        ):
            data_headings.add(a["id"])
            demoted.add(a["id"])
        prev_nx = a

    toc_ids = _compute_toc_ids(anns, excluded)

    # a confidential-treatment-redacted filing: "[***]" / "[•]" placeholders
    # (they eat clause enumerators, leaving marker-less sibling clauses)
    # DENSITY-gated: held-out ablation showed a single stray "[***]" in a
    # 126-page lease flipping the whole doc's clause splitting — require
    # placeholders on >=3 distinct pages so only genuinely redacted filings
    # (whose clause markers the redaction really did eat) enter this mode.
    redacted_pages = {
        a.get("page")
        for a in anns
        if "[***]" in (a.get("rawText") or "") or "[•]" in (a.get("rawText") or "")
    }
    redacted_doc = len(redacted_pages) >= 3

    # --- OCR-garble table pages collapse to one flat unit per page ---------
    garble_root = {}
    for pg in _garble_pages(anns):
        first = next(
            (a["id"] for a in anns if a.get("page") == pg and a["id"] not in excluded),
            None,
        )
        if first is not None:
            garble_root[pg] = first

    def heading_ok(i):
        return (
            i in by_id
            and i not in excluded
            and by_id[i]["annotationLabel"] in _HEADING_LABELS
            and i not in demoted
            and i not in toc_ids  # a TOC entry never roots a unit
            and by_id[i].get("page") not in garble_root  # garble junk
        )

    def nearest_ok_heading(a):
        cur, seen = a.get("parent_id"), set()
        while cur is not None and cur in by_id and cur not in seen:
            seen.add(cur)
            if heading_ok(cur):
                return cur
            cur = by_id[cur].get("parent_id")
        return None

    # a marker only attaches to a heading that actually roots a unit
    for m, h in list(marker_attach.items()):
        if not heading_ok(h):
            del marker_attach[m]
            excluded.add(m)

    # --- signature/approval stacks: geometric column grouping ---------------
    sig_assign, sig_seeds = _collect_sig_stacks(anns, excluded, page_dims)
    sig_pages = {by_id[s].get("page") for s in sig_seeds}
    rowsplit_ids = _repeated_prefix_rows(anns, excluded)

    # --- coarsen: each fine block -> its unit-root (a real heading / itself) ---
    root_of = {}
    for a in anns:
        if a["id"] in excluded:
            continue
        if a.get("page") in garble_root:
            root_of[a["id"]] = garble_root[a.get("page")]
            continue
        if a["id"] in sig_assign:
            root_of[a["id"]] = sig_assign[a["id"]]
            continue
        if a["id"] in marker_attach:
            root_of[a["id"]] = marker_attach[a["id"]]
            continue
        root_of[a["id"]] = (
            a["id"] if heading_ok(a["id"]) else (nearest_ok_heading(a) or a["id"])
        )
    # --- stitch: headingless section pseudo-roots adopt their body ---------
    # A charter/bylaws "SECTION 1. ANNUAL MEETING." rendered as a plain
    # paragraph is a headingless singleton root in the fine tree, and so is
    # the body paragraph after it.  Stitch, in reading order: a section-start
    # pseudo-root adopts the following headingless singleton paragraphs until
    # the next heading / section-start / boundary; it also remembers the last
    # ARTICLE-heading unit so the section can nest under it (the subtree the
    # golden expects).
    pseudo_parent = {}
    open_pseudo = None
    last_article = None
    for a in anns:
        aid = a["id"]
        if aid in excluded:
            continue
        if aid in sig_assign:
            open_pseudo = None
            continue
        t = (a.get("rawText") or "").strip()
        if heading_ok(aid):
            open_pseudo = None
            if re.match(r"^ARTICLE\b", t):
                last_article = aid
            continue
        if root_of.get(aid) != aid:
            open_pseudo = None
            continue
        secnum = _section_number(t)
        if _CAPS_SECTION_RE.match(t) or (
            # a bare dotted-number section title ("1.11 Action without
            # Meeting.") rendered as a plain line is likewise a headingless
            # section pseudo-root
            secnum is not None
            and "." in secnum
            and len(t.split()) <= 6
            and aid not in toc_ids
        ):
            open_pseudo = aid
            if last_article is not None:
                pseudo_parent[aid] = last_article
            continue
        if open_pseudo is not None and a.get("annotationLabel") in (
            "Paragraph",
            "List Item",
        ):
            lvl = _enum_level(t)
            if (
                _HARD_BOUNDARY_RE.match(t)
                or _PREAMBLE_RE.match(t)
                or _FRONT_BOUNDARY_RE.match(t)
            ):
                open_pseudo = None
            elif lvl is not None:
                # a letter/roman sub-item under a section-numbered pseudo-root
                # ("(a) Taking of Action by Consent" under "1.11 Action
                # without Meeting.") belongs to that section — the numbering
                # split then files it as a child clause; a same-level number
                # is a sibling section and closes the pseudo-root
                if lvl >= 2 and _section_number(
                    (by_id[open_pseudo].get("rawText") or "").strip()
                ):
                    root_of[aid] = open_pseudo
                else:
                    open_pseudo = None
            else:
                root_of[aid] = open_pseudo
            continue
        open_pseudo = None

    # a TOC entry that ended up self-rooted (its heading label was vetoed, or
    # it has no heading ancestor) joins the unit before the run instead of
    # shattering the TOC into singletons; a headingless singleton whose text
    # starts lowercase is a broken continuation and joins the previous unit
    # caption lines orphaned under an excluded bare signature marker ("X" /
    # "By" sign-here lines): a line hugging its marker captions the field
    # group above, and a sibling caption under the same marker follows the
    # caption directly before it ("Signature(s)" / "Guaranteed:" pairs).
    marker_captions = set()
    prev_nx = None
    for a in anns:
        if a["id"] in excluded:
            continue
        par = by_id.get(a.get("parent_id"))
        if (
            par is not None
            and par["id"] in excluded
            and _is_bare_marker(par.get("rawText"))
            and (
                (_row_gap(par, a) is not None and _row_gap(par, a) < 12.0)
                or (
                    prev_nx is not None
                    and prev_nx["id"] in marker_captions
                    and prev_nx.get("parent_id") == a.get("parent_id")
                )
            )
        ):
            marker_captions.add(a["id"])
        prev_nx = a

    prev_root = None
    prev_ann = None
    for a in anns:
        aid = a["id"]
        if aid in excluded:
            continue
        if aid in sig_assign:
            prev_root = None
            prev_ann = None
            continue
        t = (a.get("rawText") or "").strip()
        if aid in toc_ids and root_of.get(aid) == aid:
            if prev_root is not None:
                root_of[aid] = prev_root
        elif (
            root_of.get(aid) == aid
            and not heading_ok(aid)
            and prev_root is not None
            and (
                t[:1].islower()
                or (_is_signature_field(t) and a.get("page") not in sig_pages)
                or (
                    # a paragraph orphaned by a corner-stamp exclusion joins
                    # the open flow — unless that flow's root is itself a
                    # sibling stamp-orphan paragraph on the same page (two
                    # cover disclaimers under one exhibit stamp stand alone)
                    a.get("parent_id") in stamp_ids
                    and not (
                        by_id[prev_root].get("parent_id") in stamp_ids
                        and by_id[prev_root].get("page") == a.get("page")
                        and by_id[prev_root].get("annotationLabel")
                        not in _HEADING_LABELS
                    )
                )
                or aid in marker_captions
                or aid in data_headings
                or (
                    _colon_lead(t) is not None
                    and _colon_lead((by_id[prev_root].get("rawText") or ""))
                    == _colon_lead(t)
                )
                or (
                    # a colon-label row hugging the colon-label row above it
                    # (<14pt) continues that metadata field group even with a
                    # different lead ("Quote Amount:" under "Quote Date:",
                    # "Client ID:" under "Quote Amount:")
                    _colon_lead(t) is not None
                    and prev_ann is not None
                    and _colon_lead((prev_ann.get("rawText") or "").strip()) is not None
                    and _row_gap(prev_ann, a) is not None
                    and _row_gap(prev_ann, a) < 14.0
                )
                or (
                    # a welded two-column signature row ("For and on behalf
                    # of X For and on behalf of Y") follows the conformed
                    # "/s/ ..." row above it
                    _is_welded_twin(t)
                    and prev_ann is not None
                    and (prev_ann.get("rawText") or "").strip().startswith("/s/")
                )
            )
        ):
            # a broken lowercase continuation, a stray signature-field
            # fragment ("Name: Prineha Narang") below a conformed signature
            # (on pages with geometric stacks the stack grouper owns sig
            # fields instead), or a paragraph orphaned by the exclusion of
            # the corner stamp that used to root it — all join the previous
            # unit, whose split loop re-applies the clause boundaries.
            root_of[aid] = prev_root
        else:
            prev_root = root_of.get(aid)
        prev_ann = a

    # conformed-signature grouping: "/s/ Name" plus the short city/date lines
    # after it form ONE signature unit ("/s/" is the universal conformed-
    # signature format).  A short heading absorbed here ("Tysons, Virginia")
    # drags along the blocks that coarsened under it (its date line).
    ann_index = {a["id"]: i for i, a in enumerate(anns)}
    for a in anns:
        t = (a.get("rawText") or "").strip()
        if not t.startswith("/s/") or a["id"] in excluded or a["id"] in sig_assign:
            continue
        target = root_of.get(a["id"])
        if target is None:
            continue
        absorbed = 0
        for b in anns[ann_index[a["id"]] + 1 :]:
            if b["id"] in excluded:
                continue
            bt = (b.get("rawText") or "").strip()
            if (
                absorbed >= 3
                or b.get("page") != a.get("page")
                or bt.startswith("/s/")
                or len(bt.split()) > 6
                or _enum_level(bt) is not None
                or root_of.get(b["id"]) not in (b["id"], target)
            ):
                break
            old = b["id"]
            root_of[old] = target
            for c in anns:  # descendants that coarsened under an absorbed heading
                if root_of.get(c["id"]) == old and c["id"] != old:
                    root_of[c["id"]] = target
            absorbed += 1

    members = {}
    for a in anns:
        if a["id"] in excluded:
            continue
        members.setdefault(root_of[a["id"]], []).append(a)
    coarse_roots = [
        a["id"] for a in anns if a["id"] not in excluded and root_of[a["id"]] == a["id"]
    ]

    # --- build the nested unit tree (coarse roots + numbering split) ---
    nodes = []  # ordered list of {"key", "parent_key", "members"}
    node_by_key = {}

    def add_node(key, parent_key):
        nd = {"key": key, "parent_key": parent_key, "members": []}
        nodes.append(nd)
        node_by_key[key] = nd
        return nd

    for rid in coarse_roots:
        root_ann = by_id[rid]
        if rid in sig_seeds or rid in garble_root.values():
            # a signature stack / garble-page blob is one flat unit:
            # no nesting, no splitting
            add_node(rid, None)["members"].extend(members[rid])
            continue
        parent_heading = pseudo_parent.get(rid) or nearest_ok_heading(root_ann)
        parent_key = parent_heading if heading_ok(parent_heading or "") else None
        # Heading-root units nest only under a structural container: the
        # stitched ARTICLE parent, an ARTICLE-style heading, or a heading whose
        # outline number is genuinely shallower ("1.1 ..." under "1. ...").
        # Everything else (USC note-heading chains, signature-page header
        # chains) is a SIBLING — the fine tree's level chain must not merge
        # their spans into one subtree.
        if parent_key is not None and rid not in pseudo_parent:
            ptxt = (by_id[parent_key].get("rawText") or "").strip()
            ctxt = (root_ann.get("rawText") or "").strip()
            pnum, cnum = _section_number(ptxt), _section_number(ctxt)
            outline_deeper = (
                pnum is not None
                and cnum is not None
                and cnum.count(".") > pnum.count(".")
            )
            # a data-ish "heading" (a mis-promoted table fragment: it carries
            # a comma/decimal magnitude like "34,362", or is a short
            # parenthesized unit like "(MW)") is not a sibling section — leave
            # it under the table's heading.  A year ("SHORT TITLE OF 2012
            # AMENDMENT") is not data.
            data_ish = bool(re.search(r"\d[\d,]*[.,]\d", ctxt)) or bool(
                re.fullmatch(r"\([^)]{1,12}\)", ctxt)
            )
            if not (_CONTAINER_HEADING_RE.match(ptxt) or outline_deeper or data_ish):
                parent_key = None
        root_node = add_node(rid, parent_key)
        # A table of contents is ONE unit: its entries look like numbered clauses
        # but are navigation, so never numbering-split under a TOC heading.  A
        # Notes block is likewise one unit — but it owns only its enumerated
        # items: the first fresh body paragraph after them ends the notes and
        # opens its own top-level unit (what follows joins that unit).
        if _is_toc_heading(root_ann.get("rawText")):
            cut = None
            if _NOTES_RE.match((root_ann.get("rawText") or "").strip()):
                seen_item = False
                for mi, a in enumerate(members[rid]):
                    t = (a.get("rawText") or "").strip()
                    if a["id"] != rid and (
                        a.get("annotationLabel") == "List Item"
                        or _enum_level(t) is not None
                    ):
                        seen_item = True
                    elif (
                        seen_item
                        and a.get("annotationLabel") == "Paragraph"
                        and t[:1].isupper()
                    ):
                        cut = mi
                        break
            if cut is None:
                root_node["members"].extend(members[rid])
            else:
                root_node["members"].extend(members[rid][:cut])
                add_node(f"{rid}#notesbody", None)["members"].extend(members[rid][cut:])
            continue
        weak_root = _is_weak_root(root_ann, page_dims)
        stack = [(0, rid)]  # (enum_level, node_key); level 0 == the root unit
        n_split = 0
        last_letter2 = None  # last single-letter marker seen at depth 2
        prev_member = None
        last_label_lead = None  # first word of the open colon-label field row
        row_zone_left = None  # left edge of an open outdented def-table zone
        row_zone_parent = None  # heading the zone's rows nest under (or None)
        runin_keys = set()  # run-in clause units (may own numbered sub-items)
        mem_list = members[rid]
        for mi, a in enumerate(mem_list):
            if a["id"] == rid or a["id"] in marker_attach:
                root_node["members"].append(a)
                continue
            txt = (a.get("rawText") or "").strip()

            def open_top_unit(ann):
                # a fresh TOP-LEVEL unit (execution clause, recital, field row):
                # trailing plain content joins it via the stack.
                nonlocal n_split, last_label_lead
                n_split += 1
                last_label_lead = None
                nk = f"{rid}#{n_split}"
                add_node(nk, None)["members"].append(ann)
                del stack[1:]
                stack.append((0, nk))

            # hard terminator (execution/testimonium/letter closing/conformed
            # signature): close open clauses and open a fresh top-level unit.
            if _HARD_BOUNDARY_RE.match(txt):
                open_top_unit(a)
                prev_member = a
                continue
            # front-matter boundary: a recital / operative lead-in / all-caps
            # preamble opens its own unit — except a recital under an explicit
            # RECITALS/WITNESSETH heading (the heading already delimits it).
            if _PREAMBLE_RE.match(txt) or (
                _FRONT_BOUNDARY_RE.match(txt)
                and not (
                    txt.lower().startswith("whereas")
                    and _is_recitals_heading(root_ann.get("rawText"))
                )
            ):
                open_top_unit(a)
                prev_member = a
                continue
            # a quoted-defined-term paragraph is a definition of its own even
            # when its outline number was lost to the block segmentation.
            if _DEF_START_RE.match(txt):
                open_top_unit(a)
                prev_member = a
                continue
            # a form question row ("...? * Yes No ...") is its own field.
            if _FORM_QUESTION_RE.search(txt):
                open_top_unit(a)
                prev_member = a
                continue
            # a short TitleCase caption sitting directly over an e-signature /
            # signature-field line ("Client Approval" over "XDianna (Jul 24,
            # 2025 ...)") opens the signature block's own unit.
            nxt2 = mem_list[mi + 1 : mi + 3]
            if (
                len(txt.split()) <= 4
                and not txt.endswith((".", ";", ",", ":"))
                and not _is_signature_field(txt)  # a field is never a caption
                and "/" not in txt
                and all(w[:1].isupper() for w in txt.split() if w[:1].isalpha())
                and any(
                    _ESIG_RE.search(x.get("rawText") or "")
                    or _is_signature_field(x.get("rawText"))
                    for x in nxt2
                )
            ):
                open_top_unit(a)
                prev_member = a
                continue
            # a table-of-contents entry is navigation: keep it in the open
            # unit, never numbering-split it into a clause — but inside a
            # unit rooted at a TABLE OF CONTENTS title, an ARTICLE/SECTION
            # group row opens a new TOC group unit (the golden groups a
            # contents listing per article).
            if a["id"] in toc_ids:
                if _CAPS_SECTION_RE.match(txt) and "table of contents" in (
                    (root_ann.get("rawText") or "").lower()
                ):
                    open_top_unit(a)
                else:
                    node_by_key[stack[-1][1]]["members"].append(a)
                prev_member = a
                continue
            # a signature-marker caption ("Signature(s)" / "Guaranteed:" under
            # an excluded "X" sign-here line) stays with the open unit — the
            # colon-label row splitter must not tear it off.
            if a["id"] in marker_captions:
                node_by_key[stack[-1][1]]["members"].append(a)
                prev_member = a
                continue
            # a weak (letter/cover) root owns no body: each fresh uppercase
            # body paragraph directly under it is its own top-level unit (a
            # cover title splits only its own page's body).
            if (
                weak_root
                and stack[-1][0] == 0
                and a.get("annotationLabel") == "Paragraph"
                and _enum_level(txt) is None
                and txt[:1].isupper()
                and (weak_root == "salutation" or a.get("page") == root_ann.get("page"))
            ):
                open_top_unit(a)
                prev_member = a
                continue
            # a Table Row that OUTDENTS to the page's left margin, far left
            # of the running body level, is a new outer label|value def-table
            # row ("Deliverables Operation and management ...", "Acceptance
            # Tests N/A"): each is its own unit, and consecutive rows at the
            # same outdented left are sibling rows.  A lowercase start is a
            # wrapped continuation and stays in the open row; an enumerated
            # row belongs to the numbering split instead.  Rows arriving
            # while the root unit is still open (a heading directly over its
            # table) nest under that heading; rows after intermediate clause
            # splits are the page's own top-level entries.
            if (
                a.get("annotationLabel") == "Table Row"
                and _enum_level(txt) is None
                and not _CITATION_ROW_RE.search(txt)  # data/citation rows
            ):
                fr = _page_frac(a, page_dims)
                pfr = (
                    _page_frac(prev_member, page_dims)
                    if prev_member is not None
                    else None
                )
                same_pg = (
                    fr is not None
                    and pfr is not None
                    and _row_gap(prev_member, a) is not None
                )
                opens = False
                if fr is not None and txt[:1].isupper():
                    if same_pg and fr[2] < 0.16 and (pfr[2] - fr[2]) > 0.12:
                        row_zone_left = fr[2]
                        row_zone_parent = (
                            rid
                            if (
                                stack[-1][1] == rid
                                and heading_ok(rid)
                                and by_id[rid].get("page") == a.get("page")
                            )
                            else None
                        )
                        opens = True
                    elif (
                        row_zone_left is not None and abs(fr[2] - row_zone_left) < 0.02
                    ):
                        opens = True
                if opens:
                    n_split += 1
                    last_label_lead = None
                    nk = f"{rid}#{n_split}"
                    add_node(nk, row_zone_parent)["members"].append(a)
                    del stack[1:]
                    stack.append((0, nk))
                    prev_member = a
                    continue
                if fr is not None and row_zone_left is not None:
                    if fr[2] > row_zone_left + 0.1:
                        row_zone_left = None
            # a TALL (multi-line) Table Row starting at the page's left margin
            # is a welded label|value definition row ("Authority This report
            # was authorized by ...", "Source of Information Information used
            # in ..."): each opens its own top-level unit.  Data rows
            # (citation/number formats) and indented rows (a field's wrapped
            # body) stay in the open unit.
            if (
                a.get("annotationLabel") == "Table Row"
                and _enum_level(txt) is None
                and not _CITATION_ROW_RE.search(txt)
                and _colon_lead(txt) is None  # colon rows use the label rule
                and txt[:1].isupper()
                and prev_member is not None
            ):
                tb_a = _ann_top_bottom(a)
                fr_a = _page_frac(a, page_dims)
                if (
                    tb_a is not None
                    and (tb_a[2] - tb_a[1]) > 20.0
                    and fr_a is not None
                    and fr_a[2] < 0.15
                ):
                    open_top_unit(a)
                    prev_member = a
                    continue
            # a table/field row opens its own top-level unit when it starts a
            # new semantic entry: an alpha-only "Label:" lead, a statute-style
            # row enumerator, or a large vertical gap from the previous member
            # (a spaced definition-table row); numeric/data rows and tight
            # continuation lines stay in the open unit.
            if (
                a.get("annotationLabel") in ("Table Row", "Paragraph")
                and _enum_level(txt) is None
                and not _is_signature_field(txt)
            ):
                gap = _row_gap(prev_member, a) if prev_member is not None else None
                label_lead = None
                m_lbl = _ROW_LABEL_RE.match(txt)
                if m_lbl and a.get("annotationLabel") == "Table Row":
                    label_lead = txt.split()[0].lower()
                row_enum = a.get("annotationLabel") == "Table Row" and bool(
                    _ROW_ENUM_RE.match(txt)
                )
                if (
                    (label_lead is not None and label_lead != last_label_lead)
                    or row_enum
                    or a["id"] in rowsplit_ids
                    or (
                        gap is not None
                        and gap > 32.0
                        and not txt[:1].islower()  # never split a continuation
                    )
                ):
                    caps_root = heading_ok(rid) and (
                        (by_id[rid].get("rawText") or "").strip().isupper()
                    )
                    # an enumerated data-table row nests under its table's
                    # heading only on the heading's own page — a continuation
                    # page's rows are that page's top-level entries
                    if (label_lead is not None and caps_root) or (
                        row_enum
                        and caps_root
                        and by_id[rid].get("page") == a.get("page")
                    ):
                        # a labelled field row under an ALL-CAPS section
                        # heading ("BASIC LEASE INFORMATION") is that
                        # section's field: its unit nests there so the
                        # section's rolled span stays whole
                        n_split += 1
                        nk = f"{rid}#{n_split}"
                        add_node(nk, rid)["members"].append(a)
                        del stack[1:]
                        stack.append((0, nk))
                    else:
                        open_top_unit(a)
                    last_label_lead = label_lead
                    prev_member = a
                    continue
                if label_lead is not None:
                    # a colon-label row repeating the previous label's first
                    # word ("Quote NO.:" / "Quote Date:") extends that field
                    # group rather than opening a new one
                    node_by_key[stack[-1][1]]["members"].append(a)
                    prev_member = a
                    continue
            # letter-format boundary lines: a salutation ("Dear Mr. Upreti:"
            # — short mixed-case colon-terminated line) opens the letter-body
            # unit; a reference line ("Re: Siting Study ...") opens its own
            # subject unit.
            if a.get("annotationLabel") == "Paragraph" and (
                (
                    txt.rstrip().endswith(":")
                    and len(txt.split()) <= 4
                    and _SALUTATION_RE.match(txt)
                    and not _is_signature_field(txt)
                )
                or re.match(r"^(re|subject)\s*:", txt, re.I)
            ):
                open_top_unit(a)
                prev_member = a
                continue
            # a run-in clause heading (TitleCase lead + ". " + new sentence)
            # opens its own clause under the current heading root — the
            # enumerator was lost, but the bold run-in format survives.
            if a.get("annotationLabel") == "Paragraph" and _is_runin(txt):
                n_split += 1
                nk = f"{rid}#{n_split}"
                add_node(nk, rid if heading_ok(rid) else None)["members"].append(a)
                runin_keys.add(nk)
                del stack[1:]
                stack.append((0, nk))
                prev_member = a
                continue
            lvl = _enum_level(txt)
            if lvl is None:
                # in a REDACTED doc (confidential-treatment "[***]"/"[•]"
                # placeholders — a filing-format artifact that eats clause
                # markers), a fresh long uppercase body paragraph inside an
                # open enumerated clause is a sibling sub-clause whose marker
                # was lost — it opens its own clause node at the same level
                # (and may own sub-items) rather than extending the item.  A
                # colon-ended previous line introduces the paragraph
                # (":" = "as follows:"), so that one stays a member.
                if (
                    redacted_doc
                    and stack[-1][0] > 0
                    and a.get("annotationLabel") == "Paragraph"
                    and txt[:1].isupper()
                    and len(txt.split()) >= 12
                    and not (
                        prev_member is not None
                        and (prev_member.get("rawText") or "").rstrip().endswith(":")
                    )
                    # a paragraph directly after a short title-like item
                    # ("5.3.5 Confidentiality") is that item's BODY — only a
                    # clause that already reads as a full sentence can have
                    # lost a sibling's marker to redaction
                    and prev_member is not None
                    and len((prev_member.get("rawText") or "").split()) >= 12
                ):
                    cur_lvl, cur_key = stack[-1]
                    n_split += 1
                    nk = f"{rid}#{n_split}"
                    add_node(nk, node_by_key[cur_key]["parent_key"])["members"].append(
                        a
                    )
                    runin_keys.add(nk)
                    stack.pop()
                    stack.append((cur_lvl, nk))
                else:
                    node_by_key[stack[-1][1]]["members"].append(a)
                prev_member = a
                continue
            prev_member = a
            # disambiguate letter "(i)" (sibling of "(h)") from roman "(i)/(ii)"
            m = re.match(r"^\(([a-z])\)", txt, re.I)
            letter = m.group(1).lower() if m else None
            if letter == "i" and last_letter2 == "h":
                lvl = 2
            # a parenthesized integer "(1)" under an open letter clause "(f)"
            # is a sub-item of it, not a new top-level section
            if (
                lvl == 1
                and re.match(r"^\([0-9]{1,2}\)", txt)
                and stack
                and stack[-1][0] == 2
            ):
                lvl = 3
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            pkey = stack[-1][1] if stack else rid
            # a top-level numbered clause is a SIBLING — never a child — of a
            # boundary unit (execution / preamble / field row) or of a
            # non-heading, un-numbered root (a lease preamble that adopted the
            # body): nesting would merge the whole document into one span.
            # Under a real heading root ("1. DEFINITIONS.") the clause stays a
            # child (the golden nests it there too) — and a run-in clause
            # lead-in ("Additional Milestone Terms. ...") likewise owns its
            # enumerated sub-items.
            if stack and stack[-1][0] == 0:
                if pkey != rid:
                    if pkey not in runin_keys:
                        pkey = None  # boundary units never own numbered clauses
                elif weak_root or (
                    not heading_ok(rid)
                    and _section_number((root_ann.get("rawText") or "").strip()) is None
                ):
                    pkey = None
            n_split += 1
            nk = f"{rid}#{n_split}"
            add_node(nk, pkey)["members"].append(a)
            stack.append((lvl, nk))
            if letter and lvl == 2:
                last_letter2 = letter

    # --- staircase fix: a section-numbered unit (heading root OR numbered
    # split-clause) nested under a same-or-shallower-numbered unit is a sibling;
    # promote it up. Processed in reading order so corrections cascade
    # (16 -> parent-of-15 -> parent-of-14; a clause 15 mis-filed under 14 climbs
    # out to be 14's sibling).
    def _node_secnum(nd):
        return (
            _section_number(nd["members"][0].get("rawText")) if nd["members"] else None
        )

    def _node_text(nd):
        return nd["members"][0].get("rawText") if nd["members"] else ""

    for nd in nodes:
        cnum = _node_secnum(nd)
        if cnum is None:
            continue
        pk = nd["parent_key"]
        while pk is not None and pk in node_by_key:
            p = node_by_key[pk]
            if _should_promote(_node_secnum(p), cnum) or _is_recitals_heading(
                _node_text(p)
            ):
                nd["parent_key"] = p["parent_key"]
                pk = nd["parent_key"]
            else:
                break

    # a Notes/footnotes unit annotates the element before it; it is never that
    # element's child (nesting would merge their spans)
    for nd in nodes:
        if nd["parent_key"] is not None and _NOTES_RE.match(_node_text(nd) or ""):
            nd["parent_key"] = None

    # a RUN of >=2 consecutive enumerated pseudo-heading roots directly after
    # a colon lead-in line ("... routed for CSO processing in the following
    # order:" over "1. Katherine Cenicola (Approver)" / "2. ..." / "3. ...")
    # is an approver/routing list: each entry nests under the lead-in's unit.
    # A real numbered section after "as follows:" never forms such a run —
    # its body always sits between the headings.
    node_of_ann = {}
    for nd in nodes:
        for m in nd["members"]:
            node_of_ann[m["id"]] = nd["key"]
    seq = [a for a in anns if a["id"] not in excluded]
    i = 0
    while i < len(seq):
        a = seq[i]
        t = (a.get("rawText") or "").strip()
        if (
            a["annotationLabel"] in _HEADING_LABELS
            and _enum_level(t) is not None
            and a["id"] in node_by_key  # roots its own unit
        ):
            j = i
            while (
                j + 1 < len(seq)
                and seq[j + 1]["annotationLabel"] in _HEADING_LABELS
                and _enum_level((seq[j + 1].get("rawText") or "").strip()) is not None
                and seq[j + 1]["id"] in node_by_key
            ):
                j += 1
            lead = seq[i - 1] if i > 0 else None
            if (
                j > i
                and lead is not None
                and (lead.get("rawText") or "").rstrip().endswith(":")
                and lead["id"] in node_of_ann
            ):
                for k in range(i, j + 1):
                    node_by_key[seq[k]["id"]]["parent_key"] = node_of_ann[lead["id"]]
            i = j + 1
        else:
            i += 1

    # --- emit: annotations + relationships (deterministic su ids) ---
    # Units are FLAT annotations (parent_id is always None): the nested clause
    # hierarchy is carried entirely by OC_PARENT_CHILD relationship edges among
    # the "Semantic Unit"-tagged annotations, and membership by OC_SEMANTIC_UNIT
    # edges. Nothing in this layer relies on the parent_id FK.
    su_by_key = {nd["key"]: f"su-{i}" for i, nd in enumerate(nodes)}
    new_anns, member_edges, child_by_parent = [], [], {}
    for i, nd in enumerate(nodes):
        su_id = su_by_key[nd["key"]]
        parent_su = (
            su_by_key.get(nd["parent_key"]) if nd["parent_key"] is not None else None
        )
        mem = nd["members"]
        unit_text = " ".join((m.get("rawText") or "") for m in mem).strip()
        # union of member modalities (["TEXT"] unless a member carries images)
        unit_mods = sorted(
            {md for m in mem for md in (m.get("content_modalities") or ("TEXT",))}
        ) or ["TEXT"]
        new_anns.append(
            {
                "id": su_id,
                "annotationLabel": _UNIT_LABEL,
                "rawText": unit_text,
                "page": mem[0].get("page", 0) if mem else 0,
                "annotation_json": _union_annotation_json(mem),
                "parent_id": None,  # hierarchy is via relationships, not parent_id
                "annotation_type": "TOKEN_LABEL",
                "structural": True,
                "content_modalities": unit_mods,
            }
        )
        member_edges.append(
            {
                "id": f"surel-{i}",
                "relationshipLabel": _MEMBER_REL,
                "source_annotation_ids": [su_id],
                "target_annotation_ids": [m["id"] for m in mem],
                "structural": True,
            }
        )
        if parent_su is not None:
            child_by_parent.setdefault(parent_su, []).append(su_id)

    export.setdefault("labelled_text", []).extend(new_anns)
    rels = export.setdefault("relationships", [])
    rels.extend(member_edges)
    base = len(nodes)
    for j, (parent_su, kids) in enumerate(child_by_parent.items()):
        rels.append(
            {
                "id": f"surel-{base + j}",
                "relationshipLabel": _PARENT_CHILD_REL,
                "source_annotation_ids": [parent_su],
                "target_annotation_ids": kids,
                "structural": True,
            }
        )
