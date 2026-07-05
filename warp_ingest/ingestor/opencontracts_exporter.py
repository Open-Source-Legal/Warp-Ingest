"""Convert Warp-Ingest parse output into an ``OpenContractDocExport``.

This reads only already-produced engine outputs — the Tika-format **XHTML**
(per-page word boxes) and the finalized **blocks** — and never touches the
layout engine, preserving the XHTML contract the rest of the codebase relies on.

Target format: ``docs/opencontracts_export_format.md``.
Design notes: ``docs/superpowers/specs/2026-06-27-opencontracts-export-design.md``.
"""

import hashlib
import logging
import re
import string
from collections import defaultdict
from difflib import SequenceMatcher

from bs4 import BeautifulSoup

from warp_ingest.ingestor.image_tokens import extract_page_images
from warp_ingest.ingestor.line_parser import stop_words
from warp_ingest.ingestor.semantic_units import append_semantic_units

logger = logging.getLogger(__name__)

# block_type -> annotation label name (get-or-created downstream)
_LABEL_BY_BLOCK_TYPE = {
    "header": "Section Header",
    "header_modified": "Section Header",
    "header_modified_to_para": "Section Header",
    "inline_header": "Section Header",
    "para": "Paragraph",
    "list_item": "List Item",
    "table_row": "Table Row",
}

_POS_SPLIT = "), ("
_EPS = 0.01

_WORD_STRIP = string.punctuation + "’‘“”—–"


def _norm_word(w):
    """Lightweight word key for text alignment: case- and edge-punctuation-folded
    so a token like ``"Numbers."`` matches a ``block_text`` word ``"Numbers"``."""
    return w.strip(_WORD_STRIP).lower()


# Annotation labels that may legitimately parent other annotations. A block whose
# resolved label is not one of these (a Paragraph, List Item, Table Row — or a
# run-in header demoted to Paragraph) is spliced out of the parent chain so the
# hierarchy never has a non-heading bearing children.
_HEADER_ANNOTATION_LABELS = frozenset({"Section Header", "Title"})

# A genuine section header is short. The visual engine sometimes emits a
# ``header`` block for a *run-in* heading — a bolded "6.6 Taxes." lead-in that
# is visually fused with the sentence(s) that follow — and absorbs the whole
# section body into one block. Such a block is body text, not a heading: a
# structure oracle (Docling) reads these spans as paragraphs, and a 100+ word
# "Section Header" is meaningless for retrieval. Above this many words we relabel
# a header-ish block as a paragraph. Empirically (Docling cross-engine study over
# 10 fixtures) header blocks up to ~12 words are still real headings; essentially
# every longer one is mislabeled body/list.
_HEADER_MAX_WORDS = 12

# The word-count rule above misses the *worst* run-in headers: a bold lead-in
# like "6.7 Audits." whose ``block_text`` is only two words but whose annotation
# absorbs the entire clause body via its token box (188 PAWLS tokens for a
# 2-word "header"). The diverse-sample audit found these are the single most
# common cleanly-exporter-fixable defect, so a header whose *assigned token
# count* exceeds this is demoted to Paragraph regardless of its short text — a
# real heading carries only a handful of tokens. (A further class — decorative /
# metadata text the engine mislabeled as a header: filenames, lone connectors,
# parenthetical form captions, watermarks — is also exporter-demotable, but
# de-anchoring it perturbs the table-less Docling head-ancestor oracle beyond its
# margin, so it is deferred to a baseline-regeneration follow-up. See the
# heuristic-audit writeup in docs/.)
_HEADER_MAX_TOKENS = 20


# A ``header`` block whose *entire text is a single non-prose structural token* —
# a page folio, a bare email, or a bare URL — is page furniture, not a section
# heading; labeling it ``Section Header`` (worse) makes it a parent-chain ancestor
# that adopts the page body as children. Demoting it to ``Paragraph`` fixes the
# label *and* the parenting cascade (the tree-consistency rule then splices the
# non-heading out). Page-folio demotion was the 60-page audit's highest-frequency
# clean win (an "F-18"/"iii"/"105" folio splitting a section across a page break).
#
# These are **structural / format** detectors — they recognize a structural
# element (pagination, a contact token) by its universal format, never by the
# *content* (the words/semantics) of any particular document. We deliberately do
# NOT match content phrases (recital openers, signature-field words, "TABLE OF
# CONTENTS", e-signature watermarks) or jurisdiction-specific clerk/filing stamps:
# those are content-based / corpus-specific and were dropped. The general,
# structural way to catch margin furniture (running headers/footers, per-page
# banners) is *geometric + cross-page repetition* — now implemented in
# ``_furniture_demotions`` (a short header repeated verbatim in a page margin
# across many pages), alongside ``_cover_overrides`` (centered pseudo-headers on a
# sparse cover page) and ``_embedded_fragment_demotions`` (a header mis-promoted
# inside a table). All three are content-free (geometry, repetition, raw
# ``block_type`` adjacency) and are the generalizable replacement for the reverted
# EDGARx2 engine rules; see
# docs/superpowers/specs/2026-06-29-edgarx2-hierarchy-structural-rehome-design.md.
# Anchored so genuine short headings ("ARTICLE 9", "WITNESSETH", "Definitions")
# are never caught — see the export tests. Verified neutral vs the Docling oracle.
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
# canonical roman numeral (lookahead forbids the empty match); applied only to
# all-lowercase tokens so real all-caps words made of {i,v,x,l,c,d,m} are safe.
_ROMAN_RE = re.compile(
    r"(?=[ivxlcdm])m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})"
)
# Page folios: bare arabic ("105"), dashed ("- 33 -"), letter-dash financial
# folios ("F-18", "C-17"). NB: the "Page N of M" footer form is deliberately not
# demoted here — it isn't among the 60-page audit's defects and demoting it
# perturbs the Docling head-ancestor oracle on cerebras_ex1013e beyond its margin
# (same baseline-regeneration deferral as the other oracle-sensitive metadata).
# 1-3 digits only: page folios are small; 4-digit numbers are years/amounts, not
# folios (avoids demoting a standalone "2002").
_PAGE_FOLIO_RE = re.compile(r"\d{1,3}|-\s*\d{1,3}\s*-|[A-Za-z]-\d{1,3}")


def _is_nonheading_furniture(text):
    """True if a header block's *entire text* is a single non-prose structural
    token — a page folio, a bare email, or a bare URL. Structural/format only;
    never matches document content/semantics."""
    t = " ".join((text or "").split())
    if not t:
        return False
    low = t.lower()
    if _EMAIL_RE.fullmatch(t) or _URL_RE.fullmatch(t):
        return True
    if _PAGE_FOLIO_RE.fullmatch(low):
        return True
    if t == low and _ROMAN_RE.fullmatch(low):  # lowercase roman folio (i, ii, iii)
        return True
    return False


# Geometric "corner furniture": a header pinned in the page's extreme top/bottom
# margin AND horizontally offset past the page center is a running header/footer,
# page-number, or corner filing stamp — never a section heading (which starts at
# the left text margin or is centered). This is the *structural* (position-only,
# content-free) way to catch the corner stamps the text filter deliberately won't
# name (the FortWorth "CSC No." / "OFFICIAL RECORD" clerk stamps). The 36-doc
# corner scan + per-fixture Docling A/B set the bands: ``left_frac > 0.55`` excludes
# every left-aligned/centered real heading by construction; the *top* band is tight
# (~0.055) so it catches the extreme-top-edge stamps (top ≲0.05) but NOT the
# slightly-lower top-right "Exhibit X.X" identifier line (top ≳0.06) — demoting
# those is defensible but perturbs the small-doc Docling oracle (exyn_ex211), so it
# is left to a baseline-regeneration follow-up. Keeping the band tight makes the
# rule exactly Docling-neutral. See test_top_right_corner_header_demoted_by_geometry.
_MARGIN_TOP_FRAC = 0.055  # block bottom above this frac of page height = top margin
_MARGIN_BOT_FRAC = 0.91  # block top below this frac = bottom margin
_CORNER_LEFT_FRAC = 0.55  # block left edge past this frac of page width = offset
# Deep far-corner footer: a block in the bottom margin AND deep on the right
# (further right than a centered heading can reach) is a corner stamp/footer even
# if it sits slightly above the strict bottom band used for the general case.
# NOTE: the band is kept tight (0.88/0.70). The legal-100 eval surfaced FortWorth
# clerk stamps just above it (bottom-frac 0.84-0.88), but widening the *single-page
# geometric* gate to reach them also demotes genuine bottom-right headings (the
# hetero-100 eval caught "Integrated Reports Inquiries" at bottom 0.88/left 0.73)
# — pure position cannot separate the two. Those stamps repeat across pages and
# are instead the target of the cross-page position-repetition detector
# (_repeated_position_furniture).
_CORNER_DEEP_BOT_FRAC = 0.88
_CORNER_DEEP_LEFT_FRAC = 0.70


def _is_corner_furniture(geom):
    """geom = (top_frac, bottom_frac, left_frac) of a block's token box vs the page,
    or None. True when the box is in a top/bottom margin band and right-offset."""
    if geom is None:
        return False
    topf, botf, lf = geom
    in_band = topf < _MARGIN_TOP_FRAC or botf > _MARGIN_BOT_FRAC
    if in_band and lf > _CORNER_LEFT_FRAC:
        return True
    return botf > _CORNER_DEEP_BOT_FRAC and lf > _CORNER_DEEP_LEFT_FRAC


# A heading is a label/noun-phrase, never a sentence or clause. These tells are
# content-free and structural; the recital-trigger set is the one (tunable)
# generic-keyword backstop (default = universal legal boilerplate openers).
_DEFAULT_RECITAL_TRIGGERS = frozenset({"whereas", "now therefore"})
# Coordinating conjunctions (FANBOYS): a heading never ends on one.
_PROSE_END_CONJUNCTIONS = frozenset({"and", "or", "nor", "but", "for", "yet", "so"})


def _first_alpha_word(text):
    """First whitespace token whose first char is a letter (skips leading
    numbers/punctuation like '52004' or '(a)'); '' if none."""
    for tok in text.split():
        s = tok.strip(string.punctuation)
        if s and s[0].isalpha():
            return s
    return ""


def _has_lowercase_content_word(text):
    """True if any whitespace token (stripped of surrounding punctuation) has a
    lowercase first letter AND its lowercased form is not a stopword — i.e. a
    genuine lowercase *content* word, the signature of real prose. A Title-Case
    name ("Subsidiaries of Exyn Technologies, Inc.") has no such word: its only
    lowercase token ("of") is a stopword."""
    for tok in (text or "").split():
        s = tok.strip(string.punctuation)
        if s and s[0].islower() and s.lower() not in stop_words:
            return True
    return False


def _is_prose_header(text, triggers=_DEFAULT_RECITAL_TRIGGERS):
    """True if a header-resolved block's text is a sentence/clause, not a label."""
    t = " ".join((text or "").split())
    if not t:
        return False
    low = t.lower()
    words = low.split()
    first = words[0].strip(string.punctuation)
    first2 = " ".join(w.strip(string.punctuation) for w in words[:2])
    if first in triggers or first2 in triggers:
        return True
    if t[-1] in ",;":  # clause continuation
        return True
    alpha = _first_alpha_word(t)
    if alpha and alpha[0].islower() and alpha.isalpha():  # mid-sentence continuation
        return True
    # a sentence: ends with a period and carries a genuine lowercase content word.
    # The content-word test (not a bare comma) keeps Title-Case names with an
    # "Inc."-style trailing period ("Subsidiaries of Exyn Technologies, Inc.")
    # from being mistaken for prose.
    if t[-1] == "." and len(words) >= 5 and _has_lowercase_content_word(t):
        return True
    if words[-1].strip(string.punctuation) in _PROSE_END_CONJUNCTIONS:
        return True
    return False


# A cross-reference to a numbered provision of another instrument — "Section 201
# of the … Act", "Article 4 of the Agreement" — is body text, not a section
# heading. The structural tell is a section word + an (arabic) enumerator that is
# *immediately* followed by "of": a reference construction. A genuine heading puts
# a title after its enumerator ("Section 5. Effectiveness", "6.6 Taxes", "ARTICLE
# II. TERM OF AGREEMENT"), never "of" straight after the number. This matters once
# a run-in citation's box is correct (its token count no longer trips the
# absorbed-body rule): universal legal-citation structure, never corpus vocabulary.
_XREF_HEADER_RE = re.compile(
    r"^(section|sec|article|art|title|rule|clause|paragraph|para|subsection|"
    r"schedule|exhibit|annex|appendix)\.?\s+\d[\d.()a-z-]*\s+of\s+",
    re.I,
)


def _is_cross_reference_header(text):
    """True if a header-resolved block is a numbered cross-reference to another
    instrument ("Section 201 of the … Act") rather than a section heading."""
    return bool(_XREF_HEADER_RE.match(" ".join((text or "").split())))


# Raw engine block types that map to a heading label (see _LABEL_BY_BLOCK_TYPE).
# The structural-correction pass keys on the RAW block_type, not the resolved
# label, so it has no ordering dependency on _resolve_label.
_HEADERISH_BLOCK_TYPES = frozenset(
    {"header", "header_modified", "header_modified_to_para", "inline_header"}
)


def _norm_text(text):
    return " ".join((text or "").split()).lower()


# Cross-page repeated furniture: a short header whose text repeats verbatim in a
# page margin across many pages is a running header/footer or a per-page banner
# ("TABLE OF CONTENTS" repeated on every page), never a section heading. This is
# the structural margin-furniture detector the module docstring named as deferred
# work. Verbatim cross-page repetition is the gate; a one-off real heading (or a
# real TOC heading appearing once) is never caught. The band is looser than
# _is_corner_furniture because repetition itself is strong evidence.
_FURNITURE_MAX_WORDS = 6
_FURNITURE_MIN_PAGE_FRAC = 0.04
_FURNITURE_TOP_FRAC = 0.12
_FURNITURE_BOT_FRAC = 0.88


def _furniture_demotions(blocks, geom_by_idx, page_count):
    """block_idx set of header-ish blocks demoted as repeated margin furniture."""
    if page_count < 3:
        return set()
    min_pages = max(3, round(_FURNITURE_MIN_PAGE_FRAC * page_count))
    by_text = defaultdict(list)
    for b in blocks:
        if b["block_type"] not in _HEADERISH_BLOCK_TYPES:
            continue
        t = _norm_text(b["block_text"])
        if not t or len(t.split()) > _FURNITURE_MAX_WORDS:
            continue
        geom = geom_by_idx.get(b["block_idx"])
        if geom is None:
            continue
        topf, botf, lf = geom
        if (
            topf < _FURNITURE_TOP_FRAC
            or botf > _FURNITURE_BOT_FRAC
            or lf > _CORNER_LEFT_FRAC
        ):
            by_text[t].append(b)
    demote = set()
    for repeated in by_text.values():
        if len({b["page_idx"] for b in repeated}) >= min_pages:
            demote.update(b["block_idx"] for b in repeated)
    return demote


# Cross-page repeated *position* furniture: a header-ish block pinned in the
# bottom-right corner at the SAME position across >=2 pages is a clerk/filing
# stamp or running corner mark. The verbatim-text detector above misses these on
# **scanned** docs because OCR re-reads the stamp differently each page
# ("OFFICIAL RECORD" / "FFICIAL RECORD" / "RECORD CITY") — so we key on repeated
# *geometry*, not text. Surfaced by the legal-100 eval (FortWorth clerk stamps).
# Strictly position-repetition + bottom-right band: a one-off real bottom-right
# heading (the hetero-100 "Integrated Reports Inquiries" the single-page geometric
# rule would wrongly demote) is never caught, because it does not repeat. Only the
# bottom band is used (the top-right "Exhibit X.X" identifier is deliberately left
# alone — demoting it perturbs the small-doc Docling oracle; a baseline-regen
# follow-up).
_REPEAT_POS_MIN_PAGES = 2
_REPEAT_POS_BOT_FRAC = 0.84  # bottom-margin band (above the strict deep-corner band)
_REPEAT_POS_LEFT_FRAC = 0.62  # right-offset, past page center
_REPEAT_POS_GRID = 0.06  # position quantization: boxes within ~6% are "the same spot"


def _repeated_position_furniture(blocks, geom_by_idx, page_count):
    """block_idx set of header-ish blocks demoted as repeated bottom-right stamps."""
    if page_count < 2:
        return set()
    by_pos = defaultdict(list)
    for b in blocks:
        if b["block_type"] not in _HEADERISH_BLOCK_TYPES:
            continue
        geom = geom_by_idx.get(b["block_idx"])
        if geom is None:
            continue
        _topf, botf, lf = geom
        if botf > _REPEAT_POS_BOT_FRAC and lf > _REPEAT_POS_LEFT_FRAC:
            key = (round(lf / _REPEAT_POS_GRID), round(botf / _REPEAT_POS_GRID))
            by_pos[key].append(b)
    demote = set()
    for grp in by_pos.values():
        if len({b["page_idx"] for b in grp}) >= _REPEAT_POS_MIN_PAGES:
            demote.update(b["block_idx"] for b in grp)
    return demote


# Sparse-cover front-matter: on an early, sparse page (a title/cover page),
# centered+isolated pseudo-headers (a company name, "Form ...", a date) are not
# section headings — but the single most prominent one is the document Title.
# Demote the rest so they do not parent the body; relabel the most prominent as
# Title (an allowed parent) so a real title still anchors the tree. Generalizes
# to any cover/title page. Requires >=2 centered candidates (a genuine cover
# stack) so a lone centered heading is never spuriously promoted.
#
# _COVER_MAX_BLOCKS is LOAD-BEARING, not a free tunable: a dense first page is
# deliberately NOT treated as a cover. Real SEC cover pages are dense (~24 blocks)
# and structurally indistinguishable from dense first pages of contracts/statutes
# that carry legitimate centered headings — verified: fw_nctcog page 0 centers
# "INTERLOCAL AGREEMENT FOR" + "WITNESSETH"; USC Title 1 page 0 centers real
# statutory headings ("REPEALS", "WRITS OF ERROR"). Relaxing the gate to catch the
# dense SEC boilerplate ("FORM 8-K", the SEC name) would demote those real
# headings. The boilerplate can only be told from a real heading by *content*, so
# dense covers are left alone (prefer-structural-not-content-rules). This detector
# therefore fires only on genuinely sparse title/cover pages.
_COVER_MAX_BLOCKS = 12
_COVER_MIN_CANDIDATES = 2


def _cover_overrides(blocks, blocks_by_page, block_tokens, page_box, pawls):
    """{block_idx: "Title"|"Paragraph"} for centered pseudo-headers on cover pages."""
    block_by_idx = {b["block_idx"]: b for b in blocks}
    # cover pages: page 0 if sparse; page 1 only if page 0 is also a sparse cover
    cover_pages = []
    ids0 = blocks_by_page.get(0, [])
    if ids0 and len(ids0) <= _COVER_MAX_BLOCKS:
        cover_pages.append(0)
        ids1 = blocks_by_page.get(1, [])
        if ids1 and len(ids1) <= _COVER_MAX_BLOCKS:
            cover_pages.append(1)

    overrides = {}
    for pidx in cover_pages:
        page_tokens = pawls[pidx]["tokens"]
        page_width = page_box[pidx][0]
        candidates = []  # (block_idx, median_token_height, top_y)
        for bidx in blocks_by_page.get(pidx, []):
            if block_by_idx[bidx]["block_type"] not in _HEADERISH_BLOCK_TYPES:
                continue
            toks = block_tokens.get(bidx, [])
            if not toks:
                continue
            if not _is_centered_heading_line(toks, page_tokens, page_width):
                continue
            heights = sorted(page_tokens[t]["height"] for t in toks)
            top_y = min(page_tokens[t]["y"] for t in toks)
            candidates.append((bidx, heights[len(heights) // 2], top_y))
        if len(candidates) < _COVER_MIN_CANDIDATES:
            continue
        title_bidx = max(candidates, key=lambda c: (c[1], -c[2]))[0]
        for bidx, _h, _y in candidates:
            overrides[bidx] = "Title" if bidx == title_bidx else "Paragraph"
    return overrides


def _tok_box(toks, page_tokens):
    """(left, top, right, bottom) union box of a token-index list."""
    xs0 = [page_tokens[t]["x"] for t in toks]
    xs1 = [page_tokens[t]["x"] + page_tokens[t]["width"] for t in toks]
    ys0 = [page_tokens[t]["y"] for t in toks]
    ys1 = [page_tokens[t]["y"] + page_tokens[t]["height"] for t in toks]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _embedded_fragment_demotions(blocks, blocks_by_page, block_tokens, pawls):
    """block_idx set of header blocks interleaved INSIDE a table-row run.

    A real heading never sits inside a table body: if a header block's immediate
    previous and next blocks on the page (reading order) are both Table Rows, and
    its token box is within their combined column span, it is a mis-promoted table
    fragment (e.g. a fee-table cell). A heading directly above/below a table (a
    Table Row only on one side) is NOT embedded and is kept.

    Only demote a fragment that has a heading ancestor (non-empty ``level_chain``):
    its children then climb to that existing heading instead of being orphaned into
    non-heading roots. A top-level pseudo-header inside a table (no heading above)
    is left alone — demoting it could only orphan its subtree.
    """
    block_by_idx = {b["block_idx"]: b for b in blocks}
    demote = set()
    for pidx, ids in blocks_by_page.items():
        ordered = sorted(ids)
        page_tokens = pawls[pidx]["tokens"]
        for i in range(1, len(ordered) - 1):
            bidx = ordered[i]
            if block_by_idx[bidx]["block_type"] not in _HEADERISH_BLOCK_TYPES:
                continue
            if not block_by_idx[bidx].get("level_chain"):
                continue
            prev_b, next_b = ordered[i - 1], ordered[i + 1]
            if (
                block_by_idx[prev_b]["block_type"] != "table_row"
                or block_by_idx[next_b]["block_type"] != "table_row"
            ):
                continue
            b_toks = block_tokens.get(bidx, [])
            prev_toks = block_tokens.get(prev_b, [])
            next_toks = block_tokens.get(next_b, [])
            if not (b_toks and prev_toks and next_toks):
                continue
            bl, _bt, br, _bb = _tok_box(b_toks, page_tokens)
            pl, _pt, pr, _pb = _tok_box(prev_toks, page_tokens)
            nl, _nt, nr, _nb = _tok_box(next_toks, page_tokens)
            region_left, region_right = min(pl, nl), max(pr, nr)
            if bl >= region_left - _EPS and br <= region_right + _EPS:
                demote.add(bidx)
    return demote


def _structural_corrections(
    blocks, blocks_by_page, block_tokens, geom_by_idx, page_box, pawls
):
    """Content-free structural label overrides computed once over the whole doc.

    Returns {block_idx: "Paragraph" | "Title"}. Keys on geometry, cross-page
    repetition, and raw block_type only — never document content. Demotion takes
    precedence over Title promotion. Extended by later detectors.
    """
    page_count = len(pawls)
    overrides = {}
    overrides.update(
        _cover_overrides(blocks, blocks_by_page, block_tokens, page_box, pawls)
    )
    demote = set()
    demote |= _furniture_demotions(blocks, geom_by_idx, page_count)
    demote |= _repeated_position_furniture(blocks, geom_by_idx, page_count)
    demote |= _embedded_fragment_demotions(blocks, blocks_by_page, block_tokens, pawls)
    for bidx in demote:
        overrides[bidx] = "Paragraph"  # demotion wins over Title promotion
    return overrides


def _resolve_label(
    block, ntok=0, geom=None, recital_triggers=_DEFAULT_RECITAL_TRIGGERS
):
    """block_type -> annotation label, demoting header-ish blocks that are really
    body: over-long text, a run-in lead-in that absorbed the clause body (seen via
    its assigned token count), page furniture / processing metadata (by text), a
    geometric corner-furniture block (by position), or a sentence/clause opener
    (by structural prose tells + tunable trigger set)."""
    label = _LABEL_BY_BLOCK_TYPE.get(
        block["block_type"],
        block["block_type"].replace("_", " ").title(),
    )
    if label == "Section Header" and (
        len(block["block_text"].split()) > _HEADER_MAX_WORDS
        or ntok > _HEADER_MAX_TOKENS
        or _is_nonheading_furniture(block["block_text"])
        or _is_corner_furniture(geom)
        or _is_prose_header(block["block_text"], recital_triggers)
        or _is_cross_reference_header(block["block_text"])
    ):
        return "Paragraph"
    return label


def _assign_page_tokens(token_texts, ordered_block_ids, block_words, rects, centers):
    """Assign each page token to one block: geometry first, text only to *recover*
    tokens geometry drops.

    Pure geometry ("smallest block rect containing the token center") silently
    loses every token whose center lands inside *no* block rect — the wrapped tail
    of a table cell whose ``box_style`` undercovers its text, for instance, which
    is how a table page ends up ~50% uncovered. Those uncovered tokens are the
    only ones text alignment decides here: the page's token-text stream is aligned
    to the blocks' concatenated word streams (``difflib``) and an uncovered token
    is given the block its text matches. Tokens that already sit inside one (or
    more) block rects keep the geometric assignment, so the well-behaved and the
    pathological single-column layouts the regression baselines were captured on
    are untouched. Returns a list ``owner[token_index] -> block_idx | None``.
    """
    n = len(token_texts)
    owner = [None] * n
    candidates = [
        (bid, rects[bid][0], rects[bid][1], rects[bid][2], rects[bid][3], rects[bid][5])
        for bid in ordered_block_ids
    ]

    # geometric assignment: smallest rect whose box contains the token center.
    any_uncovered = False
    for ti, (cx, cy) in enumerate(centers):
        best, best_area = None, float("inf")
        for bid, top, left, right, bottom, area in candidates:
            if (
                left - _EPS <= cx <= right + _EPS
                and top - _EPS <= cy <= bottom + _EPS
                and area < best_area
            ):
                best, best_area = bid, area
        owner[ti] = best
        any_uncovered = any_uncovered or best is None

    # recover uncovered tokens (in no rect) by aligning their text to a block.
    if any_uncovered:
        b_words, b_owner = [], []
        for bid in ordered_block_ids:
            for w in block_words[bid]:
                b_words.append(w)
                b_owner.append(bid)
        if b_words:
            sm = SequenceMatcher(None, token_texts, b_words, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    continue
                for k in range(i2 - i1):
                    ti = i1 + k
                    if owner[ti] is None:  # only ever fill, never reassign
                        owner[ti] = b_owner[j1 + k]
    return owner


def _runin_overflow_tokens(cur_tokens, token_texts, raw):
    """Trailing tokens of ``cur_tokens`` (reading order) beyond what ``raw`` needs.

    A run-in header's line-box covers the heading *and* the body that follows it on
    the same line, so the heading block is assigned ``[heading tokens][body tokens]``
    while its ``block_text`` is only the heading. Aligning the token texts to ``raw``
    finds the last token that belongs to ``raw``; everything after it is overflow
    that structurally belongs to a later block. Returns ``[]`` when the block's
    tokens already end where its text does (the common, faithful case) — so this is
    a no-op for every block that geometry got right.
    """
    if not cur_tokens or not raw:
        return []
    av_txt = [_norm_word(token_texts[ti]) for ti in cur_tokens]
    raw_norm = [_norm_word(w) for w in raw]
    last_matched = -1
    matched_raw = 0
    sm = SequenceMatcher(None, av_txt, raw_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            last_matched = max(last_matched, i2 - 1)
            matched_raw += i2 - i1
    if last_matched < 0 or last_matched >= len(cur_tokens) - 1:
        return []  # nothing trails the block's own text
    # Require the block to actually *cover* its text before we treat the tail as
    # foreign: if most of ``raw`` is unmatched the mismatch is tokenisation noise
    # (OCR word-gluing, a table cell), not a run-in — leave it to geometry.
    if matched_raw < 0.75 * len(raw):
        return []
    return cur_tokens[last_matched + 1 :]


def _reconcile_runin_overflow(
    owner, token_texts, ordered_block_ids, block_words, header_donors=None
):
    """Repair run-in token collisions by donating a header's overflow to the block
    it belongs to, guaranteeing per-block text faithfulness.

    Geometric assignment gives every token to the smallest rect containing its
    center. When a heading and the body that follows it share one physical line (a
    *run-in* header — "SECTION 13. …", "6.6 Taxes. Each Party…", ubiquitous in
    numbered contracts), the heading's line-box covers both, so the heading absorbs
    the body's tokens and the body block is left starved (sometimes with none). The
    block *texts* are already correctly segmented, so we use them as the authority:
    a header's trailing tokens that its own ``block_text`` does not claim, and that
    the *next* block's ``block_text`` begins with, are moved to that next block.

    Deliberately conservative — a token moves only when **all** of these hold, so on
    every clean or merely-noisy page (tables, OCR, multi-column enterprise reports)
    this is a strict no-op and the regression baselines are untouched:
      * the donor is a **header** block (``header_donors``) — the reported bug is a
        heading swallowing its paragraph; restricting to headers keeps dense table
        / footnote layouts, where wrapped cells legitimately span rows, untouched,
      * the donor covers ≥75% of its own text (it is a real block, not OCR mush),
      * the donated run is a contiguous *tail* of the donor (a genuine run-in), and
      * that run matches the *start* of the immediately-following block, which is
        currently short of tokens.
    No token is ever dropped or double-assigned: a token either stays put or moves
    to exactly one adjacent block.
    """
    if header_donors is None:  # restrict to nothing -> no-op (defensive default)
        header_donors = frozenset()
    own = defaultdict(list)  # block_idx -> [token_index] in reading order
    for ti, bid in enumerate(owner):
        if bid is not None:
            own[bid].append(ti)

    new = list(owner)
    for i in range(len(ordered_block_ids) - 1):
        bid = ordered_block_ids[i]
        if bid not in header_donors:
            continue
        nxt = ordered_block_ids[i + 1]
        cur = [ti for ti in own.get(bid, ()) if new[ti] == bid]
        overflow = _runin_overflow_tokens(cur, token_texts, block_words.get(bid, ()))
        if not overflow:
            continue
        nxt_raw = [_norm_word(w) for w in block_words.get(nxt, ())]
        if not nxt_raw:
            continue
        nxt_have = [ti for ti in own.get(nxt, ()) if new[ti] == nxt]
        # Don't feed a block that already holds (at least) its whole text — that
        # would duplicate words; the run-in victim is always *short* of tokens.
        if len(nxt_have) >= len(nxt_raw):
            continue
        # How much of the next block's text does it already cover from the start?
        # Its own tokens are the wrapped tail, so they align to ``nxt_raw`` at some
        # offset ``first_covered`` > 0; the block is missing ``nxt_raw[:first_covered]``.
        # (A first_covered of 0 means it already starts at its own text — skip; this
        # is checked by alignment, not a single token, so a mid-line word that merely
        # *looks* like the opening — "(the" vs "The" — cannot spoof it.)
        first_covered = len(nxt_raw)
        if nxt_have:
            nxt_have_txt = [_norm_word(token_texts[ti]) for ti in nxt_have]
            sm = SequenceMatcher(None, nxt_have_txt, nxt_raw, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    first_covered = min(first_covered, j1)
        if first_covered == 0:
            continue
        # Donate the overflow prefix that reproduces exactly the missing opening
        # ``nxt_raw[:first_covered]`` — never past where the block's own tokens
        # resume, so a token is never assigned to two blocks.
        donated = []
        for j, ti in enumerate(overflow):
            if j < first_covered and _norm_word(token_texts[ti]) == nxt_raw[j]:
                donated.append(ti)
            else:
                break
        if not donated:
            continue
        for ti in donated:
            new[ti] = nxt
            own[nxt].append(ti)
        own[nxt].sort()
    return new


# A block's assigned tokens should reconstruct its own ``block_text``. Above this
# ratio of tokens-to-words (with a meaningful absolute excess) a block has almost
# certainly swallowed a neighbour's text — the collision this module repairs. Post
# repair it should be silent on clean input; a warning means an unhandled variant.
_OVERSPAN_TOKEN_RATIO = 3.0
_OVERSPAN_MIN_EXCESS = 15


def _warn_unfaithful_blocks(blocks, block_text_by_idx, block_tokens):
    """Log (never raise) blocks whose emitted tokens clearly don't match their
    ``block_text`` — a block that absorbed a neighbour's tokens, or a non-trivial
    block left with none. Cheap observability for a failure mode that was
    previously silent; OCR word-gluing can trip it, so it is advisory only."""
    if not logger.isEnabledFor(logging.WARNING):
        return
    for b in blocks:
        bidx = b["block_idx"]
        n_words = len(block_text_by_idx[bidx].split())
        n_tok = len(block_tokens.get(bidx, ()))
        if n_words == 0:
            continue
        if n_tok == 0:
            logger.warning(
                "OpenContracts export: block %s (%r) has text but no assigned "
                "tokens — its span is ungroundable.",
                bidx,
                block_text_by_idx[bidx][:60],
            )
        elif (
            n_tok > n_words * _OVERSPAN_TOKEN_RATIO
            and n_tok - n_words >= _OVERSPAN_MIN_EXCESS
        ):
            logger.warning(
                "OpenContracts export: block %s (%r) was assigned %d tokens for "
                "%d words — it likely absorbed a neighbouring block's text.",
                bidx,
                block_text_by_idx[bidx][:60],
                n_tok,
                n_words,
            )


# Re-segmentation thresholds. visual_ingestor sometimes fuses a centered heading
# line (a title, "RECITALS", "General Provisions:") into the body paragraph that
# follows it, emitting one block where there should be a heading + a body — the
# diverse-sample audit's single most frequent defect class. The per-visual-line
# geometry that distinguishes a centered heading from justified body survives in
# the token boxes, so the exporter re-splits such a (non-header) block: tokens are
# clustered into lines by vertical position; a line that is narrow and has
# symmetric side margins is a centered heading; maximal runs of heading vs body
# lines become separate annotations. This recovers structure the engine
# over-merged without touching it.
_CENTER_NARROW_FRAC = 0.6  # a heading line spans < this fraction of page width
_CENTER_MARGIN_FRAC = 0.12  # both side margins exceed this fraction of page width
_CENTER_SYM_FRAC = 0.12  # |left margin - right margin| under this fraction


def _cluster_lines(tok_idxs, page_tokens):
    """Group a block's tokens into visual lines by vertical position (reading order)."""
    items = sorted((page_tokens[t]["y"], page_tokens[t]["x"], t) for t in tok_idxs)
    lines, cur, cur_y = [], [], None
    for y, _x, t in items:
        h = page_tokens[t]["height"] or 1.0
        if cur_y is None or abs(y - cur_y) <= max(2.0, 0.6 * h):
            cur.append(t)
        else:
            lines.append(cur)
            cur = [t]
        cur_y = y
    if cur:
        lines.append(cur)
    return lines


def _is_centered_heading_line(
    line_toks, page_tokens, page_width, recital_triggers=_DEFAULT_RECITAL_TRIGGERS
):
    # furniture (a centered email/recital/address/folio) is never a heading, even
    # when geometrically centered — keeps the split path from re-promoting what the
    # label filter demotes.
    text = " ".join(
        page_tokens[t]["text"]
        for t in sorted(line_toks, key=lambda t: page_tokens[t]["x"])
    )
    if _is_nonheading_furniture(text):
        return False
    # a prose/recital line ("WHEREAS, the current renewal term expires …; and") is
    # never a heading either, even when geometrically centered — without this, the
    # split path re-promotes the very recital lines _resolve_label already demoted.
    if _is_prose_header(text, recital_triggers):
        return False
    left = min(page_tokens[t]["x"] for t in line_toks)
    right = max(page_tokens[t]["x"] + page_tokens[t]["width"] for t in line_toks)
    left_margin, right_margin = left, page_width - right
    return (
        (right - left) < _CENTER_NARROW_FRAC * page_width
        and left_margin > _CENTER_MARGIN_FRAC * page_width
        and right_margin > _CENTER_MARGIN_FRAC * page_width
        and abs(left_margin - right_margin) < _CENTER_SYM_FRAC * page_width
        and len(line_toks) <= _HEADER_MAX_WORDS
    )


def _segment_block(
    tok_idxs, page_tokens, page_width, recital_triggers=_DEFAULT_RECITAL_TRIGGERS
):
    """Split a fused non-header block into a leading centered heading + its body.

    Returns ``[{'tokens': [idx...], 'heading': bool}]`` — a single element (no
    split) unless the block *begins* with a run of centered heading line(s)
    followed by body. Only a *leading* heading is split (the dominant
    heading-fused-into-body case); the heading run must be short enough to be a
    real heading (≤ ``_HEADER_MAX_WORDS`` words) or the block is left whole, which
    keeps the split precise and never manufactures an over-long "Section Header".
    """
    tok_idxs = list(tok_idxs)
    if len(tok_idxs) < 2:
        return [{"tokens": tok_idxs, "heading": False}]
    lines = _cluster_lines(tok_idxs, page_tokens)
    flags = [
        _is_centered_heading_line(ln, page_tokens, page_width, recital_triggers)
        for ln in lines
    ]
    if not flags[0]:  # block doesn't start with a centered heading
        return [{"tokens": tok_idxs, "heading": False}]
    j = 0
    while j < len(lines) and flags[j]:
        j += 1
    if j == len(lines):  # all heading, no body to separate
        return [{"tokens": tok_idxs, "heading": False}]
    head_tokens = sorted(t for k in range(0, j) for t in lines[k])
    if len(head_tokens) > _HEADER_MAX_WORDS:  # too long to be a real heading
        return [{"tokens": tok_idxs, "heading": False}]
    body_tokens = sorted(t for k in range(j, len(lines)) for t in lines[k])
    return [
        {"tokens": head_tokens, "heading": True},
        {"tokens": body_tokens, "heading": False},
    ]


# Relationship label for explicit parent→child structural edges. OpenContracts
# honors this convention in its subtree-group walker alongside the parent FK.
_PARENT_CHILD_LABEL = "OC_PARENT_CHILD"


def _list_intro_parents(blocks, label_by_idx):
    """Map list-item block_idx -> id of the lead-in paragraph that introduces it.

    A contiguous run of ``List Item`` blocks whose immediately-preceding block is
    a colon-terminated ``Paragraph`` ("… as follows:") is a child of that
    paragraph — the classic "intro: A. … B. …" structure the level-chain (which
    only tracks headings) cannot express. Returns ``{block_idx: parent_id_str}``.
    """
    out = {}
    n = len(blocks)
    i = 0
    while i < n:
        if label_by_idx.get(blocks[i]["block_idx"]) != "List Item":
            i += 1
            continue
        run_start = i
        while i < n and label_by_idx.get(blocks[i]["block_idx"]) == "List Item":
            i += 1
        if run_start == 0:
            continue
        intro = blocks[run_start - 1]
        intro_idx = intro["block_idx"]
        if label_by_idx.get(intro_idx) == "Paragraph" and (
            intro["block_text"].rstrip().endswith(":")
        ):
            for k in range(run_start, i):
                out[blocks[k]["block_idx"]] = str(intro_idx)
    return out


def _parent_child_relationships(labelled_text):
    """Emit one ``OC_PARENT_CHILD`` relationship per parent: source = the parent,
    targets = its direct children (reading order). Mirrors the ``parent_id`` tree
    as explicit annotation-to-annotation edges (OpenContracts honors both)."""
    ids = {a["id"] for a in labelled_text}
    children = defaultdict(list)
    for a in labelled_text:
        pid = a["parent_id"]
        if pid is not None and pid in ids:
            children[pid].append(a["id"])
    relationships = []
    for a in labelled_text:  # stable: parents in reading order
        kids = children.get(a["id"])
        if kids:
            relationships.append(
                {
                    "id": f"rel-{a['id']}",
                    "relationshipLabel": _PARENT_CHILD_LABEL,
                    "source_annotation_ids": [a["id"]],
                    "target_annotation_ids": kids,
                    "structural": True,
                }
            )
    return relationships


# Image attach gate: an image whose box lies at least this fraction inside an
# existing annotation's bounds joins that annotation (["IMAGE","TEXT"]) rather
# than becoming a standalone "Image" annotation. Mirrors the smallest-rect rule
# used for text-token assignment.
_IMG_ATTACH_FRAC = 0.9


def _append_image_layer(pawls, labelled_text, images_by_page):
    """Append image tokens + ``Image`` annotations (design spec 2026-07-03).

    Tokens are appended after each page's text tokens so existing indices never
    shift. An image ≥ ``_IMG_ATTACH_FRAC`` inside an existing annotation's
    bounds joins that annotation; otherwise it becomes a standalone ``Image``
    annotation parented like a text block at its position (last preceding
    annotation by top-y: a heading parents it, anything else lends its parent).
    """
    anns_by_page = defaultdict(list)
    for order, a in enumerate(labelled_text):
        anns_by_page[a["page"]].append((order, a))

    for pidx, images in sorted(images_by_page.items()):
        if not (0 <= pidx < len(pawls)):
            continue
        page_tokens = pawls[pidx]["tokens"]
        for k, tok in enumerate(images):
            tidx = len(page_tokens)
            page_tokens.append(tok)
            ref = {"pageIndex": pidx, "tokenIndex": tidx}
            il, it_ = tok["x"], tok["y"]
            ir, ib = tok["x"] + tok["width"], tok["y"] + tok["height"]
            area = max(_EPS, (ir - il) * (ib - it_))

            host, host_area = None, float("inf")
            for _, a in anns_by_page[pidx]:
                pj = a["annotation_json"].get(str(pidx))
                if pj is None:
                    continue
                b = pj["bounds"]
                ov_w = min(ir, b["right"]) - max(il, b["left"])
                ov_h = min(ib, b["bottom"]) - max(it_, b["top"])
                if ov_w <= 0 or ov_h <= 0:
                    continue
                if ov_w * ov_h < _IMG_ATTACH_FRAC * area:
                    continue
                b_area = max(_EPS, (b["right"] - b["left"]) * (b["bottom"] - b["top"]))
                if b_area < host_area:
                    host, host_area = a, b_area

            if host is not None:
                pj = host["annotation_json"][str(pidx)]
                pj["tokensJsons"].append(ref)
                b = pj["bounds"]
                b["left"], b["top"] = min(b["left"], il), min(b["top"], it_)
                b["right"] = max(b["right"], ir)
                b["bottom"] = max(b["bottom"], ib)
                host["content_modalities"] = sorted(
                    {*(host.get("content_modalities") or ()), "IMAGE", "TEXT"}
                )
                continue

            parent = None
            preceding = []
            for order, a in anns_by_page[pidx]:
                pj = a["annotation_json"].get(str(pidx))
                if pj is not None and pj["bounds"]["top"] <= it_:
                    preceding.append((pj["bounds"]["top"], order, a))
            if preceding:
                prev = max(preceding, key=lambda p: (p[0], p[1]))[2]
                if prev["annotationLabel"] in _HEADER_ANNOTATION_LABELS:
                    parent = prev["id"]
                else:
                    parent = prev["parent_id"]

            labelled_text.append(
                {
                    "id": f"img-{pidx}-{k}",
                    "annotationLabel": "Image",
                    "rawText": "",
                    "page": pidx,
                    "annotation_json": {
                        str(pidx): {
                            "bounds": {
                                "top": it_,
                                "left": il,
                                "right": ir,
                                "bottom": ib,
                            },
                            "tokensJsons": [ref],
                            "rawText": "",
                        }
                    },
                    "parent_id": parent,
                    "annotation_type": "TOKEN_LABEL",
                    "structural": True,
                    "content_modalities": ["IMAGE"],
                }
            )


class ExportValidationError(ValueError):
    """Raised when an export violates an OpenContractDocExport invariant (spec §6)."""


# --------------------------------------------------------------------------- #
# XHTML / style parsing
# --------------------------------------------------------------------------- #
def _style_kv(style_str):
    out = {}
    for part in style_str.split(";"):
        key, sep, val = part.partition(":")
        if sep:
            out[key.strip()] = val.strip()
    return out


def _parse_positions(raw):
    """'[(72,100), (114,100)]' -> [(72.0, 100.0), (114.0, 100.0)]."""
    inner = raw.strip()[2:-2] if raw.strip().startswith("[(") else ""
    if not inner:
        return []
    pairs = []
    for chunk in inner.split(_POS_SPLIT):
        nums = chunk.split(",")
        pairs.append((float(nums[0]), float(nums[1])))
    return pairs


def _parse_px(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val[:-2]) if val.endswith("px") else float(val)
    except (ValueError, TypeError):
        return default


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _page_dims(page_div):
    style = _style_kv(page_div.get("style", ""))
    return _parse_px(style.get("width"), 612.0), _parse_px(style.get("height"), 792.0)


def _page_tokens(page_div, width, height):
    """Build word-level PAWLS tokens for one page from its <p> word boxes."""
    tokens = []
    for p in page_div.find_all("p"):
        style = _style_kv(p.get("style", ""))
        starts = _parse_positions(style.get("word-start-positions", ""))
        ends = _parse_positions(style.get("word-end-positions", ""))
        if not starts:
            continue
        words = p.get_text().split()
        line_h = _parse_px(style.get("height")) or _parse_px(style.get("font-size"))
        for i, text in enumerate(words):
            # contract: one (x,y) per word; degrade gracefully if it ever drifts
            sx, sy = starts[i] if i < len(starts) else starts[-1]
            ex = ends[i][0] if i < len(ends) else sx
            x = _clamp(sx, 0.0, width)
            y = _clamp(sy, 0.0, height)
            w = _clamp(max(0.0, ex - sx), 0.0, width - x)
            h = _clamp(line_h, 0.0, height - y)
            tokens.append({"x": x, "y": y, "width": w, "height": h, "text": text})
    return tokens


def _title_from_xhtml(soup):
    for meta in soup.find_all("meta"):
        name = meta.get("name", "")
        if name.endswith(":title") and meta.get("content"):
            return meta["content"]
    return ""


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def to_opencontracts_export(
    xhtml,
    blocks,
    *,
    title=None,
    file_type="application/pdf",
    pdf_bytes=None,
    recital_triggers=None,
    semantic_units=False,
    include_images=False,
):
    """Build an OpenContractDocExport dict from XHTML + finalized blocks."""
    triggers = (
        _DEFAULT_RECITAL_TRIGGERS
        if recital_triggers is None
        else frozenset(t.lower() for t in recital_triggers)
    )
    soup = (
        xhtml
        if isinstance(xhtml, BeautifulSoup)
        else BeautifulSoup(str(xhtml), "html.parser")
    )
    page_divs = soup.find_all("div", class_="page")

    # --- PAWLS token layer (one entry per page, position == index) ---
    pawls = []
    page_centers = []  # per-page: list of (token_index, cx, cy)
    page_box = []  # per-page: (width, height)
    for idx, div in enumerate(page_divs):
        width, height = _page_dims(div)
        tokens = _page_tokens(div, width, height)
        pawls.append(
            {"page": {"width": width, "height": height, "index": idx}, "tokens": tokens}
        )
        page_centers.append(
            [(t["x"] + t["width"] / 2.0, t["y"] + t["height"] / 2.0) for t in tokens]
        )
        page_box.append((width, height))
    page_count = len(page_divs)

    # --- assign each token to the smallest block rect whose box contains it ---
    rects = {}  # block_idx -> (top, left, right, bottom, page_idx, area)
    blocks_by_page = defaultdict(list)
    for b in blocks:
        pidx = int(b["page_idx"])
        if not (0 <= pidx < page_count):
            continue
        width, height = page_box[pidx]
        bs = b["box_style"]
        top = _clamp(float(bs[0]), 0.0, height)
        left = _clamp(float(bs[1]), 0.0, width)
        right = _clamp(float(bs[2]), 0.0, width)
        bottom = _clamp(float(bs[0]) + float(bs[4]), 0.0, height)
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        area = max(_EPS, (right - left) * (bottom - top))
        rects[b["block_idx"]] = (top, left, right, bottom, pidx, area)
        blocks_by_page[pidx].append(b["block_idx"])

    block_text_by_idx = {b["block_idx"]: b["block_text"] for b in blocks}
    # Two assignments per page. ``block_tokens_geo`` is the pure geometric one the
    # structural layer (labels, hierarchy, corner/furniture geometry) is scored on
    # by every regression baseline — left untouched. ``block_tokens`` is that same
    # assignment with run-in collisions repaired by text (``_reconcile_runin_overflow``)
    # and is what the *emitted* annotations carry, so the exported token layer is
    # faithful to each block's ``rawText`` without disturbing the structural scoring.
    # only header blocks may donate run-in overflow (see _reconcile_runin_overflow)
    header_donor_ids = frozenset(
        b["block_idx"]
        for b in blocks
        if _LABEL_BY_BLOCK_TYPE.get(b["block_type"]) == "Section Header"
    )
    block_tokens_geo = defaultdict(list)  # block_idx -> [token_index] (geometry only)
    block_tokens = defaultdict(list)  # block_idx -> [token_index] (emitted, repaired)
    for pidx, centers in enumerate(page_centers):
        ordered_ids = sorted(blocks_by_page.get(pidx, ()))  # block_idx == reading order
        block_words = {bid: block_text_by_idx[bid].split() for bid in ordered_ids}
        token_texts = [t["text"] for t in pawls[pidx]["tokens"]]
        owner = _assign_page_tokens(
            token_texts, ordered_ids, block_words, rects, centers
        )
        owner_emit = _reconcile_runin_overflow(
            list(owner), token_texts, ordered_ids, block_words, header_donor_ids
        )
        for tidx, (bid, ebid) in enumerate(zip(owner, owner_emit)):
            if bid is not None:
                block_tokens_geo[bid].append(tidx)
            if ebid is not None:
                block_tokens[ebid].append(tidx)

    _warn_unfaithful_blocks(blocks, block_text_by_idx, block_tokens)

    # --- annotation layer (one annotation per block) ---
    emitted_ids = {str(b["block_idx"]) for b in blocks}

    def _block_geom(b):
        """(top_frac, bottom_frac, left_frac) of a block's assigned-token box vs its
        page, for the geometric corner-furniture check; None if untokened/off-page.
        Uses the *geometric* assignment so structural scoring is baseline-stable."""
        toks = block_tokens_geo.get(b["block_idx"], ())
        pidx = int(b["page_idx"])
        if not toks or not (0 <= pidx < page_count):
            return None
        pw, ph = page_box[pidx]
        if not (pw > 0 and ph > 0):
            return None
        pt = pawls[pidx]["tokens"]
        top = min(pt[t]["y"] for t in toks)
        bottom = max(pt[t]["y"] + pt[t]["height"] for t in toks)
        left = min(pt[t]["x"] for t in toks)
        return (top / ph, bottom / ph, left / pw)

    geom_by_idx = {b["block_idx"]: _block_geom(b) for b in blocks}
    overrides = _structural_corrections(
        blocks, blocks_by_page, block_tokens_geo, geom_by_idx, page_box, pawls
    )
    label_by_idx = {}
    for b in blocks:
        bidx = b["block_idx"]
        base = _resolve_label(
            b,
            ntok=len(block_tokens_geo.get(bidx, ())),
            geom=geom_by_idx[bidx],
            recital_triggers=triggers,
        )
        ov = overrides.get(bidx)
        if ov == "Paragraph":
            label_by_idx[bidx] = "Paragraph"
        elif ov == "Title" and base in _HEADER_ANNOTATION_LABELS:
            label_by_idx[bidx] = "Title"
        else:
            label_by_idx[bidx] = base
    # a run of list items is a child of the lead-in paragraph that introduces it
    # ("… agree as follows:" → A. … B. …): the colon-terminated paragraph right
    # before the run becomes those items' parent (instead of the nearest header).
    intro_parent_by_idx = _list_intro_parents(blocks, label_by_idx)
    labelled_text = []
    for b in blocks:
        bidx = b["block_idx"]
        rect_top, rect_left, rect_right, rect_bottom, pidx, _ = rects[bidx]
        base_label = label_by_idx[bidx]
        tok_idxs = block_tokens.get(bidx, ())
        # parent: an introducing lead-in paragraph for a list item, else the
        # nearest still-heading ancestor in the level chain (demoted run-in
        # headers are skipped so they never silently bear children).
        block_parent = intro_parent_by_idx.get(bidx)
        if block_parent is None:
            for level in b.get("level_chain") or []:
                cand = str(level["block_idx"])
                if (
                    cand in emitted_ids
                    and cand != str(bidx)
                    and label_by_idx.get(level["block_idx"])
                    in _HEADER_ANNOTATION_LABELS
                ):
                    block_parent = cand
                    break

        # split a fused Paragraph block into a centered heading + its body. Only
        # Paragraph blocks: header blocks are referenced as parents by id, and
        # Table Row / List Item blocks have centered cells/markers that look like
        # headings but aren't (the split would mangle table rows).
        if base_label != "Paragraph" or len(tok_idxs) < 2:
            segments = [{"tokens": list(tok_idxs), "heading": False}]
        else:
            segments = _segment_block(
                tok_idxs, pawls[pidx]["tokens"], page_box[pidx][0], triggers
            )
        split = len(segments) > 1

        last_heading_id = None
        for k, seg in enumerate(segments):
            seg_id = str(bidx) if k == 0 else f"{bidx}-{k}"
            seg_tokens = seg["tokens"]
            if split:
                label = "Section Header" if seg["heading"] else base_label
                seg_text = " ".join(
                    pawls[pidx]["tokens"][t]["text"] for t in seg_tokens
                )
                seg_parent = (
                    block_parent if k == 0 else (last_heading_id or block_parent)
                )
                if seg["heading"]:
                    last_heading_id = seg_id
            else:
                label = base_label
                seg_text = b["block_text"]
                seg_parent = block_parent

            tok_refs = [{"pageIndex": pidx, "tokenIndex": t} for t in seg_tokens]
            if seg_tokens:
                # spec §6.6: bounds = union of the segment's tokens' boxes
                toks = [pawls[pidx]["tokens"][t] for t in seg_tokens]
                left = min(t["x"] for t in toks)
                right = max(t["x"] + t["width"] for t in toks)
                top = min(t["y"] for t in toks)
                bottom = max(t["y"] + t["height"] for t in toks)
            else:
                top, left, right, bottom = rect_top, rect_left, rect_right, rect_bottom
            labelled_text.append(
                {
                    "id": seg_id,
                    "annotationLabel": label,
                    "rawText": seg_text,
                    "page": pidx,
                    "annotation_json": {
                        str(pidx): {
                            "bounds": {
                                "top": top,
                                "left": left,
                                "right": right,
                                "bottom": bottom,
                            },
                            "tokensJsons": tok_refs,
                            "rawText": seg_text,
                        }
                    },
                    "parent_id": seg_parent,
                    "annotation_type": "TOKEN_LABEL",
                    "structural": True,
                    "content_modalities": ["TEXT"],
                }
            )

    if include_images:
        if pdf_bytes is None:
            raise ValueError("include_images=True requires pdf_bytes")
        _append_image_layer(
            pawls, labelled_text, extract_page_images(pdf_bytes, page_box)
        )

    export = {
        "title": title if title is not None else _title_from_xhtml(soup),
        "content": "\n".join(b["block_text"] for b in blocks),
        "description": None,
        "pawls_file_content": pawls,
        "page_count": page_count,
        "doc_labels": [],
        "labelled_text": labelled_text,
        "relationships": _parent_child_relationships(labelled_text),
        "file_type": file_type,
    }
    if pdf_bytes is not None:
        export["pdf_file_hash"] = hashlib.sha256(pdf_bytes).hexdigest()
    if semantic_units:
        append_semantic_units(export)
    return export


# --------------------------------------------------------------------------- #
# validation (spec §6)
# --------------------------------------------------------------------------- #
_PAGE_KEY_RE = re.compile(r"^\d+$")
_OPTIONAL_TOKEN_FIELDS = (
    "is_image",
    "image_path",
    "base64_data",
    "format",
    "content_hash",
    "original_width",
    "original_height",
    "image_type",
)
_IMAGE_ONLY_TOKEN_FIELDS = _OPTIONAL_TOKEN_FIELDS[1:]  # everything but is_image
_IMAGE_FORMATS = ("jpeg", "png")
_IMAGE_TYPES = ("embedded", "cropped")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_MODALITIES = frozenset({"TEXT", "IMAGE"})


def validate_export(export):
    """Assert the OpenContractDocExport invariants; raise on the first breach."""

    def fail(msg):
        raise ExportValidationError(msg)

    for key in ("title", "content"):
        if not isinstance(export.get(key), str):
            fail(f"'{key}' must be a string")
    if "description" not in export:
        fail("'description' key is required (may be null)")

    pawls = export["pawls_file_content"]
    page_count = export["page_count"]
    if len(pawls) != page_count:
        fail(f"page_count {page_count} != pawls pages {len(pawls)}")

    page_token_counts = []
    page_image_idx = []  # per page: set of token indices that are image tokens
    for i, page in enumerate(pawls):
        pb = page["page"]
        if pb["index"] != i:
            fail(f"page {i} has index {pb['index']}")
        w, h = pb["width"], pb["height"]
        if not (w > 0 and h > 0):
            fail(f"page {i} has non-positive dims {w}x{h}")
        for j, t in enumerate(page["tokens"]):
            for f in _OPTIONAL_TOKEN_FIELDS:
                if f in t and t[f] is None:
                    fail(f"page {i} token {j} has null optional field '{f}'")
            if t.get("is_image"):
                if t["text"] != "":
                    fail(f"page {i} image token {j} has non-empty text")
                if not (t.get("image_path") or t.get("base64_data")):
                    fail(f"page {i} image token {j} needs image_path or base64_data")
                if "format" in t and t["format"] not in _IMAGE_FORMATS:
                    fail(f"page {i} image token {j} has bad format {t['format']!r}")
                if "image_type" in t and t["image_type"] not in _IMAGE_TYPES:
                    fail(
                        f"page {i} image token {j} has bad image_type "
                        f"{t['image_type']!r}"
                    )
                for f_ in ("original_width", "original_height"):
                    if f_ in t and not (isinstance(t[f_], int) and t[f_] > 0):
                        fail(f"page {i} image token {j} has bad {f_}")
                if "content_hash" in t and not _HEX64_RE.match(t["content_hash"]):
                    fail(f"page {i} image token {j} has bad content_hash")
            else:
                for f_ in _IMAGE_ONLY_TOKEN_FIELDS:
                    if f_ in t:
                        fail(f"page {i} text token {j} has image-only field '{f_}'")
            if not (
                t["x"] >= -_EPS
                and t["y"] >= -_EPS
                and t["x"] + t["width"] <= w + 1.0
                and t["y"] + t["height"] <= h + 1.0
            ):
                fail(f"page {i} token {j} box out of page bounds")
        page_token_counts.append(len(page["tokens"]))
        page_image_idx.append(
            {j for j, t in enumerate(page["tokens"]) if t.get("is_image")}
        )

    ids = [a["id"] for a in export["labelled_text"]]
    if len(ids) != len(set(ids)):
        fail("annotation ids are not unique")
    id_set = set(ids)

    for a in export["labelled_text"]:
        if a["annotation_type"] != "TOKEN_LABEL":
            fail(f"annotation {a['id']} annotation_type != TOKEN_LABEL")
        if a["structural"] is not True:
            fail(f"annotation {a['id']} is not structural")
        aj = a["annotation_json"]
        has_image_ref = has_text_ref = False
        for pkey, page_ann in aj.items():
            if not _PAGE_KEY_RE.match(pkey):
                fail(f"annotation {a['id']} has non-int page key {pkey!r}")
            b = page_ann["bounds"]
            if b["left"] > b["right"] + _EPS or b["top"] > b["bottom"] + _EPS:
                fail(f"annotation {a['id']} has inverted bounds")
            for ref in page_ann["tokensJsons"]:
                pi, ti = ref["pageIndex"], ref["tokenIndex"]
                if not (0 <= pi < page_count):
                    fail(f"annotation {a['id']} token ref page {pi} out of range")
                if not (0 <= ti < page_token_counts[pi]):
                    fail(f"annotation {a['id']} token ref index {ti} out of range")
                if ti in page_image_idx[pi]:
                    has_image_ref = True
                else:
                    has_text_ref = True
        mods = a.get("content_modalities")
        if mods is not None and (not mods or not set(mods) <= _MODALITIES):
            fail(f"annotation {a['id']} has bad content_modalities {mods!r}")
        if has_image_ref and "IMAGE" not in (mods or ()):
            fail(f"annotation {a['id']} references image tokens without IMAGE modality")
        if has_text_ref and mods is not None and "TEXT" not in mods:
            fail(f"annotation {a['id']} references text tokens without TEXT modality")
        parent = a["parent_id"]
        if parent is not None and parent not in id_set:
            fail(f"annotation {a['id']} parent_id {parent!r} does not exist")

    # token partition (fine layer): no page token may belong to two structural
    # block annotations. This is what makes the exported text layer faithful to a
    # blind flat extract — every token grounds to at most one block, so a block's
    # span never silently contains a neighbour's text. The additive "Semantic Unit"
    # layer deliberately re-references its members' tokens, so it is excluded.
    token_owner = {}
    for a in export["labelled_text"]:
        if a["annotationLabel"] == "Semantic Unit":
            continue
        for pkey, page_ann in a["annotation_json"].items():
            for ref in page_ann["tokensJsons"]:
                key = (ref["pageIndex"], ref["tokenIndex"])
                if key in token_owner:
                    fail(
                        f"token {key} claimed by two annotations "
                        f"({token_owner[key]!r} and {a['id']!r})"
                    )
                token_owner[key] = a["id"]

    # cycle detection on the parent_id tree
    parent_of = {a["id"]: a["parent_id"] for a in export["labelled_text"]}
    for start in parent_of:
        seen = set()
        node = start
        while node is not None:
            if node in seen:
                fail(f"cycle in parent_id tree at {node!r}")
            seen.add(node)
            node = parent_of.get(node)

    # relationships (spec §4): typed annotation-to-annotation edges
    for rel in export.get("relationships", []):
        if (
            not isinstance(rel.get("relationshipLabel"), str)
            or not rel["relationshipLabel"]
        ):
            fail(f"relationship {rel.get('id')!r} has no relationshipLabel")
        if rel.get("structural") not in (True, False):
            fail(f"relationship {rel.get('id')!r} structural must be bool")
        src = rel.get("source_annotation_ids") or []
        tgt = rel.get("target_annotation_ids") or []
        if not src or not tgt:
            fail(f"relationship {rel.get('id')!r} has empty source/target")
        for aid in (*src, *tgt):
            if aid not in id_set:
                fail(
                    f"relationship {rel.get('id')!r} references unknown annotation {aid!r}"
                )
        if set(src) & set(tgt):
            fail(
                f"relationship {rel.get('id')!r} has an annotation as its own source+target"
            )
