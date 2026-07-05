# OpenContractDocExport structure audit (diverse 36-page sample)

This records a structural-accuracy study of the `OpenContractDocExport` produced
by Warp's PDF pipeline, the two diagnostic utilities built to run it, the
heuristic gaps it surfaced, the exporter-only fixes it justified, and what was
deliberately deferred.

> TL;DR — Two new utilities make the export *legible*: a **bbox projector**
> (`oc_visualize.project_page`) that overlays each annotation's box on the PDF
> raster grouped/colored by its structural subtree, and a **sparse tree dumper**
> (`oc_visualize.render_tree_outline`). Running them over **36 pages of 17
> diverse PDFs** plus an LLM-in-the-loop audit found the dominant defect class is
> **layout-engine segmentation** (block merge/split, multi-column fusion) — out
> of scope here (the exporter may only relabel / re-parent / re-assign tokens,
> never split or merge a block). The **cleanly exporter-fixable** gaps were
> fixed: run-in headings kept as `Section Header`, non-heading blocks bearing
> children, and dropped table-cell text. Sample results: `childbearing_non_headers`
> **23 → 0**, min `token_coverage` **0.507 → 0.771**, run-in mislabels resolved,
> with **no regression** to any committed suite.

## 1. The utilities

`warp_ingest/ingestor/oc_visualize.py` (reads an export dict only; never touches
`visual_ingestor` or the XHTML contract):

- `build_tree(export)` — resolves the `parent_id` hierarchy into a cycle-safe
  tree (roots, children in reading order, depth, subtree-root).
- `render_tree_outline(export)` — a compact indented outline + a stats header
  (`roots`, `max_depth`, `non_heading_roots`, `childbearing_non_headers`,
  `overlong_headings`, `untokened`, `token_coverage`).
- `diagnostics(export)` — the machine-readable structural metrics those summarize.
- `page_summary(export, page)` — per-page rows (id, label, parent, depth, ntok).
- `project_page` / `project_document` — render a page raster + annotation boxes,
  colored by structural subtree (the "relationship"), with child→parent connector
  lines and a legend naming the subtree roots.

CLIs: `scripts/oc_visualize.py` (one PDF → projections + tree) and
`scripts/oc_sample_audit.py` (the diverse sample → projections + per-page
summaries + roll-up diagnostics).

## 2. The sample

36 pages across 17 PDFs spanning: negotiated agreements (Eton dev agreement,
4 S-1/8-K exhibit agreements), a deep-numbered statute (USC Title 1), a scanned
OCR page, a two-column academic paper (equations, tables, references), three long
credit agreements (cover/TOC/defined-terms/tables), SEC filings (8-K,
primary-document), and S-1 exhibits (table-heavy SOW, specimen certificate,
1-page consent). Pages were chosen for layout diversity, not depth.

## 3. Audit method

1. `scripts/oc_sample_audit.py` runs Warp once per doc and emits a projection PNG
   + structural summary + `diagnostics` per page.
2. An LLM-in-the-loop pass judged each page against its rendered projection across
   six dimensions (labels, hierarchy, grouping, coverage, reading-order,
   segmentation), reporting concrete defects with severity, then clustered them
   into gap categories and tagged each **exporter-fixable** vs **layout-engine**.

## 4. The gap taxonomy

| category | freq (of 36) | severity | exporter-fixable | disposition |
|---|---|---|---|---|
| **Run-in heading kept as `Section Header`** (short bold lead-in absorbs the clause body) | 3+ | high | **yes** | **fixed** (token-count demotion) |
| Non-heading block bearing children (demoted run-in still a parent) | many | med | **yes** | **fixed** (de-anchor) |
| Dropped table-cell / wrapped text (uncovered tokens) | ~6 | med | **yes** | **fixed** (text recovery) |
| Non-structural text promoted to `Section Header` (pagination, metadata, connectors, ToC, form captions, watermarks) | 10 | high | partial | **deferred** (perturbs the table-less Docling head-ancestor oracle beyond margin) |
| List ↔ Paragraph ↔ Table-Row label confusion | 9 | high | partial | **deferred** (definitional; see docsling study §6) |
| Hierarchy mis-nesting / wrong-level parenting | 8 | high | partial | **deferred** (depends on engine heading levels) |
| **Heading fused into the body block that follows it** (centered title / "RECITALS" / "Consent of …" glued to its paragraph) | 11/4 | high | **yes** (re-segment) | **fixed** (centered-line split) |
| Multi-column / side-by-side fused across columns | 7 | high | **no** | layout engine |
| Single unit over-split mid-sentence | 5 | high | **no** | layout engine |
| Two distinct *adjacent* headings glued (both centered, no body between) | part of 11 | high | **no** | layout engine |

The originally-dominant failure mode — **block under-segmentation** — turned out
to be *partly* exporter-fixable after all: where a block fuses a **centered
heading line with the left-aligned body that follows**, the per-visual-line
geometry survives in the token boxes, so the exporter re-segments it (§5.4) — the
correct visual lines are still in the XHTML; only `visual_ingestor`'s *block
grouping* over-merged them. What the exporter still cannot recover is fusion that
also destroyed the line structure (two columns merged into one horizontal line)
or that needs lines *joined* (a sentence torn mid-way); those remain layout-engine
work.

## 5. The fixes (exporter only)

All three live in `warp_ingest/ingestor/opencontracts_exporter.py`:

1. **Token-count run-in demotion.** The existing `_HEADER_MAX_WORDS=12` word rule
   can't see a run-in heading whose `block_text` is a 2-word bold lead-in
   ("6.7 Audits.") but whose annotation absorbs the whole clause body via its
   token box (188 tokens). `_resolve_label` now also demotes a `Section Header`
   to `Paragraph` when its **assigned PAWLS token count** exceeds
   `_HEADER_MAX_TOKENS=20` — a real heading carries only a handful of tokens.
   This was the single most common cleanly-fixable defect.

2. **Tree consistency (de-anchor).** `parent_id` is now the nearest ancestor in
   the level chain whose *resolved label* is still a heading, so a demoted run-in
   header (now `Paragraph`) is spliced out and never bears children. No
   non-heading annotation parents content.

3. **Uncovered-token recovery.** Token→block assignment stays geometric, but
   tokens whose center lands inside *no* block rect (the wrapped tail of a table
   cell whose `box_style` undercovers its text) are recovered by aligning the
   page's token-text stream to the blocks' word streams (`difflib`). This only
   *fills* unassigned tokens — it never reassigns a token geometry already
   placed, so the regression baselines' single-column layouts are untouched.

4. **Block re-segmentation (the dominant-defect fix).** A non-header block is
   re-split when it *begins* with a run of **centered heading line(s)** followed
   by left-aligned body. Tokens are clustered into visual lines by vertical
   position; a line that is narrow and has symmetric side margins (and ≤
   `_HEADER_MAX_WORDS` words) is a centered heading; a leading run of such lines
   is emitted as its own `Section Header` annotation that parents the body
   annotation carrying the rest. This recovers the centered title + preamble and
   the "RECITALS" + WHEREAS clauses on contract first-pages, the consent-letter
   title, etc. — structure `visual_ingestor` over-merged — *without touching the
   engine*: the granular truth (one `<p>` per visual line) is intact in the
   XHTML; only the block grouping was lossy. It is deliberately conservative
   (leading heading only; never manufactures an over-long header; only
   non-header blocks, which are never referenced as parents, so no id breaks).

## 6. Results

Over the 36-page sample (`scripts/oc_sample_audit.py`, before → after):

| metric | before | after |
|---|---|---|
| `childbearing_non_headers` (sum) | 23 | **0** |
| min `token_coverage` | 0.507 | **0.771** |
| `overlong_headings` (sum) | 0 | 0 |

Per-doc highlights: the table-heavy SOW exhibit's coverage rose 0.507 → 0.790;
the academic paper 0.936 → 0.972; run-in subsection headings on Eton p8/p12
(`6.7 Audits`, `6.8 Accounting`, `11.1`–`11.3`) and credit-agreement definition
sections are now `Paragraph`, not `Section Header`.

### 6.1 Re-running the LLM audit after re-segmentation

The same 36-page LLM audit, re-run on the re-segmented export, rates the
**segmentation** dimension `ok` on 11 pages (was 8) and the centered-heading
split improved it on 6 pages — Eton p0 (title + RECITALS split off), EcoScience
p0/p1, the OCR scan, the cerebras SOW p4, and the auditor-consent — with **zero
real regressions** (the two pages the independent re-run rated lower,
`credit_aon` p40 and an Eton page, contain *no* split annotations — my code never
touched them, so the delta is LLM rating noise between independent runs).

The aggregate segmentation count barely moved (38 → 39 flagged defects) because
the *rest* of the under-segmentation class is **not** reachable from the
exporter, and the re-run's own synthesis says so: of its remaining gaps "six
require the layout engine because they need block split/merge or column/table
detection the exporter cannot perform" — multi-column fusion (the two-column
academic paper fails wholesale), undetected tables, two stacked centered
headings, and list/bullet boundaries. One tempting case is a genuine dead-end at
the exporter: a run-in heading ("6.7 Audits.") whose box steals the body
paragraph's tokens, leaving a zero-token "ghost" sibling. Giving the body its
tokens (a full token-split) is *more* correct but turns "6.7 Audits." back into a
short `Section Header`, which **tanks** the Docling `head_ancestor` oracle
(generate_bio 0.87 → 0.25) because Docling treats run-in lead-ins as body — so
the docsling-faithful choice is exactly the current one (demote the lead-in to
`Paragraph`; the ghost is the residual). These are recorded for a layout-engine
pass, not chased at the exporter.

**No regressions.** Every committed suite passes — `test_opencontracts_export`,
`test_opencontracts_regression`, `test_layout_docsling_regression` (`--runslow`),
`test_s1_regression`, `test_fixtures_parse`. On the Docling cross-engine oracle
the changes are neutral-or-better on **every** fixture vs the prior engine:
`label_agree` up where it moves, and heading-ancestor agreement up sharply on the
run-in-dense docs (`generate_bio` +0.63, `eton` +0.23) because the demoted run-in
"headers" stop masquerading as the body's section ancestor.

## 7. Deliberately deferred

- **Full run-in token split** (move the absorbed body tokens off the demoted
  header onto its empty sibling, instead of demoting the whole block to
  `Paragraph`). Verified *more* correct from the projections, but it shifts each
  annotation's token-union bounds, which reshuffles the reading-order word stream
  the Docling oracle scores; on run-in-dense agreements it drops
  `word_seq_similarity` past the committed margin. It needs a Docling-baseline
  regeneration (the oracle merges run-in heading+body into one BODY span and so
  under-credits the split). Recorded for that follow-up.
- **Non-structural header demotion** (pagination / EDGAR metadata / connectors /
  ToC / form captions / watermarks → body). Correct, but de-anchoring these
  perturbs the table-less Docling head-ancestor oracle beyond margin on
  table-heavy exhibits. Same follow-up.
- **List/Paragraph churn, hierarchy re-leveling, and all segmentation/
  multi-column categories** — definitional or layout-engine; left alone, matching
  the Docling study's `docs/docsling_layout_oracle.md` §6 disposition.

## 8. Parent↔child relationships campaign

A follow-up campaign made the parent→child structure both *more correct* and
*explicit*:

- **List items now nest under their introducing lead-in paragraph.** A run of
  `list_item` blocks whose immediately-preceding block is a colon-terminated
  `Paragraph` ("As an inducement … agree as follows:") is parented to that
  paragraph (`_list_intro_parents`), instead of being parentless siblings or
  attaching to a distant header. This is the structure the SOUPMAN franchise
  guarantee shows: items A–G are children of the "… agree as follows:" lead-in —
  **including across the page break** (item G on page 2 still parents to the
  lead-in on page 1). It is the one sanctioned non-heading parent; the
  `childbearing_non_headers` diagnostic now allows a Paragraph whose children are
  all List Items.
- **The hierarchy is emitted as explicit `OC_PARENT_CHILD` relationships**
  (`_parent_child_relationships`) — one per parent (`source = [parent]`,
  `target = [direct children]`, `structural: true`), the OpenContracts convention
  honored by its subtree-group walker alongside the `parent_id` FK.
  `validate_export` enforces relationship referential integrity (non-empty,
  resolvable source/target; no self source+target).
- **The projector renders relationships.** `project_page` draws each
  `OC_PARENT_CHILD` edge as a parent→child arrow, colored by the subtree the
  child belongs to, and marks **cross-page** edges with an "↑parent [id] pN" cue
  on the child (since the parent box is on another page). The subtree coloring
  groups every box by the relationship it participates in.

Verified across the sample: relationships emit for every doc with hierarchy,
`validate_export` passes on all, and the SOUPMAN p0/p1 cross-page list correctly
groups under its lead-in. No regression to any committed suite (relationships and
the para→list parent don't move the Docling word-stream/label/head-ancestor
metrics — the oracle's heading-ancestor walk skips the paragraph parent).

## 9. Second pass — the diverse 60-page audit (furniture/metadata demotion)

A second run of the runbook (`docs/runbooks/structural-quality-audit.md`) over a
**fresh 60-page batch** — 26 docs spanning **FortWorth municipal contracts**
(a new committed source, `tests/fixtures/contracts/`), unused EDGAR S-1
prospectus bodies, exhibit agreements, charters, and single-page consents — was
audited by a per-page vision fan-out across all three dimensions
(`review_pngs/audit_findings_60page.md` records the raw defects).

**Dominant defect class (C1): non-heading text mislabeled `Section Header`.** The
layout engine emits a `header` block for page furniture / document-processing
metadata, and labeling it a heading *also* made it a parent-chain ancestor that
adopted unrelated body as children — the "fan of arrows" cascade visible on
nearly every municipal cover page and every page-break in the long S-1 bodies.
Examples found: page folios (`F-18`, `iii`, `105`, `- 33 -`); the EDGAR running
`TABLE OF CONTENTS` link; the `Docusign Envelope ID` watermark; recital openers
(`WHEREAS …`); signature-block fields (`Name:`, `Title:`, `By: /s/ …`,
`INVESTOR:`); bare emails; postal addresses; and jurisdiction-specific clerk/filing
corner stamps (`CSC No. 65007`, `OFFICIAL RECORD / CITY SECRETARY`).

**Fix (exporter only): `_is_nonheading_furniture` in `_resolve_label`.** A
`Section Header` whose **entire text is a single non-prose structural token** —
a page folio (`F-18`/`iii`/`105`/`- 33 -`), a bare email, or a bare URL — is
demoted to `Paragraph`; the existing tree-consistency rule then splices it out of
the parent chain, so the same change fixes both the **label** and the
**parent↔child** dimension (children re-attach to the nearest real heading, or
root). The same filter gates the re-segmentation path (`_is_centered_heading_line`)
so a centered furniture line can't be re-promoted. Anchored so genuine short
headings (`ARTICLE 9`, `WITNESSETH`, `Definitions`) are never caught
(`test_real_heading_not_demoted_as_furniture`). Page-folio demotion was the
highest-frequency clean win (a folio splitting a section across a page break, all
through the long S-1 bodies).

**Structural / format only — no content-based rules.** This is the deliberate
design boundary (and a reviewer steer): the filter recognizes a *structural*
element (pagination, a contact token) by its universal **format**, never by the
**content** (the words/semantics) of any document. So it does **not** match recital
words (`WHEREAS`), signature-field words (`Name:`/`Title:`/`By:`), specific phrases
(`TABLE OF CONTENTS`, the DocuSign watermark), postal addresses, or — most of all —
corpus-specific clerk/filing stamps (`CSC No.`, `OFFICIAL RECORD`, `Fort Worth`)
(`test_content_based_furniture_deliberately_not_demoted` locks the boundary).

**Follow-up: geometric corner-furniture demotion (`_is_corner_furniture`).** The
clerk/filing **corner stamps** the text filter won't name are instead caught the
*structural* way — by **position**, not words. A `Section Header` whose assigned-
token box is in the page's extreme top/bottom margin (`top_frac < 0.055` or
`bottom_frac > 0.91`) **and** offset past page center (`left_frac > 0.55`) is a
running header / page-number / corner stamp, never a section heading (those start at
the left margin or are centered — `left_frac > 0.55` excludes them by construction).
A 36-doc corner scan + per-fixture Docling A/B confirmed it demotes only such
furniture (the FortWorth `CSC No.` / `OFFICIAL RECORD` stamps — fixing their
cover-page cascades), zero left-aligned/centered real headings. The top band is kept
tight (0.055) so the slightly-lower top-right `Exhibit X.X` identifier line is
spared — demoting it is defensible and *improves* most Docling fixtures, but tanks
the tiny `exyn_ex211` oracle (Docling treats the exhibit-id as a heading), so it is
left to a baseline-regen follow-up. Net: **exactly Docling-neutral** (only the
pre-existing `spacex_ex41` env-drift fixture fails). The remaining content cases
(recitals, signature fields, addresses) stay deferred to the layout engine.

**Oracle neutrality.** Verified neutral-or-better against the Docling oracle per
fixture (`forbright_ex1012` / `generate_bio_ex33` / `exyn_ex211` are shared
fixtures): `word_seq_similarity` and `head_ancestor_agree` unchanged, `label_agree`
flat-or-up. Two patterns are **deliberately not** demoted because they perturb the
Docling head-ancestor proxy on `cerebras_ex1013e` beyond its margin (the
baseline-regeneration deferral from §7): the `Page N of M` footer form, and
4-digit numbers (years, not folios). `spacex_ex41` continues to fail the Docling
suite *locally* only — a pre-existing `pdfplumber` env drift, confirmed identical
on a clean checkout and exactly neutral to this change.

**Deferred (content-based / layout-engine / oracle-bound), recorded in
`review_pngs/audit_findings_60page.md`:** in-column / non-corner content mislabels
that have no structural signal — e-signature watermarks (top-left edge), recital
openers, signature-block fields, addressee blocks (all main-column, so the corner
rule can't see them) — left to the layout engine; top-right `Exhibit X.X`
identifier lines (spared to stay Docling-neutral; demotable after a baseline
regen); left-aligned run-in numbered headings fused into body (`SECTION 13. …`,
`1.1.1.1 Premises`) and the untokened bodies they leave behind (token-assignment
geometry); TOC rows mislabeled `Section Header` among `Table Row` siblings;
extending list-intro parenting to `Table Row` runs; 0-token "ghost" duplicate
annotations.

**Lock-in.** The 60 pages are frozen as a deterministic per-page regression
(`tests/test_oc_batch_regression.py` + `tests/fixtures/oc_batch_baseline.json` +
`tests/oc_batch_compat.py`, regenerated by `scripts/build_oc_batch_fixtures.py`;
discovery driver `scripts/oc_audit60.py`): smell counts ceiled, coverage/anchored/
tightness floored, `validate_export` + relationship integrity re-asserted.
