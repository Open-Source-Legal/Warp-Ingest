# Docsling cross-engine layout study (Warp-Ingest vs Docling)

This document records a structural comparison between **Warp-Ingest** (this
repo's pure-Python engine) and **Docling** (the `jscrudato/docsling-local`
microservice), the gaps it surfaced, the one layout change it justified, and the
regression suite that locks the result in.

> TL;DR — Docling is a useful *structure oracle* but **not** ground truth. It
> drops tables entirely, under-labels numbered legal headings as `text`, and
> over-applies `list_item`. The study found exactly one **clean, demonstrable,
> Docling-aligned** Warp defect — run-in "header" blocks that swallow a whole
> section body and get labeled `Section Header` — and fixed it. Everything else
> in the gap is either definitional or a Docling defect, and was left alone.

## 1. Setup

Both engines emit the same `OpenContractsDocExport` shape
(`docs/opencontracts_export_format.md`), so they are directly comparable:

| | Warp-Ingest | Docling microservice |
|---|---|---|
| keys | snake_case (`labelled_text`) | camelCase (`labelledText`) |
| labels | `Section Header`, `Paragraph`, `List Item`, `Table Row` | raw `DocItemLabel` (`section_header`, `text`, `paragraph`, `list_item`, `title`, `page_header`, `page_footer`, `caption`, `picture`, …) |
| tokens | visual-line word boxes from `pdf_plumber_parser` | `pdfplumber.extract_words` |
| hierarchy | `parent_id` from each block's `level_chain` | `HierarchicalChunker`: each body item's `parent_id` = the section header it falls under |
| page `index` | 0-based | **1-based** (quirk; annotation `page` refs are 0-based) |

**10 fixtures** (~94 pages), chosen for diversity and bounded Docling runtime:
three 1-page S-1 exhibits (auditor consent, two subsidiary lists), the USC Title 1
statute (deep numbering), the Eton Development Agreement and three more S-1
exhibits/agreements, a certificate of incorporation, and a 3-page specimen with a
graphic. (`needs_ocr.pdf` was dropped: with the optional `[ocr]` backend absent,
Warp returns no tokens, so the comparison would measure OCR-backend presence, not
layout.)

## 2. Methodology

The two token grids differ (different `pdfplumber` settings, different line
grouping), so token-index equality is useless. Instead we compare **text-anchored
word streams**:

1. Map both label vocabularies onto a canonical set: `TITLE, HEADING, BODY, LIST,
   TABLE, CAPTION, PICTURE, …` (Docling's generic `text` and `paragraph` both →
   `BODY`).
2. Flatten each export to a `(word, canon_label, annotation_id)` stream in reading
   order (sort annotations by page→top→left).
3. Align the two streams with `difflib.SequenceMatcher`; on `equal` runs compare
   the canonical labels word-by-word → a **confusion matrix** robust to the
   engines' different segmentation.
4. For aligned BODY/LIST words, compare the **heading ancestor** (walk `parent_id`
   to the nearest heading) → hierarchy agreement.

Implemented in `tests/docsling_compat.py`; regenerate fixtures/metrics with
`scripts/build_docsling_fixtures.py` (needs the microservice).

## 3. What Docling gets wrong (why it is an oracle, not truth)

- **No tables at all.** The microservice only converts `doc.texts` and
  `doc.pictures`; `doc.tables` is never exported. Across all 10 docs the Docling
  label histogram is `text, list_item, section_header, page_footer, page_header,
  checkbox_unselected, picture, caption` — zero `table`. → **Table detection
  cannot be judged against this oracle**, so the suite ignores it.
- **Under-labels numbered legal headings.** "11.1 Term", "15.7 Severability",
  "3.3 ETON's NDC Numbers" are all labeled `text` by Docling. Warp correctly calls
  these `Section Header`. So a high BODY→HEADING count is *not* prima facie a Warp
  bug.
- **Over-applies `list_item`.** Statutory amendment notes ("Pub. L. 112-231, §1,
  Dec. 28, 2012, …") are paragraphs, but Docling tags them `list_item`.
- **Reading-order quirks.** e.g. a top-right "Exhibit 23.1" marker is emitted
  last.

## 4. The gap (aggregate word-level confusion, rows = Docling, cols = Warp)

**Before the fix** (overall label agreement **0.454**, heading-ancestor **0.300**):

```
D\W        HEADING    BODY    LIST   TABLE
HEADING        512      23      13      20
BODY          2697   12005    9603     238
LIST           625    5249    2948      64
```

The dominant cells, and the verdict after reading the actual spans:

| cell | words | verdict |
|---|---|---|
| BODY→LIST | 9603 | **definitional** — Warp's indent heuristics call numbered clauses list items; Docling calls them body. Both defensible. |
| LIST→BODY | 5249 | **definitional** — the inverse, and Docling itself mislabels note paragraphs as lists. |
| **BODY→HEADING** | **2697** | **Warp defect (partly).** Two sub-cases — see below. |
| LIST→HEADING | 625 | same root cause as BODY→HEADING. |

### The one clean defect: run-in headings absorbing the section body

Bucketing Warp `HEADING` annotations by word count, with the Docling-majority
label of their words:

| words/heading | anns | Docling majority |
|---|---|---|
| 1–12 | 259 | mixed HEADING/BODY/LIST — *definitional noise (Docling under-labels headings)* |
| 13–25 | 9 | 7 BODY, 1 LIST, 1 none |
| 26+ | 18 | **15 BODY, 3 LIST** |

The 27 `HEADING` blocks longer than 12 words hold **2,750 words** — essentially
the entire BODY→HEADING disagreement — and ~26/27 are body/list per the oracle.
These are **run-in headings**: a bolded "6.6 Taxes." lead-in fused with the
sentence(s) that follow, which Warp's header detector marks as a single `header`
block. The result is a 139- or 260-word "Section Header":

```
[header  p8  nwords=139]  6.6 Taxes. Each Party shall be responsible for and shall pay all Taxes payable …
[header  p9  nwords=260]  7.3 Patent Prosecution. Each Party shall be responsible, at its own expense, …
```

A 260-word section header is meaningless for retrieval, and Docling agrees it's
body. **This is demonstrable and Docling-aligned.**

## 5. The fix

`opencontracts_exporter._resolve_label`: a header-ish block whose text exceeds
`_HEADER_MAX_WORDS` (= 12) is labeled **`Paragraph`** instead of `Section Header`.
The threshold is empirical — header blocks up to ~12 words are still real
headings (the 1–12 bucket); essentially every longer one is mislabeled body.
It is a pure word-count heuristic, so it can in principle demote a genuinely long
title (a 15-word "Risk Factors Relating to …" heading); in the 10-doc study no
such false positive occurred (the 13+ bucket was ~26/27 body/list per the oracle),
and the regression suite floors against any net loss, but the tradeoff is real.

This lives in the **exporter only**. It does not touch `visual_ingestor` or the
XHTML contract, so the `json`/`html`/`all` renders and the S-1 / fixture
regression suites are unaffected (they compare engine output, not the export).

**After the fix** (overall agreement **0.454 → 0.522**, +6.8 pts):

```
D\W        HEADING    BODY    LIST
HEADING        512      23      13
BODY           382   14320    9603     (BODY→HEADING 2697 → 382)
LIST           237    5637    2948     (LIST→HEADING  625 → 237)
```

Per-doc `label_agree`, the run-in-header docs move most:

| fixture | before | after |
|---|---|---|
| eton_dev_agreement | 0.444 | **0.564** |
| generate_bio_ex33 | 0.416 | **0.546** |
| exyn_ex1010 | 0.371 | 0.394 |
| usc_title1 | 0.710 | 0.716 |

`overlong_heading_count` (HEADING annotations with >12 `rawText.split()` words)
goes to **0** on every fixture.

## 6. Gaps deliberately left alone

- **LIST ↔ BODY** churn (the two biggest cells): definitional, and Docling is
  inconsistent here. Chasing it risks regressing the tuned S-1 suite for no
  demonstrable correctness gain.
- **Tables:** the oracle has none; unverifiable here.
- **Pictures:** Docling emits `picture`; Warp defers image tokens (issue #1).
- **Page header/footer:** Docling keeps them; Warp strips running headers/footers
  by design (good for RAG). Not a defect.

## 7. The regression suite

- `tests/fixtures/docsling_targets/<slug>.json` — committed **slimmed** Docling
  oracle (annotations only, no token grid; ~412 KB total).
- `tests/fixtures/docsling_layout_baseline.json` — per-doc floored metrics.
- `tests/test_layout_docsling_regression.py` — runs Warp live, floors
  `word_seq_similarity` (symmetric difflib ratio of the two word streams — a
  content-preservation proxy, not one-sided recall), `label_agree`, and
  `head_ancestor_agree` (with margins) and **ceils** `overlong_heading_count` at 0;
  `page_count` exact. `head_ancestor_agree` is floored only where the oracle
  baseline is meaningful (≥ 0.10) — on a couple of docs Docling's own hierarchy is
  too sparse/defective to floor against, and the tiny 1-page exhibits act mainly
  as `page_count`/`word_seq_similarity` content guards. Large docs are `@slow`.
- `scripts/build_docsling_fixtures.py` — regenerates oracle + baseline (needs the
  microservice; the test does not).

Only Warp runs in CI — the 14 GB Docling image is needed only to regenerate the
oracle, exactly like the S-1 suite's upstream-Tika dependency.

```bash
pytest tests/test_layout_docsling_regression.py            # fast docs
pytest tests/test_layout_docsling_regression.py --runslow  # + eton, generate_bio
python scripts/build_docsling_fixtures.py                  # regenerate (service up)
```
