# Text-faithfulness evaluation: does Warp-Ingest ever lose text?

**Date:** 2026-07-06 · **Scope:** every PDF in the repo — **179 documents / 5,850
pages** (`tests/fixtures/**` incl. the 50 S-1 bodies, contracts, hetero-100,
legal-100, OCR fixtures, plus `files/pdf/`) · **Tool:**
`scripts/text_faithfulness_eval.py` · **Raw results:**
`docs/text_faithfulness_eval_results.jsonl`

## Question

Structure (labels, hierarchy, reading order) may be imperfect — but the parser
must never **lose** text. This evaluation compares Warp's exported text against
known-good independent extractors, exhaustively, and triages every page that
shows a material deficit.

## Method

- **Oracles** (what a page "really says"):
  - **pypdfium2** (PDFium — Chrome's PDF text stack). Fully independent of
    Warp's pdfplumber/pdfminer front-end, so agreement is meaningful evidence.
  - **pdfplumber flat extract** as a second opinion (same library family as the
    front-end, so it checks the pipeline *after* extraction).
- **Warp surfaces** (what Warp exports):
  - **`pawls`** — the OpenContracts PAWLS token stream, rebuilt from the
    front-end XHTML. This is the export's designed no-loss surface (compared
    page-by-page).
  - **`content`** — the OC export `content` field == the engine's
    `block_text`, i.e. the `json`/`html` render surface (compared doc-level).
- **Scoring**: multiset token recall after normalization (NFKC, case, quote/dash
  folding, bullet-glyph removal), with a *re-tokenization forgiveness* pass — a
  missing token is forgiven when its whitespace-free form still appears
  contiguously in Warp's whitespace-free character stream (different word
  segmentation is not loss). Missing tokens with no alphanumeric content are
  counted separately as trivial.

## Headline results (2,586,748 oracle tokens)

| surface | oracle | strict recall | **effective recall** | material missing | docs fully clean | flagged pages |
|---|---|---|---|---|---|---|
| pawls | pdfium | 0.9899 | **0.999855** | 375 | 136 / 179 | 61 / 5,850 |
| pawls | pdfplumber | 0.9947 | **0.999893** | 282 | 144 / 179 | 44 / 5,850 |
| content | pdfium | 0.9865 | 0.997491 | 5,729 | 57 / 179 | — |
| content | pdfplumber | 0.9900 | 0.997497 | 5,826 | 59 / 179 | — |

**Verdict: no material body-text loss was found on the PAWLS surface.** Every
one of the 61 flagged pages (375 tokens, 0.0145% of the corpus) was triaged —
by an automated classifier over re-parsed pages plus manual inspection of every
class — and decomposes into the categories below. The residual *genuine* losses
are a handful of tokens of page-margin/decorative text, detailed in §"Genuine
losses".

## PAWLS surface triage (the no-loss guarantee)

Classifier over all flagged pages (token-weighted; big-doc singletons
pattern-matched to the same classes and spot-verified):

| class | ~tokens | verdict | example |
|---|---|---|---|
| **Doubled characters** | 126 | *garble, not absence* — fake-bold overprint (text drawn twice at a sub-pixel offset) comes through pdfplumber as `ccaappiittaall::` where PDFium dedupes to `capital:`. The words are unreadable/unsearchable in that form. | `GBFR22-value-creation-model_p1` infographic; the diagonal “SUBJECT TO COMPLETION” overlay on draft S-1 cover pages (the pdfplumber-oracle mirror of this class is the reversed/doubled watermark tokens: `seitiruces`, `sseeiittiirruucceess`) |
| **Oracle-side gluing** | 62+ | *not Warp's fault* — PDFium welds vertically-stacked table-header cells into one token (`dateexhibit`, `numberfiled`, `capitalretained`, `holdersdividend`); Warp has both words separately. Verified by splitting each glued token into two tokens present on the same Warp page. | `files/pdf/primary-document.pdf` exhibit index pp.114–117; forbright/liftoff S-1 balance-sheet headers |
| **Per-character explosion** | 45 | *garble, not absence* — wide letter-spaced headings and rotated/diagonal banner text arrive as single-char tokens (`t`,`e`,`i`,`r`,…). Text present, words destroyed. | `chapter_10_p2` (letter-spaced heads), infleqtion/exyn/swarmer S-1 p1–2 diagonal red banner (`changed.`) |
| **Unmapped CID glyphs** | 10 | *decode gap* — pdfminer emits `(cid:47)…` for fonts whose cmap it can't decode; PDFium decodes them. | `GBFR22` display-font labels |
| **Soft-hyphen compounds** | ~15 | *hyphen lost, words kept* — the visible hyphen is encoded U+00AD (`low\xadcarbon`); Warp splits the pair, dropping only the hyphen glyph. | ESEF annual-report pp.22–23, `t-co2e` |
| **Sub/superscript & math splits** | ~20 | *present, tokenized apart* — `0.01` + `%`, `co` + `2`, `kh2po4`, math glyph-order (`γ5q`, `~x∈v`); also accent-decoding disagreements (`tramèr` vs `tramer`). Spot-checked present. | `ar2024e_p49` donut-chart labels, physics/CS papers |
| **Rotated margin text** | ~10 | **GENUINE LOSS** — 90°-rotated margin/sidebar text is absent from Warp output (pdfplumber's own word extraction can't see it either; a pdfminer-stack limitation). All instances are marginal/decorative, not body text. | arXiv sidebar `arXiv:2411.18478v1 [cs.CL]`; Federal Register processing keys (`jspears on DSK121TN23PROD`, `E:\FR\FM\27MYR2.SGM`) |
| **Sparse-scan OCR reroute** | ~6 | **GENUINE LOSS (with huge offsetting gain)** — a scanned page with a tiny embedded text layer (<4 lines) is rerouted to OCR; the embedded tokens are discarded. generate_bio p111: embedded layer is just the folio `C-9`; OCR recovered **299 tokens** of drawing text the oracle can't see, but the folio was dropped. Same class: `fw_garver` p4, `fw_southern_computer` (`order:`). | `generate_bio_s1__ex1015` p111, FortWorth scans (whose remaining “missing” gibberish — `4popr`, `uoq` — is oracle noise from the scans' junk text layer) |

## Content surface (`block_text` / json render)

Doc-level effective recall 0.9975. The deficit vocabulary is dominated by the
engine's **deliberate repeated-furniture stripping** (`should_ignore_line`,
`find_true_header_footers`): running `TABLE OF CONTENTS` banners (724 tokens),
EDGAR `Source: … 10-Q, 11/14/2019` footers, `The accompanying notes are an
integral part …` banners, page folios, dot-leader runs (the trivial-token
column). Duplicated TOC lines are also deduped (`sample.pdf`). This is by
design and the text remains in the PAWLS layer — but consumers of the
`json`/`html`/markdown renders should know it is *removed there*.

One class deserves attention rather than a shrug: **repeated signature blocks**.
On `neutron_lime_s1__exhibit45sx1` (295-page guarantee supplement with ~177
near-identical consent pages), the signature lines `NEUTRON HOLDINGS, INC. /
By: /s/ Chris Maher / Title: Authorized Signatory` are stripped from
`block_text` on every page — verified present in the XHTML/PAWLS layer on all
177 pages, absent from all blocks. The furniture heuristics treat
same-position-every-page repetition as headers/footers, and a mass-signature
exhibit meets that signature. For a legal parser, `/s/` execution lines are
substantive content; this is the one place the deliberate stripping deletes
text a consumer may care about (901 tokens on that doc). Recommended follow-up
below.

## Genuine losses — complete inventory

Across 5,850 pages and 2.59M tokens, the text Warp actually fails to carry on
its no-loss (PAWLS) surface:

1. **~10 tokens of 90°-rotated margin text** on 4 pages (arXiv sidebar, Federal
   Register margin keys) — pdfminer-stack limitation, decorative text.
2. **~6 embedded-text-layer tokens on rerouted scan pages** (page folio `C-9`,
   a few label tokens on FortWorth scans) — where OCR simultaneously *added*
   hundreds of tokens per page that no text-layer extractor can see.

No sentence, clause, paragraph, table cell, or heading of body text was lost
anywhere in the corpus. Additionally, two *garble* classes (overprint
char-doubling; letter-spacing/rotation char-explosion) keep the text in the
export but in a form that defeats search — they are fidelity, not loss, but are
the highest-value fixes.

## Recommended follow-ups (not in scope of this eval)

1. **Overprint dedupe** in the front-end (pdfplumber `dedupe_chars()` or
   equivalent) — kills the doubled-char class at the root, incl. the S-1
   draft-cover watermark garble.
2. **Signature-block exemption** in the furniture stripper (never strip a line
   containing `/s/` or `By:`-style execution fields) — restores signature text
   to the json/html renders.
3. **Merge embedded text with OCR** on rerouted sparse pages (union rather than
   replace), recovering folios like `C-9`.
4. Rotated-text capture is a pdfminer-stack limitation; revisit only if
   sidebar/margin text ever matters to a consumer.

## Reproducing

```bash
uv run python scripts/text_faithfulness_eval.py --out eval_results.jsonl   # full corpus, ~40 min on 4 cores
uv run python scripts/text_faithfulness_eval.py --out eval_results.jsonl --summarize
```

`tests/test_text_faithfulness.py` locks the guarantee in CI: PAWLS
effective recall vs the PDFium oracle ≥ 0.999 on the canonical fixture docs.
