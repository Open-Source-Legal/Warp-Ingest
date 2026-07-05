# Runbook: structural-quality audit of `OpenContractDocExport` (+ regression lock-in)

A repeatable procedure for auditing the parser's structural output on a batch of
PDF pages — **parent↔child relationships**, **bbox tightness**, and **structural
labels** — fixing the exporter-level gaps it surfaces, and freezing the result
into the automated regression suite. Written so someone who has never run it can
execute it end to end.

> Prior runs of this procedure produced the bbox projector, the run-in / tree /
> coverage / re-segmentation fixes, and the `OC_PARENT_CHILD` relationship work.
> See `docs/oc_export_heuristic_audit.md` for the findings those produced.

---

## 0. Background — what you're working with

Warp-Ingest turns a PDF into an **`OpenContractDocExport`**: PAWLS word tokens,
one labelled annotation per layout block (`Section Header` / `Paragraph` /
`List Item` / `Table Row`), a `parent_id` hierarchy, and explicit
`OC_PARENT_CHILD` relationships. Format spec: `docs/opencontracts_export_format.md`.

Key code (all reachable with `.venv/bin/python`):

| What | Where |
|---|---|
| PDF → export dict | `warp_ingest.ingestor.pdf_ingestor.parse_to_opencontracts(path)` |
| The exporter (where fixes go) | `warp_ingest/ingestor/opencontracts_exporter.py` (`to_opencontracts_export`, `validate_export`, `_resolve_label`, `_segment_block`, `_list_intro_parents`, `_parent_child_relationships`) |
| Visualization / diagnostics | `warp_ingest/ingestor/oc_visualize.py` (`project_page`, `project_document`, `render_tree_outline`, `diagnostics`, `page_summary`, `build_tree`) |
| CLI: one PDF → PNGs + tree | `scripts/oc_visualize.py PDF --pages 0,2,5 --out DIR` |
| Driver: a page manifest → PNGs + summaries + diagnostics | `scripts/oc_sample_audit.py --out DIR` |
| PDFs to sample from | `tests/fixtures/`, `tests/fixtures/s1/` (50 EDGAR S-1s), `files/pdf/` (academic paper, credit agreements, contracts, 8-K, primary-document) |

**Hard constraint:** all fixes go in `opencontracts_exporter.py` (the export
layer). **Do NOT edit `visual_ingestor.py` / `indent_parser.py` / the XHTML
contract** — that is what the S-1 regression suite guards, and editing it is out
of scope for this runbook. Format with `black` + `isort --profile black` before
committing.

---

## 1. Goal

Audit **a fresh batch of diverse pages** (default: 30) on three quality
dimensions, fix the clearly exporter-fixable gaps, and **add those pages to the
automated regression suite** so the quality can't silently regress later. Store
before/after PNGs for review and update the open PR.

---

## 2. Procedure

### Step 0 — Confirm a green baseline
```bash
cd /home/jman/Code/nlm-ingestor
.venv/bin/python -m pytest tests/test_opencontracts_export.py tests/test_oc_visualize.py -q
```

### Step 1 — Choose the pages
Pick pages that **expand** coverage — different documents / page-types than those
already in `scripts/oc_sample_audit.py`'s `SAMPLE` list. Aim for variety: deeper
pages of the long credit agreements & S-1 bodies, two-column academic pages,
table pages, TOCs, defined-term sections, signature/exhibit pages, a scanned/OCR
page. Record the manifest as `(pdf_path, page_index)` tuples (copy the shape of
`SAMPLE`).

### Step 2 — Generate projections + structure for each page
Per PDF:
```bash
.venv/bin/python scripts/oc_visualize.py "<pdf>" --pages 3,7,12 --out audit_out/<slug>
```
or add the pages to a copy of `scripts/oc_sample_audit.py` and run
`--out audit_out`. Each page yields a **projection PNG** (boxes + relationship
arrows, colored by subtree) and a `tree.txt` outline; capture `diagnostics(export)`
per doc too.

### Step 3 — Judge each page on the three dimensions
Look at the projection PNG **and** the `tree.txt` / `page_summary` together:

1. **Parent↔child relationships rational** — does each `parent_id` /
   `OC_PARENT_CHILD` edge match the real structure? (list items under their
   lead-in paragraph; sub-clauses under their heading; no body orphaned at the
   document root; no mis-nesting; cross-page edges correct.)
2. **Bounding boxes tight** — does each box closely enclose its actual tokens?
   Flag boxes that **under-cover** visible text or **balloon** over
   whitespace/neighbors. (Untokened annotations falling back to the block rect
   are the usual offenders.)
3. **Structural labels good** — are the four labels correct? (no run-in heading
   mislabeled as a header; no page-number / metadata / watermark as a heading; no
   obvious list↔paragraph confusion.)

Record concrete defects: `(annotation_id, dimension, severity, one-line note)`.

*How to judge:* open the PNGs yourself, or hand each PNG + its `page_summary` to a
vision-capable model. (This repo's session tooling has a multi-agent **Workflow**
that fans one vision agent out per page and synthesizes the defects — the "same
procedure" used in prior runs. Without it, a manual or single-model pass is fine.)

### Step 4 — Fix the exporter-fixable gaps
Revise heuristics **in `opencontracts_exporter.py` only** (precedents:
`_resolve_label`, `_segment_block`, `_list_intro_parents`,
`_parent_child_relationships`). For each change:
- **Write the unit test first** (`tests/test_opencontracts_export.py`), watch it
  fail, then implement (TDD).
- Keep `validate_export` passing.
- Anything layout-engine-bound, or that trips the Docling oracle: **document it
  as deferred**, don't force it.

### Step 5 — Roll the pages into the regression suite
Add a **deterministic, floored** regression, mirroring the two existing patterns
(`tests/oc_compat.py` + `tests/fixtures/oc_export_baseline.json` +
`tests/test_opencontracts_regression.py`; and `tests/docsling_compat.py` +
`tests/fixtures/docsling_layout_baseline.json` +
`tests/test_layout_docsling_regression.py`):

1. **Metrics function** computing per-page deterministic signals — e.g.:
   - `relationship_validity` — `validate_export` passes & every relationship
     resolves (boolean; must stay true);
   - `childbearing_non_headers`, `non_heading_roots`, `overlong_headings`,
     `untokened` (counts; **ceiled** — must not grow);
   - `token_coverage`, `anchored_fraction` (**floored** — must not drop);
   - **bbox tightness** — per annotation `token_union_area / annotation_box_area`;
     emit the per-page mean (**floored**) and the count of "loose" boxes where
     tokens fill `< ~0.5` of the box (**ceiled**).
2. **Freeze** the current values into a committed baseline JSON, one entry per
   `(doc, page)`.
3. **Test** (`tests/test_<name>_regression.py`) recomputes live and fails on
   regression (floors for coverage/tightness, ceils for the smell-counts), with
   small tolerances — improvements pass, regressions fail.
4. **Regeneration script** `scripts/build_<name>_fixtures.py`; document it in
   `CLAUDE.md` alongside the other suites.

> LLM judgments are non-deterministic — keep them as the *discovery* tool, not the
> gate. The committed test floors **deterministic** metrics only.

### Step 6 — Verify, store artifacts, update the PR
```bash
.venv/bin/python -m pytest \
  tests/test_opencontracts_export.py tests/test_oc_visualize.py \
  tests/test_opencontracts_regression.py tests/test_layout_docsling_regression.py --runslow \
  tests/test_s1_regression.py tests/test_fixtures_parse.py -q
.venv/bin/python -m black . && .venv/bin/python -m isort --profile black .
```
Save before/after projection PNGs for the revised pages under `review_pngs/`
(gitignored, for human review). Commit on the working branch and update the open
PR with: the page manifest, defects per dimension, the heuristic changes, and the
new regression baseline.

---

## 3. Definition of done

- The batch of pages audited on all three dimensions; defects recorded.
- Clear exporter-fixable gaps fixed (each with a test); layout-engine /
  oracle-bound gaps documented as deferred.
- A committed deterministic baseline + a passing floored regression test covering
  the batch, plus a regeneration script.
- Full suite green. (The pre-existing `spacex_ex41` docsling fixture may fail
  *locally* due to a `pdfplumber` env drift — confirm it fails identically on a
  clean checkout and is exactly neutral to your change before dismissing it.)
- `black` / `isort` clean.
- Before/after PNGs in `review_pngs/`; PR updated.

---

## 4. Gotchas

- `poetry` may not be on `PATH`; call tools via `.venv/bin/python -m black …` /
  `… -m isort …` / `… -m pytest …` rather than `make`.
- `parse_to_opencontracts` prints engine progress to stdout — redirect it
  (`contextlib.redirect_stdout`) when scripting so it doesn't drown your output.
- Adding annotations (e.g. splitting a block) *raises* `annotation_count`; the
  regression floors are minimums, so that passes. Dropping annotations can trip
  the count floor — prefer relabel / re-parent / re-assign over delete.
- A change that shifts an annotation's **bounds** reorders the Docling word
  stream and can move `word_seq_similarity` / `label_agree` even when content is
  unchanged; always re-check the docsling suite (`--runslow`) after token/bounds
  changes, and verify neutral-or-better vs a clean checkout per fixture.
