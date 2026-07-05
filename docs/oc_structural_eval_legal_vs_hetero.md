# Structural-correctness evaluation: legal vs heterogeneous

Two committed golden-answer-set evals score Warp's OpenContracts structural
export — **labels** (`Title` / `Section Header` / `Paragraph` / `List Item` /
`Table Row`) and **relationships** (`parent_id` hierarchy) — on two 100-page
batches:

| Batch | Pages | Golden source | Suite |
|---|---|---|---|
| **hetero-100** | 100 single-page enterprise PDFs (insurance / finance / government), from ParseBench `layout` | **ParseBench human-verified `layout.jsonl`** (external truth) | `tests/test_oc_hetero100_regression.py` |
| **legal-100** | 70 EDGAR S-1 pages + 29 FortWorth municipal-contract pages | **vision-adjudicated** (two independent auditors per page) | `tests/test_oc_legal100_regression.py` |

Warp is tuned for legal contracts; the central question is whether its structural
quality is in fact higher on legal than on heterogeneous enterprise layouts, and
**which systematic issues** the heterogeneous set exposes that the legal corpus
hides.

## Method

* **Parser-independent golden.** Each page's truth is a list of regions
  `{bbox_frac, text, gold_label, gold_parent_region_id, ro_index}`. Live Warp
  annotations are aligned to gold regions geometrically (overlap + text-Jaccard,
  **many-Warp→one-gold** so per-row Warp tables roll up to one gold table
  region). Scorer: `tests/oc_golden_eval.py` (9 unit tests).
* **Metrics.** Per-label one-vs-rest **F1** over {Heading, Paragraph, List Item}
  (`struct_macro_f1` = their mean); **gold_coverage** / **spurious_frac**;
  **head-ancestor** + **parent-class** relationship agreement; **reading-order**
  agreement (hetero only). Tables are scored by **coverage** only, and gold body
  regions that Warp covered with Table Rows are pulled out into
  **`body_as_tablerow_frac`** — the multi-column-fusion / table-over-fire signal.
* **Oracle honesty.** ParseBench, like the Docling oracle, **under-labels
  tables** (it collapses a data table to one region or tags its cells as `Text`)
  and excludes images. So `Image` regions are not scored and Table↔Paragraph
  disagreement is reported as its own signal, never folded into the label F1.
* **Regression.** Both suites floor the agreement metrics and ceil the smells per
  page against a committed baseline (`tests/fixtures/{hetero,legal}100_baseline.json`);
  deterministic, so improvements pass and regressions fail.

## Heterogeneous-100 results (ParseBench oracle)

Batch means over 100 pages:

| metric | mean | reading |
|---|---:|---|
| `struct_macro_f1` | **0.469** | heading/list/paragraph labels, macro |
| heading_f1 | 0.523 | |
| list_f1 | 0.471 | |
| paragraph_f1 | 0.414 | |
| `gold_coverage` | 0.722 | 28% of gold regions get no Warp overlap |
| `spurious_frac` | 0.203 | Warp boxes over empty/invented area |
| **`body_as_tablerow_frac`** | **0.216** | **body fused into table rows** |
| table_region_coverage | 0.956 | real tables are covered well |
| head_ancestor_agreement | 0.370 | hierarchy vs reconstructed gold (noisy on grids) |
| parent_class_agreement | 0.760 | |
| reading_order_agreement | 0.914 | |
| furniture_as_heading (total) | 24 | running headers/banners promoted |

Aggregated label confusion (gold → Warp dominant):

```
Paragraph->Table Row 304   Paragraph->Paragraph 280   Heading->Table Row 196
List Item->Table Row 174   Heading->Heading    155   Table Row->Table Row 106
List Item->List Item  65   Heading->Paragraph   62   Paragraph->Heading    53
```
Spurious Warp boxes by class: **Table Row 903**, Paragraph 123, Heading 44, List 25.

### Systematic issues (heterogeneous)

1. **Multi-column fusion → body mislabeled `Table Row` (dominant issue).** On
   multi-column enterprise pages Warp reads across columns and packs the
   interleaved, scrambled text into `table_row` blocks. `Paragraph→Table Row`
   (304) + `Heading→Table Row` (196) + `List Item→Table Row` (174) dominate the
   confusion, 903 spurious table-row boxes, and `body_as_tablerow_frac` 0.22.
   *Evidence:* `chapter_10_p2` — Warp emits 58 Table Rows whose text is
   cross-column garble (`"Objectiv starch in paper es"` = "Objectives" interleaved
   with a side column) on a page whose gold is headings + a numbered list. This is
   the engine's reading-order / column problem, already noted as deferred
   (`fix2-table-row-and-multicolumn-deferred`); the heterogeneous set quantifies
   its cost. **Engine-level.**
2. **List under-detection** (`list_f1` 0.471; 174 list items fused to table rows,
   163 missed). Enumerated lists on dense pages are frequently absorbed into
   table rows or paragraphs. **Engine/exporter.**
3. **Running-header / banner promoted to heading** (`furniture_as_heading` 24):
   e.g. "THE YEAR IN REVIEW – COMPRESSOR TECHNIQUE", "MANAGEMENT'S DISCUSSION AND
   ANALYSIS", "Lancashire Holdings Limited | Annual Report". *Caveat:* the hetero
   fixtures are **single pages**, so Warp's cross-page repeated-furniture detector
   (`_furniture_demotions`, needs ≥3 pages) is structurally blind here — part of
   this is a single-page-fixture artifact, but a top-margin-banner geometric cue
   could still fire. **Exporter (geometric).**
4. **Weak hierarchy on complex pages** (`head_ancestor_agreement` 0.370). Partly
   real (multi-column fusion destroys structure), partly the reconstructed gold
   hierarchy being noisy on grid/multi-section pages (many same-depth headers →
   flat gold). Reported with that caveat; `parent_class_agreement` (0.76) is the
   more robust relationship signal.
5. **Coverage gap** (`gold_coverage` 0.722). ~28% of gold regions have no Warp
   overlap — a mix of image-caption text, fine sub-cells, and footnotes Warp
   groups differently.

Real data tables are covered well (`table_region_coverage` 0.956) — Warp's table
*detection* is strong; the problem is table *over-firing* on non-tabular
multi-column prose.

## Legal-100 results (vision-adjudicated)

Golden built by two independent vision auditors per page (opus + sonnet), 99
pages, ~1180 regions; **91.5% inter-auditor agreement**, 428 adjudicated
corrections to Warp's labels (disagreements keep Warp's label, so the floor only
penalizes Warp where *both* auditors agree it erred). Because the gold regions
are Warp's own boxes, **coverage and spurious are trivially perfect here** — the
legal suite measures Warp's **label** and **relationship** accuracy, not
segmentation (the heterogeneous suite covers segmentation against an independent
oracle).

| metric | mean | reading |
|---|---:|---|
| `struct_macro_f1` | **0.758** | heading/list/paragraph labels, macro |
| heading_f1 | 0.654 | |
| list_f1 | 0.824 | |
| paragraph_f1 | 0.797 | |
| `body_as_tablerow_frac` | 0.074 | body fused into table rows |
| head_ancestor_agreement | 0.653 | |
| parent_class_agreement | 0.518 | under-nesting (see below) |
| furniture_as_heading (total) | 37 | clerk stamps / running headers / e-sig |

Adjudicated corrections to Warp's labels (what Warp got wrong, by volume):

```
Section Header->Paragraph 92   Table Row->Paragraph 91   Paragraph->Furniture 61
Section Header->Furniture 39   Paragraph->Section Header 35   Paragraph->List Item 24
Section Header->Title 20   Section Header->List Item 13   Table Row->Furniture 12
```

**Clean-digital vs scanned/form split (the real axis).** The legal batch is not
uniform — EDGAR S-1 filings are clean digital prose; FortWorth municipal
contracts are scanned, signature/form-heavy:

| metric | EDGAR S-1 (70p) | FortWorth (29p) |
|---|---:|---:|
| struct_macro_f1 | **0.819** | 0.610 |
| heading_f1 | 0.768 | **0.378** |
| list_f1 | 0.879 | 0.689 |
| body_as_tablerow_frac | 0.038 | 0.160 |
| head_ancestor_agreement | 0.708 | 0.520 |
| furniture_as_heading | 15 | 22 |

FortWorth contracts — despite being legal — behave structurally much more like
the heterogeneous enterprise pages: form/signature grids fused into table rows,
running/clerk furniture promoted to headings, and real headings on scanned pages
under-detected (heading F1 0.38).

### Systematic issues (legal)

1. **Furniture not demoted** (112 corrections: 61 Paragraph + 39 Section Header +
   12 Table Row → `Furniture`). Clerk/filing stamps ("OFFICIAL RECORD / CITY
   SECRETARY"), e-signature audit lines, running headers/footers, and the FORT
   WORTH seal are kept as content or promoted to headings. Concentrated on
   FortWorth signature/routing pages. **Exporter-fixable (geometric/structural).**
2. **Section Header ↔ Paragraph confusion** (92 over-promoted + 35 missed). Warp
   promotes signature-block field labels ("Name:", "Date:", "By:") and run-in
   lead-ins to headers, and misses some real headings on scanned pages. Partly
   covered by existing run-in / prose-header demotion; gaps remain. **Mixed.**
3. **Table Row ↔ Paragraph** (91): signature/form grids and some multi-column
   prose fused into table rows (same root cause as the heterogeneous #1, milder
   here). **Engine-level.**
4. **Title under-emission** (20 Section Header → Title, 10 Paragraph → Title):
   Warp emits `Title` only via the cover detector, so cover/first-page document
   titles are usually left as `Section Header`. **Exporter-fixable.**
5. **Under-nesting** (`parent_class_agreement` 0.518): even where labels are
   correct, Warp frequently roots a block that the gold nests under a heading
   (especially on cover/title pages). **Hierarchy.**

## Legal vs heterogeneous

| metric | legal-100 | hetero-100 | comparable? |
|---|---:|---:|---|
| struct_macro_f1 | **0.758** | 0.469 | yes |
| heading_f1 | 0.654 | 0.523 | yes |
| list_f1 | 0.824 | 0.471 | yes |
| paragraph_f1 | 0.797 | 0.414 | yes |
| body_as_tablerow_frac | 0.074 | 0.216 | yes |
| head_ancestor_agreement | 0.653 | 0.370 | partly (gold-hierarchy noise) |
| parent_class_agreement | 0.518 | 0.760 | partly* |
| gold_coverage | 0.970 | 0.722 | **no** (legal gold = Warp's own boxes) |
| spurious_frac | 0.000 | 0.203 | **no** (same reason) |

\* the two goldens are built differently (legal = Warp-anchored vision
correction; hetero = independent ParseBench), so `parent_class_agreement` is not
strictly comparable across sets.

**Conclusion.** Warp's structural labeling is **markedly stronger on legal**
(macro-F1 0.76 vs 0.47) — it is tuned for prospectus-style legal prose, where it
nails lists (0.82), paragraphs (0.80), and headings (0.77 on EDGAR). The single
biggest cross-corpus weakness is **multi-column / table over-firing**
(`body_as_tablerow` 3× worse on heterogeneous: 0.22 vs 0.07), which is an engine
reading-order problem that surfaces on any column-heavy or form-heavy layout —
including the scanned FortWorth *legal* contracts (0.16). So the gap is better
described as **clean-digital-prose vs complex/scanned layout** than purely
legal-vs-not.

## Triage & fixes

| # | issue | locus | action |
|---|---|---|---|
| 1 | Furniture (stamps / e-sig / running headers) kept as heading/content | exporter (geometric/structural) | **fix** — extend furniture demotion |
| 4 | Cover/first-page Title under-emitted | exporter | **fix** — promote dominant cover line to Title (cover detector already exists) |
| 2 | Section Header ↔ Paragraph (signature fields, run-in) | exporter (partly covered) | **fix where format-only**; rest deferred |
| 3 | Table Row ↔ Paragraph (multi-column / form-grid fusion) | **engine** (reading order) | deferred — root cause is column reading order; large blast radius on the S-1 baselines (see `fix2-table-row-and-multicolumn-deferred`) |
| 5 | Under-nesting on cover/title pages | exporter/engine | partial via #4 (Title anchors the tree); deeper nesting deferred |

The fixes below stay **content-free / format-only** (geometry, structure,
verbatim repetition — never a corpus's words) per the project's standing rule,
and are TDD'd in `tests/test_opencontracts_export.py`.

### Fix outcome

**Shipped — cross-page repeated-position furniture demotion**
(`_repeated_position_furniture`). The FortWorth clerk stamps evade both existing
detectors: the verbatim cross-page furniture rule misses them because OCR
re-reads the stamp differently each page ("OFFICIAL RECORD" / "FFICIAL RECORD" /
"RECORD CITY"), and the single-page geometric deep-corner band cannot reach them
without collateral damage. The new detector keys on a header-ish block repeating
at the **same bottom-right position across ≥2 pages** — content-free,
OCR-robust, and structurally incapable of touching a one-off real heading.
Result: legal `furniture_as_heading` **37 → 29**; **zero regressions** across
`oc_export`, `oc_batch`, `s1`, `docsling`, and `hetero100`; the legal baseline
was re-floored to lock in the FortWorth furniture counts.

**The eval earned its keep mid-fix.** The first attempt simply widened the
single-page geometric band to reach the stamps. The **hetero-100 suite
immediately failed** on `ar2024e_p49`, catching that the widened band demotes a
genuine bottom-right heading ("Integrated Reports Inquiries") — pure position
cannot separate it from a clerk stamp. That false positive drove the redesign to
position-*repetition* (which a single-page heading can never trigger). This is
exactly the regression guard these suites exist to provide.

**Deferred (documented, not attempted here):**
- **Multi-column / table-over-fire (#3)** — the highest-value remaining item
  (`body_as_tablerow` 0.22 hetero). Root cause is the engine's column reading
  order; every retrofit tried historically corrupts real tables / reading order
  (`fix2-table-row-and-multicolumn-deferred`), and a fix re-floors the S-1 +
  docsling baselines. The two new suites now **quantify its cost and floor it**,
  so it can be attempted later with a safety net. Engine-level.
- **Signature-field Section Header ↔ Paragraph (#2)** — telling "Name:" / "Date:"
  / "By:" from "RECITALS:" / "ARTICLE 1:" requires *content*; no format-only rule
  separates them, so per the standing structural-only rule it is left alone.
- **Single-page running-banner furniture (hetero #3)** and **Title under-emission
  (#4)** — the former needs single-page top-banner geometry (same over-demotion
  risk as the reverted band); the latter is invisible to the coarse
  heading/​body metric. Both low-priority.
