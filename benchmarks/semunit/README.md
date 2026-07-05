# Semantic-Unit golden-answer benchmark (204 pages)

Measures the **`semantic_units` clause-grouping layer** against a human-granularity
golden — the honest replacement for `scripts/semunit_bench.py`, whose
`clean_unit_fraction` only checks 6 narrow defect classes and reported 1.00 while docs
were visibly mis-grouped.

## What it is
- **`golden_204_clause.json`** — the committed truth: for each of 204 pages, the *correct*
  clause grouping (which fine-block ids form each unit + nesting + furniture), adjudicated
  by 26 vision agents **blind to the processor**, at **clause granularity** (a clause =
  provision/section/recital/definition/signature-block; body paragraphs are members, only
  enumerated sub-clauses nest). 983 units.
- **`manifest.json`** — the 30 source docs / pages (20 fresh S-1 exhibits + the 10-doc
  prior audit set: agreements, a charter, a reserves report, a statute, municipal
  contracts, consents).
- **`prep.py`** — regenerates, into `audit_out/semunit_bench100/` (gitignored): per-page
  fine-block PNGs (block ids overlaid, for re-adjudication), `blocks.json`, and the
  processor's `proc_units.json`.
- **`score.py`** — scores the processor's grouping vs the golden.

## Metric
**Pairwise-membership F1** over each page's fine blocks (do golden & processor agree on
whether two blocks share a unit?), reported at two granularities:
- **subtree-F1** (roll nesting up to clause spans; Jaccard≥0.5 recovery) — the fair headline.
- **leaf-F1** (finest units) — stricter, granularity-sensitive.

Both penalize over-splitting (recall↓) *and* over-merging (precision↓).

## Run
```bash
python benchmarks/semunit/prep.py       # regen work dir (parses 30 docs, renders 204 pages)
python benchmarks/semunit/score.py      # score processor vs committed golden
```

## Benchmarking strategy (anti-overfitting governance)

The 204-page golden is the **development set**: rules are iterated directly
against it, so its score is IN-SAMPLE and must never be quoted as a
generalization claim on its own.  The honest headline is the pair:
development score + the **held-out check** (`heldout.py` /
`heldout_manifest.json` / `heldout_baseline.json`): 16 never-tuned docs / 511
pages (CUAD-class contracts, credit agreements, S-1 exhibits disjoint from
this manifest) scored by the numbering-derived oracle.  Rules of the road:

1. The held-out set is **scored, never tuned**.  A rule change that improves
   the 204-bench but regresses `heldout.py` is overfit — fix the rule.
   Re-baselining requires a commit message explaining the deliberate
   behavior change; never re-baseline to absorb a regression.
2. The oracle is a **drift detector, not truth**: it cannot see
   redaction-split siblings, letter granularity, or signature stacks.
   Chasing its per-doc numbers to zero is itself overfitting.
3. What the first litmus run caught (2026-07-02): the wave-3 salutation
   weak-root shattered ordinary colon-terminated clause headings on a CUAD
   sponsorship agreement, and one stray `[***]` flipped doc-wide redaction
   splitting in a 126-page lease; both were fixed at a cost of only
   -0.001 dev-set subtree-F1.  It also *saved* two rules the tuned-corpus
   firing census had wrongly called dead code (they fire on kardigan).
4. Corpus-word rules (municipal names, filer vocabulary) are banned even
   when they lift the dev score; the 2026-07-02 hygiene pass deleted them
   (dev 0.950 -> 0.941 — that delta is what memorization was worth).

## Current result (post-hygiene, 2026-07-02)
| | subtree-F1 (dev, in-sample) | leaf-F1 |
|---|--:|--:|
| overall (204pp) | **0.941** | 0.88 |

Held-out (numbering oracle, 511pp): windowdiff 0.206 / mean-unit-IoU 0.818 /
fragmentation 0.163 — at parity with the pre-tuning wave-1 layer (0.193 /
0.837 / 0.161), i.e. the wave-2/3 gains no longer come at held-out expense;
the residual gap is concentrated in the genuinely redacted eikon ex-10.11,
whose sibling splits the oracle cannot represent (the human golden validated
the same splits on its sister doc).

### Result before hygiene (wave 3, for reference)
| | subtree-F1 | leaf-F1 |
|---|--:|--:|
| overall (204pp) | 0.951 | 0.88 |

Per-doc subtree-F1: USC 0.99, exyn ex11 0.99, swarmer lease 0.98, quantinuum
bylaws 0.97, eikon 0.97, Eton 0.99, forbright 0.95, generate_bio 0.95,
fw_nctcog 0.95, exyn ex1010 0.94, fw_wert 0.94, fw_vertigis 0.93, fervo 0.91,
cerebras 0.86, fw_garver 0.82 (deep OCR garble — its p1 signature fields are
OCR-destroyed, the residual ceiling), spacex 0.69, consents 1.0.  (exyn ex11's
LEAF-F1 drops to 0.07: the redacted-sibling split shatters its multi-paragraph
items at leaf level while making every page-local clause span line up — the
headline subtree metric is the target.)

The 0.903 -> 0.951 wave (format/geometry only, never corpus words): run-in
leads accept dashes but not honorific periods ("Payment Method - Due Dates."
yes, "Dear Mr. Upreti:" no); repeated-line stamps (all-alpha key, trailing
folio stripped, >=40% of pages); Notes units own only their enumerated items;
redacted docs ([***]/[•]) split lost-marker sibling clauses inside open enum
items; weak roots (mixed-case colon salutations, first-page centered titles)
own no body — each paragraph is its own unit and numbered clauses are
top-level; cover disclaimers under one exhibit stamp stand alone; "/s/" rows
absorb city/date lines and welded twin rows; enum pseudo-heading runs after a
colon lead-in nest under it; sign-here marker captions stay with their field
group; sig stacks fixed (value-carrying seeds not standalone, column
adjacency, seed shadowing, DocuSigned-by never seeds); TOC runs carry group
headers, break at folios, and split per-article inside contents-titled units;
dotted-secnum pseudo-roots adopt letter sub-items; tight colon-row metadata
groups (incl. mis-promoted value headings); tall left-margin welded def rows;
letterhead runs (URL/phone token) are furniture; letter Re:/salutation lines
open their own units.

The 0.855 -> 0.903 wave (all format/geometry, never corpus words): citation
/data-row veto in the TOC detector + per-row units for disposition tables
(same-page heading nesting); form-question row splits ("...? * Yes No");
OCR-garble page collapse (junk-token + short-heading swarm, no sig fields ->
one unit per page); outdented left-margin def-table rows as per-row units
(with data-row and enumerator vetoes, root-open nesting); run-in clause
lead-ins own their enumerated sub-items; margin-stamp exclusion by repetition
+ bottom-band geometry (5-word key); junk-token upgrades (case scramble,
repeated-char runs, roman-numeral exemption); repeated-prefix heading
demotion ("Exhibit ..." index runs); id-lead data-heading demotion + joins;
same-lead colon-label field grouping; signature captions over e-sig lines;
seedless label-below-value signature grids group as one region unit.

### Result at the unit-layer revamp (wave 1, for reference)
overall 0.855 / leaf 0.79 — Eton 0.995, quantinuum 0.95, exyn 0.93–0.94,
generate_bio 0.91, USC 0.89, forbright 0.89, swarmer 0.88, eikon 0.86,
fw_wert 0.83, fw_nctcog 0.77, cerebras 0.74, fervo 0.74, fw_vertigis 0.71,
fw_garver 0.50. (The honest post-splitter baseline before the unit-layer
revamp was 0.703.)

### Result at PR #18 (pre-splitter, for reference)
| | subtree-F1 | leaf-F1 |
|---|--:|--:|
| overall (204pp) | 0.71 | 0.61 |
| clean / born-digital (900 units) | 0.75 | — |
| OCR / multi-column (197 units, 4 docs) | 0.46 | — |

Strong on clean contract prose (Eton **0.97**, exyn agreements 0.87–0.92, forbright 0.88,
generate_bio 0.85); weak on tables / signature-grids / OCR (fw_vertigis 0.35, fw_garver
0.40, cerebras 0.64). See `docs/` design notes and the PR description for the failure-mode
analysis. The multi-column XY-cut (PR #17) is a **no-op here** — it fires only on 2-column
*prose*; this benchmark's hard docs are 2-column *table/grid cells*, a distinct problem.

## Golden staleness after the grid-cell splitter

The front-end **grid-cell splitter** (`pdf_plumber_parser._detect_grid_regions`,
follow-up to PR #17) resolved that distinct problem at block formation: two-column
signature/approval/form grids no longer weld left+right cells into one block
("By:  By:", "Dana Burghdoff  Gary L. Wert"), so the fine segmentation changed on 5 docs
(fw_wert, fw_vertigis, fw_nctcog, fw_garver, exyn_s1__ex1010). `golden_204_clause.json`
still references the **old (fused) block ids** on those docs' pages, so `score.py` is only
valid for the other 25 docs until the golden is re-adjudicated (re-run `prep.py`, then the
vision-agent adjudication on the changed pages). A geometric remap of the old golden onto
the new blocks (old-member bbox overlap) scores exyn at 0.94 (up from 0.92) and the fw
docs flat-to-slightly-down — but that remap *inherits the fused-era adjudication* (e.g. a
fused row put one party's cell text inside the other party's clause), so it underestimates
per-party groupings. The remaining fw loss is now the **unit layer's** coarsening of the
(correctly separated) signature stacks — each mis-promoted cell header roots its own unit
— which was unfixable before the splitter and is the next work item, plus the deferred
data-table row granularity (cerebras, unchanged at 0.64).
