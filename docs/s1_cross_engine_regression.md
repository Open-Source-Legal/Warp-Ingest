# S-1 cross-engine regression suite

This document describes the regression suite that pins the pure-Python engine's
output against the **original** Java/Tika [`nlm-ingestor`](https://github.com/nlmatics/nlm-ingestor),
using 50 real EDGAR S-1 PDFs as gold fixtures.

## Why this exists

Warp-Ingest replaced the original Java/Tika PDF front-end with a pure-Python one
(`pdfplumber` + optional `rapidocr`). The downstream rule engine (`visual_ingestor`)
was left untouched, on the theory that the new front-end reproduces the same
intermediate XHTML contract. This suite empirically verifies the *end-to-end*
consequence of that claim: **does the new engine produce documents compatible with
what the original engine produced?**

## What "compatible" means (and why not byte-exact)

The two front-ends segment words differently — pdfplumber reconstructs words from
glyph boxes; the nlm Tika fork uses its own logic — so a byte-exact XHTML/JSON match
is impossible *by construction*. The suite therefore compares on:

- **Content** — token recall/precision of the new output against the original,
  over the whole document. Tokens are alphanumeric and lowercased, so the metric is
  robust to punctuation/whitespace spacing artifacts.
- **Structure** — exact page count, block-type (`tag`) distribution, table count,
  and the similarity of the *ordered* block-text sequence (`difflib` ratio).

Both engines emit the identical schema (`result = {blocks, styles}`), so the
comparison is apples-to-apples. The gold targets were captured from the original
Java engine's API
(`POST /api/parseDocument?renderFormat=all&useNewIndentParser=yes&applyOcr=no`);
Warp's side parses with its single (default) pathway — the optional
`NewIndentParser` re-leveling pass was removed 2026-07-02, which is metrically
invisible here because it only rewrote `block["level"]` and no S-1 metric reads
levels (they score `tag` + text only).

## The corpus

50 PDFs sampled deterministically (stratified by company and size) from the 22
`*_oc_corpus_v3.zip` S-1 corpora in `EDGARx2`. Provenance for every fixture (source
zip + member) is recorded in `tests/fixtures/s1_manifest.json`. The set spans:

- 6 full S-1 / DRS bodies (183–440 pages: TOCs, footnotes, dense financial tables),
- ~20 medium exhibits (material contracts, credit agreements, leases: 16–294 pages),
- ~24 small exhibits (subsidiary lists, auditor consents, charters: 1–35 pages).

Sizes range 6 KB → 3.8 MB (median 84 KB). The biggest outliers (>4 MB) were
excluded to keep the suite runnable.

## Results

Gold generated from the unmodified upstream image
`ghcr.io/nlmatics/nlm-ingestor:latest`. Over the **49 docs the original parsed
successfully** (see the 50th below):

| metric            |   min |  median |   mean |
|-------------------|------:|--------:|-------:|
| token recall      | 0.842 |   0.999 |  0.992 |
| token precision   | 0.919 |   0.998 |  0.990 |
| block-seq ratio   | 0.000 |   0.893 |  0.818 |

- **Page count: exact match on 49/49.**
- **Token recall ≥ 0.95 on 47/49, ≥ 0.99 on 43/49.**
- **Tables: 1,395 (original) vs 1,434 (new) in aggregate — within 3%.**

Per-doc numbers are committed in `tests/fixtures/s1_baseline.json`.

## Where the engines differ (and which is better)

### 1. The original engine *crashes* on `swarmer_s1__ex1018` — ours does not ✅
The original returns HTTP 500 `{"reason": "'visual_lines'", "status": "fail"}` (a
`KeyError` in its own `visual_ingestor`) on this 45-page exhibit. The pure-Python
engine parses it into 45 pages / 212 blocks / 17 tables. This is a **strict
improvement**; the suite records `orig_failed: true` for this doc and asserts only
that the new engine keeps producing a sane parse.

### 2. Stylized stock certificate (`spacex_s1__exhibit41`) — neither is "right" ⚪
This is a graphic share certificate with letter-spaced / decorative text. The
original reads `C U S I P X X X...` (letters split by spaces); pdfplumber glues them
`CUSIPXXXXXX`. Tokenization is inherently ambiguous on such art, which is the sole
reason recall dips to 0.842. Not a content regression — both outputs are degraded.
The absolute recall floor is waived for docs where even the gold is degraded.

### 3. Numbered-definition / recital lists — ours arguably better for RAG ⚪→✅
On documents like `eikon_s1__ex1010`, the original emits a list marker (`"1.1"`,
`"A."`) as a *separate block* from its definition text; the new engine tends to keep
the marker with its text. Content recall is 100%; only segmentation differs. Keeping
a definition and its number together is generally **more useful** for retrieval,
though it lowers the ordered-sequence similarity vs the original.

### 4. Table granularity — net-neutral
On dense tabular pages, table detection occasionally splits or merges differently
(`generate_bio_s1__ex1015`: 21 vs 50; `cerebras_s1__exhibit1013`: 11 vs 7). In
aggregate over/under-detection nearly cancels (1,395 vs 1,434, ratio 1.03), so there
is no systematic bias. The suite bounds per-doc table drift rather than requiring
exact equality.

### 5. Cosmetic spacing around quotes/brackets
pdfplumber sometimes inserts spaces around curly quotes/brackets (`" Agreement "`
vs `"Agreement"`). This is purely cosmetic — it does not change alphanumeric tokens,
so it affects neither recall/precision nor the suite.

## How the test guards against regressions

`tests/test_s1_regression.py` runs the new engine on each PDF and asserts, against
the committed gold target + baseline:

- token recall ≥ an absolute floor (0.90, waived for gold-degraded docs) **and** ≥
  `baseline − 0.02` (no regression from today's level);
- token precision ≥ `baseline − 0.03`;
- page count exactly equal to the original's;
- ordered block-sequence similarity ≥ `baseline − 0.05`;
- table count within a tolerance band of the baseline.

The baseline freezes *today's* compatibility, so engine changes that **improve**
fidelity pass freely while regressions fail. Docs where the original engine crashed
are asserted only to keep producing a sane parse.

## Running

```bash
# fast subset (small/medium docs); large bodies are skipped by default
pytest tests/test_s1_regression.py -q

# full corpus including the large multi-hundred-page S-1 bodies
pytest tests/test_s1_regression.py --runslow -q
```

## Regenerating the fixtures

Targets come from the **unmodified** upstream engine; the baseline from ours:

```bash
# 1. run the original engine (read-only upstream image, nothing is modified)
docker run -d -p 5011:5001 ghcr.io/nlmatics/nlm-ingestor:latest

# 2. rebuild gold targets from the original, then re-bless our baseline
python scripts/build_s1_fixtures.py --targets --orig-url http://localhost:5011/api/parseDocument
python scripts/build_s1_fixtures.py --baseline
```

Re-run `--baseline` after any *intended* engine change to re-bless the baseline.
