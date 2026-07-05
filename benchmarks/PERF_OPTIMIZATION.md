# Warp parse-throughput optimization (2026-06-28; rounds 2–3 2026-07-02)

Goal: make warp's parse speed competitive without regressing quality or tests.
Methodology and levers, with measured effect and regression risk.

## Round 3 (2026-07-02): single-thread — engine micro-opts + Python 3.14 runtime

Round 2's parallel front-end doesn't help single-page documents (ParseBench's
whole corpus) or `WARP_FE_WORKERS=1` deployments, so round 3 targets the
serial path. All changes are again **hash-identical** to the unmodified
pipeline (XHTML + `all`/`json`/`html` renders over the 13-doc corpus) — and
that equivalence was *also* verified across interpreters (3.12 vs 3.14), so
benchmark scores cannot move by construction.

| config (dense set, no-OCR, workers=1, back-to-back) | front-end | engine | total | **pages/s** |
|---|---|---|---|---|
| round-2 tip, Python 3.12 | 8823 ms | 2888 ms | 11711 ms | **24.5** |
| + engine micro-opts, Python 3.12 | 8943 ms | 2545 ms | 11488 ms | **25.0** |
| + **Python 3.14 runtime** | 5742 ms | 1912 ms | 7655 ms | **37.5** |

1. **Engine micro-opts** (−12% engine): `sent_tokenize` prefilter v2 anchors
   each needle to its rule's pattern shape (`^abb` → startswith; `\sabb` →
   whitespace-preceded), pruning the ~200 rules/sentence whose short
   abbreviations ("no", "st") pass a bare substring check; cached per-word
   `LineStyle` + `get_numeric_font_weight` (immutable namedtuples, a handful
   of fonts per doc); `indent_parser.get_level` replaces a `level_stack`
   deepcopy with per-entry shallow dict copies (entries are flat scalar dicts).
2. **Python 3.14 runtime** (1.50× single-thread: front-end 1.56×, engine
   1.33×): the remaining front-end floor is pdfminer's pure-Python
   content-stream interpreter — exactly what a faster CPython accelerates.
   Text-extraction output is hash-identical to 3.12 across the corpus, and
   OCR output is token-identical on the same host (verified serial and
   in-worker; container-vs-host OCR can still micro-drift on individual glue
   points — a pre-existing environment sensitivity of rapidocr, independent
   of the interpreter). The full `--runslow` suite passes on 3.14 locally
   (704/704, and ~15% faster) and in the CI matrix. Shipped as the default
   runtime: `python:3.14-bookworm` base images (add an untracked `.python-version`
   with `3.14` to opt in locally; the file is gitignored by project convention;
   supported range stays 3.10–3.14 and the experimental JIT stays off — measured
   neutral-to-negative on this workload). The images install via
   `uv sync --frozen` — pip refuses rapidocr on 3.14 (`Requires-Python <3.13`
   metadata, though it runs green), and the old `RUN ... || true` was silently
   swallowing exactly that failure. Caution for local envs: switching the
   interpreter recreates `.venv`, and a plain `uv run` rebuilds it *without*
   the `[ocr]` extra — re-run `uv sync --group dev --extra ocr`.

Not pursued: pypdfium2 extraction (rewrites token geometry → threatens
geometry-floored baselines), BS4→lxml (~3%, adds a parser-dependent output
risk), pdfminer hexstring micro-patch (~2%, third-party internals), swapping
the punkt sentence tokenizer (output-changing).

## Round 2 (2026-07-02): parallel front-end + engine caches — 2.6× over round-1 HEAD

Round 1's Lever A + `check_date` gate were already on `main` (21.9 pg/s
aggregate on this box, front-end 67% / engine 33%). Round 2 adds three
output-identical levers on top (all proven hash-identical to the unmodified
pipeline across the 13-doc dense corpus — XHTML *and* `all`/`json`/`html`
renders):

1. **Parallel page extraction** (`pdf_plumber_parser`): pages are striped
   across a persistent spawn-based process pool (default `min(8, cpus)`
   workers, docs ≥ 8 pages); each worker opens the PDF independently and
   returns rendered page divs, reassembled in order → byte-identical XHTML.
   `WARP_FE_WORKERS` (≤1 = serial) and `WARP_FE_PARALLEL_MIN_PAGES` tune it;
   any pool failure falls back to the serial loop. **~3.7× front-end** on the
   dense set.
2. **`sent_tokenize` abbreviation-rule prefilter** (`ingestor_utils/utils`):
   the ~700 precompiled abbreviation regexes are skipped unless the rule's
   literal needle occurs in the casefolded text (a pure necessary condition) —
   ~2 orders of magnitude fewer regex scans per sentence.
3. **Line/Word LRU caches** (`line_parser.bare_line` / `make_word`): the
   engine re-parses the same block text in several phases (block typing,
   organize/indent, level assignment); style-free `Line` and `Word`
   construction are pure functions of the input string and nothing mutates
   them afterwards, so instances are shared via bounded LRU caches
   (`WARP_LINE_CACHE=0` bypasses). **~1.6× engine** measured cold (caches
   cleared per parse in the harness; within-document reuse only).

| config (dense set, no-OCR, same box/session) | front-end | engine | total | **pages/s** | vs round-1 HEAD |
|---|---|---|---|---|---|
| round-1 HEAD (`main` @ 06d577c) | 8749 ms | 4351 ms | 13100 ms | **21.9** | 1.00× |
| + parallel front-end | 3029 ms | 4391 ms | 7420 ms | **38.7** | 1.77× |
| + engine levers (cold caches) | 2335 ms | 2712 ms | 5046 ms | **56.9** | **2.60×** |

Worst-case big doc (exyn, 57pp): 17.0 → 51.1 pg/s (3.0×). Small docs below the
8-page gate keep the serial path (pool startup would dominate). The timing
harness clears the line caches before every engine run so the number is what a
single real-world parse sees, and sets `WARP_DISABLE_OCR=1` / serial-forcing
env vars so `--no-ocr` / `--force-fallback` reach the spawned workers.

## How to reproduce

```bash
# Front-end vs engine attribution, per-doc pages/s (sequential, warm, same corpus)
python -m benchmarks.timing_harness --set dense --no-ocr --runs 3
# before/after the front-end change (forces the original pdfplumber path):
python -m benchmarks.timing_harness --set dense --no-ocr --force-fallback --runs 3
```

`--no-ocr` disables sparse-page OCR routing to isolate the text-extraction path
(see "Confounders" below). `--force-fallback` turns Lever A off (original path).

## Profiling first — cost attribution (no-OCR, dense docs)

| stage | share of time |
|---|---|
| **front-end** — `pdf_plumber_parser` (pdfminer/pdfplumber word/char/font extraction) | **~78%** |
| **engine** — `visual_ingestor` rule engine (+ indent/sentence) | **~22%** |

Within the front-end, ~45% of extraction cost is pdfplumber's per-object
`process_object` machinery (`resolve_all` over colour attrs we never use, unicode
normalize, graphicstate colours); the rest is pdfminer's content-stream
interpretation. Within the engine, `line_parser.parse_line` is ~59% (it builds
~50 `Word()` objects per visual line).

### Confounders that make naive timing wrong (both bit us)
1. **OCR auto-routing.** Pages with <4 text lines auto-route to neural OCR even
   with `apply_ocr=False`; one OCR'd page (~0.5–1s) swamps text-extraction cost.
   Many S-1 exhibits have a few sparse pages. Measure with OCR disabled.
2. **pdfplumber caches `page.chars`.** Re-timing a warm page reuses the cache and
   flatters the reference; the real parser opens fresh (cold) and touches each
   page once. Always measure cold / back-to-back in one process.

## Levers

### Lever A — pdfminer-direct fast extraction (SHIPPED, zero risk)
`_fast_page_objects` reads `LTChar`/`LTLine`/`LTRect` straight off pdfminer's
layout tree and builds minimal char dicts (only the keys `WordExtractor` needs),
using pdfplumber's exact mediabox top/bottom formula, then runs the *same*
`WordExtractor`. Skips `process_object` entirely.
- **Byte-identical XHTML** — proven across 16 docs / ~290 pages incl. OCR pages
  (fast path vs pdfplumber fallback produce identical output).
- **~1.73× front-end** (controlled back-to-back cold A/B; range 1.43–2.18×).
- Falls back to pdfplumber if the fast path ever raises.

### Engine — `check_date` strptime gate (SHIPPED, zero risk)
`Word.check_date` ran up to 9 `strptime` calls for any token containing `-`/`/`
— firing on common words ("well-known", "e-mail"). Every date pattern needs a
numeric field, so we skip when the token has no digit. **Output-identical**
(0 mismatches / 3920 tokens) with a new `test_check_date`.

### Python 3.14 runtime — free ~1.3–1.6× (RECOMMENDED; verify deps)
Same code on CPython 3.14 (no JIT): engine **1.33×**, full pipeline **1.58×**.
Lever A + 3.14 ≈ **2.35× over the original on dense docs** — meets the target
with no code risk. Caveat: 3.14 needs `setuptools` (distutils shim) and 3.14
wheels for all deps; verify `onnxruntime`/`rapidocr` (OCR path) before adopting.

### Experimental JIT (`PYTHON_JIT=1`) — NOT recommended
Prebuilt 3.14 (uv/python-build-standalone, registered as pyenv `3.14.4-jit`)
ships the JIT compiled in, but enabling it was **neutral-to-slightly-negative**
on the engine (1101 ms vs 1070 ms JIT-off). Not mature for this workload.

### Lever B — pypdfium2 extraction (EVALUATED; gated on re-baselining)
~4× front-end (2.2× beyond Lever A). High fidelity vs current engine: token
recall 0.985–1.0, tables exact, block counts ±1–3. **But** pdfium reports
glyph-ink/font-line boxes, not pdfminer's font box, so it cannot be
byte-compatible — it rewrites every token's geometry, which threatens the
geometrically-floored regression baselines (the docsling layout baseline is
env-fragile; OC floors `anchored_fraction`/`token_coverage`/`tightness`). Viable
only if those committed baselines are regenerated. Prototype: see the design
notes; not enabled by default.

## Before / after (dense set: 14 docs / 287 pages, no-OCR, warm-median, same session)

| config | front-end | engine | total | **pages/s** | vs original |
|---|---|---|---|---|---|
| A) Original front-end, Python 3.12 | 15094 ms | 4310 ms | 19404 ms | **14.8** | 1.00× |
| B) **Lever A**, Python 3.12 | 8920 ms | 3998 ms | 12918 ms | **22.2** | **1.50×** |
| C) **Lever A**, Python **3.14** (no JIT) | 5079 ms | 2992 ms | 8071 ms | **35.6** | **2.41×** |

- Lever A cuts front-end **1.69×** (15094→8920 ms); engine ~flat as expected.
- Python 3.14 adds **1.44×** on the engine (4310→2992) and more on the
  pure-Python front-end → combined **front-end 3.0×** (15094→5079 ms).
- **Lever A + 3.14 = 2.41× overall on dense docs** — meets the 2–3× target with
  no regression risk. Worst-case single doc (exyn, 57pp): 11.6→27.7 pg/s (2.39×).
- Front-end's time share falls 78% → 69% → 63%, confirming it was the right
  target. pdfium (Lever B) would push the front-end further (~4×) but is gated on
  baseline regeneration (see above).

## No-regression verification
Clean-HEAD `--runslow`: 1 failed (`spacex_ex41` docsling — pre-existing
env-fragile), 267 passed. With Lever A + `check_date` gate: same result (byte/
output-identical → deterministic suites unchanged). ParseBench warp scores are
unchanged by construction (identical XHTML → identical blocks → identical
markdown).
