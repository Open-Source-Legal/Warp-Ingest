# ParseBench integration for Warp-Ingest

Runs the **official** [LlamaIndex ParseBench](https://github.com/run-llama/ParseBench)
benchmark against Warp-Ingest and the local-library baselines
(LiteParse, MarkItDown, PyMuPDF, pypdf), so we get **leaderboard-comparable,
deterministic** numbers for how Warp-Ingest converts PDFs into agent-ready
structure.

This is a *faithful* recreation: we do **not** reimplement any scoring. We add a
`warp_ingest` provider to the official framework and let ParseBench's own
rule-based evaluators (GriTS/TableRecordMatch for tables, chart-data-point
matching, content-faithfulness, semantic-formatting, layout element pass-rate)
produce the scores.

**Determinism.** ParseBench's five **headline** metrics are rule-based. The one
exception is an *optional* `claude-haiku` "normalization" of **failed chart
rules** that the framework defaults *on* (`LLAMACLOUD_BENCH_LLM_NORMALIZATION`
defaults to `judge`) — but it only ever produces *separate* `*_judge` columns we
never read (the headline `avg_rule_pass_rate` is always the deterministic parent
value) and it needs an `ANTHROPIC_API_KEY` to do anything. Our `run.py` **forces
it off** (`=off`), so every reported number is produced with **zero LLM calls /
zero network** and is fully reproducible — no API keys needed.

## What's here

| File | Role |
|---|---|
| `warp_markdown.py` | Pure renderer: Warp blocks → per-page Markdown (HTML `<table>`s, ranked ATX headings, bullets) + per-block geometry. No `parse_bench` dependency, so it is unit-tested standalone. Table cells come from warp's own native table engine by default (region-aware replacement); content-stripping transforms are off by default. |
| `warp_ingest_provider.py` | ParseBench `@register_provider("warp_ingest")` PARSE provider, thin wrapper over `warp_markdown`. |
| `table_providers.py` | Pluggable table-cell providers. Default `WARP_TABLE_PROVIDER=native` = warp's own license-clean engine (`warp_ingest.ingestor.table_engine`, pure MIT stack — pdfplumber ruled grids + whitespace-channel grid inference). `pymupdf4llm` is kept as an opt-in ablation only (AGPL, and with `pymupdf-layout` installed its tables come from a Polyform-Noncommercial ONNX model); `none` disables. |
| `warp_layout_adapter.py` | Layout/visual-grounding adapter: turns Warp's per-block geometry into a `LayoutOutput` so warp is scored on the **Visual Grounding** dimension too (text-only baselines can't be). Registers a warp-keyed passthrough label mapper. |
| `run.py` | Drives the official ParseBench pipeline in-process for Warp-Ingest + baselines; prints a leaderboard-style comparison. |
| `setup_parsebench.sh` | One-time: clone pinned ParseBench + `pip install` the framework and baseline parsers. |

## Quick start

```bash
# 1) one-time setup (clones run-llama/ParseBench @ pinned commit, installs it +
#    liteparse/markitdown/pymupdf/pypdf into the current env)
bash benchmarks/parsebench/setup_parsebench.sh
# the two stronger local baselines (pymupdf4llm needs nothing extra; opendataloader needs Java 11+)
pip install pymupdf4llm opendataloader-pdf

# 2) quick directional run — 3 files/category (~12 pages), same-harness compare.
#    Default renderer mode is faithful: Warp output only (including warp's own
#    native table engine — no external parser), no CJK/header-footer stripping.
python -m benchmarks.parsebench.run --test

# 3) full, leaderboard-comparable run over all deterministic local parsers
python -m benchmarks.parsebench.run --full --force \
  --pipelines warp_ingest_faithful,liteparse_markdown,pymupdf4llm_markdown,opendataloader_markdown,markitdown,pymupdf_html,pymupdf_text,pypdf_baseline

# comprehensive table + faithfulness cross-check vs the published leaderboard.
# Missing in-run cells are not filled from published scores by default.
python scripts/parsebench_summarize.py --output-dir parsebench_work/output
```

Outputs land under `./parsebench_work/output/<pipeline>/` (per-category
`_evaluation_report.json` + HTML reports + a cross-pipeline leaderboard HTML).
The run is forced fully deterministic (`LLAMACLOUD_BENCH_LLM_NORMALIZATION=off`).
`--renderer-mode quality` is available for experiments that explicitly opt into
the local table-provider path (`WARP_TABLE_PROVIDER=auto`) and render-boundary
content stripping; do not use it as the primary faithful score.

Current faithful run (2026-07-02, native table engine): `warp_ingest_faithful`
scores **39.0 Overall** (Tables 57.4, Charts 7.1, Content Faith. 68.3,
Sem. Format. 43.5, Visual Ground. 18.6) — above every deterministic local
parser measured in this harness (pymupdf4llm 37.5, liteparse 32.8).

## The five dimensions

| Dimension | Metric | What it tests |
|---|---|---|
| Tables | GTRM (GriTS + TableRecordMatch) | merged cells, hierarchical headers |
| Charts | chart-data-point pass rate | exact values + labels from charts |
| Content Faithfulness | content-faithfulness score | omissions, hallucinations, reading order |
| Semantic Formatting | semantic-formatting score | bold / strike / super-/sub-script / titles |
| Visual Grounding | layout element pass rate | bbox localization + classification |

## Honest caveats

- **Warp is a layout/RAG parser, not a Markdown-fidelity parser.** It surfaces
  **bold** from the engine's own per-word font weight (the renderer wraps the exact
  bold runs in `**…**`; **no** synthetic formatting is invented), which makes it
  strong on *Semantic Formatting* (47.1 in the faithful rerun) — but it carries
  no strikethrough/super-/sub-script, so those rule types score low.
- **Within the local deterministic field, warp now leads Overall (39.0)** —
  with all output produced by this repo's own code (the native table engine is
  warp code, not an external parser). The comparison class is local
  deterministic parsers; geometry-native VLM/commercial systems score far
  higher on Visual Grounding. See [RESULTS.md](RESULTS.md) for the full
  breakdown, the native-vs-pymupdf4llm head-to-head, and honest deltas
  (Sem. Format. −3.7 vs the pre-table-engine run: bold table regions now render
  as cells, not `**` prose).
- **OCR.** Scanned (image-only) pages are routed to Warp's built-in auto-OCR
  (`rapidocr-onnxruntime`, the optional `[ocr]` extra). Install it before `--full`
  so the comparison matches `liteparse`, which OCRs scanned pages even in its
  "no-OCR" mode. The faithful full run parsed all PDF inputs; 42 layout inputs
  are `.jpg`/`.png` image files and are counted as zero by the official aggregate
  because the Warp provider is PDF-only.
- **Charts** are images; Warp (like every local parser) extracts no chart data
  points, so all sit near the floor — Warp's 7.1 is still low.
- **Visual Grounding** is scored via `warp_layout_adapter.py` from Warp's real
  per-block geometry, through the **official** `@register_layout_adapter` extension
  point (the same one LlamaParse/Docling/Azure use). The scorer strictly penalizes
  missing/extra/mislabeled elements, so it **cannot be inflated**. Warp's 18.6 beats
  the text-only local floor (~10, where pymupdf/pypdf/markitdown sit because they
  emit no geometry) but is **far below** geometry-native models (Docling 66.1,
  Azure DI 73.8, LlamaParse 80.6) — warp wins VG *within the local class only*.
- **`--test`** is 3 files/category (~12 pages): a directional, same-harness
  comparison only. Only **`--full`** is comparable to the published leaderboard.
- The official `liteparse` provider shells out to a Rust workspace binary; we
  redirect it to the PATH `lit` console script shipped by the `liteparse` wheel
  (same engine, same flags).

## Published leaderboard baselines (full dataset, for context)

Source: [`run-llama/ParseBench/leaderboard.csv`](https://github.com/run-llama/ParseBench/blob/main/leaderboard.csv)
(pinned `b74caa14`). Our run reproduces the **local/deterministic** rows below
exactly (except pymupdf4llm, where we run the newer `1.28.0` wheel) — see the
faithfulness cross-check in [RESULTS.md](RESULTS.md).

| Provider | Overall | Tables | Charts | Content Faith. | Sem. Format. | Visual Ground. |
|---|--:|--:|--:|--:|--:|--:|
| LiteParse (no OCR) | 32.8 | 40.3 | 3.4 | 68.6 | 44.6 | 10.7 |
| PyMuPDF4LLM (older build) | 30.9 | 36.7 | 1.6 | 60.9 | 44.6 | 10.7 |
| OpenDataLoader | 29.4 | 35.2 | 0.9 | 66.1 | 34.1 | 10.8 |
| pdf-inspector | 26.6 | 26.6 | 5.3 | 56.1 | 35.1 | 9.9 |
| MarkItDown | 18.6 | 15.8 | 2.0 | 64.5 | 0.9 | 9.9 |
| PyMuPDF (HTML) | 16.6 | 0.0 | 0.0 | 55.6 | 18.3 | 9.2 |
| PyMuPDF (Text) | 16.0 | 0.0 | 0.0 | 68.3 | 0.9 | 10.9 |
| pypdf | 14.9 | 0.0 | 0.0 | 62.5 | 0.9 | 10.9 |
| Docling-models (VLM, for scale) | 50.7 | 66.4 | 52.8 | 66.9 | 1.0 | 66.1 |

Warp-Ingest's own row (and the comprehensive 8-parser comparison) is produced by
the full run + `scripts/parsebench_summarize.py` above; see [RESULTS.md](RESULTS.md).
