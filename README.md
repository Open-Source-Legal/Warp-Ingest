<img width="620" height="423" alt="Warp-Ingest" align="center" src="https://github.com/user-attachments/assets/dffc8c8b-d7fc-4b64-8442-6ecce16a244f" />

# Warp-Ingest

**Warp-Ingest is a state-of-the-art, deterministic PDF parser.** It turns a PDF into
layout-aware structure — accurate word- and block-level bounding boxes, structural
labels (section header, paragraph, list item, table row), and the relationships
between them (the section/sub-section/paragraph **parent ↔ child** hierarchy) — and
renders it as blocks, JSON, HTML, or an [OpenContracts](https://github.com/JSv4/OpenContracts)
structural export.

It is **rule-based, not model-based**: structure comes from text coordinates,
graphics, and font data — no GPU, no per-page rasterization, no training set. That
makes it fast and predictable on long, text-layer documents (hundreds of pages), which
is where vision parsers are slowest and least stable. See
[Rule-based vs. model-based](#rule-based-vs-model-based) for the rationale.

Warp-Ingest is a pure-Python rewrite of the nlmatics `nlm-ingestor` engine, with the
Java/Apache-Tika and Tesseract dependencies removed.

## How the pipeline works

```
PDF ──pdfplumber──► per-word boxes + fonts ──► Tika-format XHTML ──► visual_ingestor ──► blocks / JSON / HTML / OpenContracts
       (scanned pages ──rapidocr──► OCR words ──► same XHTML ──┘)
```

- **`pdfplumber`** (MIT) extracts each word's real bounding box and font data in
  absolute, top-left-origin PDF points.
- A scanned or sparse page is detected automatically and routed to the optional
  **`rapidocr-onnxruntime`** OCR backend (Apache-2.0, no Tesseract binary, no GPU),
  which emits the **same** word-box format — so a scanned page and a born-digital page
  flow through the identical layout engine.
- The front-end emits an intermediate **Tika-format XHTML** (one `<p>` per visual line,
  carrying per-word positions and fonts). `visual_ingestor` — the ~6,000-line rule
  engine — groups those lines into typed blocks, detects tables from rule-line
  graphics, strips repeating headers/footers, removes watermarks, and fixes reading
  order.

The word-level boxes use the same technique as OpenContracts (see
[docs/bbox_architecture.md](docs/bbox_architecture.md)).

## What the parser produces

1. Sections and sub-sections with their nesting levels.
2. Paragraphs (lines joined into coherent blocks).
3. The parent ↔ child links between sections and paragraphs.
4. Tables, with the section each table sits in.
5. Lists and nested lists.
6. Content joined across page breaks.
7. Removal of repeating headers and footers.
8. Watermark removal.
9. OCR with bounding boxes for scanned pages.
10. An [OpenContracts structural export](docs/opencontracts_export_format.md): PAWLS
    word tokens, one structural annotation per block, and the `parent_id` heading
    hierarchy as explicit relationships.

## Benchmarks

Warp-Ingest is scored on the official
[LlamaIndex ParseBench](https://github.com/run-llama/ParseBench) — **2,078
human-verified pages** of real enterprise documents — by running it *through the
official framework* with deterministic, rule-based metrics (no LLM-as-a-judge). Among
the **8 deterministic, local, no-API parsers**, Warp-Ingest is a **top-2 result
(2nd overall)**, it is the **only** local parser that carries real **visual
grounding**, and it **leads on Charts**. Full numbers, methodology, and the
reproduction commands are in [benchmarks/parsebench/RESULTS.md](benchmarks/parsebench/RESULTS.md).

## Installation

```bash
# install the project and dev tools with uv
uv sync --group dev

# include the optional OCR backend for scanned PDFs
uv sync --group dev --extra ocr

# one-time NLTK data download
uv run python -m nltk.downloader punkt punkt_tab stopwords
```

## Running the service

```bash
python -m warp_ingest.ingestion_daemon      # or: ./run.sh   (FastAPI/uvicorn, port 5001)
```

The launcher budgets concurrency automatically from the CPUs actually available
(CPU affinity **and** the container cgroup quota, so a docker `--cpus` / K8s
`limits.cpu` deployment uses exactly its slice): one uvicorn worker per
effective CPU, with the front-end page-striping pool and the OCR session
threads sized so the layers never oversubscribe the box. Every knob can be
overridden via environment variables:

| Env var | Default | Effect |
|---|---|---|
| `WARP_API_KEY` | `abc123` | API key required by `/api/parse` (send as `X-API-Key` or `Authorization: Bearer`) |
| `WARP_WEB_WORKERS` (or `WEB_CONCURRENCY`) | effective CPUs | uvicorn worker processes |
| `WARP_FE_WORKERS` | `max(1, min(8, cpus // workers))` | per-worker front-end page-striping pool (≤1 = serial) |
| `WARP_OCR_THREADS` | `max(1, cpus // (workers × fe_workers))` | onnxruntime intra-op threads per OCR session |
| `WARP_WORKER_PARSE_SLOTS` | `1` | concurrent parses allowed inside one worker |
| `WARP_HOST` / `WARP_PORT` (or `PORT`) | `0.0.0.0` / `5001` | bind address |

`POST /api/parse` with a `file` form field and the API key; the response body is
the parse result (`{"page_dim": ..., "num_pages": ..., "result": ...}`) and
errors are standard `{"detail": ...}` bodies. Interactive OpenAPI docs at `/docs`.

```
curl -H "X-API-Key: abc123" -F file=@document.pdf \
  "http://localhost:5001/api/parse?render_format=all"
```

Query parameters (booleans accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`):

| Param | Values | Effect |
|---|---|---|
| `render_format` | `all` \| `json` \| `html` \| `opencontracts` | output rendering |
| `apply_ocr` | bool | force OCR on every page (scanned pages are OCR'd automatically regardless) |
| `disable_ocr` | bool | keep every page on its embedded text layer — no OCR for this request |
| `semantic_units` | bool | append the additive Semantic-Unit clause layer (`render_format=opencontracts`) |

`GET /` and `GET /healthz` are unauthenticated health endpoints (`/healthz`
reports the resolved concurrency settings and OCR availability). Warp-Ingest
parses **PDF only**; a non-PDF upload returns HTTP 415.

### Docker

```bash
docker build -t warp-ingest .
docker run -p 5010:5001 -e WARP_API_KEY=change-me warp-ingest
```

Every GitHub release publishes a versioned multi-arch image to GHCR
(`ghcr.io/open-source-legal/warp-ingest:X.Y.Z` / `:X.Y` / `:X` / `:latest`,
cosign-signed) via `.github/workflows/docker-publish.yml`.

### Library use

```python
from warp_ingest.ingestor import pdf_ingestor

export = pdf_ingestor.parse_to_opencontracts("document.pdf")   # OpenContracts export
ingestor = pdf_ingestor.PDFIngestor("document.pdf", {"render_format": "all"})
blocks = ingestor.blocks
```

The notebook
[pdf_visual_ingestor_step_by_step.ipynb](notebooks/pdf_visual_ingestor_step_by_step.ipynb)
walks the whole pipeline on a sample PDF.

## Testing

No Java or Tika is needed.

```bash
make test                  # full pytest suite (unit + fixture parsing, incl. OCR)
uv run pytest tests/       # same, directly
```

Beyond unit tests, the suite includes cross-engine regression against the original
Java/Tika engine (S-1), an OpenContracts-export regression, a Docling layout oracle,
and vision-adjudicated structural-correctness suites — all floored against committed
baselines so engine changes can only improve, never silently regress. See
[CLAUDE.md](CLAUDE.md) for the map of which suite guards what.

## Rule-based vs. model-based

Over four years the nlmatics team evaluated many options, including a YOLO-based vision
parser, and settled on the rule-based approach for these reasons:

1. **Speed.** It is ~100× faster than a vision parser, which must rasterize every page
   (even text-layer ones). A vision parser is the better tool for scanned PDFs with no
   text layer or small form-like documents; for large text-layer PDFs spanning hundreds
   of pages, a rule-based parser is far more practical.
2. **No special hardware.** It runs on CPU.
3. **Fixable.** Vision-parser errors are fixed either by adding training examples (which
   can degrade previously-correct behavior) or by layering on rules anyway — at which
   point you are writing rules again.

## Credits

The PDF parser
[visual_ingestor](warp_ingest/ingestor/visual_ingestor/visual_ingestor.py) and its
indent parsers were
written by Ambika Sukla, with contributions from Reshav Abraham, Tom Liu (the original
Indent Parser), and Kiran Panicker (parsing speed, table-parsing, indent-parsing, and
reordering accuracy). The core
[line_parser](warp_ingest/ingestor/line_parser.py) was written by Ambika Sukla.

Thanks to the `pdfplumber` / `pdfminer.six`, `pypdfium2`, and `RapidOCR` open-source
communities, and to the Apache PDFBox and Tika developers whose XHTML format the engine
is built around.

## History

Earlier versions depended on an nlmatics-modified Apache Tika (`nlm-tika`) on the JVM
plus Tesseract for OCR. That is gone — the pure-Python front-end
([`pdf_plumber_parser.py`](warp_ingest/file_parser/pdf_plumber_parser.py) and
[`ocr_parser.py`](warp_ingest/file_parser/ocr_parser.py)) reproduces the same
intermediate XHTML contract, so the layout engine did not have to change.
