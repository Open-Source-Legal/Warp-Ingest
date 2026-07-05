# How to builds word-level bboxes for paragraph or line level bbox outputs (real example from LlamaParse)

## The key thing to understand first

**LlamaParse does *not* give us word-level bboxes.** It gives us *element-level* bboxes — one rectangle per layout block (a title, a paragraph, a table, a figure). There is exactly one bbox for a whole paragraph, not one per word.

So the word-level bboxes are **not derived by subdividing LlamaParse's bbox** (e.g. evenly distributing N words across the rectangle). That approach was explicitly rejected because it produces misaligned highlights — see the comment at `opencontractserver/pipeline/parsers/llamaparse_parser.py:1151-1155`.

Instead, the real word boxes come from a **second, independent source — `pdfplumber`** — and LlamaParse's element bbox is used only as a **query rectangle** to decide *which* of those real word boxes belong to each element. The mechanism is a spatial join.

```
LlamaParse element bbox ──┐
(normalized to absolute)  ├──► spatial intersection (STRtree) ──► token refs
pdfplumber word boxes ────┘                                       for that element
```

Two coordinate systems are reconciled into one set of absolute, top-left-origin PDF points, and then geometry does the matching.

---

## The data structures

- **LlamaParse element bbox** (input): a dict, in one of several shapes — `{x, y, w, h}` (its actual primary format), `{x1,y1,x2,y2}`, `{left,top,right,bottom}`, or a 4-array. Values may be **fractional (0–1)** or **absolute points**. Defined/handled in `_create_pawls_tokens_from_bbox`.
- **PAWLS word token** (`PawlsTokenPythonType`, `opencontractserver/types/dicts.py:75`): `{x, y, width, height, text}` in absolute top-left-origin points. This is the word-level box.
- **`BoundingBoxPythonType`** (`dicts.py:133`): `{top, bottom, left, right}` — the normalized element rectangle.
- **`TokenIdPythonType`** (`dicts.py:144`): `{pageIndex, tokenIndex}` — a *reference* from an annotation to a word token (annotations don't copy token geometry, they point at it).
- **Annotation** carries `bounds` (the element rect) + `tokensJsons` (list of token refs) + `content_modalities`.

---

## The pipeline, step by step

All of this lives in `LlamaParseParser._convert_json_to_opencontracts` (`llamaparse_parser.py:389`).

### Step 1 — Collect page dimensions from LlamaParse
First pass over `pages` reads `width`/`height` (with `w`/`h`/`pageWidth` fallbacks), validating they're positive and falling back to US-Letter `612×792` otherwise (`llamaparse_parser.py:425-451`). These dimensions matter because they (a) de-fractionalize LlamaParse coords and (b) are the target scale for the pdfplumber tokens.

### Step 2 — Extract real word tokens from the PDF (the actual word bboxes)
`extract_pawls_tokens_from_pdf(pdf_bytes, page_dimensions)` (`opencontractserver/utils/pdf_token_extraction.py:159`) runs `pdfplumber` and is where word boxes are born:

```python
words = pdf_page.extract_words(
    x_tolerance=2, y_tolerance=2,
    keep_blank_chars=False, use_text_flow=True,
)
for token_index, word in enumerate(words):
    x0 = float(word["x0"]) * scale_x      # pdfplumber: x0, top, x1, bottom (top-left origin)
    top = float(word["top"]) * scale_y
    x1 = float(word["x1"]) * scale_x
    bottom = float(word["bottom"]) * scale_y
    token = {"x": x0, "y": top, "width": x1 - x0, "height": bottom - top, "text": word["text"]}
```
(`pdf_token_extraction.py:249-296`)

Two important details:
- **Coordinate reconciliation via scaling** (`pdf_token_extraction.py:228-238`): pdfplumber's native page size is compared to LlamaParse's reported size, and every word coordinate is multiplied by `scale_x = llamaparse_width / native_width` (same for y). This guarantees the word boxes live in the *same coordinate space* as the normalized element bbox, which is the whole point — otherwise the spatial join would miss.
- **A spatial index is built per page** using Shapely's `STRtree` (`pdf_token_extraction.py:300-318`). Each word becomes a `shapely.box`, indexed for O(log n) intersection queries. The parallel `token_indices` array maps an index entry back to its position in the page's token list.

If `pdfplumber` fails or no `pdf_bytes` are supplied, the parser degrades gracefully to empty PAWLS pages, and annotations end up with empty `tokensJsons` (`llamaparse_parser.py:475-500`).

### Step 3 — Normalize each LlamaParse element bbox to absolute points
For each element (`items`, or `layout` as a fallback — `llamaparse_parser.py:716`, `:814`), `_create_pawls_tokens_from_bbox` (`llamaparse_parser.py:1004`) converts whatever shape LlamaParse gave into a single `{left, top, right, bottom}` rectangle in absolute points. It:

1. **Detects the bbox shape** by key sniffing (`x1`/`x`/`left`/array — `:1053-1125`).
2. **Decides fractional vs. absolute by heuristic.** For LlamaParse's primary `{x,y,w,h}` form, it treats the box as fractional only if *all four corners* `x, y, x+w, y+h` sit within `[0,1]`; otherwise it's absolute (`:1081-1095`). Fractional coords are de-normalized by multiplying by page width/height.
3. **Sanity-clamps the result** (`:1132-1149`): swaps inverted edges, clamps to page bounds, and enforces a ≥1pt minimum width/height.

Crucially, **this function returns `(tokens=[], bounds)` — an *empty* token list.** It never fabricates word tokens. Its only job is to produce a clean rectangle (`:1176-1177`).

### Step 4 — Spatial join: map the element rect onto real word boxes
This is where word-level bboxes get associated with the element. `find_tokens_in_bbox(bounds, page_idx, spatial_index, token_indices, tokens)` (`pdf_token_extraction.py:355`) is called immediately after normalization (`llamaparse_parser.py:736-742`):

1. Build a Shapely `box(left, top, right, bottom)` from the normalized element rect.
2. `spatial_index.query(query_bbox)` returns candidate words whose *bounding boxes* overlap (a coarse, fast filter).
3. Filter candidates by a true `geom.intersects(query_bbox)` test (the STRtree query is only a bbox-overlap prefilter).
4. Map surviving geometry indices back through `token_indices` to token positions, sorted, and emit `{pageIndex, tokenIndex}` refs.

The output is the set of **real pdfplumber word tokens that fall inside the LlamaParse element's rectangle** — i.e., the word-level resolution the element didn't natively have.

### Step 5 — (Figures/images) overlap or crop
For `figure`/`image`/`chart`/`diagram` elements, instead of text words it looks for image tokens overlapping the bounds via `find_image_tokens_in_bounds` (AABB test), and if none exists it crops the region from the PDF and registers a new image token (`llamaparse_parser.py:744-794`). Text and image refs are then combined.

### Step 6 — Assemble the annotation
`_create_annotation` (`llamaparse_parser.py:1213`) stores:
- `bounds` → the normalized element rectangle (Step 3),
- `tokensJsons` → the word/image token refs from the spatial join (Steps 4–5),
- `content_modalities` → `["TEXT"]`, `["IMAGE"]`, or both, driven by which kinds of refs were found (`:1258-1263`).

These structural annotations (`structural: True`, `annotation_type = TOKEN_LABEL`) are what the frontend highlights. When `tokensJsons` is empty (no PDF / extraction failed), the frontend just renders the bounding box without per-word highlights.

---

## Why it's built this way (the design rationale)

| Decision | Reason |
|---|---|
| Don't synthesize word boxes from the element bbox | Even distribution doesn't match real glyph positions → visibly wrong highlights (`llamaparse_parser.py:1151`). |
| Get word boxes from `pdfplumber`, not LlamaParse | LlamaParse has no token-level geometry; pdfplumber gives true per-word boxes. |
| Scale pdfplumber coords to LlamaParse page dims | Puts both sources in one coordinate space so the spatial join is valid (`pdf_token_extraction.py:232-238`). |
| Fractional-vs-absolute heuristic per format | LlamaParse is inconsistent about whether bboxes are 0–1 or points; the all-corners-in-`[0,1]` test disambiguates (`:1081-1095`). |
| `STRtree` spatial index + true-intersection recheck | O(log n) candidate lookup, then exact geometry test, instead of O(n) per element. |
| Graceful empty-token fallback | A missing/again-unparseable PDF still yields usable element-level annotations. |

---

## Where to look / tests

- Parser orchestration: `opencontractserver/pipeline/parsers/llamaparse_parser.py`
  - `_convert_json_to_opencontracts:389`, element loop `:690`, bbox normalize `_create_pawls_tokens_from_bbox:1004`, annotation build `_create_annotation:1213`.
- Word extraction + spatial join: `opencontractserver/utils/pdf_token_extraction.py`
  - `extract_pawls_tokens_from_pdf:159`, `find_tokens_in_bbox:355`, `find_image_tokens_in_bounds:~1184`.
- Types: `opencontractserver/types/dicts.py` (`PawlsTokenPythonType:75`, `BoundingBoxPythonType:133`, `TokenIdPythonType:144`).
- Tests: `opencontractserver/tests/test_doc_parser_llamaparse.py` — `TestLlamaParseParserBboxConversion:432` covers each bbox format and the fractional/absolute heuristic (e.g. `test_bbox_llamaparse_format` asserts `{x,y,w,h}` → `right = x+w`, `bottom = y+h`).
- Background on the target format: `docs/architecture/pawls-format.md`.

---

A one-line summary: **the word-level bboxes are pdfplumber's real per-word boxes; LlamaParse's element bbox is only normalized to absolute coordinates and then used as a geometric query to select which of those word boxes belong to each element via an STRtree spatial join — no word boxes are ever invented from the element bbox itself.**

Want me to turn this into a committed doc (e.g. a section under `docs/pipelines/` or `docs/architecture/`) on the `claude/llamaparse-word-bboxes-odw40v` branch, or keep it here in chat?
