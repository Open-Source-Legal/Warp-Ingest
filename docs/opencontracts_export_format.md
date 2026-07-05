# OpenContracts `OpenContractDocExport` — output format spec

Self-contained contract for the dictionary a parser/structure-engine must
produce for one document. Three layers: **document metadata**, a **PAWLS token
layer** (geometric word/image grid), and an **annotation layer** (labeled
regions + hierarchy + relationships). Serializes to plain JSON.

> This is the canonical reference Warp-Ingest's exporter targets. The exporter
> lives in `warp_ingest/ingestor/opencontracts_exporter.py`; design notes in
> `docs/superpowers/specs/2026-06-27-opencontracts-export-design.md`.

## 0. Conventions (read first)

- **Units & coordinates:** PDF points (1 pt = 1/72 inch). Origin is **top-left**;
  `x` increases right, `y` increases down (matches pdfplumber/PDFium, *not* PDF's
  native bottom-left origin).
- **Pages are 0-indexed.** `pawls_file_content` is a list where **list position
  == page index**. Don't compact it (no gaps); include an entry per page even if
  empty.
- **Everything spatial is absolute** page-point coordinates, clamped to the page
  box (`0 ≤ x ≤ page.width`, `0 ≤ y ≤ page.height`).
- **Tokens are referenced by index**, never duplicated: an annotation points at
  `(pageIndex, tokenIndex)` pairs into `pawls_file_content[pageIndex].tokens`.
- **IDs are export-local strings** (e.g. `"0"`, `"b12"`), unique within this
  document. They are *not* database IDs; the importer remaps them. `parent_id` /
  relationship references use these same local IDs.

## 1. Top-level: `OpenContractDocExport`

```ts
interface OpenContractDocExport {
  // ---- document metadata ----
  title: string;                 // required; "" allowed
  content: string;               // required; full plaintext of the doc (see note)
  description: string | null;    // required key; may be null

  // ---- PAWLS token layer (geometry) ----
  pawls_file_content: PawlsPage[]; // required; one entry per page, in page order
  page_count: number;            // required; integer page count

  // ---- annotation layer ----
  doc_labels: string[];          // required; document-level label names (usually [])
  labelled_text: Annotation[];   // required; the blocks/regions (your structure output)
  relationships?: Relationship[];// optional; typed edges between annotations

  // ---- optional provenance (safe to omit) ----
  file_type?: string | null;        // MIME, e.g. "application/pdf"
  structural_set_hash?: string | null;
  pdf_file_hash?: string | null;    // sha-256 of source bytes
}
```

**`content` note:** for PDF documents that include `pawls_file_content`, the
canonical text layer is rebuilt from token order downstream, so `content` is
effectively a fallback/secondary copy — but it is still required and must be a
string (use your joined token text or markdown).

## 2. PAWLS token layer

```ts
interface PawlsPage { page: PageBoundary; tokens: PawlsToken[]; }

interface PageBoundary {
  width: number; height: number; index: number;  // index 0-based == list position
}

interface PawlsToken {
  x: number; y: number; width: number; height: number;  // top-left origin, abs points
  text: string;                                          // "" for image tokens

  // image-token fields (present ONLY when is_image === true)
  is_image?: boolean;
  image_path?: string;
  base64_data?: string;
  format?: string;            // "jpeg" | "png"
  content_hash?: string;      // sha-256 of image bytes (dedup)
  original_width?: number;
  original_height?: number;
  image_type?: string;        // "embedded" | "cropped"
}
```

- **Tokens are word-level** for text (one token ≈ one whitespace-delimited word
  with its box).
- **Images are tokens too**, in the same array with `is_image: true`. Optional
  image fields must be **omitted entirely when absent** (never `null`).
- `tokenIndex` (used by annotations) is the **position in this `tokens` array**.

## 3. Annotation layer

```ts
interface Annotation {
  id: string | number | null;
  annotationLabel: string;       // label NAME, e.g. "Section Header"
  rawText: string;
  page: number;                  // 0-based primary page index
  annotation_json: AnnotationJson;
  parent_id: string | number | null;   // parent annotation id, or null for roots
  annotation_type: string | null;      // "TOKEN_LABEL" for PDF; "SPAN_LABEL" for text
  structural: boolean;                  // true = inherent document structure

  content_modalities?: string[];        // ["TEXT"] | ["IMAGE"] | ["TEXT","IMAGE"]
  long_description?: string | null;
  link_url?: string | null;
  data?: object | null;
}
```

- **`annotation_type`**: `"TOKEN_LABEL"` for PDF/token-based annotations.
- **`structural`**: `true` for parser/structure output.
- **`parent_id`**: the hierarchy — parent block's `id`, or `null` for roots.
- **`content_modalities`**: omit if purely geometric; otherwise declare it.

### 3.1 `annotation_json` shapes

**(A) Verbose token-based (v1) — emit this for PDFs.** Map from stringified page
index to per-page geometry:

```ts
type AnnotationJsonV1 = { [pageIndexStr: string]: SinglePageAnnotation };
interface SinglePageAnnotation {
  bounds: BoundingBox;                 // {top, bottom, left, right} abs points
  tokensJsons: TokenId[];              // {pageIndex, tokenIndex}[] (may be [])
  rawText: string;
}
interface BoundingBox { top: number; bottom: number; left: number; right: number; }
interface TokenId { pageIndex: number; tokenIndex: number; }
```

**(B) Compact token-based (v2)** — `{v:2, p:{<page>:{b:[top,left,right,bottom], t:"35-37,40"}}}`.
Read-only optimization; the system can compact v1 for you.

**(C) Span-based (text docs only)** — `{start, end, text}` char offsets into `content`.

> PDF structure engine: **use shape (A)**, `annotation_type: "TOKEN_LABEL"`.

## 4. Relationships (optional)

```ts
interface Relationship {
  id: string | number | null;
  relationshipLabel: string;                 // e.g. "OC_PARENT_CHILD"
  source_annotation_ids: (string|number)[];
  target_annotation_ids: (string|number)[];
  structural: boolean;
}
```

Parent-child hierarchy is *also* expressed via each annotation's `parent_id`, and
the system derives subtree groupings from that tree. Warp-Ingest additionally
emits the hierarchy **explicitly** as `OC_PARENT_CHILD` relationships — one per
parent, `source_annotation_ids = [parent]`, `target_annotation_ids = [its direct
children]`, `structural: true`. `OC_PARENT_CHILD` is the OpenContracts convention
for explicit parent→child edges, honored by the subtree-group walker alongside
the `parent_id` FK (so shipping both is redundant-but-safe, and makes the
structure legible to consumers that read relationships rather than walk
`parent_id`). `relationshipLabel` references a `RELATIONSHIP_LABEL` the importer
get-or-creates by name. `validate_export` checks each relationship has non-empty,
resolvable source/target ids and no annotation that is both its own source and
target.

## 5. Hierarchy semantics

- The tree is defined entirely by `parent_id` pointers among `labelled_text`.
- `parent_id` must reference an `id` that exists in the same export. Roots = `null`.
- IDs unique per document (strings recommended). Order of `labelled_text` doesn't
  matter; the importer resolves by id, reading order comes from geometry.
- The importer materializes, for each non-leaf node, a grouping of that node +
  **all transitive descendants** ("whole clause + subsections" retrieval).

## 6. Producer invariants / checklist

1. `pawls_file_content.length === page_count`; `page.index === array position`.
2. Every `TokenId` resolves on its page.
3. All bounds/token boxes within page dims; `left ≤ right`, `top ≤ bottom`.
4. Every `parent_id` references an existing annotation `id`; roots `null`; no cycles.
5. `annotation_type: "TOKEN_LABEL"`, `structural: true`, shape (A) for PDF output.
6. Block `bounds` = union of its tokens' boxes; `tokensJsons` = exactly those
   tokens (empty list allowed for box-only annotations).
7. Optional token/image fields **omitted when absent**, never `null`.
8. `title`, `content`, `description` keys all present (`description` may be `null`).

## 7. Minimum valid (no structure)

```json
{
  "title": "x", "content": "…", "description": null,
  "page_count": 1,
  "pawls_file_content": [ { "page": {"width":612,"height":792,"index":0}, "tokens": [] } ],
  "doc_labels": [], "labelled_text": [], "relationships": []
}
```

A full worked example (heading + two child clauses + a logo image, with the
`parent_id` tree) is in the project history / design notes.
