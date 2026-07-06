"""Markdown export helpers for Warp-Ingest PDF output.

The functions in this module are generic library utilities: they convert Warp's
typed block stream into Markdown pages or a whole-document Markdown string. They
do not depend on any external framework or scoring schema.
"""

from __future__ import annotations

import contextlib
import gc
import io
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PARSE_OPTIONS = {
    "apply_ocr": False,
    "render_format": "all",
}

SERIALIZED_BLOCK_FIELDS = (
    "page_idx",
    "block_type",
    "block_class",
    "block_text",
    "level",
    "table_idx",
    "cell_values",
    "header_cell_values",
    "bold_ratio",
    "bold_mask",
    "italic_ratio",
    "italic_mask",
)

TABLES_BY_PAGE_KEY = "tables_by_page"

__all__ = (
    "DEFAULT_PARSE_OPTIONS",
    "SERIALIZED_BLOCK_FIELDS",
    "TABLES_BY_PAGE_KEY",
    "box_xywh",
    "parse_to_markdown",
    "parse_to_markdown_pages",
    "parse_to_markdown_payload",
    "render_markdown",
    "render_pages",
    "serialize_blocks",
)


def box_xywh(box_style: Any) -> list[float] | None:
    """Convert a Warp BoxStyle to ``[left, top, width, height]``."""
    if box_style is None:
        return None
    try:
        left = float(getattr(box_style, "left", box_style[1]))
        top = float(getattr(box_style, "top", box_style[0]))
        width = float(getattr(box_style, "width", box_style[3]))
        height = float(getattr(box_style, "height", box_style[4]))
    except (TypeError, IndexError, ValueError):
        return None
    return [left, top, width, height]


def serialize_blocks(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact JSON-serializable projection of Warp blocks."""
    serialized: list[dict[str, Any]] = []
    for block in blocks:
        kept = {key: block.get(key) for key in SERIALIZED_BLOCK_FIELDS}
        kept["box"] = box_xywh(block.get("box_style"))
        serialized.append(kept)
    return serialized


def _table_regions(
    blocks: list[dict[str, Any]],
) -> dict[int, list[tuple[float, float, float, float]]]:
    regions: dict[int, list[tuple[float, float, float, float]]] = {}
    cur: tuple[int, Any, float, float, float, float] | None = None

    def flush() -> None:
        nonlocal cur
        if cur:
            regions.setdefault(cur[0], []).append(cur[2:])
        cur = None

    for block in blocks:
        if block.get("block_type") != "table_row" or block.get("table_idx") is None:
            continue
        box = block.get("box")
        if not box:
            continue
        page = int(block.get("page_idx", 0) or 0)
        table_idx = block.get("table_idx")
        x0, top, x1, bottom = box[0], box[1], box[0] + box[2], box[1] + box[3]
        if cur and cur[0] == page and cur[1] == table_idx:
            cur = (
                page,
                table_idx,
                min(cur[2], x0),
                min(cur[3], top),
                max(cur[4], x1),
                max(cur[5], bottom),
            )
        else:
            flush()
            cur = (page, table_idx, x0, top, x1, bottom)
    flush()
    return regions


def _normalize_tables_by_page(tables: dict[Any, Any]) -> dict[int, list[list[Any]]]:
    normalized: dict[int, list[list[Any]]] = {}
    for key, items in tables.items():
        page_tables: list[list[Any]] = []
        for item in items:
            if isinstance(item, str):
                page_tables.append([None, item])
            elif (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], str)
            ):
                bbox, html = item
                if bbox is None:
                    page_tables.append([None, html])
                elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    page_tables.append([[float(value) for value in bbox], html])
        if page_tables:
            normalized[int(key)] = page_tables
    return normalized


def parse_to_markdown_payload(
    doc_location: str | Path,
    parse_options: dict[str, Any] | None = None,
    *,
    include_native_tables: bool = True,
) -> dict[str, Any]:
    """Parse a PDF and return a JSON-friendly payload for Markdown rendering.

    The payload contains ``num_pages``, ``page_dim``, serialized ``blocks``, and
    when requested, native table HTML keyed by page. The original parser state is
    released before native table extraction to keep peak memory bounded.
    """
    from warp_ingest.ingestor.pdf_ingestor import PDFIngestor

    options = dict(DEFAULT_PARSE_OPTIONS)
    if parse_options:
        options.update(parse_options)
    options["render_format"] = "all"

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        ingestor = PDFIngestor(str(doc_location), options)

    blocks = serialize_blocks(ingestor.blocks)
    page_indexes = [block.get("page_idx", 0) or 0 for block in blocks]
    num_pages = max(page_indexes) + 1 if page_indexes else 0
    page_dim = ingestor.return_dict.get("page_dim") or [612.0, 792.0]

    payload: dict[str, Any] = {
        "num_pages": num_pages,
        "page_dim": list(page_dim),
        "blocks": blocks,
    }

    if include_native_tables:
        try:
            from warp_ingest.ingestor.table_engine import extract_pdf_tables

            regions = _table_regions(blocks)
            del ingestor
            gc.collect()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                tables = extract_pdf_tables(str(doc_location), regions_by_page=regions)
            payload[TABLES_BY_PAGE_KEY] = _normalize_tables_by_page(tables)
        except Exception:
            # Markdown rendering can still faithfully fall back to Warp's own
            # table-row blocks if the supplemental table extractor abstains.
            pass

    return payload


def _heading_prefix(level: Any, level_rank: dict[Any, int] | None = None) -> str:
    if level_rank and level in level_rank:
        return "#" * level_rank[level]
    try:
        raw_level = int(level)
    except (TypeError, ValueError):
        raw_level = 1
    return "#" * max(1, min(6, raw_level))


def _decorate_emphasis(text: str, block: dict[str, Any]) -> str:
    if not text:
        return text
    words = text.split()
    word_count = len(words)
    bold_mask = block.get("bold_mask")
    italic_mask = block.get("italic_mask")
    bold_ok = bool(bold_mask) and len(bold_mask) == word_count
    italic_ok = bool(italic_mask) and len(italic_mask) == word_count
    if not bold_ok and not italic_ok:
        ratio = block.get("bold_ratio")
        if ratio is not None and ratio >= 0.9:
            return f"**{text}**"
        return text

    bold_flags = [bold_ok and bold_mask[idx] == "1" for idx in range(word_count)]
    italic_flags = [italic_ok and italic_mask[idx] == "1" for idx in range(word_count)]
    out: list[str] = []
    idx = 0
    while idx < word_count:
        bold = bold_flags[idx]
        italic = italic_flags[idx]
        end = idx
        while (
            end < word_count and bold_flags[end] == bold and italic_flags[end] == italic
        ):
            end += 1
        run = " ".join(words[idx:end])
        if bold and italic:
            out.append(f"***{run}***")
        elif bold:
            out.append(f"**{run}**")
        elif italic:
            out.append(f"*{run}*")
        else:
            out.append(run)
        idx = end
    return " ".join(out)


def _table_html(header: list[str] | None, rows: list[list[str]]) -> str:
    parts = ["<table>"]
    if header:
        cells = "".join(f"<th>{escape(str(cell))}</th>" for cell in header)
        parts.append(f"<tr>{cells}</tr>")
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _bbox_overlap_frac(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return (ix1 - ix0) * (iy1 - iy0) / area_a


def _emit_nontable(
    out: list[str],
    block: dict[str, Any],
    level_rank: dict[Any, int] | None,
) -> None:
    text = (block.get("block_text") or "").strip()
    if not text:
        return
    block_type = block.get("block_type")
    if block_type == "header":
        out.append(f"{_heading_prefix(block.get('level'), level_rank)} {text}")
    elif block_type == "list_item":
        out.append(f"- {_decorate_emphasis(text, block)}")
    else:
        out.append(_decorate_emphasis(text, block))


def _next_table_row(
    blocks: list[dict[str, Any]],
    start: int,
    table_idx: Any,
) -> dict[str, Any] | None:
    for idx in range(start, len(blocks)):
        next_block = blocks[idx]
        if next_block.get("block_type") == "table_row":
            if next_block.get("table_idx") != table_idx:
                return None
            return next_block
        if next_block.get("table_idx") != table_idx:
            return None
    return None


def _render_block_stream(
    blocks: list[dict[str, Any]],
    level_rank: dict[Any, int] | None = None,
    tables: list[list[Any]] | None = None,
) -> str:
    table_pairs = None
    if tables:
        table_pairs = list(tables)

    covered_by: list[int | None] = [None] * len(blocks)
    if table_pairs:
        for idx, block in enumerate(blocks):
            box = block.get("box")
            if not box:
                continue
            block_bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            in_table = (
                block.get("block_type") == "table_row"
                or block.get("table_idx") is not None
            )
            threshold = 0.5 if in_table else 0.7
            for table_idx, (table_bbox, _html) in enumerate(table_pairs):
                if table_bbox is None:
                    continue
                if _bbox_overlap_frac(block_bbox, tuple(table_bbox)) >= threshold:
                    covered_by[idx] = table_idx
                    break

    emitted_tables: set[int] = set()
    span_class_counts: dict[Any, Counter[Any]] = {}
    for block in blocks:
        if (
            block.get("block_type") == "table_row"
            and block.get("table_idx") is not None
        ):
            span_class_counts.setdefault(block.get("table_idx"), Counter())[
                block.get("block_class")
            ] += 1
    dominant_class = {
        table_idx: counts.most_common(1)[0][0]
        for table_idx, counts in span_class_counts.items()
    }

    out: list[str] = []
    cur_table_idx: Any = None
    cur_header: list[str] | None = None
    cur_rows: list[list[str]] = []
    have_table = False

    def flush_table() -> None:
        nonlocal cur_table_idx, cur_header, cur_rows, have_table
        if have_table:
            header = [str(cell) for cell in cur_header] if cur_header else None
            body = [row for row in cur_rows if not (header and row == header)]
            out.append(_table_html(header, body))
        cur_table_idx = None
        cur_header = None
        cur_rows = []
        have_table = False

    for idx, block in enumerate(blocks):
        block_type = block.get("block_type")
        text = (block.get("block_text") or "").strip()
        table_idx = block.get("table_idx")

        coverage = covered_by[idx]
        if coverage is not None:
            if have_table:
                flush_table()
            if coverage not in emitted_tables:
                out.append(table_pairs[coverage][1])
                emitted_tables.add(coverage)
            continue

        if block_type == "table_row":
            cells = [
                str(cell)
                for cell in (block.get("cell_values") or ([text] if text else []))
            ]
            repeat_header = bool(
                have_table and cur_header and cur_rows and cells == cur_header
            )
            if have_table and (table_idx != cur_table_idx or repeat_header):
                flush_table()
            cur_table_idx = table_idx
            header = block.get("header_cell_values")
            if cur_header is None and header:
                cur_header = [str(cell) for cell in header]
            cur_rows.append(cells)
            have_table = True
            continue

        dominant = dominant_class.get(cur_table_idx)
        next_row = _next_table_row(blocks, idx + 1, cur_table_idx)
        next_cells = (
            [str(cell) for cell in (next_row.get("cell_values") or [])]
            if next_row
            else None
        )
        class_ok = block.get("block_class") == dominant and (
            next_row is None or next_row.get("block_class") == dominant
        )
        keep_open = (
            have_table
            and text
            and table_idx is not None
            and table_idx == cur_table_idx
            and class_ok
        )
        if keep_open:
            if cur_header and next_cells is not None and next_cells == cur_header:
                flush_table()
                _emit_nontable(out, block, level_rank)
            else:
                cells = block.get("cell_values") or [text]
                cur_rows.append([str(cell) for cell in cells])
            continue

        if have_table:
            flush_table()
        _emit_nontable(out, block, level_rank)

    if cur_rows or cur_header:
        flush_table()

    if table_pairs:
        for idx, (_table_bbox, html) in enumerate(table_pairs):
            if idx not in emitted_tables:
                out.append(html)

    return "\n\n".join(out)


def render_pages(payload: dict[str, Any]) -> list[tuple[int, str]]:
    """Render a serialized Markdown payload into per-page Markdown."""
    num_pages = int(payload.get("num_pages", 0) or 0)
    blocks = payload.get("blocks", []) or []

    by_page: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        by_page.setdefault(int(block.get("page_idx", 0) or 0), []).append(block)

    header_levels = sorted(
        {
            block.get("level")
            for block in blocks
            if block.get("block_type") == "header" and block.get("level") is not None
        }
    )
    level_rank = {level: min(6, idx + 1) for idx, level in enumerate(header_levels)}

    tables_by_page = _normalize_tables_by_page(payload.get(TABLES_BY_PAGE_KEY) or {})
    bounds = [num_pages - 1] + list(by_page.keys()) + list(tables_by_page.keys())
    last = max(bounds) if (num_pages or by_page or tables_by_page) else -1

    pages: list[tuple[int, str]] = []
    for page_index in range(0, last + 1):
        pages.append(
            (
                page_index,
                _render_block_stream(
                    by_page.get(page_index, []),
                    level_rank,
                    tables=tables_by_page.get(page_index),
                ),
            )
        )
    return pages


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a serialized Markdown payload into whole-document Markdown."""
    return "\n\n".join(markdown for _, markdown in render_pages(payload))


def parse_to_markdown_pages(
    doc_location: str | Path,
    parse_options: dict[str, Any] | None = None,
    *,
    include_native_tables: bool = True,
) -> list[tuple[int, str]]:
    """Parse a PDF and return ``[(page_index, markdown), ...]``."""
    payload = parse_to_markdown_payload(
        doc_location,
        parse_options=parse_options,
        include_native_tables=include_native_tables,
    )
    return render_pages(payload)


def parse_to_markdown(
    doc_location: str | Path,
    parse_options: dict[str, Any] | None = None,
    *,
    include_native_tables: bool = True,
) -> str:
    """Parse a PDF and return whole-document Markdown."""
    payload = parse_to_markdown_payload(
        doc_location,
        parse_options=parse_options,
        include_native_tables=include_native_tables,
    )
    return render_markdown(payload)
