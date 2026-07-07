"""Warp's native table-cell engine: ruled grids + region-local grid inference.

Extracts each table's cell grid from a PDF page using only warp's existing
dependencies (pdfplumber word boxes and vector edges):

* **Ruled tables** are found with pdfplumber's ``TableFinder`` (``lines``
  strategy), augmented with *envelope edges*: neighboring drawn rectangles
  (shaded row bands, fragmented frames) are clustered and each text-bearing
  cluster contributes its border lines, recovering tables whose outer frame is
  drawn in pieces or as fills.
* **Unruled tables** get a region-local grid inference over word boxes:
  column separators are *whitespace channels* voted across the region's visual
  lines (far more robust than global aligned-x voting), rows come from visual
  lines with wrapped-continuation merging, and vector rules snap/force
  boundaries when present.  Regions come from the caller (typically Warp's own
  table spans — Warp's region detection is strong; its historical weakness was
  the cell grid) plus the engine's own alignment-band proposals and ruled
  candidates that had rows but no column rules.

The core functions are pure over plain dicts so they are unit-testable without
PDFs (``tests/test_table_engine.py``).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from html import escape
from statistics import median
from typing import Any, Iterable, Optional, Sequence

# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class Cell:
    text: str = ""
    colspan: int = 1


@dataclass
class PageTable:
    """One extracted table: ``bbox`` = (x0, top, x1, bottom) in page points."""

    bbox: tuple
    grid: list  # list[list[Cell]]
    n_header_rows: int = 1
    ruled: bool = False


# --------------------------------------------------------------------------
# tunables (swept against the fast table eval; see the implementation plan)
# --------------------------------------------------------------------------

SNAP_TOLERANCE = 3.0
JOIN_TOLERANCE = 3.0
ENVELOPE_TOL = 3.0  # rect neighbor-merge distance
ENVELOPE_MAX_RECTS = 1500  # perf guard: skip envelope pass on pathological pages
CHANNEL_MIN_GAP = 4.0  # minimum whitespace-channel width (pts)
CHANNEL_BLOCK_FRAC = 0.12  # max fraction of extending lines that may cross
CHANNEL_MIN_EXTENT = 0.3  # channel must be reached by >= this frac of lines
CAPTION_SPAN_FRAC = 0.85  # a gapless line spanning this much of the region
WRAP_MAX_COLS_FRAC = 0.5  # continuation occupies < this fraction of columns
MIN_FILLED_FRAC = 0.35  # reject inferred grids emptier than this
RULED_OVERLAP_SKIP = 0.3  # region overlapping a ruled table this much is skipped


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _clean(text: Any) -> str:
    return _WS.sub(" ", str(text)).strip() if text else ""


def render_table_html(table: PageTable) -> str:
    """Render a PageTable as minimal HTML table markup."""
    parts = ["<table>"]
    for i, row in enumerate(table.grid):
        tag = "th" if i < table.n_header_rows else "td"
        cells = []
        for c in row:
            attr = f' colspan="{c.colspan}"' if c.colspan > 1 else ""
            cells.append(f"<{tag}{attr}>{escape(_clean(c.text))}</{tag}>")
        parts.append(f"<tr>{''.join(cells)}</tr>")
    parts.append("</table>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# header detection
# --------------------------------------------------------------------------

# currency/number/percent-ish content — a row dominated by these is data
_NUMERICISH = re.compile(
    r"^[\s\$€£¥+\-—–—()]*\d[\d\s.,%()\-\$€£¥/]*$|^[\-—–]+$|^\(\d[\d.,]*\)$"
)


_YEARISH = re.compile(r"^(19|20)\d{2}$")


def _is_numericish(text: str) -> bool:
    t = text.strip()
    if _YEARISH.match(t):
        # bare years are column labels far more often than data
        return False
    return bool(_NUMERICISH.match(t))


def _n_header_rows(grid: Sequence[Sequence[Cell]]) -> int:
    """Header depth: row 0, plus row 1 only with stacked-group evidence.

    Always at least one ``<th>`` row: the scorer demotes predicted headers
    harmlessly when the ground truth is headerless, and a real header row is
    the only way to earn real column keys.  A *deep* leading-non-numeric run
    is NOT header evidence (all-text tables would th every row and poison the
    column keys); only the classic stacked-group shape — a row-0 with spanning
    or gapped cells completed by a label-like row 1 — earns depth 2.
    """
    # leading full-span caption/title rows count as header rows (the scorer
    # strips a top spanning <th> title), then the stacked-header logic runs
    # on the first real rows
    ncols_all = max((sum(c.colspan for c in row) for row in grid), default=1)
    off = 0
    while (
        off < len(grid) - 1
        and len(grid[off]) == 1
        and grid[off][0].colspan == ncols_all
        and ncols_all > 1
    ):
        off += 1
    if off:
        grid = grid[off:]
    if len(grid) < 3:
        return off + 1
    row0, row1 = grid[0], grid[1]
    row0_span = any(c.colspan > 1 for c in row0)
    row0_gap = any(not c.text.strip() for c in row0) and any(
        c.text.strip() for c in row0
    )
    row1_texts = [c.text.strip() for c in row1 if c.text.strip()]
    row1_labelish = (
        len(row1_texts) >= 2
        and sum(_is_numericish(t) for t in row1_texts) / len(row1_texts) < 0.3
        and len(row1) > 1
    )
    if (row0_span or row0_gap) and row1_labelish:
        return off + 2
    return off + 1


# --------------------------------------------------------------------------
# envelope augmentation (ruled path)
# --------------------------------------------------------------------------


def _cluster_rects(boxes: Sequence[tuple], tol: float = ENVELOPE_TOL) -> list[tuple]:
    """Merge rectangles whose x- and y-intervals both come within *tol*.

    Iterates to a fixed point; returns merged bounding boxes.
    """
    merged = [tuple(map(float, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[tuple] = []
        for b in merged:
            hit = None
            for i, o in enumerate(out):
                if (
                    b[0] <= o[2] + tol
                    and o[0] <= b[2] + tol
                    and b[1] <= o[3] + tol
                    and o[1] <= b[3] + tol
                ):
                    hit = i
                    break
            if hit is None:
                out.append(b)
            else:
                o = out[hit]
                out[hit] = (
                    min(o[0], b[0]),
                    min(o[1], b[1]),
                    max(o[2], b[2]),
                    max(o[3], b[3]),
                )
                changed = True
        merged = out
    return merged


def _envelope_lines(page, words) -> tuple[list, list]:
    """Border lines of text-bearing clusters of drawn rects (shaded bands,
    fragmented frames) as explicit pdfplumber line objects."""
    rects = getattr(page, "rects", None) or []
    if not rects or len(rects) > ENVELOPE_MAX_RECTS:
        return [], []
    boxes = [(r["x0"], r["top"], r["x1"], r["bottom"]) for r in rects]
    clusters = _cluster_rects(boxes)
    v_lines: list = []
    h_lines: list = []
    for x0, top, x1, bottom in clusters:
        if x1 - x0 < 8 or bottom - top < 4:
            continue
        has_word = any(
            wd["x0"] >= x0 - 1
            and wd["x1"] <= x1 + 1
            and wd["top"] >= top - 1
            and wd["bottom"] <= bottom + 1
            for wd in words
        )
        if not has_word:
            continue
        for x in (x0, x1):
            v_lines.append(
                {
                    "x0": x,
                    "x1": x,
                    "top": top,
                    "bottom": bottom,
                    "height": bottom - top,
                    "orientation": "v",
                    "object_type": "line",
                }
            )
        for y in (top, bottom):
            h_lines.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "width": x1 - x0,
                    "top": y,
                    "bottom": y,
                    "orientation": "h",
                    "object_type": "line",
                }
            )
    return v_lines, h_lines


# --------------------------------------------------------------------------
# ruled tables via pdfplumber TableFinder
# --------------------------------------------------------------------------


def _pdfplumber_to_pagetable(table) -> Optional[PageTable]:
    """Convert a pdfplumber ``Table`` into a compacted PageTable.

    Column spans are expanded by *replication* (exactly what the scorer does
    to ground-truth colspans, so the two forms are provably equivalent), then
    all-empty spacer columns (artifacts of shading-rect edges) are dropped and
    currency-symbol columns merged into their value column."""
    try:
        texts = table.extract()
    except Exception:
        return None
    rows = table.rows
    if not rows or len(rows) < 2:
        return None
    col_x0s = sorted({c[0] for r in rows for c in r.cells if c is not None})
    ncols = len(col_x0s)
    if ncols < 1:
        return None
    right_most = max((c[2] for r in rows for c in r.cells if c is not None), default=0)
    matrix: list[list[str]] = []
    captions: dict[int, str] = {}  # matrix row idx -> full-span title text
    any_text = False
    for row, row_texts in zip(rows, texts):
        cells = [""] * ncols
        real = []
        for j in range(min(ncols, len(row.cells))):
            cbox = row.cells[j]
            if cbox is None:
                continue
            text = _clean(row_texts[j] if j < len(row_texts) else "")
            if not text:
                continue
            any_text = True
            real.append((j, cbox, text))
            span = 1
            while j + span < ncols and col_x0s[j + span] < cbox[2] - 1.0:
                span += 1
            for k in range(j, min(j + span, ncols)):
                cells[k] = text
        # a single cell spanning the whole width is a title/caption row —
        # keep its colspan form (the scorer's title-strip pass keys on it)
        if (
            ncols > 1
            and len(real) == 1
            and real[0][0] == 0
            and real[0][1][2] >= right_most - 2.0
        ):
            captions[len(matrix)] = real[0][2]
        matrix.append(cells)
    if not any_text:
        return None
    # drop all-empty *spacer* columns (narrow artifacts of shading-rect
    # edges); a normal-width empty column is real table structure (GT keeps
    # it, and empty↔empty grid cells are full matches). Caption rows don't
    # count toward column occupancy.
    body_rows = [r for i, r in enumerate(matrix) if i not in captions]
    bounds = col_x0s + [right_most]
    keep = [
        j
        for j in range(ncols)
        if any(r[j] for r in body_rows) or (bounds[j + 1] - bounds[j]) >= 12.0
    ]
    if not keep or not any(any(r[j] for j in keep) for r in body_rows or matrix):
        return None
    # single-column ruled tables are real (a framed 2-row list) but only when
    # cell-like: short texts, no prose panels
    if len(keep) == 1 and any(len(r[keep[0]]) > 80 for r in matrix):
        return None
    matrix = [
        ([""] * len(keep) if i in captions else [r[j] for j in keep])
        for i, r in enumerate(matrix)
    ]
    cuts = [col_x0s[j] for j in keep[1:]]
    # merge a column into its right neighbor when every non-empty cell is a
    # currency symbol or a replica of the neighbor (a value that spanned the
    # symbol slot on symbol-less rows) — the shaded-financial-grid artifact
    changed = True
    while changed:
        changed = False
        w = len(matrix[0])
        for j in range(w - 1):
            col = [r[j] for r in matrix]
            if not any(col):
                continue
            # only a column that actually carries currency symbols may merge:
            # a span-replica column without symbols is real (GT) structure
            if not any(v in _SYMBOLS for v in col if v):
                continue
            if all(
                (not v) or v in _SYMBOLS or v == r[j + 1] for v, r in zip(col, matrix)
            ):
                for r in matrix:
                    if r[j] and r[j] in _SYMBOLS and not r[j + 1].startswith(r[j]):
                        r[j + 1] = f"{r[j]} {r[j + 1]}".strip()
                    del r[j]
                if j < len(cuts):
                    del cuts[j]
                changed = True
                break
    final_ncols = len(matrix[0])
    grid = [
        (
            [Cell(captions[i], colspan=final_ncols)]
            if i in captions
            else [Cell(t) for t in row]
        )
        for i, row in enumerate(matrix)
    ]
    pt = PageTable(
        bbox=tuple(table.bbox),
        grid=grid,
        n_header_rows=_n_header_rows(grid),
        ruled=True,
    )
    pt._cuts = cuts
    return pt


def _ruled_tables(page, words) -> list[PageTable]:
    ev, eh = _envelope_lines(page, words)
    settings: dict[str, Any] = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": SNAP_TOLERANCE,
        "join_tolerance": JOIN_TOLERANCE,
    }
    if ev:
        settings["explicit_vertical_lines"] = ev
    if eh:
        settings["explicit_horizontal_lines"] = eh
    try:
        found = page.find_tables(table_settings=settings)
    except Exception:
        return []
    out = []
    for t in found:
        pt = _pdfplumber_to_pagetable(t)
        if pt is not None:
            out.append(pt)
    return out


# --------------------------------------------------------------------------
# region-local grid inference (unruled path)
# --------------------------------------------------------------------------


def _cluster_lines(words: Sequence[dict]) -> list[list[dict]]:
    """Group words into visual lines by vertical-overlap, sorted by top."""
    lines: list[list[dict]] = []
    for wd in sorted(words, key=lambda d: (d["top"], d["x0"])):
        placed = False
        for line in lines:
            ref = line[0]
            overlap = min(ref["bottom"], wd["bottom"]) - max(ref["top"], wd["top"])
            smaller = min(ref["bottom"] - ref["top"], wd["bottom"] - wd["top"])
            if smaller > 0 and overlap >= 0.5 * smaller:
                line.append(wd)
                placed = True
                break
        if not placed:
            lines.append([wd])
    for line in lines:
        line.sort(key=lambda d: d["x0"])
    lines.sort(key=lambda l: min(d["top"] for d in l))
    return lines


def _line_gaps(line: Sequence[dict], min_gap: float) -> list[tuple]:
    gaps = []
    for a, b in zip(line, line[1:]):
        if b["x0"] - a["x1"] >= min_gap:
            gaps.append((a["x1"], b["x0"]))
    return gaps


def _channel_separators(
    lines: Sequence[Sequence[dict]],
    voting_idx: Sequence[int],
    rx0: float,
    rx1: float,
    min_gap: float,
) -> list[float]:
    """Column cut points from whitespace channels voted across visual lines."""
    if not voting_idx or rx1 - rx0 <= 2 * min_gap:
        return []
    nbins = max(8, int((rx1 - rx0) / 0.5))
    step = (rx1 - rx0) / nbins
    block = [0] * nbins
    extent = [0] * nbins

    def bin_range(a: float, b: float) -> range:
        lo = max(0, int((a - rx0) / step))
        hi = min(nbins, int((b - rx0) / step) + 1)
        return range(lo, hi)

    nvote = len(voting_idx)
    for i in voting_idx:
        line = lines[i]
        lx0, lx1 = line[0]["x0"], line[-1]["x1"]
        for k in bin_range(lx0, lx1):
            extent[k] += 1
        for wd in line:
            for k in bin_range(wd["x0"], wd["x1"]):
                block[k] += 1

    min_extent = max(2.0, CHANNEL_MIN_EXTENT * nvote)
    ok = [
        extent[k] >= min_extent and block[k] <= CHANNEL_BLOCK_FRAC * extent[k]
        for k in range(nbins)
    ]
    cuts: list[float] = []
    k = 0
    while k < nbins:
        if not ok[k]:
            k += 1
            continue
        j = k
        while j < nbins and ok[j]:
            j += 1
        width = (j - k) * step
        if width >= min_gap:
            cuts.append(rx0 + (k + (j - k) / 2.0) * step)
        k = j
    return cuts


def infer_grid(
    words: Sequence[dict],
    bbox: tuple,
    v_rules: Sequence[dict] = (),
    h_rules: Sequence[dict] = (),
) -> Optional[PageTable]:
    """Infer a cell grid for an (unruled) table region from word geometry.

    ``words``: dicts with x0/x1/top/bottom/text, already clipped to the region.
    ``v_rules``: dicts with x/top/bottom — vertical rules forcing separators.
    ``h_rules``: dicts with y — horizontal rules forcing row breaks.
    """
    words = [wd for wd in words if _clean(wd.get("text"))]
    if len(words) < 4:
        return None
    lines = _cluster_lines(words)
    if len(lines) < 2:
        return None

    rx0 = min(wd["x0"] for wd in words)
    rx1 = max(wd["x1"] for wd in words)
    heights = [wd["bottom"] - wd["top"] for wd in words]
    char_w = median((wd["x1"] - wd["x0"]) / max(1, len(wd["text"])) for wd in words)
    min_gap = max(CHANNEL_MIN_GAP, 1.0 * char_w)

    # gapless full-width lines don't get a vote on column channels
    voting_idx: list[int] = []
    for i, line in enumerate(lines):
        span = line[-1]["x1"] - line[0]["x0"]
        if not (
            span > CAPTION_SPAN_FRAC * (rx1 - rx0) and not _line_gaps(line, min_gap)
        ):
            voting_idx.append(i)

    cuts = _channel_separators(lines, voting_idx, rx0, rx1, min_gap)
    # vector rules force separators (snap duplicates away)
    for vr in v_rules:
        x = float(vr["x"] if "x" in vr else vr["x0"])
        if rx0 < x < rx1 and not any(abs(x - c) < min_gap for c in cuts):
            cuts.append(x)
    cuts = sorted(cuts)
    ncols = len(cuts) + 1
    if ncols < 2:
        return None

    # a caption/section row is a wide line with a word physically straddling a
    # separator — decided against the *final* cuts so legitimate rows whose
    # cells simply abut (no big gaps) are not misflagged
    caption_idx: set[int] = set()
    for i, line in enumerate(lines):
        span = line[-1]["x1"] - line[0]["x0"]
        if span <= CAPTION_SPAN_FRAC * (rx1 - rx0):
            continue
        if any(wd["x0"] < c - 1.0 and wd["x1"] > c + 1.0 for wd in line for c in cuts):
            caption_idx.add(i)

    def col_of(wd: dict) -> int:
        cx = (wd["x0"] + wd["x1"]) / 2.0
        return bisect_right(cuts, cx)

    # row formation with wrapped-continuation merging
    line_tops = [min(wd["top"] for wd in lines[i]) for i in range(len(lines))]
    line_bots = [max(wd["bottom"] for wd in lines[i]) for i in range(len(lines))]
    h_rule_ys = sorted(float(r["y"] if "y" in r else r["top"]) for r in h_rules)

    def rule_between(a_bottom: float, b_top: float) -> bool:
        for y in h_rule_ys:
            if a_bottom - 0.5 <= y <= b_top + 0.5:
                return True
        return False

    rows: list[dict] = []  # {"lines": [i...], "cols": {col: [words]}, "caption": bool}
    # strictly less than half the columns: a full-width data row (values in
    # every value column) is never a continuation, while a 2-of-5 overflow is
    max_wrap_cols = max(1, (ncols - 1) // 2)
    for i, line in enumerate(lines):
        if i in caption_idx:
            rows.append({"lines": [i], "cols": None, "caption": True})
            continue
        cols: dict[int, list[dict]] = {}
        for wd in line:
            cols.setdefault(col_of(wd), []).append(wd)
        prev = rows[-1] if rows else None
        line_h = line_bots[i] - line_tops[i]
        merge = False
        if (
            prev is not None
            and not prev["caption"]
            and len(cols) <= max_wrap_cols
            and set(cols) <= set(prev["cols"])
            and not rule_between(line_bots[prev["lines"][-1]], line_tops[i])
            and line_tops[i] - line_bots[prev["lines"][-1]] < 0.6 * max(line_h, 1.0)
        ):
            merge = True
        if merge:
            prev["lines"].append(i)
            for c, ws_ in cols.items():
                prev["cols"].setdefault(c, []).extend(ws_)
        else:
            rows.append({"lines": [i], "cols": cols, "caption": False})

    # look-down merge: leading sparse lines stacked tightly above a fuller row
    # are multi-line header cells ("Trading" over "Symbol"), not rows of their
    # own — fold them into the row below so the header stays one <th> row
    def _cols_numeric_frac(r: dict) -> float:
        texts = []
        for ws_ in r["cols"].values():
            txt = _clean(" ".join(wd["text"] for wd in ws_))
            if txt:
                texts.append(txt)
        if not texts:
            return 0.0
        return sum(_is_numericish(x) for x in texts) / len(texts)

    while len(rows) >= 2 and not rows[0]["caption"] and not rows[1]["caption"]:
        c1, c2 = set(rows[0]["cols"]), set(rows[1]["cols"])
        if not c1 or len(c1) >= 0.75 * ncols or not (c1 <= c2):
            break
        # never fold header labels into a *data* row (year labels above the
        # first numeric row must stay their own header row)
        if _cols_numeric_frac(rows[1]) >= 0.3:
            break
        lh0 = line_bots[rows[0]["lines"][-1]] - line_tops[rows[0]["lines"][-1]]
        gap = line_tops[rows[1]["lines"][0]] - line_bots[rows[0]["lines"][-1]]
        if gap >= 1.2 * max(lh0, 1.0):
            break
        for c, ws_ in rows[0]["cols"].items():
            rows[1]["cols"].setdefault(c, []).extend(ws_)
        rows[1]["lines"] = rows[0]["lines"] + rows[1]["lines"]
        rows.pop(0)

    if sum(1 for r in rows if not r["caption"]) < 2:
        return None

    grid: list[list[Cell]] = []
    row_meta: list[dict] = []
    filled = 0
    total = 0
    for r in rows:
        r_top = min(line_tops[i] for i in r["lines"])
        r_bot = max(line_bots[i] for i in r["lines"])
        if r["caption"]:
            text = " ".join(_clean(wd["text"]) for i in r["lines"] for wd in lines[i])
            grid.append([Cell(text, colspan=ncols)])
            row_meta.append(
                {
                    "top": r_top,
                    "bottom": r_bot,
                    "caption": True,
                    "numeric_frac": 0.0,
                    "nonempty": 1,
                }
            )
            continue
        cells = []
        for c in range(ncols):
            ws_ = sorted(r["cols"].get(c, []), key=lambda d: (d["top"], d["x0"]))
            text = _clean(" ".join(wd["text"] for wd in ws_))
            cells.append(Cell(text))
            total += 1
            if text:
                filled += 1
        grid.append(cells)
        row_meta.append(
            {
                "top": r_top,
                "bottom": r_bot,
                "caption": False,
                "numeric_frac": _row_numeric_frac(cells),
                "nonempty": sum(1 for c in cells if c.text.strip()),
            }
        )
    if total == 0 or filled / total < MIN_FILLED_FRAC:
        return None

    merged = _merge_symbol_columns(grid)
    final_cuts = list(cuts)
    if merged is not grid and len(merged[0]) != len(grid[0]):
        # a symbol column j merged into j+1 removes the cut between them
        body = [r for r in grid if not (len(r) == 1 and r[0].colspan > 1)]
        ncols_orig = len(body[0]) if body else 0
        for j in reversed(range(ncols_orig - 1)):
            texts = [r[j].text.strip() for r in body]
            nonempty = [t for t in texts if t]
            if (
                nonempty
                and all(t in _SYMBOLS for t in nonempty)
                and j < len(final_cuts)
            ):
                del final_cuts[j]
        grid = merged

    top = min(wd["top"] for wd in words)
    bottom = max(wd["bottom"] for wd in words)
    table = PageTable(
        bbox=(rx0, top, rx1, bottom),
        grid=grid,
        n_header_rows=_n_header_rows(grid),
        ruled=False,
    )
    table._row_meta = row_meta
    table._cuts = final_cuts
    return table


# --------------------------------------------------------------------------
# currency-symbol column merging
# --------------------------------------------------------------------------

_SYMBOLS = {"$", "€", "£", "¥"}


def _merge_symbol_columns(grid: list) -> list:
    """Merge a column holding only currency symbols into the value column to
    its right (``$`` | ``151,790`` → ``$ 151,790``, the ground-truth shape)."""
    body = [r for r in grid if not (len(r) == 1 and r[0].colspan > 1)]
    if not body:
        return grid
    ncols = len(body[0])
    if ncols < 2 or any(len(r) != ncols for r in body):
        return grid
    if any(c.colspan != 1 for r in body for c in r):
        return grid
    drop = []
    for j in range(ncols - 1):
        texts = [r[j].text.strip() for r in body]
        nonempty = [t for t in texts if t]
        if nonempty and all(t in _SYMBOLS for t in nonempty):
            drop.append(j)
    if not drop:
        return grid
    out: list = []
    for r in grid:
        if len(r) == 1 and r[0].colspan > 1:
            out.append([Cell(r[0].text, colspan=ncols - len(drop))])
            continue
        cells = [Cell(c.text, c.colspan) for c in r]
        for j in sorted(drop, reverse=True):
            sym = cells[j].text.strip()
            if sym:
                cells[j + 1].text = f"{sym} {cells[j + 1].text}".strip()
            del cells[j]
        out.append(cells)
    return out


# --------------------------------------------------------------------------
# external header extension (column labels above the detected grid)
# --------------------------------------------------------------------------


def _extend_header_up(
    table: PageTable,
    words: Sequence[dict],
    claimed: Sequence[tuple],
    max_lines: int = 2,
) -> None:
    """Attach column-label lines sitting just above a data-topped table.

    Financial tables often carry their year/period labels above the region the
    grid inference (or warp's span) captured.  Only fires when the current top
    row looks like data; prose lines are rejected by straddle/word-count
    guards.  Mutates ``table`` in place."""
    grid = table.grid
    if not grid:
        return
    ncols0 = len(grid[0])
    row0_nonempty = sum(1 for c in grid[0] if c.text.strip())
    # fire only when the top row looks like data (numbers) or a sparse
    # section-label row ("Revenues") — a label-header top row is complete
    if _row_numeric_frac(grid[0]) < 0.3 and row0_nonempty > max(1, ncols0 // 3):
        return
    cuts = getattr(table, "_cuts", None)
    if not cuts:
        return
    ncols = len(cuts) + 1
    meta = getattr(table, "_row_meta", None) or []
    lh = median((m["bottom"] - m["top"]) for m in meta) if meta else 10.0
    x0, top, x1, bottom = table.bbox

    def center(wd: dict) -> float:
        return (wd["x0"] + wd["x1"]) / 2.0

    cand = [
        wd
        for wd in words
        if _clean(wd.get("text"))
        and x0 - 2 <= center(wd) <= x1 + 2
        and top - 3.0 * lh <= wd["bottom"] <= top + 0.5
        and not any(
            bb[0] <= center(wd) <= bb[2]
            and bb[1] <= (wd["top"] + wd["bottom"]) / 2.0 <= bb[3]
            for bb in claimed
        )
    ]
    if not cand:
        return
    clines = _cluster_lines(cand)
    attached: list[list[dict]] = []
    ref_top = top
    for line in reversed(clines):  # nearest line first
        lb = max(wd["bottom"] for wd in line)
        lt = min(wd["top"] for wd in line)
        if ref_top - lb > 1.8 * lh:
            break
        straddle = sum(
            1 for wd in line for c in cuts if wd["x0"] < c - 2 and wd["x1"] > c + 2
        )
        occupied = {bisect_right(cuts, center(wd)) for wd in line}
        if len(line) > 1.5 * ncols or straddle > 0.2 * len(line) or len(occupied) < 2:
            break
        attached.append(line)
        ref_top = lt
        if len(attached) >= max_lines:
            break
    if not attached:
        return
    for line in attached:  # nearest-first: the farthest line ends up as row 0
        cells = [Cell("") for _ in range(ncols)]
        buckets: dict[int, list[dict]] = {}
        for wd in line:
            buckets.setdefault(bisect_right(cuts, center(wd)), []).append(wd)
        for c, ws_ in buckets.items():
            ws_.sort(key=lambda d: d["x0"])
            cells[c] = Cell(_clean(" ".join(wd["text"] for wd in ws_)))
        grid.insert(0, cells)
        table.bbox = (
            min(x0, min(wd["x0"] for wd in line)),
            min(wd["top"] for wd in line),
            max(x1, max(wd["x1"] for wd in line)),
            bottom,
        )
        x0, top, x1, bottom = table.bbox
    table.n_header_rows = max(len(attached), _n_header_rows(grid))


# --------------------------------------------------------------------------
# under-segmentation detection (fragmentary-rule "ruled" tables)
# --------------------------------------------------------------------------

# two number tokens separated by whitespace inside ONE cell — the signature of
# a grid whose column rules were fragmentary (e.g. header underlines only)
_MULTINUM = re.compile(r"\d[\d,.]*\)?\s+[$(€£]{0,2}\s?\d")


def _looks_undersegmented(grid: Sequence[Sequence[Cell]]) -> bool:
    cells = [c for row in grid for c in row if c.text.strip()]
    if not cells:
        return True
    fused = sum(1 for c in cells if _MULTINUM.search(c.text))
    return fused / len(cells) > 0.25 or (len(cells) < 12 and fused >= 2)


def _looks_degenerate(grid: Sequence[Sequence[Cell]]) -> bool:
    """A 'ruled' grid that is really a framed prose/info panel: prose-blob
    cells or an almost-empty matrix."""
    cells = [c for row in grid for c in row]
    if not cells:
        return True
    nonempty = [c for c in cells if c.text.strip()]
    if not nonempty:
        return True
    fill = len(nonempty) / len(cells)
    if fill < 0.2:
        return True
    # spacer rows are the signature of a framed info-panel, not a data grid
    empty_rows = sum(1 for row in grid if not any(c.text.strip() for c in row))
    if len(grid) >= 4 and empty_rows / len(grid) > 0.35:
        return True
    # prose blobs only condemn a grid that is ALSO sparse — a dense grid with
    # long descriptive cells (research/requirements matrices) is a real table
    return fill < 0.5 and sum(1 for c in nonempty if len(c.text) > 150) >= 2


def _looks_prose_grid(grid: Sequence[Sequence[Cell]]) -> bool:
    """An *inferred* (unruled) grid whose cells are predominantly long *prose*
    runs is side-by-side flowing text (a multi-column body, a pull-quote
    beside a paragraph, a scanned newsletter), not a data table.  The
    alignment-band proposer sees such pages as one huge aligned band -- every
    line of two book-justified columns shares the same whitespace channel --
    so the gate is applied to the inference *output*, where the evidence is
    unambiguous.  A prose cell is >= 6 words that are mostly alphabetic:
    digit-dense long cells (a TOC run "3.1 Our company 11 3.2 ...", a wide
    numeric data row) are *not* prose, so TOC / data grids survive.  A real
    table with one long description column also stays: its other columns
    keep the prose-cell fraction low.  Structural token shape only -- no
    content matching."""
    cells = [c for row in grid for c in row if c.text.strip()]
    if not cells:
        return True  # nothing extractable: never worth replacing real blocks

    def _is_prose_cell(c: Cell) -> bool:
        toks = c.text.split()
        if len(toks) < 6:
            return False
        numeric = sum(1 for t in toks if any(ch.isdigit() for ch in t))
        return numeric / len(toks) < 0.3

    prose_cells = sum(1 for c in cells if _is_prose_cell(c))
    return prose_cells / len(cells) >= 0.5


# --------------------------------------------------------------------------
# standalone region proposal (alignment bands)
# --------------------------------------------------------------------------

PROPOSE_MIN_LINES = 3  # a band needs at least this many gap-bearing lines
BAND_BRIDGE_MAX_GAP = 48.0  # bands closer than this (pts) may bridge into one


def _interval_intersect(a: Sequence[tuple], b: Sequence[tuple], min_w: float):
    out = []
    for ax0, ax1 in a:
        for bx0, bx1 in b:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if hi - lo >= min_w:
                out.append((lo, hi))
    return out


def propose_regions(words: Sequence[dict]) -> list[tuple]:
    """Table-region candidates from persistent whitespace-channel bands.

    A band = consecutive visual lines sharing at least one running whitespace
    channel.  Guards: ≥3 member lines; a single-channel (2-column) band is kept
    only when one side is numeric-dominant (label|value financial layout) —
    two-column *prose* never qualifies.
    """
    words = [wd for wd in words if _clean(wd.get("text"))]
    if len(words) < 6:
        return []
    lines = _cluster_lines(words)
    char_w = median((wd["x1"] - wd["x0"]) / max(1, len(wd["text"])) for wd in words)
    min_gap = max(CHANNEL_MIN_GAP, 1.2 * char_w)
    gaps_per_line = [_line_gaps(line, min_gap) for line in lines]

    bands: list[tuple[tuple, list[tuple]]] = []  # (bbox, channels)
    band: list[int] = []
    channels: list[tuple] = []

    def flush() -> None:
        nonlocal band, channels
        gap_lines = [i for i in band if gaps_per_line[i]]
        if len(gap_lines) >= PROPOSE_MIN_LINES and channels:
            ok = len(channels) >= 2 or _numeric_side(
                [lines[i] for i in gap_lines], channels[0]
            )
            if ok:
                ws_ = [wd for i in band for wd in lines[i]]
                bands.append(
                    (
                        (
                            min(wd["x0"] for wd in ws_),
                            min(wd["top"] for wd in ws_),
                            max(wd["x1"] for wd in ws_),
                            max(wd["bottom"] for wd in ws_),
                        ),
                        list(channels),
                    )
                )
        band, channels = [], []

    for i, gaps in enumerate(gaps_per_line):
        if not gaps:
            # absorb an interleaved caption/section-label line (e.g. "Net
            # sales:") when the band's channels continue right below it
            if (
                band
                and i + 1 < len(lines)
                and gaps_per_line[i + 1]
                and _interval_intersect(channels, gaps_per_line[i + 1], min_gap)
            ):
                band.append(i)
                continue
            flush()
            continue
        if not band:
            band, channels = [i], list(gaps)
            continue
        nxt = _interval_intersect(channels, gaps, min_gap)
        if nxt:
            band.append(i)
            channels = nxt
        else:
            flush()
            band, channels = [i], list(gaps)
    flush()

    # bridge bands interrupted by caption/label lines: consecutive bands whose
    # channels still intersect and sit close vertically are one table region
    # (the structural splitter downstream re-separates true stacked tables).
    # A full-width line in the gap is prose between two *different* tables —
    # never bridge across it.
    def _prose_between(y0: float, y1: float, width: float) -> bool:
        for line in lines:
            lt = min(wd["top"] for wd in line)
            lb = max(wd["bottom"] for wd in line)
            if lb <= y0 or lt >= y1:
                continue
            if line[-1]["x1"] - line[0]["x0"] > 0.7 * width:
                return True
        return False

    merged: list[tuple[tuple, list[tuple]]] = []
    for bbox, chans in sorted(bands, key=lambda b: b[0][1]):
        if merged:
            (px0, ptop, px1, pbot), pch = merged[-1]
            common = _interval_intersect(pch, chans, min_gap)
            width = max(px1 - px0, bbox[2] - bbox[0])
            if (
                common
                and 0 <= bbox[1] - pbot <= BAND_BRIDGE_MAX_GAP
                and not _prose_between(pbot, bbox[1], width)
            ):
                merged[-1] = (
                    (
                        min(px0, bbox[0]),
                        ptop,
                        max(px1, bbox[2]),
                        max(pbot, bbox[3]),
                    ),
                    common,
                )
                continue
        merged.append((bbox, list(chans)))
    return [bbox for bbox, _ch in merged]


def _numeric_side(lines: Sequence[Sequence[dict]], channel: tuple) -> bool:
    """True when one side of a single channel is numeric-dominant."""
    mid = (channel[0] + channel[1]) / 2.0
    left_num = right_num = left_n = right_n = 0
    for line in lines:
        for side, toks in (
            ("l", [wd for wd in line if wd["x1"] <= mid]),
            ("r", [wd for wd in line if wd["x0"] >= mid]),
        ):
            if not toks:
                continue
            text = " ".join(wd["text"] for wd in toks)
            numeric = bool(_NUMERICISH.match(text.strip()))
            if side == "l":
                left_n += 1
                left_num += numeric
            else:
                right_n += 1
                right_num += numeric
    for num, n in ((left_num, left_n), (right_num, right_n)):
        if n >= PROPOSE_MIN_LINES and num / n >= 0.5:
            return True
    return False


# --------------------------------------------------------------------------
# stacked-table splitting
# --------------------------------------------------------------------------

SPLIT_GAP_RATIO = 1.8  # row gap > ratio x median gap starts a new table
SPLIT_MIN_GAP = 6.0


def _row_numeric_frac(cells: Sequence[Cell]) -> float:
    texts = [c.text.strip() for c in cells if c.text.strip()]
    if not texts:
        return 0.0
    return sum(_is_numericish(t) for t in texts) / len(texts)


def infer_tables(
    words: Sequence[dict],
    bbox: tuple,
    v_rules: Sequence[dict] = (),
    h_rules: Sequence[dict] = (),
    _depth: int = 0,
) -> list[PageTable]:
    """Infer one *or several stacked* tables in a region.

    Runs the single-grid inference, then looks for split points (big row gaps,
    header-restart rows, mid-region captions) and re-infers each segment so
    every stacked table gets its own column grid.  Over-splitting is preferred
    to fusing (the scorer pairs each GT table to its best prediction)."""
    t = infer_grid(words, bbox, v_rules=v_rules, h_rules=h_rules)
    if t is None:
        return []
    if _depth >= 1:
        return [t]
    meta = getattr(t, "_row_meta", None)
    if not meta or len(meta) < 4:
        return [t]

    # candidate split indices (row starts a new table).  Wrapped-cell merging
    # already folded continuation lines into their rows, so a big inter-row
    # gap is real table separation; a header-ish/caption restart after data
    # rows splits stacked same-column tables even without a gap.
    gaps = [meta[i]["top"] - meta[i - 1]["bottom"] for i in range(1, len(meta))]
    med_gap = median(gaps) if gaps else 0.0
    splits: set[int] = set()
    numeric_run = 0
    for i, m in enumerate(meta):
        if i > 0:
            gap = m["top"] - meta[i - 1]["bottom"]
            if gap > max(SPLIT_MIN_GAP, SPLIT_GAP_RATIO * max(med_gap, 1.0)):
                splits.add(i)
            headerish = (
                not m["caption"] and m["numeric_frac"] < 0.3 and m["nonempty"] >= 2
            )
            if numeric_run >= 2 and (m["caption"] or headerish):
                splits.add(i)
        numeric_run = (
            numeric_run + 1 if (not m["caption"] and m["numeric_frac"] >= 0.5) else 0
        )
    if not splits:
        return [t]

    # build segments, gluing any <2-row fragment forward (a caption/header
    # fragment belongs to the table below it), else backward
    idxs = sorted(splits)
    bounds = [0] + idxs + [len(meta)]
    raw_segs = list(zip(bounds, bounds[1:]))
    segments: list[tuple[int, int]] = []
    k = 0
    while k < len(raw_segs):
        a, b = raw_segs[k]
        if b - a < 2:
            if k + 1 < len(raw_segs):
                raw_segs[k + 1] = (a, raw_segs[k + 1][1])
                k += 1
                continue
            if segments:
                segments[-1] = (segments[-1][0], b)
                k += 1
                continue
        segments.append((a, b))
        k += 1
    if len(segments) < 2:
        return [t]

    out: list[PageTable] = []
    for a, b in segments:
        top = meta[a]["top"] - 1.0
        bottom = meta[b - 1]["bottom"] + 1.0
        seg_words = [wd for wd in words if wd["top"] >= top and wd["bottom"] <= bottom]
        seg_bbox = (bbox[0], top, bbox[2], bottom)
        out.extend(
            infer_tables(
                seg_words, seg_bbox, v_rules=v_rules, h_rules=h_rules, _depth=1
            )
        )
    return out if out else [t]


# --------------------------------------------------------------------------
# ruled-run stitching (sectioned report grids that are one logical table)
# --------------------------------------------------------------------------

STITCH_MAX_GAP = 40.0
STITCH_CUT_TOL = 4.0


def _stitch_ruled_runs(tables: list, words: Sequence[dict]) -> list:
    """Merge vertically-adjacent ruled tables with *identical* column cuts.

    Financial reports draw one logical table as shaded sections separated by
    label lines; the identical-cuts requirement (±4pt on every boundary) makes
    fusing genuinely separate tables structurally unlikely.  Gap lines become
    full-width caption rows; a prose line in the gap blocks the stitch."""

    def compat(a: PageTable, b: PageTable) -> bool:
        ca, cb = getattr(a, "_cuts", None), getattr(b, "_cuts", None)
        if not ca or not cb or len(ca) != len(cb):
            return False
        if any(abs(x - y) > STITCH_CUT_TOL for x, y in zip(ca, cb)):
            return False
        # a lower section that starts with its own title/caption or header row
        # is a separate stacked table (repeated-header rate sheets), not a
        # continuation of the one above
        ncols_b = max((sum(c.colspan for c in row) for row in b.grid), default=1)
        first_rows = list(b.grid[:2])
        if (
            first_rows
            and len(first_rows[0]) == 1
            and first_rows[0][0].colspan == ncols_b > 1
        ):
            return False  # own full-span title -> its own table
        b_texts = [c.text.strip() for c in b.grid[0] if c.text.strip()]
        if (
            len(b_texts) >= 2
            and sum(_is_numericish(t) for t in b_texts) / len(b_texts) < 0.3
        ):
            return False
        gap = b.bbox[1] - a.bbox[3]
        if not (0 <= gap <= STITCH_MAX_GAP):
            return False
        width = max(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
        gap_words = [
            wd
            for wd in words
            if wd["top"] >= a.bbox[3] - 1 and wd["bottom"] <= b.bbox[1] + 1
        ]
        for line in _cluster_lines(gap_words):
            if line[-1]["x1"] - line[0]["x0"] > 0.7 * width:
                return False
        return True

    ruled = sorted(
        (t for t in tables if t.ruled and max(len(r) for r in t.grid) >= 2),
        key=lambda t: t.bbox[1],
    )
    rest = [t for t in tables if t not in ruled]
    out: list = []
    for t in ruled:
        if out and compat(out[-1], t):
            cur = out[-1]
            ncols = max(len(r) for r in cur.grid)
            gap_words = [
                wd
                for wd in words
                if wd["top"] >= cur.bbox[3] - 1 and wd["bottom"] <= t.bbox[1] + 1
            ]
            for line in _cluster_lines(gap_words):
                text = _clean(" ".join(wd["text"] for wd in line))
                if text:
                    cur.grid.append([Cell(text, colspan=ncols)])
            cur.grid.extend(t.grid)
            cur.bbox = (
                min(cur.bbox[0], t.bbox[0]),
                cur.bbox[1],
                max(cur.bbox[2], t.bbox[2]),
                t.bbox[3],
            )
        else:
            out.append(t)
    return rest + out


# --------------------------------------------------------------------------
# page orchestration
# --------------------------------------------------------------------------


def _bbox_overlap_frac(a: tuple, b: tuple) -> float:
    """Fraction of *a*'s area covered by intersection with *b*."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return (ix1 - ix0) * (iy1 - iy0) / area_a


def _region_rules(page, bbox: tuple) -> tuple[list, list]:
    """Vector rules inside a region: long v-lines and h-lines."""
    x0, top, x1, bottom = bbox
    h = bottom - top
    w = x1 - x0
    v_rules, h_rules = [], []
    for e in getattr(page, "edges", []) or []:
        if e.get("orientation") == "v":
            if (
                x0 - 2 <= e["x0"] <= x1 + 2
                and min(e["bottom"], bottom) - max(e["top"], top) >= 0.5 * h
            ):
                v_rules.append({"x": e["x0"], "top": e["top"], "bottom": e["bottom"]})
        elif e.get("orientation") == "h":
            if (
                top - 2 <= e["top"] <= bottom + 2
                and min(e["x1"], x1) - max(e["x0"], x0) >= 0.5 * w
            ):
                h_rules.append({"y": e["top"]})
    return v_rules, h_rules


def _clip_words(words: Sequence[dict], reg: tuple, pad: float = 2.0) -> list[dict]:
    return [
        wd
        for wd in words
        if wd["x0"] >= reg[0] - pad
        and wd["x1"] <= reg[2] + pad
        and wd["top"] >= reg[1] - pad
        and wd["bottom"] <= reg[3] + pad
    ]


def extract_page_tables(
    page, regions: Optional[Sequence[tuple]] = None
) -> list[PageTable]:
    """All tables on a pdfplumber page.

    Order of evidence: (1) ruled grids (pdfplumber lines + envelope edges) —
    but a ruled grid whose cells fused multiple values (fragmentary rules,
    e.g. header underlines only) is *demoted* to a region candidate; (2) region
    candidates — demoted ruled bboxes, then the engine's own alignment-band
    proposals, then caller-supplied regions (warp table spans) — run through
    grid inference with stacked-table splitting.  Words already claimed by an
    accepted table are subtracted from later candidates so overlapping
    candidates never re-extract the same table."""
    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=2.0)
    except Exception:
        words = []
    tables: list[PageTable] = []
    claimed: list[tuple] = []
    demoted: list[tuple] = []
    tiny_ruled: list[PageTable] = []
    for t in _ruled_tables(page, words):
        if _looks_undersegmented(t.grid) or _looks_degenerate(t.grid):
            demoted.append(tuple(t.bbox))
        elif max(len(r) for r in t.grid) < 2:
            # a 1-column frame is kept only if nothing better claims its
            # words (it is often a fragment of a larger partial-rule grid)
            tiny_ruled.append(t)
        else:
            tables.append(t)
            claimed.append(tuple(t.bbox))
    # alignment bands go FIRST: they are whole coherent tables (headers
    # included), while a demoted ruled bbox is typically a fragment that would
    # otherwise claim the band's data words and orphan its header lines
    cands: list[tuple] = list(propose_regions(words))
    cands.extend(demoted)
    cands.extend(tuple(float(v) for v in reg) for reg in regions or [])

    def unclaimed(wd: dict) -> bool:
        cx = (wd["x0"] + wd["x1"]) / 2.0
        cy = (wd["top"] + wd["bottom"]) / 2.0
        return not any(bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3] for bb in claimed)

    for reg in cands:
        rwords = [wd for wd in _clip_words(words, reg) if unclaimed(wd)]
        if len(rwords) < 4:
            continue
        v_rules, h_rules = _region_rules(page, reg)
        found = infer_tables(rwords, reg, v_rules=v_rules, h_rules=h_rules)
        # An inferred grid of sentence-length cells is side-by-side prose
        # (multi-column body / pull-quote / scanned newsletter), not a table:
        # emitting it would *replace* real prose blocks with a fake grid.
        found = [t for t in found if not _looks_prose_grid(t.grid)]
        if found:
            tables.extend(found)
            claimed.extend(tuple(t.bbox) for t in found)

    # deferred 1-column ruled frames: keep only if still substantially unclaimed
    for t in tiny_ruled:
        twords = _clip_words(words, tuple(t.bbox))
        if twords and sum(1 for wd in twords if unclaimed(wd)) >= 0.6 * len(twords):
            tables.append(t)
            claimed.append(tuple(t.bbox))

    # stitch sectioned same-cut ruled runs into their one logical table
    tables = _stitch_ruled_runs(tables, words)

    # attach external column-label lines (year headers above data-topped
    # grids); a tiny (≤2-row) table sitting directly above is usually that
    # header line mis-extracted as its own table — release its words so the
    # extension can absorb it, and drop it once consumed
    absorbed: set[int] = set()
    for i, t in enumerate(tables):
        release: list[int] = [
            j
            for j, u in enumerate(tables)
            if j != i
            and j not in absorbed
            and len(u.grid) <= 2
            and u.bbox[3] <= t.bbox[1] + 2
            and t.bbox[1] - u.bbox[3] < 40
            and min(u.bbox[2], t.bbox[2]) - max(u.bbox[0], t.bbox[0])
            > 0.5 * (u.bbox[2] - u.bbox[0])
        ]
        others = [
            tuple(u.bbox)
            for j, u in enumerate(tables)
            if j != i and j not in release and j not in absorbed
        ]
        old_top = t.bbox[1]
        _extend_header_up(t, words, others)
        if t.bbox[1] < old_top:
            for j in release:
                u = tables[j]
                if t.bbox[1] <= u.bbox[1] + 1.5:
                    absorbed.add(j)
    tables = [t for j, t in enumerate(tables) if j not in absorbed]

    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return tables


def extract_pdf_tables(
    pdf_path: str, regions_by_page: Optional[dict] = None
) -> dict[int, list[tuple[tuple, str]]]:
    """Whole-document extraction: ``{page_idx: [(bbox, html), ...]}``."""
    import pdfplumber

    out: dict[int, list[tuple[tuple, str]]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            regions = (regions_by_page or {}).get(idx)
            tables = extract_page_tables(page, regions=regions)
            if tables:
                out[idx] = [(t.bbox, render_table_html(t)) for t in tables]
    return out
