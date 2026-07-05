"""Visualize + diagnose an ``OpenContractDocExport``.

Two diagnostic views over a *finished* export (this module reads export dicts
only — it never touches ``visual_ingestor`` or the XHTML contract):

* :func:`render_tree_outline` — a compact, indented text outline of the
  ``parent_id`` annotation tree plus structural-health stats, so a document's
  structure is legible at a glance.
* :func:`project_page` / :func:`project_document` — overlay a page's annotation
  bounding boxes onto the PDF raster, **grouped by relationship** (each box is
  colored by the root of its structural subtree, with child→parent connector
  lines and a legend), so the engine's structural decisions can be *seen*.

:func:`diagnostics` exposes the machine-readable structural metrics the two
views summarize; it is what the heuristic-gap audit consumes.

Format reference: ``docs/opencontracts_export_format.md``.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Labels that may legitimately parent other annotations. Everything else that
# bears children is a structural smell (e.g. a run-in header demoted to body but
# still acting as a parent).
HEADER_LABELS = frozenset({"Section Header", "Title"})

# Kept in sync with the exporter's run-in-header threshold.
try:  # pragma: no cover - trivial import guard
    from warp_ingest.ingestor.opencontracts_exporter import _HEADER_MAX_WORDS
except Exception:  # pragma: no cover
    _HEADER_MAX_WORDS = 12


# --------------------------------------------------------------------------- #
# annotation tree model
# --------------------------------------------------------------------------- #
@dataclass
class Tree:
    """Resolved ``parent_id`` hierarchy over an export's annotations."""

    nodes: dict = field(default_factory=dict)  # id -> annotation
    parent: dict = field(default_factory=dict)  # id -> parent id or None
    children: dict = field(default_factory=dict)  # id -> [child id] (reading order)
    roots: list = field(default_factory=list)  # root ids (reading order)
    depth: dict = field(default_factory=dict)  # id -> int (root == 0)
    subtree_root: dict = field(default_factory=dict)  # id -> top-level ancestor id


def _primary_page_bounds(a):
    """(page_index, top, left, right, bottom) for an annotation's primary page."""
    aj = a.get("annotation_json") or {}
    pkey = str(a.get("page"))
    page_ann = aj.get(pkey)
    if page_ann is None and aj:
        pkey = sorted(aj, key=lambda k: int(k))[0]
        page_ann = aj[pkey]
    page = int(pkey) if pkey.lstrip("-").isdigit() else 0
    if not page_ann:
        return page, 0.0, 0.0, 0.0, 0.0
    b = page_ann["bounds"]
    return (
        page,
        float(b["top"]),
        float(b["left"]),
        float(b["right"]),
        float(b["bottom"]),
    )


def _reading_key(a):
    page, top, left, _r, _b = _primary_page_bounds(a)
    return (page, top, left)


def _token_count(a):
    return sum(
        len(pg.get("tokensJsons", ()))
        for pg in (a.get("annotation_json") or {}).values()
    )


def build_tree(export) -> Tree:
    """Resolve the ``parent_id`` hierarchy into a cycle-safe :class:`Tree`."""
    anns = export.get("labelled_text", [])
    nodes = {a["id"]: a for a in anns}

    parent = {}
    for a in anns:
        p = a.get("parent_id")
        # a parent that doesn't exist / points at self is no parent (a root).
        parent[a["id"]] = p if (p in nodes and p != a["id"]) else None

    children = defaultdict(list)
    for aid, p in parent.items():
        if p is not None:
            children[p].append(aid)
    for p in children:
        children[p].sort(key=lambda c: _reading_key(nodes[c]))

    roots = sorted(
        (aid for aid, p in parent.items() if p is None),
        key=lambda c: _reading_key(nodes[c]),
    )

    def depth_of(aid):
        d, seen, node = 0, set(), parent[aid]
        while node is not None and node not in seen:
            seen.add(node)
            d += 1
            node = parent.get(node)
        return d

    def root_of(aid):
        seen, node = set(), aid
        while parent.get(node) is not None and node not in seen:
            seen.add(node)
            node = parent[node]
        return node

    depth = {aid: depth_of(aid) for aid in nodes}
    subtree_root = {aid: root_of(aid) for aid in nodes}
    return Tree(
        nodes=nodes,
        parent=parent,
        children=dict(children),
        roots=roots,
        depth=depth,
        subtree_root=subtree_root,
    )


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def diagnostics(export) -> dict:
    """Structural-health metrics for one export (used to find heuristic gaps)."""
    tree = build_tree(export)
    anns = export.get("labelled_text", [])

    label_histogram = Counter(a["annotationLabel"] for a in anns)
    depth_histogram = Counter(tree.depth[a["id"]] for a in anns)

    # a non-heading that bears children is a smell — EXCEPT a lead-in paragraph
    # whose children are all List Items ("intro: A. … B. …"), which is intended.
    def _bad_parent(aid):
        if tree.nodes[aid]["annotationLabel"] in HEADER_LABELS:
            return False
        kids = tree.children.get(aid) or []
        if tree.nodes[aid]["annotationLabel"] == "Paragraph" and all(
            tree.nodes[c]["annotationLabel"] == "List Item" for c in kids
        ):
            return False
        return True

    childbearing_non_header_ids = [
        aid for aid in tree.nodes if tree.children.get(aid) and _bad_parent(aid)
    ]
    non_heading_roots = [
        rid
        for rid in tree.roots
        if tree.nodes[rid]["annotationLabel"] not in HEADER_LABELS
    ]
    overlong_heading_ids = [
        a["id"]
        for a in anns
        if a["annotationLabel"] == "Section Header"
        and len(a["rawText"].split()) > _HEADER_MAX_WORDS
    ]
    untokened_ids = [a["id"] for a in anns if _token_count(a) == 0]

    pawls = export.get("pawls_file_content", [])
    total_tokens = sum(len(p["tokens"]) for p in pawls)
    claimed = set()
    for a in anns:
        for pg in (a.get("annotation_json") or {}).values():
            for r in pg.get("tokensJsons", ()):
                claimed.add((r["pageIndex"], r["tokenIndex"]))

    return {
        "n_annotations": len(anns),
        "n_roots": len(tree.roots),
        "max_depth": max(tree.depth.values(), default=0),
        "label_histogram": dict(label_histogram),
        "depth_histogram": dict(depth_histogram),
        "childbearing_non_headers": len(childbearing_non_header_ids),
        "childbearing_non_header_ids": childbearing_non_header_ids,
        "non_heading_roots": len(non_heading_roots),
        "non_heading_root_ids": non_heading_roots,
        "overlong_headings": len(overlong_heading_ids),
        "overlong_heading_ids": overlong_heading_ids,
        "untokened": len(untokened_ids),
        "untokened_ids": untokened_ids,
        "token_coverage": (
            round(len(claimed) / total_tokens, 4) if total_tokens else 1.0
        ),
    }


def page_summary(export, page_index) -> list:
    """Per-page annotation rows (reading order) with parent label + depth.

    The compact, page-scoped structural view handed to the audit alongside the
    projection PNG.
    """
    tree = build_tree(export)
    page_key = str(page_index)
    rows = []
    for a in export.get("labelled_text", []):
        if page_key not in (a.get("annotation_json") or {}):
            continue
        pid = tree.parent.get(a["id"])
        rows.append(
            {
                "id": a["id"],
                "label": a["annotationLabel"],
                "parent_id": pid,
                "parent_label": tree.nodes[pid]["annotationLabel"] if pid else None,
                "depth": tree.depth.get(a["id"], 0),
                "n_tokens": _token_count(a),
                "text": " ".join((a.get("rawText") or "").split()),
            }
        )
    rows.sort(key=lambda r: _reading_key(tree.nodes[r["id"]]))
    return rows


# --------------------------------------------------------------------------- #
# text outline
# --------------------------------------------------------------------------- #
def _truncate(text, max_text):
    text = " ".join(text.split())
    if len(text) > max_text:
        return text[:max_text].rstrip() + "…"
    return text


def render_tree_outline(export, *, max_text=80) -> str:
    """A compact, indented text outline of the annotation tree + stats header."""
    tree = build_tree(export)
    diag = diagnostics(export)

    title = export.get("title") or "(untitled)"
    n_tokens = sum(len(p["tokens"]) for p in export.get("pawls_file_content", []))
    labels = ", ".join(f"{k}={v}" for k, v in sorted(diag["label_histogram"].items()))

    header = [
        f"# {title}",
        f"# page_count={export.get('page_count')} tokens={n_tokens} "
        f"annotations={diag['n_annotations']}",
        f"# labels: {labels}",
        f"# roots={diag['n_roots']} max_depth={diag['max_depth']} "
        f"non_heading_roots={diag['non_heading_roots']} "
        f"childbearing_non_headers={diag['childbearing_non_headers']} "
        f"overlong_headings={diag['overlong_headings']} "
        f"untokened={diag['untokened']} token_coverage={diag['token_coverage']}",
        "#",
    ]

    lines = []
    visited = set()

    def walk(aid, depth):
        if aid in visited:
            return
        visited.add(aid)
        a = tree.nodes[aid]
        page, *_ = _primary_page_bounds(a)
        indent = "  " * depth
        lines.append(
            f"{indent}- [{aid}] {a['annotationLabel']} "
            f"(p{page}, {_token_count(a)}t) {_truncate(a['rawText'], max_text)}"
        )
        for child in tree.children.get(aid, ()):
            walk(child, depth + 1)

    for root in tree.roots:
        walk(root, 0)
    # any node not reachable from a root (a cycle member) still gets listed.
    for aid in tree.nodes:
        if aid not in visited:
            walk(aid, 0)

    return "\n".join(header + lines)


# --------------------------------------------------------------------------- #
# raster projection
# --------------------------------------------------------------------------- #
# visually distinct palette (RGB), reused cyclically across subtree groups.
_PALETTE = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (166, 86, 40),
    (247, 129, 191),
    (0, 162, 162),
    (153, 153, 0),
    (106, 61, 154),
    (31, 120, 180),
    (178, 34, 34),
]
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _load_font(size):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_pdf_page(pdf_bytes, page_index, scale):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page = pdf[page_index]
        return page.render(scale=scale).to_pil().convert("RGB")
    finally:
        pdf.close()


def _color_for(key, color_map):
    if key not in color_map:
        color_map[key] = _PALETTE[len(color_map) % len(_PALETTE)]
    return color_map[key]


def _darken(color, factor=0.72):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _text_bbox(draw, xy, text, font):
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:  # pragma: no cover - Pillow < 8 compatibility
        w, h = draw.textsize(text, font=font)
        x, y = xy
        return (x, y, x + w, y + h)


def _text_size(draw, text, font):
    x0, y0, x1, y1 = _text_bbox(draw, (0, 0), text, font)
    return x1 - x0, y1 - y0


def _draw_label_pill(draw, xy, ident, text, color, font, scale):
    """Draw a white-backed annotation callout with a colored id chip."""
    w, h, chip_w, ident_h, text_h = _label_pill_metrics(draw, ident, text, font, scale)
    x, y = xy
    radius = min(7, max(2, int(3 * scale)))
    box = (x, y, x + w, y + h)

    draw.rounded_rectangle(
        box, radius=radius, fill=(255, 255, 255), outline=color, width=1
    )
    draw.rounded_rectangle(
        (x, y, x + chip_w, y + h),
        radius=radius,
        fill=color,
        outline=color,
        width=1,
    )
    chip_pad = max(5, int(5 * scale))
    gap = max(4, int(3 * scale))
    draw.text(
        (x + chip_pad, y + (h - ident_h) / 2), ident, fill=(255, 255, 255), font=font
    )
    draw.text(
        (x + chip_w + gap, y + (h - text_h) / 2),
        text,
        fill=(25, 25, 25),
        font=font,
    )
    return box


def _label_pill_metrics(draw, ident, text, font, scale):
    pad_x = max(4, int(4 * scale))
    pad_y = max(2, int(2 * scale))
    chip_pad = max(5, int(5 * scale))
    gap = max(4, int(3 * scale))
    ident_w, ident_h = _text_size(draw, ident, font)
    text_w, text_h = _text_size(draw, text, font)
    h = max(ident_h, text_h) + 2 * pad_y
    chip_w = ident_w + 2 * chip_pad
    w = chip_w + gap + text_w + 2 * pad_x
    return w, h, chip_w, ident_h, text_h


def _label_candidates(box, label_w, label_h, img_w, img_h, scale):
    x0, y0, x1, y1 = box
    gap = max(4, int(4 * scale))
    candidates = [
        (x0, y0 - label_h - gap),
        (x0, y1 + gap),
        (x1 - label_w, y0 - label_h - gap),
        (x1 - label_w, y1 + gap),
        (x1 + gap, y0),
        (x0 - label_w - gap, y0),
        (x0 + gap, y0 + gap),
    ]
    for x, y in candidates:
        yield (
            max(0, min(img_w - label_w, x)),
            max(0, min(img_h - label_h, y)),
        )


def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(box, label_size, img_size, occupied, scale):
    label_w, label_h = label_size
    img_w, img_h = img_size
    for x, y in _label_candidates(box, label_w, label_h, img_w, img_h, scale):
        candidate = (x, y, x + label_w, y + label_h)
        if not any(_overlaps(candidate, taken) for taken in occupied):
            occupied.append(candidate)
            return x, y

    # Crowded pages are common; keep moving down in small steps until a readable
    # slot appears, then fall back to the first clamped candidate.
    x, y = next(_label_candidates(box, label_w, label_h, img_w, img_h, scale))
    step = max(label_h + 2, int(10 * scale))
    start_y = y
    for _ in range(max(1, img_h // max(1, step))):
        candidate = (x, y, x + label_w, y + label_h)
        if not any(_overlaps(candidate, taken) for taken in occupied):
            occupied.append(candidate)
            return x, y
        y = (y + step) % max(1, img_h - label_h)
        if y == start_y:
            break
    occupied.append((x, start_y, x + label_w, start_y + label_h))
    return x, start_y


def _rect_anchor(rect, toward):
    """Point on ``rect`` edge in the direction of ``toward`` from rect center."""
    import math

    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    half_w, half_h = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
    t = min(half_w / abs(dx) if dx else math.inf, half_h / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


def _draw_arrow(draw, src, dst, color, scale):
    """Draw a haloed parent→child connector with an arrowhead."""
    import math

    color = _darken(color)
    width = max(2, int(scale * 1.35))
    halo = width + max(3, int(scale * 1.75))
    draw.line([src, dst], fill=(255, 255, 255), width=halo)
    draw.line([src, dst], fill=color, width=width)
    ang = math.atan2(dst[1] - src[1], dst[0] - src[0])
    size = max(6, 7 * scale)
    for da in (math.radians(150), math.radians(-150)):
        end = (
            dst[0] + size * math.cos(ang + da),
            dst[1] + size * math.sin(ang + da),
        )
        draw.line([dst, end], fill=(255, 255, 255), width=halo)
        draw.line([dst, end], fill=color, width=width)


def _draw_relation_label(draw, src, dst, label, color, font, occupied, img_size, scale):
    if not label:
        return
    text = str(label)
    text_w, text_h = _text_size(draw, text, font)
    pad_x = max(4, int(3 * scale))
    pad_y = max(2, int(2 * scale))
    w = text_w + 2 * pad_x
    h = text_h + 2 * pad_y
    mx = (src[0] + dst[0]) / 2.0 - w / 2.0
    my = (src[1] + dst[1]) / 2.0 - h / 2.0
    x = max(0, min(img_size[0] - w, mx))
    y = max(0, min(img_size[1] - h, my))
    candidate = (x, y, x + w, y + h)
    if any(_overlaps(candidate, taken) for taken in occupied):
        return
    occupied.append(candidate)
    draw.rounded_rectangle(
        candidate,
        radius=min(6, max(2, int(3 * scale))),
        fill=(255, 255, 255),
        outline=_darken(color),
        width=1,
    )
    draw.text((x + pad_x, y + pad_y), text, fill=_darken(color), font=font)


def project_page(export, pdf_bytes, page_index, *, scale=2.0, mode="relationship"):
    """Render one PDF page with its annotation boxes overlaid (grouped by relationship).

    ``mode='relationship'`` colors each box by the root of its structural subtree
    and draws labelled parent→child connectors; ``mode='label'`` colors by
    annotation label. Returns a ``PIL.Image`` (the page raster plus a legend strip).
    """
    from PIL import Image, ImageDraw

    base = _render_pdf_page(pdf_bytes, page_index, scale)
    tree = build_tree(export)

    # annotations that touch this page, with their per-page boxes
    page_key = str(page_index)
    on_page = []
    for a in export.get("labelled_text", []):
        page_ann = (a.get("annotation_json") or {}).get(page_key)
        if page_ann is None:
            continue
        b = page_ann["bounds"]
        on_page.append((a, b))
    on_page.sort(key=lambda ab: _reading_key(ab[0]))

    color_map = {}

    def key_of(a):
        if mode == "label":
            return a["annotationLabel"]
        return tree.subtree_root.get(a["id"], a["id"])

    box_overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    box_draw = ImageDraw.Draw(box_overlay)
    label_font = _load_font(max(11, int(7.5 * scale)))
    rel_font = _load_font(max(9, int(6.25 * scale)))
    centers = {}
    boxes = {}
    display_ids = {a["id"]: f"{i + 1:02d}" for i, (a, _b) in enumerate(on_page)}

    for a, b in on_page:
        color = _color_for(key_of(a), color_map)
        x0, y0 = b["left"] * scale, b["top"] * scale
        x1, y1 = b["right"] * scale, b["bottom"] * scale
        stroke_w = max(
            2, int(scale * (1.5 if a["annotationLabel"] in HEADER_LABELS else 1.0))
        )
        box_draw.rectangle(
            [x0, y0, x1, y1],
            fill=(*color, 18),
            outline=(*color, 230),
            width=stroke_w,
        )
        centers[a["id"]] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        boxes[a["id"]] = (x0, y0, x1, y1)

    composed = Image.alpha_composite(base.convert("RGBA"), box_overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    occupied_labels = []

    # Relationship edges: an arrow from each parent (source) to each child
    # (target), labelled with the relationship name. Falls back to ``parent_id``
    # when the export predates explicit ``OC_PARENT_CHILD`` relationships.
    if mode == "relationship":
        edges = []
        rels = export.get("relationships") or []
        if rels:
            for rel in rels:
                for src in rel.get("source_annotation_ids", ()):
                    for tgt in rel.get("target_annotation_ids", ()):
                        edges.append(
                            (src, tgt, rel.get("relationshipLabel") or "relationship")
                        )
        else:
            edges = [
                (tree.parent.get(a["id"]), a["id"], "parent_id") for a, _b in on_page
            ]
        page_of = {a["id"]: a.get("page") for a in export.get("labelled_text", [])}
        visible_edges = [
            (src, tgt, rel_label) for src, tgt, rel_label in edges if tgt in centers
        ]
        label_every_edge = len(visible_edges) <= 12
        labelled_sources = set()
        drawn_to = set()
        for src, tgt, rel_label in edges:
            if not src or not tgt or tgt not in tree.nodes:
                continue
            if src in centers and tgt in centers:
                color = _color_for(key_of(tree.nodes[tgt]), color_map)
                src_pt = _rect_anchor(boxes[src], centers[tgt])
                dst_pt = _rect_anchor(boxes[tgt], centers[src])
                _draw_arrow(draw, src_pt, dst_pt, color, scale)
                should_label_edge = label_every_edge or src not in labelled_sources
                if should_label_edge:
                    labelled_sources.add(src)
                    _draw_relation_label(
                        draw,
                        src_pt,
                        dst_pt,
                        rel_label,
                        color,
                        rel_font,
                        occupied_labels,
                        composed.size,
                        scale,
                    )
            elif tgt in centers and src not in centers and tgt not in drawn_to:
                # Cross-page edge: draw a margin-origin cue into the child so the
                # page still shows that the visible node has an off-page parent.
                drawn_to.add(tgt)
                color = _color_for(key_of(tree.nodes[tgt]), color_map)
                parent_page = page_of.get(src, "?")
                margin_y = max(4.0, 5.0 * scale)
                if isinstance(parent_page, int) and parent_page > page_index:
                    margin_y = composed.height - margin_y
                start = (centers[tgt][0], margin_y)
                end = _rect_anchor(boxes[tgt], start)
                _draw_arrow(draw, start, end, color, scale)
                _draw_relation_label(
                    draw,
                    start,
                    end,
                    f"{rel_label} from p{parent_page}",
                    color,
                    rel_font,
                    occupied_labels,
                    composed.size,
                    scale,
                )

    # Annotation callouts are drawn last, outside the translucent overlay, so
    # IDs/labels stay crisp while the underlying PDF text remains visible.
    for a, _b in on_page:
        color = _color_for(key_of(a), color_map)
        aid = a["id"]
        pid = tree.parent.get(aid)
        ident = display_ids[aid]
        label_text = f"d{tree.depth.get(aid, 0)} {a['annotationLabel']}"
        if mode == "relationship":
            if pid in display_ids:
                label_text += f" <- {display_ids[pid]}"
            elif pid is not None:
                label_text += f" <- p{tree.nodes.get(pid, {}).get('page', '?')}"
            else:
                label_text += " root"
        w, h, *_ = _label_pill_metrics(draw, ident, label_text, label_font, scale)
        x, y = _place_label(
            boxes[aid],
            (w, h),
            composed.size,
            occupied_labels,
            scale,
        )
        _draw_label_pill(draw, (x, y), ident, label_text, color, label_font, scale)

    return _append_legend(composed, color_map, tree, mode, scale)


def _append_legend(img, color_map, tree, mode, scale):
    from PIL import Image, ImageDraw

    if not color_map:
        return img
    font = _load_font(max(11, int(7 * scale)))
    line_h = int(16 * max(1.0, scale / 2))
    pad = 8
    entries = []
    notes = []
    if mode == "relationship":
        notes = [
            "Callout: NN dD Label <- parentNN; root = no parent on tree",
            "Arrow: parent -> child; arrow pill = relationshipLabel",
        ]
    for key, color in color_map.items():
        if mode == "label":
            label = str(key)
        else:
            node = tree.nodes.get(key, {})
            txt = " ".join((node.get("rawText") or "").split())[:48]
            label = f"[{key}] {node.get('annotationLabel', '?')}: {txt}"
        entries.append((color, label))

    strip_h = pad * 2 + line_h * (len(notes) + len(entries))
    out = Image.new("RGB", (img.width, img.height + strip_h), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    y = img.height + pad
    for note in notes:
        draw.text((pad, y), note, fill=(35, 35, 35), font=font)
        y += line_h
    for color, label in entries:
        draw.rectangle([pad, y + 2, pad + line_h - 6, y + line_h - 4], fill=color)
        draw.text((pad + line_h, y), label, fill=(0, 0, 0), font=font)
        y += line_h
    return out


def project_document(
    export, pdf_bytes, out_dir, *, pages=None, scale=2.0, mode="relationship"
):
    """Write a projection PNG per page and the tree outline; return written paths."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    written = []
    n_pages = export.get("page_count", len(export.get("pawls_file_content", [])))
    targets = pages if pages is not None else range(n_pages)
    for i in targets:
        img = project_page(export, pdf_bytes, i, scale=scale, mode=mode)
        path = os.path.join(out_dir, f"page_{i:03d}.png")
        img.save(path)
        written.append(path)
    tree_path = os.path.join(out_dir, "tree.txt")
    with open(tree_path, "w") as fh:
        fh.write(render_tree_outline(export))
    written.append(tree_path)
    return written
