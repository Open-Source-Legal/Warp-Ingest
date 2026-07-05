"""Golden-agreement scoring core for the structural-correctness evals.

Both the legal-100 and heterogeneous-100 suites score Warp's live
``OpenContractDocExport`` against a **parser-independent golden answer set** — a
per-page list of regions ``{region_id, bbox_frac, text, gold_label,
gold_parent_region_id, ro_index}``. Annotation IDs are never used.

**Why overlap-based, table-lenient scoring.** Warp and the golden annotate at
different *granularity*: Warp emits one annotation per visual line / table row,
while ParseBench groups a whole table (or a multi-line block) into one region —
and, like the Docling oracle, ParseBench **under-labels tables** (it often tags a
data table's cells as ``Text`` or collapses the table to a single region). So we
(a) assign each Warp annotation to the gold region it geometrically overlaps most
(**many Warp → one gold**, not 1:1), and (b) keep tables out of the headline
label metric, scoring them only by *coverage* and reporting the
heading/list/paragraph quality where the oracle is reliable.

Metrics (all pure functions of ``(export, page, gold_regions)``):

* **heading_f1 / list_f1 / paragraph_f1** — one-vs-rest over gold
  {Heading, Paragraph, List Item} regions; ``struct_macro_f1`` is their mean
  (the headline). Gold body regions whose dominant Warp overlap is *Table Row*
  are pulled OUT of these and counted in ``body_as_tablerow_frac`` instead — that
  is the multi-column-fusion / table-over-firing signal, tracked on its own so
  oracle table-noise never distorts the label metric.
* **gold_coverage** — scored gold regions with any Warp overlap.
* **spurious_frac** — Warp annotations overlapping *no* gold region (truly
  invented boxes).
* **table_region_coverage** — gold Table Row regions covered by >=1 Warp Table
  Row (descriptive; the oracle is weak here).
* **body_as_tablerow_frac** — gold Heading/Paragraph/List predicted Table Row by
  Warp (multi-column fusion smell; ceiled).
* **furniture_as_heading** — gold Page-header/footer promoted to a heading.
* **head_ancestor_agreement / parent_class_agreement** — relationship metrics.
* **reading_order_agreement** — Warp document order vs gold ``ro_index``.

``Image`` gold regions are excluded entirely (Warp defers image tokens, issue
#1). ``Furniture`` is excluded from the label metrics (tracked only via
``furniture_as_heading``).
"""

import json

from warp_ingest.ingestor.segmentation_metrics import pk as _pk
from warp_ingest.ingestor.segmentation_metrics import windowdiff as _windowdiff

_HEADING_LABELS = frozenset({"Section Header", "Title"})
# gold labels that carry a reliable ParseBench structural signal
_BODY_LABELS = frozenset({"Paragraph", "List Item"})
_LABEL_CLASSES = ("Heading", "Paragraph", "List Item")  # one-vs-rest F1 space
_OVERLAP_MIN = 0.5
_JACCARD_MIN = 0.5


def load_golden(path):
    """Load a golden JSON: ``{page_key: {regions: [...], ...}}``."""
    with open(path) as fh:
        return json.load(fh)


def coarse(label):
    """Collapse the label space to the heading/body distinction we score."""
    return "Heading" if label in _HEADING_LABELS else label


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _tokens(text):
    return frozenset(
        w
        for w in "".join(
            c.lower() if c.isalnum() else " " for c in (text or "")
        ).split()
    )


def _jaccard(t1, t2):
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


# --------------------------------------------------------------------------- #
# live-export accessors
# --------------------------------------------------------------------------- #
def _page_dims(export, page):
    pawls = export.get("pawls_file_content", [])
    if 0 <= page < len(pawls):
        meta = pawls[page].get("page", {})
        return float(meta.get("width") or 1.0), float(meta.get("height") or 1.0)
    return 1.0, 1.0


def warp_annotations(export, page):
    """Warp annotations on ``page`` with normalized box + doc order."""
    w, h = _page_dims(export, page)
    pkey = str(page)
    out = []
    for order, a in enumerate(export.get("labelled_text", [])):
        aj = a.get("annotation_json") or {}
        if pkey not in aj:
            continue
        b = aj[pkey].get("bounds") or {}
        box = [
            b.get("left", 0.0) / w,
            b.get("top", 0.0) / h,
            b.get("right", 0.0) / w,
            b.get("bottom", 0.0) / h,
        ]
        out.append(
            {
                "id": a["id"],
                "label": a["annotationLabel"],
                "coarse": coarse(a["annotationLabel"]),
                "box": box,
                "text": aj[pkey].get("rawText") or a.get("rawText") or "",
                "order": order,
                "parent_id": a.get("parent_id"),
            }
        )
    return out


_UNIT_LABEL = "Semantic Unit"


def warp_units(export, page):
    """Per-page projection of the Semantic-Unit layer (see semantic_units.py)."""
    w, h = _page_dims(export, page)
    pkey = str(page)
    # member fine ids per unit (OC_SEMANTIC_UNIT edges) and unit->parent-unit
    # (OC_PARENT_CHILD edges among "su-*" units) — the hierarchy is in the
    # relationships, never parent_id.
    members = {}
    unit_parent = {}
    for r in export.get("relationships", []):
        label = r.get("relationshipLabel")
        src = r["source_annotation_ids"][0]
        if label == "OC_SEMANTIC_UNIT":
            members[src] = list(r["target_annotation_ids"])
        elif (
            label == "OC_PARENT_CHILD"
            and isinstance(src, str)
            and src.startswith("su-")
        ):
            for tgt in r["target_annotation_ids"]:
                unit_parent[tgt] = src
    # per-page reading-order index over FINE annotations, matching the golden
    # builder (enumerate over this page's non-unit annotations in doc order) so
    # the Pk/WindowDiff masks share one axis.
    order_of = {}
    _idx = 0
    for a in export.get("labelled_text", []):
        if a["annotationLabel"] == _UNIT_LABEL:
            continue
        if pkey in (a.get("annotation_json") or {}):
            order_of[a["id"]] = _idx
            _idx += 1
    out = []
    for a in export.get("labelled_text", []):
        if a["annotationLabel"] != _UNIT_LABEL:
            continue
        aj = a.get("annotation_json") or {}
        if pkey not in aj:
            continue
        b = aj[pkey].get("bounds") or {}
        box = [
            b.get("left", 0.0) / w,
            b.get("top", 0.0) / h,
            b.get("right", 0.0) / w,
            b.get("bottom", 0.0) / h,
        ]
        mem = members.get(a["id"], [])
        orders = [order_of[m] for m in mem if m in order_of]
        out.append(
            {
                "unit_id": a["id"],
                "unit_label": a["annotationLabel"],
                "box": box,
                "member_ann_ids": mem,
                "order_start": min(orders) if orders else order_of.get(a["id"], 0),
                "order_end": max(orders) if orders else order_of.get(a["id"], 0),
                "parent_unit_id": unit_parent.get(a["id"]),
                "text": aj[pkey].get("rawText") or a.get("rawText") or "",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# assignment: each Warp annotation -> the gold region it overlaps most
# --------------------------------------------------------------------------- #
def assign(warp_anns, gold_regions):
    """Assign each Warp annotation to a gold region (many Warp -> one gold).

    A Warp box assigns to the gold region with the largest *directional*
    overlap (max of fraction-of-Warp and fraction-of-gold) when that is >= 0.5,
    else by text-Jaccard >= 0.5, else it is spurious (assigned to None). Returns
    ``w2g`` (warp index -> gold index or None) and ``g2ws`` (gold index ->
    [warp indices], best-overlap first).
    """
    w2g = {}
    g2ws = {j: [] for j in range(len(gold_regions))}
    gold_box = [g["bbox_frac"] for g in gold_regions]
    gold_tok = [_tokens(g["text"]) for g in gold_regions]
    gold_area = [_area(b) for b in gold_box]
    for i, wa in enumerate(warp_anns):
        wt = _tokens(wa["text"])
        wa_area = _area(wa["box"]) or 1e-9
        best_j, best_score = None, 0.0
        for j, g in enumerate(gold_regions):
            if g["gold_label"] == "Image":
                continue
            inter = _inter(wa["box"], gold_box[j])
            cover = max(inter / wa_area, inter / (gold_area[j] or 1e-9))
            jac = _jaccard(wt, gold_tok[j])
            score = max(cover, jac)
            if score > best_score:
                best_j, best_score = j, score
        if best_j is not None and best_score >= _OVERLAP_MIN:
            w2g[i] = best_j
            g2ws[best_j].append((i, best_score))
        else:
            # fall back to a pure text match when geometry is weak
            tj, ts = None, _JACCARD_MIN
            for j, g in enumerate(gold_regions):
                if g["gold_label"] == "Image":
                    continue
                s = _jaccard(wt, gold_tok[j])
                if s >= ts:
                    tj, ts = j, s
            if tj is not None:
                w2g[i] = tj
                g2ws[tj].append((i, ts))
            else:
                w2g[i] = None
    for j in g2ws:
        g2ws[j].sort(key=lambda t: -t[1])
        g2ws[j] = [i for i, _ in g2ws[j]]
    return w2g, g2ws


def _dominant_label(warp_anns, idxs):
    """Area-weighted dominant coarse label among assigned Warp annotations."""
    if not idxs:
        return None
    weight = {}
    for i in idxs:
        wa = warp_anns[i]
        weight[wa["coarse"]] = weight.get(wa["coarse"], 0.0) + (
            _area(wa["box"]) or 1e-9
        )
    return max(weight, key=weight.get)


# --------------------------------------------------------------------------- #
# relationships
# --------------------------------------------------------------------------- #
def _warp_heading_ancestor(ann_id, by_id):
    seen = set()
    cur = by_id.get(ann_id, {}).get("parent_id")
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        if by_id[cur]["annotationLabel"] in _HEADING_LABELS:
            return cur
        cur = by_id[cur].get("parent_id")
    return None


def _gold_heading_ancestor(region, by_id):
    seen = set()
    cur = region.get("gold_parent_region_id")
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        if by_id[cur]["gold_label"] in _HEADING_LABELS:
            return cur
        cur = by_id[cur].get("gold_parent_region_id")
    return None


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0


# --------------------------------------------------------------------------- #
# main scorer
# --------------------------------------------------------------------------- #
def score_page(export, page, gold_regions, *, have_ro=False):
    """Deterministic golden-agreement metrics for one (export, page, golden)."""
    warp = warp_annotations(export, page)
    w2g, g2ws = assign(warp, gold_regions)
    by_id = {a["id"]: a for a in export.get("labelled_text", [])}
    rid_to_idx = {g["region_id"]: idx for idx, g in enumerate(gold_regions)}
    region_by_id = {g["region_id"]: g for g in gold_regions}

    # predicted coarse label per gold region (None if uncovered)
    pred = {j: _dominant_label(warp, g2ws[j]) for j in range(len(gold_regions))}

    # ---- one-vs-rest F1 over {Heading, Paragraph, List Item} --------------
    # gold body/heading regions whose dominant prediction is Table Row are pulled
    # out into body_as_tablerow (multi-column / table-over-fire), not scored here.
    tp = {c: 0 for c in _LABEL_CLASSES}
    fp = {c: 0 for c in _LABEL_CLASSES}
    fn = {c: 0 for c in _LABEL_CLASSES}
    label_gold = 0
    covered_gold = 0
    body_total = body_as_tablerow = 0
    for j, g in enumerate(gold_regions):
        gc = coarse(g["gold_label"])
        if gc not in _LABEL_CLASSES:
            continue  # Table Row / Furniture / Image handled elsewhere
        label_gold += 1
        p = pred[j]
        if p is not None:
            covered_gold += 1
        if gc != "Heading":
            body_total += 1
            if p == "Table Row":
                body_as_tablerow += 1
        if p == "Table Row":
            # excluded from label F1 (oracle table-noise / multi-column fusion)
            continue
        if p == gc:
            tp[gc] += 1
        else:
            fn[gc] += 1
            if p in fp:
                fp[p] += 1
    # spurious / cross Warp predictions land as false positives for their class
    spurious = [i for i, j in w2g.items() if j is None]
    for i in spurious:
        c = warp[i]["coarse"]
        if c in fp:
            fp[c] += 1

    per_label = {c: _f1(tp[c], fp[c], fn[c]) for c in _LABEL_CLASSES}
    struct_macro_f1 = round(sum(per_label.values()) / len(per_label), 4)

    # ---- table + furniture signals ----------------------------------------
    table_gold = [
        j for j, g in enumerate(gold_regions) if g["gold_label"] == "Table Row"
    ]
    table_covered = sum(
        1 for j in table_gold if any(warp[i]["label"] == "Table Row" for i in g2ws[j])
    )
    table_region_coverage = (
        round(table_covered / len(table_gold), 4) if table_gold else 1.0
    )
    furniture_as_heading = sum(
        1
        for j, g in enumerate(gold_regions)
        if g["gold_label"] == "Furniture" and pred[j] == "Heading"
    )

    # ---- relationships (representative Warp ann = best-overlap assigned) ----
    rep = {j: (g2ws[j][0] if g2ws[j] else None) for j in range(len(gold_regions))}
    ha_total = ha_ok = pc_total = pc_ok = 0
    for j, g in enumerate(gold_regions):
        if coarse(g["gold_label"]) not in _LABEL_CLASSES or rep[j] is None:
            continue
        wa = warp[rep[j]]
        # head-ancestor agreement (only for body regions)
        if coarse(g["gold_label"]) != "Heading":
            gold_anc = _gold_heading_ancestor(g, region_by_id)
            gold_anc_idx = rid_to_idx.get(gold_anc) if gold_anc is not None else None
            warp_anc = _warp_heading_ancestor(wa["id"], by_id)
            warp_anc_gold = None
            if warp_anc is not None:
                for wi, wann in enumerate(warp):
                    if wann["id"] == warp_anc and w2g.get(wi) is not None:
                        warp_anc_gold = w2g[wi]
                        break
            ha_total += 1
            if gold_anc_idx is None and warp_anc_gold is None:
                ha_ok += 1
            elif gold_anc_idx is not None and warp_anc_gold == gold_anc_idx:
                ha_ok += 1
        # direct parent-class agreement
        gp = g.get("gold_parent_region_id")
        gp_class = (
            coarse(region_by_id[gp]["gold_label"])
            if gp is not None and gp in region_by_id
            else None
        )
        wp = wa["parent_id"]
        wp_class = coarse(by_id[wp]["annotationLabel"]) if wp in by_id else None
        pc_total += 1
        if gp_class == wp_class:
            pc_ok += 1

    # ---- reading order ----------------------------------------------------
    ro_pairs = []
    if have_ro:
        for j, g in enumerate(gold_regions):
            if coarse(g["gold_label"]) in _LABEL_CLASSES and rep[j] is not None:
                if g.get("ro_index") is not None:
                    ro_pairs.append((g["ro_index"], warp[rep[j]]["order"]))

    return {
        "n_warp": len(warp),
        "n_gold_label": label_gold,
        "struct_macro_f1": struct_macro_f1,
        "heading_f1": per_label["Heading"],
        "list_f1": per_label["List Item"],
        "paragraph_f1": per_label["Paragraph"],
        "gold_coverage": round(covered_gold / label_gold, 4) if label_gold else 1.0,
        "spurious_frac": round(len(spurious) / len(warp), 4) if warp else 0.0,
        "body_as_tablerow_frac": (
            round(body_as_tablerow / body_total, 4) if body_total else 0.0
        ),
        "table_region_coverage": table_region_coverage,
        "furniture_as_heading": furniture_as_heading,
        "head_ancestor_agreement": round(ha_ok / ha_total, 4) if ha_total else 1.0,
        "parent_class_agreement": round(pc_ok / pc_total, 4) if pc_total else 1.0,
        "reading_order_agreement": (
            _reading_order_agreement(ro_pairs) if have_ro else None
        ),
    }


def _reading_order_agreement(pairs):
    """Fraction of gold-adjacent matched pairs that keep order in Warp."""
    if len(pairs) < 2:
        return 1.0
    pairs = sorted(pairs)
    concordant = total = 0
    for (g1, w1), (g2, w2) in zip(pairs, pairs[1:]):
        if g1 == g2:
            continue
        total += 1
        if w2 >= w1:
            concordant += 1
    return round(concordant / total, 4) if total else 1.0


# --------------------------------------------------------------------------- #
# regression floor/ceil (shared by the legal100 / hetero100 suites)
# --------------------------------------------------------------------------- #
# floored: a quality metric that must not drop (improvements pass)
FLOOR_KEYS = (
    "struct_macro_f1",
    "heading_f1",
    "list_f1",
    "paragraph_f1",
    "gold_coverage",
    "table_region_coverage",
    "head_ancestor_agreement",
    "parent_class_agreement",
    "reading_order_agreement",
)
# ceiled: a smell that must not grow
CEIL_FRAC_KEYS = ("spurious_frac", "body_as_tablerow_frac")
CEIL_COUNT_KEYS = ("furniture_as_heading",)

_TOL = 0.03  # metrics are deterministic; a small slack absorbs rounding only


def page_regressions(live, base, tol=_TOL):
    """Human-readable regression messages for one page (empty == passing)."""
    out = []
    for k in FLOOR_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is None or bv is None:
            continue
        if lv < bv - tol:
            out.append(f"{k} {lv} < {bv - tol:.4f} (baseline {bv})")
    for k in CEIL_FRAC_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is not None and bv is not None and lv > bv + tol:
            out.append(f"{k} {lv} > {bv + tol:.4f} (baseline {bv})")
    for k in CEIL_COUNT_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is not None and bv is not None and lv > bv:
            out.append(f"{k} {lv} > baseline {bv}")
    return out


# --------------------------------------------------------------------------- #
# Semantic-Unit segmentation scoring (Phase 1)
# --------------------------------------------------------------------------- #
def _iou(a, b):
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _mask_from_seg(order_to_seg, n):
    """0/1 boundary mask over ``n`` reading-order slots (position 0 never 1)."""
    mask = [0] * n
    prev = None
    for i in range(n):
        seg = order_to_seg.get(i)
        if i > 0 and seg != prev:
            mask[i] = 1
        prev = seg
    return mask


def score_units(export, page, gold_units, *, have_ro=False):
    """Segmentation-quality metrics for the Semantic-Unit layer on one page.

    A ``gold_unit`` is ``{unit_id, bbox_frac:[l,t,r,b], text, member_order:[int]}``
    where ``member_order`` are the page reading-order positions of the gold
    unit's member blocks (used to build the Pk/WindowDiff masks).
    """
    units = warp_units(export, page)
    gold_regions = [
        {
            "bbox_frac": g["bbox_frac"],
            "text": g["text"],
            "gold_label": "Unit",
            "region_id": g["unit_id"],
        }
        for g in gold_units
    ]
    warp_as_regions = [{"box": u["box"], "text": u["text"]} for u in units]
    w2g, g2ws = assign(warp_as_regions, gold_regions)

    covered = sum(1 for j in range(len(gold_units)) if g2ws[j])
    unit_coverage = round(covered / len(gold_units), 4) if gold_units else 1.0
    # best-IoU per gold unit
    ious = []
    for j, g in enumerate(gold_units):
        best = 0.0
        for i in g2ws[j]:
            best = max(best, _iou(units[i]["box"], g["bbox_frac"]))
        ious.append(best)
    mean_unit_iou = round(sum(ious) / len(ious), 4) if ious else 1.0
    # over-segmentation: a gold unit split across >= 2 warp units
    fragmentation = sum(1 for j in range(len(gold_units)) if len(g2ws[j]) >= 2)
    fragmentation_frac = (
        round(fragmentation / len(gold_units), 4) if gold_units else 0.0
    )
    # under-segmentation: a warp unit overlapping >= 2 gold units
    merges = 0
    for u in units:
        hit = sum(1 for g in gold_units if _inter(u["box"], g["bbox_frac"]) > 0)
        if hit >= 2:
            merges += 1
    merge_frac = round(merges / len(units), 4) if units else 0.0
    spurious = sum(1 for i in range(len(units)) if w2g.get(i) is None)
    spurious_unit_frac = round(spurious / len(units), 4) if units else 0.0

    # Pk / WindowDiff over the page reading-order slots
    n = max((u["order_end"] for u in units), default=-1)
    n = (
        max(
            n,
            max(
                (max(g["member_order"]) for g in gold_units if g["member_order"]),
                default=-1,
            ),
        )
        + 1
    )
    if n >= 2:
        warp_seg = {}
        for u in units:
            for o in range(u["order_start"], u["order_end"] + 1):
                warp_seg[o] = u["unit_id"]
        gold_seg = {}
        for g in gold_units:
            for o in g["member_order"]:
                gold_seg[o] = g["unit_id"]
        ref = _mask_from_seg(gold_seg, n)
        hyp = _mask_from_seg(warp_seg, n)
        wd = _windowdiff(ref, hyp)
        pkv = _pk(ref, hyp)
    else:
        wd = pkv = 0.0

    return {
        "unit_coverage": unit_coverage,
        "mean_unit_iou": mean_unit_iou,
        "fragmentation_frac": fragmentation_frac,
        "merge_frac": merge_frac,
        "spurious_unit_frac": spurious_unit_frac,
        "windowdiff": round(wd, 4),
        "pk": round(pkv, 4),
        "n_warp_units": len(units),
        "n_gold_units": len(gold_units),
    }


UNIT_FLOOR_KEYS = ("unit_coverage", "mean_unit_iou")
UNIT_CEIL_FRAC_KEYS = (
    "fragmentation_frac",
    "merge_frac",
    "spurious_unit_frac",
    "windowdiff",
    "pk",
)
UNIT_CEIL_COUNT_KEYS = ()


def unit_regressions(live, base, tol=_TOL):
    """Floors UNIT_FLOOR_KEYS, ceils UNIT_CEIL_* (empty == passing)."""
    out = []
    for k in UNIT_FLOOR_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is not None and bv is not None and lv < bv - tol:
            out.append(f"{k} {lv} < {bv - tol:.4f} (baseline {bv})")
    for k in UNIT_CEIL_FRAC_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is not None and bv is not None and lv > bv + tol:
            out.append(f"{k} {lv} > {bv + tol:.4f} (baseline {bv})")
    for k in UNIT_CEIL_COUNT_KEYS:
        lv, bv = live.get(k), base.get(k)
        if lv is not None and bv is not None and lv > bv:
            out.append(f"{k} {lv} > baseline {bv}")
    return out
