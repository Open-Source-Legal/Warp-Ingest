"""Metrics + baseline comparison for the OpenContractDocExport regression suite.

Mirrors ``tests/s1_compat.py``: the suite captures a small set of per-document
metrics into a committed baseline and asserts the live exporter never drops
below it (improvements pass, regressions fail).
"""

from collections import Counter

# Canonical doc set for the export regression suite: (filename, is_slow).
# USC = text legal doc; Eton = longer agreement w/ tables & run-in headers;
# needs_ocr = scanned doc exercising the OCR-through-XHTML token path (slow);
# with_images = synthetic single-pager exercising the image-token layer (issue #1).
# (sample.pdf is byte-identical to the USC fixture, so it is intentionally omitted.)
FIXTURE_DOCS = [
    ("USC Title 1 - CHAPTER 1.pdf", False),
    (
        "EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_"
        "EX-10.1_Development_Agreement_ZrZJLLv.pdf",
        False,
    ),
    ("needs_ocr.pdf", True),
    ("with_images.pdf", False),
]

# Per-doc parse options: exporter flags a fixture exercises (default: none).
FIXTURE_PARSE_OPTIONS = {
    "with_images.pdf": {"include_images": True},
}

# tolerances: live must stay within these of the committed baseline
_RECALL_DROP = 0.01  # content recall may dip at most 1pt
_FRACTION_DROP = 0.02  # anchored / coverage fractions may dip at most 2pt
_COUNT_FLOOR = 0.98  # token / annotation counts must be >= 98% of baseline


def export_metrics(export: dict) -> dict:
    """Compute the regression metrics for one OpenContractDocExport."""
    pawls = export["pawls_file_content"]
    token_count = sum(len(p["tokens"]) for p in pawls)

    have = Counter(t["text"] for p in pawls for t in p["tokens"])
    want = Counter(export["content"].split())
    recall = (
        sum(min(c, have.get(w, 0)) for w, c in want.items()) / sum(want.values())
        if want
        else 1.0
    )

    anns = export["labelled_text"]
    anchored = sum(
        1
        for a in anns
        if any(pg["tokensJsons"] for pg in a["annotation_json"].values())
    )
    claimed = set()
    for a in anns:
        for pg in a["annotation_json"].values():
            for r in pg["tokensJsons"]:
                claimed.add((r["pageIndex"], r["tokenIndex"]))

    image_token_count = sum(1 for p in pawls for t in p["tokens"] if t.get("is_image"))
    # counts IMAGE-*modality* annotations (standalone "Image" + attach hosts), so
    # the metric is invariant under attach-vs-standalone flips of the same image
    image_annotation_count = sum(
        1 for a in anns if "IMAGE" in (a.get("content_modalities") or ())
    )

    return {
        "page_count": export["page_count"],
        "token_count": token_count,
        "annotation_count": len(anns),
        "content_recall": round(recall, 4),
        "anchored_fraction": round(anchored / len(anns), 4) if anns else 1.0,
        "token_coverage": round(len(claimed) / token_count, 4) if token_count else 1.0,
        "root_count": sum(1 for a in anns if a["parent_id"] is None),
        "image_token_count": image_token_count,
        "image_annotation_count": image_annotation_count,
    }


def regressions(live: dict, base: dict) -> list[str]:
    """Return human-readable regression messages (empty list == passing)."""
    out = []

    if live["page_count"] != base["page_count"]:
        out.append(f"page_count {live['page_count']} != baseline {base['page_count']}")

    for key in ("token_count", "annotation_count"):
        floor = base[key] * _COUNT_FLOOR
        if live[key] < floor:
            out.append(f"{key} {live[key]} < floor {floor:.1f} (baseline {base[key]})")

    if live["content_recall"] < base["content_recall"] - _RECALL_DROP:
        out.append(
            f"content_recall {live['content_recall']} < "
            f"{base['content_recall'] - _RECALL_DROP:.4f}"
        )

    for key in ("anchored_fraction", "token_coverage"):
        if live[key] < base[key] - _FRACTION_DROP:
            out.append(
                f"{key} {live[key]} < {base[key] - _FRACTION_DROP:.4f} "
                f"(baseline {base[key]})"
            )

    for key in ("image_token_count", "image_annotation_count"):
        if live[key] < base[key]:
            out.append(f"{key} {live[key]} < baseline {base[key]}")

    if live["root_count"] < 1:
        out.append("root_count < 1 (no hierarchy roots)")

    return out
