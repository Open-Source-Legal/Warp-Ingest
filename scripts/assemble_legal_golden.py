#!/usr/bin/env python
"""Assemble the legal-100 golden from the vision-adjudication votes.

Consumes the ``adjudicate-legal-golden`` Workflow's return — a list of
``{key, a: {blocks: [...]}, b: {blocks: [...]}}`` (two independent vision
auditors per page) — together with the per-page annotation sidecars written by
``build_legal_vision_tasks.py``, and produces ``tests/fixtures/legal100_golden.json``.

Adjudication is **conservative**: a block's gold label/parent is the auditors'
*agreed* answer; on disagreement we keep Warp's own label/parent (and flag
``needs_review``). So the eval only penalizes Warp where BOTH auditors agree it
is wrong — a high-precision floor, never a coin-flip.

    python scripts/assemble_legal_golden.py --votes audit_out/legal/votes.json

The votes file is the Workflow result saved to disk (the Workflow returns the
votes array; save it there, then run this).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VISION = REPO / "audit_out" / "legal" / "vision"
OUT = REPO / "tests" / "fixtures" / "legal100_golden.json"

_LABELS = {
    "Title",
    "Section Header",
    "Paragraph",
    "List Item",
    "Table Row",
    "Furniture",
}


def _slug(key):
    return key.replace("::p", "__p").replace("/", "__").replace(" ", "_")


def _votes_by_n(blocks):
    out = {}
    for b in blocks or []:
        try:
            out[int(b["n"])] = b
        except (KeyError, TypeError, ValueError):
            continue
    return out


def assemble(votes):
    golden = {}
    stats = Counter()
    for page in votes:
        key = page["key"]
        sidecar = VISION / f"{_slug(key)}.task.json"
        if not sidecar.exists():
            stats["missing_sidecar"] += 1
            continue
        task = json.loads(sidecar.read_text())
        anns = {a["n"]: a for a in task["annotations"]}
        a_by = _votes_by_n((page.get("a") or {}).get("blocks"))
        b_by = _votes_by_n((page.get("b") or {}).get("blocks"))
        regions = []
        for n in sorted(anns):
            ann = anns[n]
            warp_label = ann["warp_label"]
            warp_label = "Section Header" if warp_label == "Title" else warp_label
            va, vb = a_by.get(n), b_by.get(n)
            la = va["label"] if va and va.get("label") in _LABELS else None
            lb = vb["label"] if vb and vb.get("label") in _LABELS else None
            if la is not None and la == lb:
                gold_label, agreed = la, True
            else:
                gold_label, agreed = warp_label, False
            pa = va.get("parent_n") if va else None
            pb = vb.get("parent_n") if vb else None
            if pa is not None and pa == pb:
                gold_parent_n, p_agreed = pa, True
            else:
                gold_parent_n, p_agreed = ann.get("warp_parent_n") or 0, False
            stats["blocks"] += 1
            stats["label_agreed" if agreed else "label_fallback"] += 1
            if agreed and gold_label != warp_label:
                stats["correction"] += 1
                stats[f"corr:{warp_label}->{gold_label}"] += 1
            regions.append(
                {
                    "region_id": ann["id"],
                    "n": n,
                    "bbox_frac": ann["bbox_frac"],
                    "text": ann["text"],
                    "gold_label": gold_label,
                    "_gold_parent_n": int(gold_parent_n) if gold_parent_n else 0,
                    "ro_index": n,
                    "needs_review": not (agreed and p_agreed),
                }
            )
        # resolve parent_n -> region_id
        n_to_id = {r["n"]: r["region_id"] for r in regions}
        for r in regions:
            pn = r.pop("_gold_parent_n")
            r.pop("n", None)
            r["gold_parent_region_id"] = n_to_id.get(pn) if pn else None
        golden[key] = {"regions": regions}
    return golden, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--votes", default=str(REPO / "audit_out" / "legal" / "votes.json"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    votes = json.loads(Path(args.votes).read_text())
    golden, stats = assemble(votes)
    Path(args.out).write_text(json.dumps(golden, indent=1, sort_keys=True))
    n_regions = sum(len(v["regions"]) for v in golden.values())
    print(f"wrote {args.out}: {len(golden)} pages, {n_regions} gold regions")
    print("adjudication stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    return golden


if __name__ == "__main__":
    main()
