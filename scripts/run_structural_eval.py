#!/usr/bin/env python
"""Run a structural-correctness eval: parse pages with Warp, score vs golden.

Drives one batch (``--set hetero`` or ``--set legal``):

* loads the golden (hetero: ``hetero100_golden.json``; legal:
  ``legal100_golden.json`` once adjudicated) and the page manifest,
* parses each distinct PDF **once** with Warp (``parse_to_opencontracts``),
  capped at ``--jobs`` (default 2 — the engine is memory-heavy; >4 OOMs),
  caching each export JSON under ``audit_out/<set>/exports/``,
* scores every manifest page against the golden (``tests.oc_golden_eval``), and
* writes per-page metrics, an aggregate roll-up, an aggregated label-confusion,
  and a ranked **defect list** (the discovery surface for systematic issues).

    python scripts/run_structural_eval.py --set hetero --jobs 2
    python scripts/run_structural_eval.py --set legal  --jobs 2   # needs golden

Outputs (gitignored) under ``audit_out/<set>/``:
  exports/<slug>.json   metrics.json   aggregate.json   defects.json
"""

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests import oc_golden_eval as G  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

SETS = {
    "hetero": {
        "golden": REPO / "tests" / "fixtures" / "hetero100_golden.json",
        "have_ro": True,
    },
    "legal": {
        "golden": REPO / "tests" / "fixtures" / "legal100_golden.json",
        "manifest": REPO / "tests" / "fixtures" / "legal100_manifest.json",
        "have_ro": False,
    },
}

_FLOOR = (
    "struct_macro_f1",
    "heading_f1",
    "list_f1",
    "gold_coverage",
    "head_ancestor_agreement",
)


def _slug(relpath):
    return relpath.replace("/", "__").replace(" ", "_")


def _manifest(set_name):
    """Return list of (relpath, [pages]) and a {(relpath,page): regions} golden."""
    cfg = SETS[set_name]
    golden = json.load(open(cfg["golden"])) if cfg["golden"].exists() else {}
    if set_name == "hetero":
        # golden keys ARE the page fixtures (single page each)
        pages = {rel: [0] for rel in golden}
        regions = {(rel, 0): golden[rel]["regions"] for rel in golden}
    else:
        manifest = json.load(open(cfg["manifest"]))
        pages = defaultdict(list)
        for r in manifest:
            pages[r["relpath"]].append(r["page_index"])
        pages = {k: sorted(set(v)) for k, v in pages.items()}
        regions = {}
        for key, entry in golden.items():
            rel, _, pg = key.rpartition("::p")
            regions[(rel, int(pg))] = entry["regions"]
    return pages, regions, cfg["have_ro"]


def _parse_one(task):
    relpath, out_dir = task
    slug = _slug(relpath)
    cache = Path(out_dir) / "exports" / f"{slug}.json"
    try:
        if cache.exists():
            return relpath, str(cache), None
        with contextlib.redirect_stdout(io.StringIO()):
            export = parse_to_opencontracts(str(REPO / relpath))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(export))
        return relpath, str(cache), None
    except Exception:
        return relpath, None, traceback.format_exc()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_name", choices=sorted(SETS), required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-slow", action="store_true")
    args = ap.parse_args(argv)
    args.jobs = max(1, min(args.jobs, 4))  # OOM guard

    out_dir = Path(args.out or REPO / "audit_out" / args.set_name)
    (out_dir / "exports").mkdir(parents=True, exist_ok=True)

    pages, regions, have_ro = _manifest(args.set_name)
    relpaths = sorted(pages)
    print(
        f"[{args.set_name}] {len(relpaths)} docs, "
        f"{sum(len(v) for v in pages.values())} pages, jobs={args.jobs}"
    )

    # parse (capped parallelism)
    export_path = {}
    errors = []
    tasks = [(rel, str(out_dir)) for rel in relpaths]
    with Pool(processes=args.jobs) as pool:
        for rel, path, err in pool.imap_unordered(_parse_one, tasks):
            if err:
                errors.append({"relpath": rel, "error": err.splitlines()[-1]})
                print(f"  !! parse failed: {rel}", file=sys.stderr)
            else:
                export_path[rel] = path
            done = len(export_path) + len(errors)
            if done % 10 == 0:
                print(f"  parsed {done}/{len(relpaths)}")

    # score
    per_page = {}
    confusion = Counter()
    missed = Counter()
    spurious = Counter()
    defects = []
    for rel in relpaths:
        if rel not in export_path:
            continue
        export = json.loads(Path(export_path[rel]).read_text())
        for pg in pages[rel]:
            gold = regions.get((rel, pg))
            if gold is None:
                continue
            m = G.score_page(export, pg, gold, have_ro=have_ro)
            key = f"{rel}::p{pg}"
            per_page[key] = m
            # aggregate confusion / misses / spurious for the report
            warp = G.warp_annotations(export, pg)
            w2g, g2ws = G.assign(warp, gold)
            for j, g in enumerate(gold):
                gc = G.coarse(g["gold_label"])
                if gc not in G._LABEL_CLASSES and g["gold_label"] != "Table Row":
                    continue
                p = G._dominant_label(warp, g2ws[j])
                if p is not None:
                    confusion[f"{gc}->{p}"] += 1
                else:
                    missed[gc] += 1
            for i, j in w2g.items():
                if j is None:
                    spurious[warp[i]["coarse"]] += 1
            # flag low-scoring pages
            worst = min((m[k] for k in _FLOOR), default=1.0)
            if (
                worst < 0.85
                or m["furniture_as_heading"]
                or m["spurious_frac"] > 0.25
                or m["body_as_tablerow_frac"] > 0.25
            ):
                defects.append(
                    {
                        "page": key,
                        "worst": round(worst, 3),
                        **{k: m[k] for k in _FLOOR},
                        "furniture_as_heading": m["furniture_as_heading"],
                        "spurious_frac": m["spurious_frac"],
                        "body_as_tablerow_frac": m["body_as_tablerow_frac"],
                    }
                )

    # aggregate
    def avg(k):
        vals = [m[k] for m in per_page.values() if m.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg = {
        "set": args.set_name,
        "n_pages_scored": len(per_page),
        "n_parse_errors": len(errors),
        "means": {
            k: avg(k)
            for k in (
                "struct_macro_f1",
                "heading_f1",
                "list_f1",
                "paragraph_f1",
                "gold_coverage",
                "spurious_frac",
                "body_as_tablerow_frac",
                "table_region_coverage",
                "head_ancestor_agreement",
                "parent_class_agreement",
                "reading_order_agreement",
            )
        },
        "furniture_as_heading_total": sum(
            m["furniture_as_heading"] for m in per_page.values()
        ),
        "confusion": dict(confusion.most_common()),
        "missed_gold": dict(missed.most_common()),
        "spurious_by_class": dict(spurious.most_common()),
        "errors": errors,
    }
    defects.sort(key=lambda d: d["worst"])

    (out_dir / "metrics.json").write_text(
        json.dumps(per_page, indent=1, sort_keys=True)
    )
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out_dir / "defects.json").write_text(json.dumps(defects, indent=2))
    print("\n=== aggregate ===")
    print(json.dumps(agg["means"], indent=2))
    print("furniture_as_heading_total:", agg["furniture_as_heading_total"])
    print("confusion (top):", dict(confusion.most_common(8)))
    print("missed_gold:", agg["missed_gold"])
    print("spurious_by_class:", agg["spurious_by_class"])
    print(f"{len(defects)} flagged pages -> {out_dir}/defects.json")
    return agg


if __name__ == "__main__":
    main()
