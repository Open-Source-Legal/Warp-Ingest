"""Score the processor's clause grouping against the agent-built golden.

Honest metric = PAIRWISE-MEMBERSHIP F1 over each page's fine blocks: for every
pair of blocks, do golden and processor agree on whether they belong to the same
leaf unit? This penalizes over-splitting (fragmentation -> recall down) AND
over-merging (precision down). Also reports unit-recovery F1 (Jaccard>=0.5
matched leaf units) and exact-unit match. Reads:
  audit_out/semunit_bench100/golden.json     (from the golden workflow)
  audit_out/semunit_bench100/<slug>/proc_units.json , blocks.json
"""

import json
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "audit_out" / "semunit_bench100"


def leaf_map_proc(proc_units, onpage):
    """block id -> processor leaf-unit id (singleton if in no unit)."""
    m = {}
    for uid, d in proc_units.items():
        for b in d["members"]:
            if b in onpage:
                m[b] = uid
    for b in onpage:
        m.setdefault(b, f"_solo_{b}")
    return m


def leaf_map_gold(gpage, onpage):
    m = {}
    for u in gpage.get("units", []):
        for b in u["member_block_ids"]:
            if b in onpage:
                m[b] = "g_" + u["gold_id"]
    # furniture + unassigned -> singletons
    for b in onpage:
        m.setdefault(b, f"_solo_{b}")
    return m


def root_map_gold(gpage, onpage):
    """block -> its TOP-LEVEL gold unit (roll nesting up), for a granularity-robust
    clause-boundary comparison."""
    units = {u["gold_id"]: u for u in gpage.get("units", [])}

    def root(gid):
        seen = set()
        while (
            gid in units
            and units[gid].get("parent_gold_id") in units
            and gid not in seen
        ):
            seen.add(gid)
            gid = units[gid]["parent_gold_id"]
        return gid

    m = {}
    for u in gpage.get("units", []):
        r = "g_" + root(u["gold_id"])
        for b in u["member_block_ids"]:
            if b in onpage:
                m[b] = r
    for b in onpage:
        m.setdefault(b, f"_solo_{b}")
    return m


def root_map_proc(proc_units, gparent, onpage):
    def root(uid):
        seen = set()
        while gparent.get(uid) and uid not in seen:
            seen.add(uid)
            uid = gparent[uid]
        return uid

    m = {}
    for uid, d in proc_units.items():
        r = root(uid)
        for b in d["members"]:
            if b in onpage:
                m[b] = r
    for b in onpage:
        m.setdefault(b, f"_solo_{b}")
    return m


def pairwise(gmap, pmap, blocks):
    tp = fp = fn = 0
    for a, b in combinations(blocks, 2):
        gs = gmap[a] == gmap[b]
        ps = pmap[a] == pmap[b]
        if ps and gs:
            tp += 1
        elif ps and not gs:
            fp += 1
        elif gs and not ps:
            fn += 1
    return tp, fp, fn


def clusters(m, blocks):
    c = {}
    for b in blocks:
        c.setdefault(m[b], set()).add(b)
    return list(c.values())


def recovery(gmap, pmap, blocks):
    """gold leaf units recovered by a proc unit with member-Jaccard>=0.5; exact too."""
    gcl = clusters(gmap, blocks)
    pcl = clusters(pmap, blocks)
    # ignore singletons that are furniture-like on both sides (size 1)
    gcl = [c for c in gcl if len(c) >= 1]
    rec = exact = 0
    for g in gcl:
        best = 0.0
        ex = False
        for p in pcl:
            inter = len(g & p)
            union = len(g | p)
            j = inter / union if union else 0
            best = max(best, j)
            if g == p:
                ex = True
        if best >= 0.5:
            rec += 1
        if ex:
            exact += 1
    # proc precision: proc units matching some gold unit
    pmatch = 0
    for p in pcl:
        if any(len(p & g) / len(p | g) >= 0.5 for g in gcl):
            pmatch += 1
    return len(gcl), len(pcl), rec, exact, pmatch


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def gold_subtrees(gpage, onpage):
    units = {u["gold_id"]: u for u in gpage.get("units", [])}
    kids = {}
    for u in units.values():
        kids.setdefault(u.get("parent_gold_id"), []).append(u["gold_id"])

    def sub(gid, seen):
        if gid in seen:
            return set()
        seen.add(gid)
        s = {b for b in units[gid]["member_block_ids"] if b in onpage}
        for c in kids.get(gid, []):
            s |= sub(c, seen)
        return s

    return [sub(g, set()) for g in units if units[g].get("parent_gold_id") not in units]


def proc_subtrees(proc_units, gparent, onpage):
    kids = {}
    for uid in proc_units:
        kids.setdefault(gparent.get(uid), []).append(uid)

    def sub(uid, seen):
        if uid in seen or uid not in proc_units:
            return set()
        seen.add(uid)
        s = {b for b in proc_units[uid]["members"] if b in onpage}
        for c in kids.get(uid, []):
            s |= sub(c, seen)
        return s

    roots = [u for u in proc_units if gparent.get(u) not in proc_units]
    return [sub(r, set()) for r in roots]


def subtree_recovery(gsubs, psubs):
    """gold clause-spans (subtrees) recovered by a proc span at Jaccard>=0.5, and
    vice versa (precision). Nesting/granularity-robust."""

    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    gsubs = [s for s in gsubs if s]
    psubs = [s for s in psubs if s]
    rec = sum(1 for g in gsubs if any(jac(g, p) >= 0.5 for p in psubs))
    pm = sum(1 for p in psubs if any(jac(p, g) >= 0.5 for g in gsubs))
    return len(gsubs), len(psubs), rec, pm


def main():
    golden = json.loads(
        (Path(__file__).resolve().parent / "golden_204_clause.json").read_text()
    )
    gold_by = {(g["doc"], g["page"]): g for g in golden}
    manifest = json.loads((BASE / "manifest.json").read_text())

    per_doc = {}
    agg = {
        k: 0
        for k in (
            "ltp",
            "lfp",
            "lfn",
            "rtp",
            "rfp",
            "rfn",
            "gu",
            "pu",
            "rec",
            "exact",
            "pmatch",
            "gsn",
            "psn",
            "srec",
            "spm",
        )
    }
    for m in manifest:
        slug = m["slug"]
        proc_all = json.loads((BASE / slug / "proc_units.json").read_text())
        blocks_all = json.loads((BASE / slug / "blocks.json").read_text())
        gparent = {
            uid: dd.get("parent") for pg in proc_all.values() for uid, dd in pg.items()
        }
        d = {k: 0 for k in agg}
        d["pages"] = 0
        for p in m["pages"]:
            key = (slug, p)
            if key not in gold_by:
                continue
            onpage = [b["id"] for b in blocks_all.get(str(p), [])]
            if len(onpage) < 2:
                continue
            pu = proc_all.get(str(p), {})
            # leaf-level (fine; granularity-confounded)
            ltp, lfp, lfn = pairwise(
                leaf_map_gold(gold_by[key], onpage), leaf_map_proc(pu, onpage), onpage
            )
            # root-level (roll nesting up; granularity-robust clause boundaries)
            rtp, rfp, rfn = pairwise(
                root_map_gold(gold_by[key], onpage),
                root_map_proc(pu, gparent, onpage),
                onpage,
            )
            gu, pun, rec, exact, pmatch = recovery(
                leaf_map_gold(gold_by[key], onpage), leaf_map_proc(pu, onpage), onpage
            )
            gsn, psn, srec, spm = subtree_recovery(
                gold_subtrees(gold_by[key], onpage), proc_subtrees(pu, gparent, onpage)
            )
            for k, v in (
                ("ltp", ltp),
                ("lfp", lfp),
                ("lfn", lfn),
                ("rtp", rtp),
                ("rfp", rfp),
                ("rfn", rfn),
                ("gu", gu),
                ("pu", pun),
                ("rec", rec),
                ("exact", exact),
                ("pmatch", pmatch),
                ("gsn", gsn),
                ("psn", psn),
                ("srec", srec),
                ("spm", spm),
            ):
                d[k] += v
                agg[k] += v
            d["pages"] += 1
        per_doc[slug] = d

    def show(name, d):
        lf1, lp, lr = f1(d["ltp"], d["lfp"], d["lfn"])
        srr = d["srec"] / d["gsn"] if d["gsn"] else 1.0
        srp = d["spm"] / d["psn"] if d["psn"] else 1.0
        sf1 = (2 * srp * srr / (srp + srr)) if (srp + srr) else 0.0
        print(
            f"{name:<38}{d.get('pages',''):>3}  SUBTREE-F1={sf1:.3f}(P{srp:.2f}/R{srr:.2f})"
            f"  leafF1={lf1:.3f}(P{lp:.2f}/R{lr:.2f})  g/p_u={d['gu']}/{d['pu']}"
        )

    print(f"{'doc':<38}{'pg':>3}  root-level (granularity-robust) | leaf | recovery")
    print("-" * 112)
    for slug in per_doc:
        show(slug, per_doc[slug])
    print("-" * 112)
    show("OVERALL (micro)", agg)


if __name__ == "__main__":
    main()
