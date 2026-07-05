"""Manifest + metric glue for the legal-100 structural-correctness suite.

The legal batch (70 EDGAR S-1 pages + 29 FortWorth municipal-contract pages,
``tests/fixtures/legal100_manifest.json``) has no external structural truth, so
its golden (``tests/fixtures/legal100_golden.json``) is built by **vision
adjudication of Warp's own annotations** (two independent auditors per page; a
block's gold label/parent is their agreed answer, else Warp's own — see
``scripts/adjudicate_legal_golden`` + ``scripts/assemble_legal_golden.py``). The
golden therefore measures Warp's **label and relationship accuracy** (the regions
are Warp's own boxes, so coverage/segmentation are not the focus here — the
heterogeneous suite covers those against an independent oracle).

Because the gold's reading order is Warp's own document order, reading-order
agreement is *not* scored (``HAVE_RO = False``) — it would be circular.
"""

import json
import os

from tests import oc_golden_eval as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "tests", "fixtures", "legal100_golden.json")
BASELINE = os.path.join(REPO, "tests", "fixtures", "legal100_baseline.json")
MANIFEST = os.path.join(REPO, "tests", "fixtures", "legal100_manifest.json")
HAVE_RO = False


def load_golden():
    with open(GOLDEN) as fh:
        return json.load(fh)


def load_baseline():
    with open(BASELINE) as fh:
        return json.load(fh)


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def page_key(relpath, page):
    return f"{relpath}::p{page}"


def manifest_pairs():
    """``[(relpath, page, is_slow, doc_class)]`` rows from the manifest."""
    return [
        (r["relpath"], r["page_index"], r["is_slow"], r["doc_class"])
        for r in load_manifest()
    ]


def page_metrics(export, page, regions):
    """Golden-agreement metrics for one legal (export, page)."""
    return G.score_page(export, page, regions, have_ro=HAVE_RO)
