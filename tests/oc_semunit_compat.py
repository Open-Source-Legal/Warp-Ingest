"""Fixture discovery + scoring glue for the Semantic-Unit regression suite.

Clone of ``oc_legal100_compat`` adapted to unit-level scoring. ``HAVE_RO`` is
False: the numbering-derived golden shares Warp's reading order, so a
reading-order agreement metric would be circular. See
``docs/superpowers/specs/2026-07-01-semantic-unit-grouping-design.md``.
"""

import json
import os

import tests.oc_golden_eval as G

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(_HERE, "fixtures")
GOLDEN = os.path.join(FIXTURES, "semunit_golden.json")
BASELINE = os.path.join(FIXTURES, "semunit_baseline.json")
MANIFEST = os.path.join(FIXTURES, "semunit_manifest.json")
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
    return [
        (m["relpath"], m["page_index"], bool(m.get("is_slow"))) for m in load_manifest()
    ]


def page_metrics(export, page, gold_units):
    return G.score_units(export, page, gold_units, have_ro=HAVE_RO)
