"""Shared helpers for the S-1 cross-engine regression suite.

The S-1 regression compares the pure-Python engine's output against *gold targets*
captured from the original (Java/Tika) ``nlm-ingestor`` engine. Because the two
front-ends segment words differently (pdfplumber vs the nlm Tika fork), a
byte-exact XHTML match is impossible by construction; the suite therefore checks
*content* (token recall/precision over the whole document) and *structure*
(page count, block-type distribution, table count, ordered block-text similarity).

A block is reduced to ``{"tag", "level", "text"}`` where ``text`` is the visible
text of the block (sentences joined, or table cell values joined). All metrics are
computed on alphanumeric tokens, so they are robust to punctuation- and
whitespace-spacing artifacts between the two backends.
"""

import difflib
import json
import os
import re

FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
S1_PDF_DIR = os.path.join(FIX_DIR, "s1")
S1_TARGET_DIR = os.path.join(FIX_DIR, "s1_targets")
S1_BASELINE = os.path.join(FIX_DIR, "s1_baseline.json")
S1_MANIFEST = os.path.join(FIX_DIR, "s1_manifest.json")

_TOK = re.compile(r"[a-z0-9]+")
_WS = re.compile(r"\s+")


def tokens(s):
    """Alphanumeric, lowercased tokens — the unit all metrics are computed on."""
    return _TOK.findall(s.lower())


def _norm(s):
    return _WS.sub(" ", s or "").strip()


def raw_block_text(b):
    """Visible text of a raw API block (header/para/list_item or table)."""
    if b.get("tag") == "table":
        parts = []
        for row in b.get("table_rows") or []:
            cells = [str(c.get("cell_value") or "") for c in (row.get("cells") or [])]
            parts.append("\t".join(cells))
        return "\n".join(parts)
    sents = b.get("sentences")
    if isinstance(sents, list):
        return " ".join(str(x) for x in sents)
    return str(sents or "")


def result_to_blocks(result):
    """Reduce an API ``result={'blocks':[...], 'styles':[...]}`` to compact blocks."""
    blocks = (result or {}).get("blocks") or []
    return [
        {"tag": b.get("tag"), "level": b.get("level"), "text": _norm(raw_block_text(b))}
        for b in blocks
    ]


def doc_tokens(blocks):
    out = []
    for b in blocks:
        out.extend(tokens(b["text"]))
    return out


def tag_counts(blocks):
    counts = {}
    for b in blocks:
        counts[b["tag"]] = counts.get(b["tag"], 0) + 1
    return counts


def compare(gold_blocks, new_blocks):
    """Compute content + structural similarity metrics for new vs gold blocks."""
    from collections import Counter

    gt, nt = doc_tokens(gold_blocks), doc_tokens(new_blocks)
    gc, nc = Counter(gt), Counter(nt)
    inter = sum((gc & nc).values())
    g_total = sum(gc.values()) or 1
    n_total = sum(nc.values()) or 1
    g_seq = [" ".join(tokens(b["text"])) for b in gold_blocks]
    n_seq = [" ".join(tokens(b["text"])) for b in new_blocks]
    seq_ratio = difflib.SequenceMatcher(None, g_seq, n_seq, autojunk=False).ratio()
    g_tags, n_tags = tag_counts(gold_blocks), tag_counts(new_blocks)
    return {
        "token_recall": inter / g_total,
        "token_precision": inter / n_total,
        "seq_ratio": seq_ratio,
        "gold_blocks": len(gold_blocks),
        "new_blocks": len(new_blocks),
        "gold_tables": g_tags.get("table", 0),
        "new_tables": n_tags.get("table", 0),
        "gold_tokens": g_total,
        "new_tokens": n_total,
    }


def load_target(name):
    with open(os.path.join(S1_TARGET_DIR, name + ".json")) as f:
        return json.load(f)


def load_baseline():
    with open(S1_BASELINE) as f:
        return json.load(f)


def target_names():
    return sorted(
        f[: -len(".json")] for f in os.listdir(S1_TARGET_DIR) if f.endswith(".json")
    )
