#!/usr/bin/env python
"""Reconstruct legal-golden adjudication votes from a Workflow's agent transcripts.

Fallback for when the ``adjudicate-legal-golden`` Workflow's return is too large
to surface cleanly: scan the per-agent ``agent-*.jsonl`` transcripts in a
Workflow run directory, recover each agent's page key (from the overlay PNG path
in its prompt) and its ``{blocks:[...]}`` StructuredOutput, and pair the two
auditors per page into the ``[{key, a, b}]`` shape that
``assemble_legal_golden.py`` consumes.

    python scripts/extract_legal_votes.py \
        --dir ~/.claude/projects/.../subagents/workflows/wf_xxxx \
        --out audit_out/legal/votes.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _walk_blocks(obj):
    if isinstance(obj, dict):
        b = obj.get("blocks")
        if isinstance(b, list) and b and isinstance(b[0], dict) and "n" in b[0]:
            return b
        for v in obj.values():
            r = _walk_blocks(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk_blocks(v)
            if r:
                return r
    return None


def _slug_of(key):
    return key.replace("::p", "__p").replace("/", "__").replace(" ", "_")


def _scan_agent(path, slug_to_key):
    """Return (key, blocks) recovered from one agent transcript, or (None, None)."""
    key = None
    blocks = None
    for line in path.open():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        blob = json.dumps(rec)
        if key is None:
            # the prompt names the overlay PNG <slug>.png; match the exact slug
            m = re.search(r"/vision/([^\s\"]+?)\.png", blob)
            if m and m.group(1) in slug_to_key:
                key = slug_to_key[m.group(1)]
        b = _walk_blocks(rec)
        if b:
            blocks = b
    return key, blocks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="Workflow run transcript dir")
    ap.add_argument("--out", default="audit_out/legal/votes.json")
    ap.add_argument(
        "--index",
        default=str(
            Path(__file__).resolve().parent.parent
            / "audit_out"
            / "legal"
            / "vision_index.json"
        ),
    )
    args = ap.parse_args(argv)

    keys = [e["key"] for e in json.loads(Path(args.index).read_text())]
    slug_to_key = {_slug_of(k): k for k in keys}

    by_key = defaultdict(list)
    for f in sorted(Path(args.dir).glob("agent-*.jsonl")):
        key, blocks = _scan_agent(f, slug_to_key)
        if key and blocks:
            by_key[key].append(blocks)

    votes = []
    for key, lists in by_key.items():
        a = {"blocks": lists[0]} if len(lists) >= 1 else None
        b = {"blocks": lists[1]} if len(lists) >= 2 else (a)
        votes.append({"key": key, "a": a, "b": b})

    Path(args.out).write_text(json.dumps(votes, indent=1))
    print(f"recovered votes for {len(votes)} pages -> {args.out}")
    print("pages with 2 auditors:", sum(1 for k in by_key if len(by_key[k]) >= 2))
    print("pages with 1 auditor:", sum(1 for k in by_key if len(by_key[k]) == 1))
    return votes


if __name__ == "__main__":
    main()
