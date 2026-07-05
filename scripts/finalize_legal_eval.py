#!/usr/bin/env python
"""One-shot: turn the legal vision-adjudication votes into the committed eval.

Runs the legal-100 pipeline end to end once the adjudication Workflow's votes are
available:

  1. assemble the golden  (scripts/assemble_legal_golden)
  2. freeze the baseline   (scripts/build_legal100_baseline, --jobs 2)
  3. score the batch       (scripts/run_structural_eval --set legal)

and prints the legal aggregate + confusion for the report. Pass the votes file
(Workflow return saved to disk, or recovered via scripts/extract_legal_votes).

    python scripts/finalize_legal_eval.py --votes audit_out/legal/votes.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(args):
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, check=True, cwd=str(REPO))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--votes", default=str(REPO / "audit_out" / "legal" / "votes.json"))
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)

    _run([PY, "scripts/assemble_legal_golden.py", "--votes", args.votes])
    _run([PY, "scripts/build_legal100_baseline.py", "--jobs", str(args.jobs)])
    _run(
        [
            PY,
            "scripts/run_structural_eval.py",
            "--set",
            "legal",
            "--jobs",
            str(args.jobs),
        ]
    )

    agg = json.loads((REPO / "audit_out" / "legal" / "aggregate.json").read_text())
    print("\n=== LEGAL aggregate (for report) ===")
    print(json.dumps(agg["means"], indent=2))
    print("furniture_as_heading_total:", agg.get("furniture_as_heading_total"))
    print("confusion (top):", dict(list(agg.get("confusion", {}).items())[:10]))
    return agg


if __name__ == "__main__":
    main()
