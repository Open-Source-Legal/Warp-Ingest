#!/usr/bin/env bash
# Set up the official LlamaIndex ParseBench framework so Warp-Ingest can be
# evaluated against it. Installs the framework (pinned commit) + the local
# baseline parsers it is compared against. Run once.
#
#   bash benchmarks/parsebench/setup_parsebench.sh
#
# Idempotent: re-running upgrades to the pinned revision. Requires git, pip, and
# an environment where `warp_ingest` already imports (this repo, installed).
set -euo pipefail

# Pinned ParseBench revision (run-llama/ParseBench main as of integration).
PARSEBENCH_SHA="b74caa14474923860014fdd732987278b63bd974"
CLONE_DIR="${PARSEBENCH_DIR:-$HOME/Code/ParseBench}"

echo ">> ParseBench framework: $CLONE_DIR @ $PARSEBENCH_SHA"
if [ ! -d "$CLONE_DIR/.git" ]; then
  git clone https://github.com/run-llama/ParseBench.git "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --depth 1 origin "$PARSEBENCH_SHA"
git -C "$CLONE_DIR" checkout -q "$PARSEBENCH_SHA"

echo ">> Installing ParseBench (base deps; the deterministic scorers — no cloud SDKs)"
pip install -e "$CLONE_DIR"

echo ">> Installing local baseline parsers compared against Warp-Ingest"
pip install "liteparse>=2.0" "markitdown[pdf]" "pymupdf>=1.24" "pypdf>=6.0"

echo ">> Verifying"
python - <<'PY'
import shutil
import parse_bench
from parse_bench.inference.providers.registry import _PROVIDER_REGISTRY  # noqa
import parse_bench.inference.providers.parse  # noqa: trigger lazy provider import
print("parse_bench OK:", parse_bench.__file__)
print("lit on PATH:", shutil.which("lit"))
import warp_ingest  # noqa
print("warp_ingest OK")
PY

cat <<'MSG'

Setup complete. Run the benchmark:

  # quick directional run (3 files/category, same-harness comparison)
  python -m benchmarks.parsebench.run --test

  # full, leaderboard-comparable run (slower; downloads the full dataset)
  python -m benchmarks.parsebench.run --full
MSG
