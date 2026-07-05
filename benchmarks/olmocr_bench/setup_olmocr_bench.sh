#!/usr/bin/env bash
# Set up the official AI2 olmOCR-bench harness + dataset + an isolated scoring
# venv so Warp-Ingest (and baselines like LiteParse) can be evaluated against it.
# Run once.
#
#   bash benchmarks/olmocr_bench/setup_olmocr_bench.sh
#
# Everything lands in $OLMOCR_BENCH_DIR (default $HOME/Code/olmocr-bench), which
# is deliberately OUTSIDE this repo — the dataset, generated markdown and venv
# are large/external and are never committed.
#
# Note: we do NOT `pip install -e` the olmocr package (that runs the external
# repo's build). Instead we install only the harness's PyPI scoring deps into a
# venv and run the harness via PYTHONPATH=<olmocr-src>. This is the scorer's
# minimal footprint (see README).
set -euo pipefail

# Pinned revisions used for the committed RESULTS.md.
OLMOCR_SHA="${OLMOCR_SHA:-f7cfe4c22098b154c76b6ec950d1c0a464eecf8d}"   # olmocr v0.4.27
DATASET_REV="${DATASET_REV:-54a96a6fb6a2bd3b297e59869491db4d3625b711}"
BENCH_DIR="${OLMOCR_BENCH_DIR:-$HOME/Code/olmocr-bench}"

mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR"

echo ">> olmOCR-bench harness: $BENCH_DIR/olmocr-src @ $OLMOCR_SHA"
if [ ! -d olmocr-src/.git ]; then
  git clone https://github.com/allenai/olmocr.git olmocr-src
fi
git -C olmocr-src fetch --depth 1 origin "$OLMOCR_SHA"
git -C olmocr-src checkout -q "$OLMOCR_SHA"

echo ">> scoring venv: $BENCH_DIR/venv-score (harness PyPI deps only — no GPU, no olmocr build)"
if [ ! -d venv-score ]; then
  python -m venv venv-score
fi
./venv-score/bin/pip install -q --upgrade pip
# Minimal deps the scorer imports: text tests, table parsing, bootstrap CI,
# KaTeX math rendering (playwright+chromium), pdf/image helpers.
./venv-score/bin/pip install -q \
  fuzzysearch rapidfuzz lxml beautifulsoup4 numpy playwright tqdm pypdf pypdfium2 Pillow
./venv-score/bin/python -m playwright install chromium

echo ">> dataset: $BENCH_DIR/olmOCR-bench @ $DATASET_REV (1403 single-page PDFs, 7019 tests)"
hf download allenai/olmOCR-bench --repo-type dataset --revision "$DATASET_REV" \
  --local-dir ./olmOCR-bench

echo ">> verifying harness + math path import"
PYTHONPATH=./olmocr-src ./venv-score/bin/python - <<'PY'
from olmocr.bench import benchmark  # noqa
from olmocr.bench.katex.render import render_equation
assert render_equation("x^2").error is None, "chromium/KaTeX not working"
print("olmocr.bench OK; KaTeX math rendering OK")
PY

cat <<MSG

Setup complete. Generate candidates + score:

  # generate one markdown per PDF (from THIS repo's env, which has nlm_ingestor + lit)
  python -m benchmarks.olmocr_bench.generate --parser warp
  python -m benchmarks.olmocr_bench.generate --parser liteparse

  # score with the OFFICIAL harness (all candidate subfolders at once)
  PYTHONPATH=$BENCH_DIR/olmocr-src $BENCH_DIR/venv-score/bin/python \\
      -m olmocr.bench.benchmark --dir $BENCH_DIR/olmOCR-bench/bench_data | tee results_both.txt

  # pretty comparison table
  python -m benchmarks.olmocr_bench.analyze_results results_both.txt
MSG
