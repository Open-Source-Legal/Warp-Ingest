# olmOCR-bench evaluation for Warp-Ingest

Runs the **official** [AI2 olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
benchmark against Warp-Ingest (nlm-ingestor) and the LiteParse baseline, so we
get **leaderboard-comparable** numbers for how Warp converts PDFs to text.

This is a *faithful* recreation: we do **not** reimplement any scoring. We render
each parser's output to Markdown and let AI2's own `olmocr.bench.benchmark`
harness (unit-test-style pass/fail checks for text presence/absence, reading
order, tables, math, headers/footers, and a baseline quality gate) produce the
scores.

**What olmOCR-bench measures vs what Warp is:** olmOCR-bench is an **OCR /
math / reading-order fidelity** benchmark built for OCR & VLM pipelines. Warp is
a rule-based **RAG layout/segmentation** parser. Two of the eight categories are
math and Warp emits no LaTeX, so it scores `0` there *by construction*. The
number is a faithful, out-of-domain baseline — not the dimension Warp is built to
win (that is ParseBench / the OpenContracts structural evals). See RESULTS.md.

## What's here

| File | Role |
|---|---|
| `setup_olmocr_bench.sh` | One-time setup: clone the harness (pinned commit), download the dataset (pinned rev), build an isolated scoring venv (harness PyPI deps + chromium for KaTeX). |
| `generate.py` | Render one Markdown file per PDF for a candidate (`--parser warp` via `benchmarks/parsebench/warp_markdown.py`; `--parser liteparse` via the `lit` CLI as ParseBench invokes it). |
| `analyze_results.py` | Parse the harness stdout into a per-category comparison table vs the published leaderboard. |
| `RESULTS.md` | The committed faithful results (Warp vs LiteParse vs leaderboard). |

Nothing heavy is committed: the dataset, the per-PDF Markdown candidates, the
scoring venv and chromium all live in `$OLMOCR_BENCH_DIR`
(default `$HOME/Code/olmocr-bench`), **outside** this repo.

## Faithful reuse

* **Scorer:** `allenai/olmocr` `olmocr.bench.benchmark`, run verbatim — no edits
  to the harness or the test data. Overall = the harness's own *average of
  per-JSONL pass rates*.
* **We do not `pip install -e` the olmocr package** (that would run the external
  repo's build). The base package imports nothing heavy (`olmocr/__init__.py` is
  a version string), so we install only the scorer's PyPI deps into a venv and
  run the harness with `PYTHONPATH=<olmocr-src>`. The KaTeX math path (chromium
  via playwright) is required even to *load* the two math JSONLs and is verified
  at setup.
* **Warp output** is produced by the same committed renderer used for ParseBench
  (`warp_markdown.py`), with service-default options — no benchmark-specific
  tuning. Deterministic → 1 repeat per PDF.
* **LiteParse** is invoked exactly as the ParseBench provider does
  (`lit parse --format markdown --quiet --no-links`, OCR on by default); raw
  markdown is scored (olmOCR-bench reads markdown tables natively).
* Every PDF (all 1403) is generated and scored — nothing sampled or hand-picked.

## Reproduce

```bash
# 0) prereqs: this repo installed (nlm_ingestor importable), `lit` on PATH for
#    the LiteParse baseline (pip install "liteparse>=2.0"), git-lfs, hf CLI.
bash benchmarks/olmocr_bench/setup_olmocr_bench.sh          # ~15 min, ~0.5 GB

# 1) generate candidates (from this repo's env)
python -m benchmarks.olmocr_bench.generate --parser warp        # ~20 min (OCR)
python -m benchmarks.olmocr_bench.generate --parser liteparse   # ~2 min

# 2) score with the official harness (all candidates at once)
BENCH=${OLMOCR_BENCH_DIR:-$HOME/Code/olmocr-bench}
PYTHONPATH=$BENCH/olmocr-src $BENCH/venv-score/bin/python \
    -m olmocr.bench.benchmark --dir $BENCH/olmOCR-bench/bench_data | tee results_both.txt

# 3) comparison table
python -m benchmarks.olmocr_bench.analyze_results results_both.txt
```

`--candidate <name>` runs a single candidate; `--sample N` scores a random N
tests; see `python -m olmocr.bench.benchmark --help` for the harness options.
