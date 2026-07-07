"""Run the official ParseBench benchmark against Warp-Ingest + local baselines.

This drives the *real* LlamaIndex ParseBench framework
(https://github.com/run-llama/ParseBench) in-process: it registers the
``warp_ingest`` provider/pipeline (defined in this repo), points the built-in
``liteparse`` provider at the PATH ``lit`` binary, then runs ParseBench's own
inference → deterministic rule-based evaluation → report pipeline for each
parser.  Nothing about ParseBench's scoring is reimplemented here, so the
numbers are directly comparable to the published leaderboard.

Prereqforms (one-time): ``benchmarks/parsebench/setup_parsebench.sh`` installs
the framework + the local parsers it compares against.

Usage:
    python -m benchmarks.parsebench.run --test
    python -m benchmarks.parsebench.run --test --pipelines warp_ingest,liteparse_markdown
    python -m benchmarks.parsebench.run --full --group table
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

# The warp provider/pipeline/layout-adapter registrations below live in *this*
# process, and ParseBench's evaluation ProcessPoolExecutor workers must inherit
# them or every layout example fails with "no provider adapter matched"
# (Visual Grounding silently collapses to ~10 while the run still exits 0).
# Python 3.14 changed the Linux default start method from fork to forkserver,
# whose fresh worker processes lose the registrations — pin fork explicitly.
if sys.platform.startswith("linux"):
    multiprocessing.set_start_method("fork", force=True)

# Default comparison set: faithful Warp-Ingest + the four local-library
# baselines that also appear on the official leaderboard.
DEFAULT_PIPELINES = [
    "warp_ingest_faithful",
    "liteparse_markdown",
    "markitdown",
    "pymupdf_text",
    "pypdf_baseline",
]

# Per-category headline metric (matches ParseBench's leaderboard defaults), as
# stored in each category's _evaluation_report.json aggregate_metrics, prefixed
# with "avg_". Values are 0..1; the leaderboard reports them ×100.
HEADLINE_METRIC = {
    "table": "avg_grits_trm_composite",
    "chart": "avg_rule_pass_rate",
    "text_content": "avg_content_faithfulness",
    "text_formatting": "avg_semantic_formatting",
    "layout": "avg_layout_element_rule_pass_rate",
}
# Display order / names mirroring the official leaderboard columns.
DIMENSIONS = [
    ("table", "Tables"),
    ("chart", "Charts"),
    ("text_content", "Content Faith."),
    ("text_formatting", "Sem. Format."),
    ("layout", "Visual Ground."),
]

# Published leaderboard baselines (full dataset) for context — source:
# run-llama/ParseBench/leaderboard.csv. Overall, Tables, Charts, ContentFaith,
# SemFormat, VisualGround.
LEADERBOARD_BASELINES = {
    "LiteParse (no OCR)": [32.8, 40.3, 3.4, 68.6, 44.6, 10.7],
    "PyMuPDF4LLM": [30.88, 36.68, 1.58, 60.85, 44.63, 10.68],
    "MarkItDown": [18.63, 15.77, 2.02, 64.54, 0.91, 9.90],
    "PyMuPDF (Text)": [16.02, 0.00, 0.00, 68.28, 0.95, 10.86],
    "pypdf": [14.87, 0.00, 0.00, 62.50, 0.91, 10.92],
    "Docling-models (VLM)": [50.65, 66.41, 52.76, 66.93, 1.03, 66.11],
}


def _repo_root() -> Path:
    # benchmarks/parsebench/run.py -> repo root is two parents up.
    return Path(__file__).resolve().parents[2]


def _ensure_importable() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import parse_bench  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: parse_bench is not installed.\n"
            "Run benchmarks/parsebench/setup_parsebench.sh first "
            "(installs the official ParseBench framework + local parsers)."
        )


def _register_warp() -> None:
    """Import the warp provider (self-registers) and register local pipelines."""
    from parse_bench.inference.pipelines import get_pipeline, register_pipeline
    from parse_bench.schemas.pipeline import PipelineSpec
    from parse_bench.schemas.product import ProductType

    import benchmarks.parsebench.warp_ingest_provider  # noqa: F401  (registers provider)
    import benchmarks.parsebench.warp_layout_adapter  # noqa: F401  (registers layout adapter)

    for pipeline_name in ("warp_ingest", "warp_ingest_faithful", "warp_ingest_quality"):
        try:
            get_pipeline(pipeline_name)
        except ValueError:
            register_pipeline(
                PipelineSpec(
                    pipeline_name=pipeline_name,
                    provider_name="warp_ingest",
                    product_type=ProductType.PARSE,
                    config={},
                )
            )


def _configure_renderer(mode: str) -> dict[str, str]:
    """Set auditable renderer env vars before importing the Warp provider."""
    keys = [
        "WARP_TABLE_PROVIDER",
        "WARP_HF_STRIP",
        "WARP_CJK_STRIP",
        "WARP_COL_REORDER",
    ]
    if mode == "env":
        return {k: os.environ.get(k, "<unset>") for k in keys}

    if mode == "faithful":
        # "native" is warp's OWN license-clean table engine
        # (warp_ingest.ingestor.table_engine, pure MIT stack) — all output is
        # still produced by this repo's code, no external parser, so the run
        # stays a faithful Warp-only score.
        desired = {
            "WARP_TABLE_PROVIDER": "native",
            "WARP_HF_STRIP": "0",
            "WARP_CJK_STRIP": "0",
            "WARP_COL_REORDER": "0",
        }
    elif mode == "quality":
        desired = {
            "WARP_TABLE_PROVIDER": "native",
            "WARP_HF_STRIP": "1",
            "WARP_CJK_STRIP": "1",
            "WARP_COL_REORDER": "0",
        }
    else:
        raise ValueError(f"unknown renderer mode: {mode}")

    os.environ.update(desired)
    return {k: os.environ[k] for k in keys}


def _patch_liteparse_binary() -> str | None:
    """Point ParseBench's liteparse provider at the PATH ``lit`` binary.

    The upstream provider hardcodes a Rust workspace build path; the pip
    ``liteparse`` wheel ships the same engine as the ``lit`` console script, so
    we redirect to it (same engine, same flags) when available.
    """
    lit = shutil.which("lit")
    if not lit:
        return None
    import parse_bench.inference.providers.parse.liteparse as lp_mod

    lp_mod._LIT_BIN = Path(lit)
    return lit


def _read_headline(output_dir: Path, pipeline: str) -> dict[str, float | None]:
    """Read each dimension's headline score (0..100) for one pipeline."""
    scores: dict[str, float | None] = {}
    pdir = output_dir / pipeline
    for cat, _label in DIMENSIONS:
        report = pdir / cat / "_evaluation_report.json"
        val: float | None = None
        if report.exists():
            try:
                data = json.loads(report.read_text())
                agg = data.get("aggregate_metrics", {})
                raw = agg.get(HEADLINE_METRIC[cat])
                if raw is not None:
                    val = round(float(raw) * 100, 2)
            except Exception:
                val = None
        scores[cat] = val
    return scores


def _print_comparison(
    output_dir: Path, pipelines: list[str], test: bool = True
) -> None:
    rows: dict[str, dict[str, float | None]] = {
        p: _read_headline(output_dir, p) for p in pipelines
    }

    name_w = max([len("Pipeline")] + [len(p) for p in pipelines]) + 2
    header = f"{'Pipeline':<{name_w}}" + "".join(f"{lbl:>16}" for _c, lbl in DIMENSIONS)
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  ParseBench results (this run) — deterministic rule-based scores, 0–100")
    print("=" * len(header))
    print(header)
    print(sep)
    for p in pipelines:
        sc = rows[p]
        cells = "".join(
            f"{(f'{sc[c]:.1f}' if sc[c] is not None else '—'):>16}"
            for c, _l in DIMENSIONS
        )
        print(f"{p:<{name_w}}{cells}")
    print(sep)
    print(
        "\n  Published full-dataset leaderboard baselines (context only — not this run):"
    )
    print(f"  {'Provider':<24}" + "".join(f"{lbl:>16}" for _c, lbl in DIMENSIONS))
    for name, vals in LEADERBOARD_BASELINES.items():
        cells = "".join(f"{v:>16.1f}" for v in vals[1:])
        print(f"  {name:<24}{cells}")
    if test:
        scope = (
            "  NB: '--test' uses 3 files/category (~15 pages) — directional, same-harness\n"
            "  comparison only; run '--full' for leaderboard-comparable numbers."
        )
    else:
        scope = "  NB: full-dataset run (~2,000 pages) — comparable to the published leaderboard."
    print(
        "\n" + scope + "\n"
        "  Metrics: Tables=GTRM, Charts=chart-rule pass-rate, Content=faithfulness,\n"
        "  Sem.Format=semantic-formatting, Visual Ground.=layout element pass-rate.\n"
        "  Visual Ground. needs a per-provider layout adapter; only warp_ingest has one\n"
        "  here (text-only baselines emit no geometry — see published baselines above)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run official ParseBench vs Warp-Ingest + baselines."
    )
    ap.add_argument(
        "--pipelines",
        default=",".join(DEFAULT_PIPELINES),
        help=f"Comma-separated pipeline names (default: {','.join(DEFAULT_PIPELINES)}).",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--test", action="store_true", help="Small test split (3 files/category)."
    )
    mode.add_argument(
        "--full", action="store_true", help="Full dataset (leaderboard-comparable)."
    )
    ap.add_argument(
        "--group",
        default=None,
        help="Single dimension only (table/chart/text_content/text_formatting/layout).",
    )
    ap.add_argument(
        "--work-dir",
        default="parsebench_work",
        help="Where ./data and ./output live (default: ./parsebench_work).",
    )
    ap.add_argument(
        "--renderer-mode",
        choices=["faithful", "quality", "env"],
        default="faithful",
        help=(
            "Warp renderer controls. faithful = Warp output only, no external "
            "table provider or content stripping (default); quality = opt into "
            "local table-provider + strip transforms; env = respect current env."
        ),
    )
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument(
        "--skip-inference",
        action="store_true",
        help="Re-evaluate existing inference results.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force fresh inference even if cached parse results exist "
        "(use after changing the parser/engine).",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the comparison from existing results.",
    )
    args = ap.parse_args()

    test = not args.full  # default to the cheap test split unless --full given
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    # Force ParseBench's optional LLM chart-normalization OFF so the run is
    # fully deterministic and offline (no Anthropic API calls). The framework
    # otherwise defaults this env var to "judge" (on) and would post-process
    # *failed chart rules* with claude-haiku IF an API key were present. That
    # path only ever adds separate ``*_judge`` columns — the headline chart
    # metric we read (``avg_rule_pass_rate``) is always the deterministic
    # parent value — so disabling it changes nothing we report while making
    # the "no LLM-as-a-judge, reproducible" guarantee literal. ``setdefault``
    # respects an explicit override if the operator really wants the judge.
    os.environ.setdefault("LLAMACLOUD_BENCH_LLM_NORMALIZATION", "off")
    renderer_env = _configure_renderer(args.renderer_mode)
    print(f"ParseBench renderer mode: {args.renderer_mode}")
    for key, value in renderer_env.items():
        print(f"  {key}={value}")

    _ensure_importable()
    _register_warp()
    lit = _patch_liteparse_binary()
    if "liteparse_markdown" in pipelines and not lit:
        print(
            "WARNING: 'lit' binary not on PATH; liteparse pipeline will be skipped/fail.",
            file=sys.stderr,
        )

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    output_dir = work / "output"

    if not args.summary_only:
        from parse_bench.pipeline.cli import PipelineCLI

        for p in pipelines:
            print(
                f"\n{'#' * 70}\n# ParseBench pipeline: {p}  ({'test' if test else 'full'})\n{'#' * 70}"
            )
            try:
                PipelineCLI().run(
                    pipeline=p,
                    test=test,
                    group=args.group,
                    max_concurrent=args.max_concurrent,
                    open_report=False,
                    skip_inference=args.skip_inference,
                    force=args.force,
                )
            except Exception as e:  # keep going so one bad parser doesn't abort the run
                print(f"[ERROR] pipeline {p} failed: {e}", file=sys.stderr)

    _print_comparison(output_dir, pipelines, test=test)

    # Canonical cross-pipeline leaderboard via ParseBench's own generator.
    try:
        from parse_bench.analysis.leaderboard_report import generate_leaderboard_report

        lb = generate_leaderboard_report(output_dir=output_dir)
        print(f"\nParseBench leaderboard HTML: {lb}")
    except Exception as e:
        print(f"(leaderboard html skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
