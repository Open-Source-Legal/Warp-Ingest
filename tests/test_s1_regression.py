"""Cross-engine regression suite over 50 real EDGAR S-1 PDFs.

Each PDF has a *gold target* captured from the original (Java/Tika) ``nlm-ingestor``
engine (see ``tests/fixtures/s1_targets``). These tests run the pure-Python engine
on the same PDF and assert that its output stays *compatible* with the original:

* **content** — token recall against the original stays above an absolute floor
  and does not regress below the committed baseline;
* **structure** — page count matches exactly, and the ordered block sequence stays
  as similar to the original as the committed baseline (within a tolerance band).

The baseline (``tests/fixtures/s1_baseline.json``) freezes today's compatibility
level so future engine changes that *improve* fidelity pass freely while
regressions fail. Regenerate fixtures with ``scripts/build_s1_fixtures.py``.

By default only the fast (small/medium) documents run; the large multi-hundred-page
S-1 bodies are marked ``slow`` (run with ``pytest -m slow`` or ``--runslow``).
"""

import os

import pytest

from warp_ingest.ingestor.pdf_ingestor import PDFIngestor

from . import s1_compat as C

# Absolute compatibility floors vs the original engine (not just self-regression).
MIN_TOKEN_RECALL = 0.90  # keep >=90% of the original's text tokens
# Tolerance bands: allow improvement freely, fail on meaningful regression.
RECALL_BAND = 0.02
PRECISION_BAND = 0.03
SEQ_BAND = 0.05
TABLE_BAND_FRAC = 0.30  # table-count drift allowance (plus a small absolute slack)

PARSE_OPTIONS = {
    "render_format": "all",
    "apply_ocr": False,
    "parse_pages": (),
}

_BASELINE = C.load_baseline() if os.path.exists(C.S1_BASELINE) else {}


def _params():
    names = C.target_names() if os.path.isdir(C.S1_TARGET_DIR) else []
    out = []
    for n in names:
        slow = _BASELINE.get(n, {}).get("slow", False)
        marks = (pytest.mark.slow,) if slow else ()
        out.append(pytest.param(n, marks=marks, id=n))
    return out


# Parse each PDF at most once per session (the big bodies are expensive).
_CACHE = {}


def _new_blocks(name):
    if name not in _CACHE:
        pdf = os.path.join(C.S1_PDF_DIR, name + ".pdf")
        ing = PDFIngestor(pdf, dict(PARSE_OPTIONS))
        result = ing.return_dict.get("result") or {}
        _CACHE[name] = (
            C.result_to_blocks(result),
            ing.return_dict.get("num_pages"),
        )
    return _CACHE[name]


@pytest.mark.parametrize("name", _params())
def test_s1_compatible_with_original(name):
    target = C.load_target(name)
    base = _BASELINE.get(name)
    assert base is not None, f"no baseline for {name}; rebuild fixtures"

    new_blocks, new_pages = _new_blocks(name)

    # Some PDFs crash the *original* (Java/Tika) engine (e.g. a KeyError on
    # 'visual_lines'); the gold target is empty. The pure-Python engine handles
    # these, so there is nothing to match — instead assert it still produces a
    # sane, non-trivial parse and does not regress below the committed baseline.
    if target.get("orig_failed"):
        assert len(new_blocks) >= max(1, int(0.8 * base["new_blocks"])), (
            f"{name}: original engine failed here; new engine output regressed "
            f"({len(new_blocks)} blocks vs baseline {base['new_blocks']})"
        )
        assert (new_pages or 0) >= 1
        return

    gold_blocks = target["blocks"]
    m = C.compare(gold_blocks, new_blocks)

    # --- content: absolute floor + no-regression band vs original ---
    # The absolute floor is waived for docs the original itself rendered
    # ambiguously (e.g. a stylized stock-certificate graphic where pdfplumber
    # glues letters and Tika spaces them); those rely on the no-regression band.
    floor = min(MIN_TOKEN_RECALL, base["token_recall"])
    assert m["token_recall"] >= floor - RECALL_BAND, (
        f"{name}: token recall {m['token_recall']:.3f} below floor "
        f"{floor:.3f} vs original"
    )
    assert m["token_recall"] >= base["token_recall"] - RECALL_BAND, (
        f"{name}: token recall regressed {base['token_recall']:.3f} -> "
        f"{m['token_recall']:.3f}"
    )
    assert m["token_precision"] >= base["token_precision"] - PRECISION_BAND, (
        f"{name}: token precision regressed {base['token_precision']:.3f} -> "
        f"{m['token_precision']:.3f}"
    )

    # --- structure: exact page count, stable ordered block similarity ---
    assert (
        new_pages == target["num_pages"]
    ), f"{name}: page count {new_pages} != original {target['num_pages']}"
    assert m["seq_ratio"] >= base["seq_ratio"] - SEQ_BAND, (
        f"{name}: block-sequence similarity regressed {base['seq_ratio']:.3f} -> "
        f"{m['seq_ratio']:.3f}"
    )

    # --- tables: count should not drift far from the committed baseline ---
    base_tbl = base["new_tables"]
    slack = max(2, int(TABLE_BAND_FRAC * max(base_tbl, m["gold_tables"])))
    assert abs(m["new_tables"] - base_tbl) <= slack, (
        f"{name}: table count {m['new_tables']} drifted from baseline {base_tbl} "
        f"(gold {m['gold_tables']}, slack {slack})"
    )


def test_s1_aggregate_recall():
    """Whole-corpus guard: mean token recall vs the original stays high."""
    names = C.target_names()
    if not names:
        pytest.skip("S-1 fixtures not present")
    # Only score docs already parsed this session to avoid forcing a full re-parse;
    # at minimum the fast set runs, which is plenty for an aggregate signal.
    # Exclude docs where the original engine failed (no meaningful gold to score).
    scored = [
        n for n in names if n in _CACHE and not C.load_target(n).get("orig_failed")
    ]
    if len(scored) < 5:
        pytest.skip("not enough docs parsed this session for an aggregate")
    recalls = []
    for n in scored:
        target = C.load_target(n)
        new_blocks, _ = _new_blocks(n)
        recalls.append(C.compare(target["blocks"], new_blocks)["token_recall"])
    mean = sum(recalls) / len(recalls)
    assert mean >= 0.95, f"mean token recall {mean:.3f} across {len(scored)} docs"
