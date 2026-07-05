"""Pure-Python OCR for scanned PDF pages.

Renders pages to images with ``pypdfium2`` (Apache-2.0 / BSD, bundled binaries,
no system dependencies) via pdfplumber, and recognizes text with
``rapidocr-onnxruntime`` (Apache-2.0) -- the PP-OCR (PaddleOCR) detection +
recognition models running on onnxruntime (MIT), with the model weights bundled
in the wheel.  No Java, no Tesseract binary, no GPU required.

The OCR output is converted into the *same* Tika-format word dictionaries that
``pdf_plumber_parser`` renders, so a scanned page and a text page flow through
the identical downstream layout engine.

OCR is an optional capability: if ``rapidocr_onnxruntime`` is not installed the
helpers degrade gracefully (``ocr_available()`` returns False) instead of
raising, so the core library is usable without the OCR dependency.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Render resolution for OCR.  200 DPI is a good speed/accuracy balance for
# document scans; the value also sets the px -> PDF-point scale (72 / DPI).
OCR_DPI = 200
# Fraction of the detected text-line box height used as the nominal font size.
OCR_FONT_RATIO = 0.72

_engine = None
_engine_failed = False


def ocr_available():
    """True if the OCR engine can be loaded.

    ``WARP_DISABLE_OCR=1`` hard-disables OCR even when rapidocr is installed.
    Being an environment variable (rather than a monkeypatch) it also reaches
    the spawned front-end worker processes, which re-import this module fresh
    and would not see an ``ocr_available`` patched in the parent."""
    if os.environ.get("WARP_DISABLE_OCR", "").lower() in ("1", "true", "yes"):
        return False
    if _engine_failed:
        return False
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def get_engine():
    """Lazily construct a process-wide RapidOCR engine (heavy: loads onnx models)."""
    global _engine, _engine_failed
    if _engine is None and not _engine_failed:
        try:
            import multiprocessing

            from rapidocr_onnxruntime import RapidOCR

            kwargs = {}
            threads = None
            env_threads = os.environ.get("WARP_OCR_THREADS")
            if env_threads is not None:
                # Explicit per-session thread budget.  The service launcher sets
                # this to cpus // (web workers x FE workers) so W concurrent
                # daemon workers cannot each spin an all-cores onnx session.
                try:
                    threads = max(1, int(env_threads))
                except ValueError:
                    logger.warning(
                        "invalid WARP_OCR_THREADS=%r; using default", env_threads
                    )
            if threads is None and multiprocessing.parent_process() is not None:
                # Inside a front-end worker process several sibling sessions
                # may run OCR concurrently; onnxruntime's default is
                # all-cores-per-session, which oversubscribes the box (workers
                # x cores runnable threads).  Give each worker session a fair
                # slice instead.  Output is unaffected by the thread count
                # (verified bit-identical det/cls/rec results at 1 vs -1).
                from warp_ingest.file_parser.pdf_plumber_parser import (
                    _fe_worker_count,
                )

                threads = max(1, (os.cpu_count() or 1) // _fe_worker_count())
            if threads is not None:
                kwargs = dict(
                    det_intra_op_num_threads=threads,
                    cls_intra_op_num_threads=threads,
                    rec_intra_op_num_threads=threads,
                )
            _engine = RapidOCR(**kwargs)
        except Exception as e:
            logger.warning("OCR engine unavailable: %s", e)
            _engine_failed = True
    return _engine


def _line_to_words(text, x0, x1, top, bottom, scale, font_size):
    """Split a detected OCR line (one box, one string) into per-word dicts with
    character-proportional x positions.  All coordinates are in PDF points."""
    text = text.strip()
    if not text:
        return []
    box_w_pt = (x1 - x0) * scale
    left_pt = x0 * scale
    top_pt = top * scale
    bottom_pt = bottom * scale
    n_chars = max(len(text), 1)
    char_w = box_w_pt / n_chars
    words = []
    cursor = left_pt
    for token in text.split(" "):
        if token == "":
            cursor += char_w  # space
            continue
        w0 = cursor
        w1 = cursor + char_w * len(token)
        words.append(
            {
                "text": token,
                "x0": w0,
                "x1": w1,
                "top": top_pt,
                "bottom": bottom_pt,
                "fontname": "OCR",
                "size": font_size,
            }
        )
        cursor = w1 + char_w  # trailing space
    return words


def ocr_page_lines(page, dpi=OCR_DPI):
    """Run OCR on a pdfplumber ``page`` and return a list of *lines*, each a list
    of word dicts in PDF-point coordinates (same schema as pdfplumber words)."""
    engine = get_engine()
    if engine is None:
        return []
    import numpy as np

    scale = 72.0 / dpi  # px -> PDF points
    try:
        pil = page.to_image(resolution=dpi).original.convert("RGB")
    except Exception as e:
        logger.warning("page render failed for OCR: %s", e)
        return []
    arr = np.asarray(pil)
    try:
        result, _ = engine(arr)
    except Exception as e:
        logger.warning("OCR inference failed: %s", e)
        return []
    if not result:
        return []

    raw = []
    for box, text, score in result:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x0, x1 = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        font_size = round((bottom - top) * scale * OCR_FONT_RATIO, 1)
        font_size = max(font_size, 1.0)
        words = _line_to_words(text, x0, x1, top, bottom, scale, font_size)
        if words:
            raw.append(words)
    # sort lines top-to-bottom, left-to-right for natural reading order
    raw.sort(key=lambda ws: (round(ws[0]["top"], 1), ws[0]["x0"]))
    return raw
