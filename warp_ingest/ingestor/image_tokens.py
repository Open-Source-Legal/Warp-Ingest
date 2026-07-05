"""Extract embedded raster images from a PDF as OpenContracts PAWLS image tokens.

Export-boundary only (issue #1; design spec 2026-07-03): reads the raw
``pdf_bytes`` the exporter already receives and emits ``is_image: true`` token
dicts in the export's top-left page-point coordinate space. Never touches the
XHTML contract or the layout engine. Optional-field convention per the format
spec (§2): omitted when absent, never null.
"""

import base64
import hashlib
import io
import logging

logger = logging.getLogger(__name__)

# An image covering nearly the whole page is a full-page scan background
# (needs_ocr, the FortWorth contracts), not a figure — its text is already
# captured via the OCR token path. Structural/format-only gate.
_PAGE_SCAN_FRAC = 0.85
_MIN_DIM_PT = 4.0  # hairline/decorative slivers are not figures
# Bound the base64 payload: rendered bitmaps longer than this on their longest
# side are downscaled (original_width/height still report native pixels).
_MAX_RENDER_PX = 2048
_OBJ_MAX_DEPTH = 4  # descend into form XObjects


def _to_top_left(bounds, pdfium_size, xhtml_size):
    """pdfium ``(left, bottom, right, top)`` (bottom-left origin) -> clamped
    ``(left, top, right, bottom)`` in the export's top-left space, rescaled
    when the XHTML page box differs from pdfium's."""
    pl, pb, pr, pt = bounds
    pw, ph = pdfium_size
    xw, xh = xhtml_size
    sx = xw / pw if pw else 1.0
    sy = xh / ph if ph else 1.0
    left, right = pl * sx, pr * sx
    top, bottom = (ph - pt) * sy, (ph - pb) * sy
    left, right = min(left, right), max(left, right)
    top, bottom = min(top, bottom), max(top, bottom)
    return (
        min(max(left, 0.0), xw),
        min(max(top, 0.0), xh),
        min(max(right, 0.0), xw),
        min(max(bottom, 0.0), xh),
    )


def _keep_box(left, top, right, bottom, page_w, page_h):
    """Structural filters: degenerate, sliver, or full-page-scan boxes are out."""
    w, h = right - left, bottom - top
    if w < _MIN_DIM_PT or h < _MIN_DIM_PT:
        return False
    if page_w <= 0 or page_h <= 0:
        return False
    return w * h < _PAGE_SCAN_FRAC * page_w * page_h


def _encode_png(pil_image):
    """``(base64_str, sha256_hex, out_w, out_h)``; downscales past _MAX_RENDER_PX."""
    from PIL import Image

    w, h = pil_image.size
    if max(w, h) > _MAX_RENDER_PX:
        scale = _MAX_RENDER_PX / max(w, h)
        pil_image = pil_image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    return (b64, hashlib.sha256(data).hexdigest(), *pil_image.size)


def extract_page_images(pdf_bytes, page_dims):
    """``{page_idx: [PAWLS image-token dict]}`` for embedded raster figures.

    ``page_dims`` is the exporter's per-page ``(width, height)`` list (the
    XHTML coordinate space). Any per-object pdfium/PIL failure logs a warning
    and skips that object — image extraction must never fail a parse.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    out = {}
    try:  # doc-level isolation: pdfium may reject a PDF pdfplumber accepted
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as e:
        logger.warning("image extraction skipped (pdfium cannot open doc): %s", e)
        return out
    try:
        for pidx in range(min(len(pdf), len(page_dims))):
            tokens = []
            try:  # per-page isolation
                page = pdf[pidx]
                pdfium_size = page.get_size()
                xhtml_size = page_dims[pidx]
                for obj in page.get_objects(
                    filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=_OBJ_MAX_DEPTH
                ):
                    try:
                        left, top, right, bottom = _to_top_left(
                            obj.get_bounds(), pdfium_size, xhtml_size
                        )
                        if not _keep_box(left, top, right, bottom, *xhtml_size):
                            continue
                        px_w, px_h = obj.get_px_size()
                        # NB: renders the native-resolution bitmap before the
                        # _MAX_RENDER_PX downscale; a MemoryError on a huge
                        # embedded image is caught by this per-object isolation.
                        b64, sha, _, _ = _encode_png(
                            obj.get_bitmap(render=True).to_pil()
                        )
                    except Exception as e:  # per-object: never fail a parse
                        logger.warning("skipping image on page %s: %s", pidx, e)
                        continue
                    tokens.append(
                        {
                            "x": left,
                            "y": top,
                            "width": right - left,
                            "height": bottom - top,
                            "text": "",
                            "is_image": True,
                            "base64_data": b64,
                            "format": "png",
                            "content_hash": sha,
                            "original_width": int(px_w),
                            "original_height": int(px_h),
                            "image_type": "embedded",
                        }
                    )
            except Exception as e:
                logger.warning("skipping images on page %s: %s", pidx, e)
                continue
            if tokens:
                out[pidx] = tokens
    finally:
        pdf.close()
    return out
