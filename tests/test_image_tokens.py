"""Unit tests for warp_ingest.ingestor.image_tokens (design spec 2026-07-03)."""

import pathlib

from warp_ingest.ingestor import image_tokens as it

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
WITH_IMAGES = FIXTURES / "with_images.pdf"


class TestToTopLeft:
    def test_flips_bottom_left_origin(self):
        # a 100x50pt box near the bottom-left of a 600x800 page
        out = it._to_top_left((10, 30, 110, 80), (600, 800), (600, 800))
        assert out == (10, 720, 110, 770)

    def test_rescales_when_xhtml_dims_differ(self):
        out = it._to_top_left((0, 0, 300, 400), (600, 800), (300, 400))
        assert out == (0, 200, 150, 400)

    def test_clamps_to_page(self):
        out = it._to_top_left((-20, -10, 700, 900), (600, 800), (600, 800))
        assert out == (0, 0, 600, 800)


class TestKeepBox:
    def test_keeps_normal_figure(self):
        assert it._keep_box(100, 100, 300, 250, 612, 792)

    def test_drops_full_page_scan(self):
        assert not it._keep_box(0, 0, 612, 792, 612, 792)

    def test_drops_slivers(self):
        assert not it._keep_box(100, 100, 103, 400, 612, 792)

    def test_drops_degenerate(self):
        assert not it._keep_box(100, 100, 100, 100, 612, 792)


class TestEncodePng:
    def test_downscale_cap_and_determinism(self):
        from PIL import Image

        b64_a, sha_a, w, h = it._encode_png(Image.new("RGB", (4096, 1024), "blue"))
        b64_b, sha_b, _, _ = it._encode_png(Image.new("RGB", (4096, 1024), "blue"))
        assert (w, h) == (2048, 512)
        assert sha_a == sha_b and b64_a == b64_b
        assert len(sha_a) == 64


class TestExtractPageImages:
    def test_fixture_yields_two_figure_tokens(self):
        out = it.extract_page_images(WITH_IMAGES.read_bytes(), [(612.0, 792.0)])
        toks = out[0]
        assert len(toks) == 2
        for t in toks:
            assert t["text"] == "" and t["is_image"] is True
            assert t["format"] == "png" and t["image_type"] == "embedded"
            assert len(t["content_hash"]) == 64 and t["base64_data"]
            assert t["original_width"] > 0 and t["original_height"] > 0
            assert t["x"] >= 0 and t["y"] >= 0
            assert t["x"] + t["width"] <= 612 and t["y"] + t["height"] <= 792
        # the 180pt figure and the 28pt logo, any order
        assert sorted(round(t["width"]) for t in toks) == [28, 180]

    def test_full_page_scan_is_filtered(self):
        import pypdfium2 as pdfium

        pdf_bytes = (FIXTURES / "needs_ocr.pdf").read_bytes()
        pdf = pdfium.PdfDocument(pdf_bytes)
        dims = [pdf[i].get_size() for i in range(len(pdf))]
        pdf.close()
        assert it.extract_page_images(pdf_bytes, dims) == {}

    def test_per_object_failure_is_isolated(self, monkeypatch):
        import pypdfium2 as pdfium

        def boom(self, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(pdfium.PdfImage, "get_bitmap", boom)
        assert it.extract_page_images(WITH_IMAGES.read_bytes(), [(612.0, 792.0)]) == {}

    def test_unreadable_pdf_is_isolated(self):
        # pdfium may reject bytes pdfplumber accepted; extraction must not raise
        assert it.extract_page_images(b"not a pdf at all", [(612.0, 792.0)]) == {}
