#!/usr/bin/env python
"""Generate tests/fixtures/with_images.pdf (one-time; the PDF is committed).

A single-page contract-ish doc exercising the image-token export (issue #1):
a heading, two multi-line paragraphs, a figure image between the paragraphs
(standalone ``Image`` annotation case) and a small logo pinned in the top-right
corner. Requires the dev group (reportlab):
``uv run python scripts/build_image_fixture.py``.
"""

import importlib.util
import io
import pathlib

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = pathlib.Path(__file__).resolve().parent.parent

# load by file path: importing the tests.fixtures package would pull in
# factories.py (needs the opencontractserver package, not a dependency here)
_spec = importlib.util.spec_from_file_location(
    "pdf_generator", ROOT / "tests" / "fixtures" / "pdf_generator.py"
)
_pdf_generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pdf_generator)
create_test_image = _pdf_generator.create_test_image

OUT = ROOT / "tests" / "fixtures" / "with_images.pdf"

PARA1 = [
    "This Agreement is entered into by the parties named below and sets",
    "out the terms on which the services described in this document are",
    "to be provided during the initial term of the engagement.",
]
PARA2 = [
    "The figure above illustrates the delivery schedule for the first",
    "phase of the project and forms part of this Agreement for all",
    "purposes, including acceptance testing and final delivery.",
]


def main():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter  # 612 x 792

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 80, "MASTER SERVICES AGREEMENT")

    c.setFont("Helvetica", 11)
    y = height - 120
    for line in PARA1:
        c.drawString(72, y, line)
        y -= 16

    # figure: 180x120pt blue image between the paragraphs
    fig = ImageReader(io.BytesIO(create_test_image(400, 267, "blue")))
    c.drawImage(fig, 200, y - 140, width=180, height=120)
    y -= 170

    c.setFont("Helvetica", 11)
    for line in PARA2:
        c.drawString(72, y, line)
        y -= 16

    # logo: small red square pinned in the top-right corner
    logo = ImageReader(io.BytesIO(create_test_image(80, 80, "red")))
    c.drawImage(logo, width - 100, height - 100, width=28, height=28)

    c.save()
    OUT.write_bytes(buf.getvalue())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
