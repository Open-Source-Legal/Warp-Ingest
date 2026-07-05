"""Prep for the honest golden-answer benchmark (fresh ~98 pages).

For each manifested page: render the page with its FINE-block boxes + ids (so a
golden-building agent can reference block ids, blind to the processor's units),
and dump the fine-block list. Separately dump the processor's unit grouping for
scoring. Outputs under audit_out/semunit_bench100/<doc>/.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from PIL import Image, ImageDraw  # noqa: E402

from warp_ingest.ingestor.oc_visualize import _load_font, _render_pdf_page  # noqa: E402
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts  # noqa: E402

OUT = REPO / "audit_out" / "semunit_bench100"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 2.0

# (relpath, [page indices]) — fresh docs (not in the prior 106 audit)
MANIFEST = [
    ("tests/fixtures/s1/fervo_s1__exhibit993sx1_6f30d7.pdf", list(range(0, 16))),
    ("tests/fixtures/s1/quantinuum_s1__exhibit33sx1_eefad8.pdf", list(range(0, 20))),
    ("tests/fixtures/s1/eikon_s1__ex1010_2e241e.pdf", list(range(0, 20))),
    ("tests/fixtures/s1/swarmer_s1__ex1018_765a25.pdf", list(range(0, 15))),
    ("tests/fixtures/s1/exyn_s1__ex11_a8500b.pdf", list(range(0, 10))),
    ("tests/fixtures/s1/spacex_s1__exhibit41sx1_331523.pdf", [0, 1, 2]),
    ("tests/fixtures/s1/eikon_s1__ex231_6fa7f0.pdf", [0]),
    ("tests/fixtures/s1/innio_s1__ex231_490e47.pdf", [0]),
    ("tests/fixtures/s1/xenergy_s1__ex232_ebd77e.pdf", [0]),
    ("tests/fixtures/s1/exyn_s1__ex211_812970.pdf", [0]),
    ("tests/fixtures/s1/hawkeye360_s1__exhibit211sx1_eed44c.pdf", [0]),
    ("tests/fixtures/s1/kardigan_s1__ex211_6d4418.pdf", [0]),
    ("tests/fixtures/s1/parabilis_s1__ex211_f13ef7.pdf", [0]),
    ("tests/fixtures/s1/solv_energy_s1__ex231_98c758.pdf", [0]),
    ("tests/fixtures/s1/yesway_s1__ex231_598e22.pdf", [0]),
    ("tests/fixtures/s1/quantinuum_s1__exhibit997sx1_bead0b.pdf", [0]),
    ("tests/fixtures/s1/liftoff_s1__ex231_c133b1.pdf", [0]),
    ("tests/fixtures/s1/pershing_square_s1__ex231_22654c.pdf", [0]),
    ("tests/fixtures/s1/neutron_lime_s1__exhibit231sx1_96d831.pdf", [0]),
    ("tests/fixtures/s1/swarmer_s1__ex211_b96bf3.pdf", [0]),
    # --- the prior 106-page audit set (complete docs) -> makes it a ~204-pg bench ---
    ("tests/fixtures/s1/exyn_s1__ex1010_de36c8.pdf", list(range(0, 7))),
    ("tests/fixtures/s1/cerebras_s1__exhibit1013esx_f124a6.pdf", list(range(0, 11))),
    (
        "tests/fixtures/EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_11893941_EX-10.1_Development_Agreement_ZrZJLLv.pdf",
        list(range(0, 23)),
    ),
    ("tests/fixtures/USC Title 1 - CHAPTER 1.pdf", list(range(0, 9))),
    ("tests/fixtures/s1/forbright_s1__exhibit1012sx1_11ea38.pdf", list(range(0, 17))),
    ("tests/fixtures/s1/generate_bio_s1__ex33_a3ab0f.pdf", list(range(0, 21))),
    ("tests/fixtures/contracts/fw_nctcog.pdf", list(range(0, 5))),
    ("tests/fixtures/contracts/fw_garver.pdf", list(range(0, 6))),
    ("tests/fixtures/contracts/fw_vertigis.pdf", list(range(0, 5))),
    ("tests/fixtures/contracts/fw_wert_bookbinding.pdf", list(range(0, 2))),
]


def fine_blocks(export, page):
    """Fine (non-unit) annotations on a page: id, label, bbox_frac, text."""
    pk = str(page)
    w = export["pawls_file_content"][page]["page"]["width"] or 1.0
    h = export["pawls_file_content"][page]["page"]["height"] or 1.0
    out = []
    for a in export["labelled_text"]:
        if a["annotationLabel"] == "Semantic Unit":
            continue
        aj = a.get("annotation_json") or {}
        if pk not in aj:
            continue
        b = aj[pk]["bounds"]
        out.append(
            {
                "id": a["id"],
                "label": a["annotationLabel"],
                "bbox_frac": [
                    round(b["left"] / w, 4),
                    round(b["top"] / h, 4),
                    round(b["right"] / w, 4),
                    round(b["bottom"] / h, 4),
                ],
                "text": " ".join((a["rawText"] or "").split()),
            }
        )
    return out


def proc_units(export, page):
    """Processor unit grouping on a page: {unit_id: {members, parent}}."""
    pk = str(page)
    mem, parent = {}, {}
    for r in export["relationships"]:
        s = r["source_annotation_ids"][0]
        if r["relationshipLabel"] == "OC_SEMANTIC_UNIT":
            mem[s] = r["target_annotation_ids"]
        elif r["relationshipLabel"] == "OC_PARENT_CHILD" and str(s).startswith("su-"):
            for t in r["target_annotation_ids"]:
                parent[t] = s
    onpage = {b["id"] for b in fine_blocks(export, page)}
    out = {}
    for u in export["labelled_text"]:
        if u["annotationLabel"] != "Semantic Unit":
            continue
        ms = [m for m in mem.get(u["id"], []) if m in onpage]
        if ms:
            out[u["id"]] = {"members": ms, "parent": parent.get(u["id"])}
    return out


def render(pdf_bytes, page, blocks):
    base = _render_pdf_page(pdf_bytes, page, SCALE).convert("RGBA")
    ov = Image.new("RGBA", base.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(ov)
    font = _load_font(max(11, int(6.0 * SCALE)))
    w, h = base.size
    for b in blocks:
        l, t, r, bot = b["bbox_frac"]
        x0, y0, x1, y1 = l * w, t * h, r * w, bot * h
        d.rectangle([x0, y0, x1, y1], outline=(30, 90, 200, 220), width=2)
        lab = f"#{b['id']}"
        tb = d.textbbox((0, 0), lab, font=font)
        d.rectangle(
            [x0, y0, x0 + (tb[2] - tb[0]) + 5, y0 + (tb[3] - tb[1]) + 4],
            fill=(30, 90, 200, 230),
        )
        d.text((x0 + 2, y0 + 1), lab, fill=(255, 255, 255), font=font)
    return Image.alpha_composite(base, ov).convert("RGB")


def main():
    total = 0
    manifest_out = []
    for relpath, pages in MANIFEST:
        slug = Path(relpath).stem[:44]
        d = OUT / slug
        d.mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            e = parse_to_opencontracts(
                str(REPO / relpath), parse_options={"semantic_units": True}
            )
        with open(REPO / relpath, "rb") as fh:
            pdf_bytes = fh.read()
        blocks_by_page, proc_by_page = {}, {}
        valid_pages = [p for p in pages if p < e["page_count"]]
        for p in valid_pages:
            blks = fine_blocks(e, p)
            blocks_by_page[p] = blks
            proc_by_page[p] = proc_units(e, p)
            render(pdf_bytes, p, blks).save(d / f"p{p:02d}.png")
        (d / "blocks.json").write_text(json.dumps(blocks_by_page, indent=1))
        (d / "proc_units.json").write_text(json.dumps(proc_by_page, indent=1))
        total += len(valid_pages)
        manifest_out.append({"slug": slug, "relpath": relpath, "pages": valid_pages})
        print(f"{slug:<44} {len(valid_pages):>2} pages", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest_out, indent=1))
    print(f"--- {total} pages prepped -> {OUT}")


if __name__ == "__main__":
    main()
