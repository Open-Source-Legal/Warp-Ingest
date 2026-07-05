import json
import logging
import re
from collections import defaultdict, namedtuple
from timeit import default_timer
from typing import Optional

from bs4 import BeautifulSoup

from warp_ingest.file_parser import pdf_file_parser
from warp_ingest.ingestor.opencontracts_exporter import to_opencontracts_export
from warp_ingest.ingestor_utils import utils
from warp_ingest.ingestor_utils.utils import (
    NpEncoder,
    detect_block_center_aligned,
    detect_block_center_of_page,
)

from .visual_ingestor import visual_ingestor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
text_only_pattern = re.compile(r"[^a-zA-Z]+")


class PDFIngestor:
    def __init__(self, doc_location, parse_options):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        parse_pages = parse_options.get("parse_pages", False) if parse_options else ()
        render_format = (
            parse_options.get("render_format", "all") if parse_options else "all"
        )

        tika_html_doc = parse_pdf(doc_location, parse_options)
        # print("tika_html_doc", tika_html_doc)
        # The OpenContracts export needs the full block set; parse with "all".
        blocks_render_format = (
            "all" if render_format == "opencontracts" else render_format
        )
        blocks, _block_texts, _sents, _file_data, result, page_dim, num_pages = (
            parse_blocks(
                tika_html_doc,
                render_format=blocks_render_format,
                parse_pages=parse_pages,
            )
        )
        print("parsed blocks")
        return_dict = {
            "page_dim": page_dim,
            "num_pages": num_pages,
        }
        if render_format == "json":
            return_dict["result"] = result[0].get("document", {})
            self.doc_result_json = result[0]
        elif render_format == "opencontracts":
            export = to_opencontracts_export(
                tika_html_doc,
                blocks,
                pdf_bytes=_read_bytes(doc_location),
                semantic_units=bool(
                    parse_options and parse_options.get("semantic_units")
                ),
                include_images=bool(
                    parse_options and parse_options.get("include_images")
                ),
            )
            return_dict["result"] = export
            self.doc_result_json = export
        elif render_format == "all":
            return_dict["result"] = result[1].get("document", {})
            self.doc_result_json = result[1]
        self.return_dict = return_dict
        self.file_data = _file_data
        self.blocks = blocks
        # Retained so downstream benchmark renderers can read the front-end's
        # per-page ``data-columns`` reading-order signal off the page divs. Purely
        # additive (a reference to a string already in scope); nothing in the
        # engine or exporter reads it.
        self.tika_html = tika_html_doc


def _read_bytes(doc_location):
    try:
        with open(doc_location, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def parse_to_opencontracts(doc_location, parse_options: Optional[dict] = None):
    """Parse a PDF straight to an ``OpenContractDocExport`` dict.

    Convenience wrapper used by the daemon's ``renderFormat=opencontracts`` path
    and the export regression suite. Reads the Tika XHTML + finalized blocks and
    hands them to :func:`to_opencontracts_export` — no engine state is mutated.
    """
    parse_options = parse_options or {}
    tika_html_doc = parse_pdf(doc_location, parse_options)
    blocks, _bt, _s, _fd, _r, _pd, _np = parse_blocks(
        tika_html_doc,
        render_format="all",
        parse_pages=parse_options.get("parse_pages", ()) or (),
    )
    return to_opencontracts_export(
        tika_html_doc,
        blocks,
        pdf_bytes=_read_bytes(doc_location),
        recital_triggers=parse_options.get("recital_triggers"),
        semantic_units=bool(parse_options.get("semantic_units")),
        include_images=bool(parse_options.get("include_images")),
    )


def parse_pdf(doc_location, parse_options):
    """Parse a PDF to Tika-compatible XHTML using the pure-Python front-end.

    Scanned/sparse pages are automatically routed to OCR by the parser; passing
    ``apply_ocr=True`` forces OCR on every page, ``disable_ocr=True`` keeps every
    page on its embedded text layer (per-request; no environment mutation).
    """
    apply_ocr = parse_options.get("apply_ocr", False) if parse_options else False
    disable_ocr = parse_options.get("disable_ocr", False) if parse_options else False
    wall_time = default_timer() * 1000
    print(
        "Parsing PDF"
        + (" (OCR forced)" if apply_ocr else "")
        + (" (OCR disabled)" if disable_ocr else "")
    )
    parsed_content = pdf_file_parser.parse_to_html(
        doc_location, do_ocr=apply_ocr, disable_ocr=disable_ocr
    )
    print(
        f"PDF Parsing finished in {default_timer() * 1000 - wall_time:.4f}ms on workspace",
    )
    return parsed_content


def parse_blocks(
    tika_html_doc,
    render_format: str = "all",
    parse_pages: tuple = (),
):
    soup = BeautifulSoup(str(tika_html_doc), "html.parser")
    meta_tags = soup.find_all("meta")
    title = None
    for tag in meta_tags:
        if tag["name"].endswith(":title"):
            title = tag["content"]
            break
    pages = soup.find_all("div", class_=lambda x: x in ["page"])
    # read ignore blocks here
    ignore_blocks = []
    if parse_pages:
        start_page_no, end_page_no = parse_pages
        pages = pages[start_page_no : end_page_no + 1]
    # Header/indent levels come solely from the engine's built-in IndentParser.
    # A second re-leveling pass (NewIndentParser, the old `useNewIndentParser`
    # option) was removed 2026-07-02 after losing a golden A/B on hierarchy
    # quality (hetero-100 parent_class_agreement -0.013, legal-100
    # head_ancestor_agreement -0.031); levels are not scored by the S-1 suite,
    # so its cross-engine baselines are unaffected.
    parsed_doc = visual_ingestor.Doc(pages, ignore_blocks, render_format)
    title_page_fonts = top_pages_info(parsed_doc)
    parsed_doc.compress_blocks()
    blocks = parsed_doc.blocks
    sents, _ = utils.blocks_to_sents(blocks)
    block_texts, _ = utils.get_block_texts(blocks)
    if render_format == "json":
        result = [
            {
                "title": title,
                "document": parsed_doc.json_dict,
                "title_page_fonts": title_page_fonts,
            }
        ]
    elif render_format == "html":
        result = [
            {
                "title": title,
                "text": parsed_doc.html_str,
                "title_page_fonts": title_page_fonts,
            }
        ]
    else:
        result = [
            {
                "title": title,
                "text": parsed_doc.html_str,
                "title_page_fonts": title_page_fonts,
            },
            {
                "title": title,
                "document": parsed_doc.json_dict,
                "title_page_fonts": title_page_fonts,
            },
        ]

    file_data = [json.dumps(res, cls=NpEncoder) for res in result]

    return (
        blocks,
        block_texts,
        sents,
        file_data,
        result,
        [parsed_doc.page_width, parsed_doc.page_height],
        len(pages) - 1,
    )


def top_pages_info(parsed_doc):
    font_freq = {}
    for idx, block in enumerate(parsed_doc.blocks):
        if block["page_idx"] > 2:  # Consider only the first 2 pages
            break
        if not block["block_type"] == "header" and not block["block_idx"] < 1:
            continue
        for line in block["visual_lines"]:
            line_font = line["line_style"][2]  # font_size
            line_text = line["text"]
            if line_font in font_freq:
                font_freq[line_font].append(
                    {
                        "text": line_text,
                        "page": block["page_idx"],
                        "block_idx": block["block_idx"],
                        "enum_idx": idx,  # Adding enum_idx as the block_idx can be wrong here.
                    }
                )
            else:
                font_freq[line_font] = [
                    {
                        "text": line_text,
                        "page": block["page_idx"],
                        "block_idx": block["block_idx"],
                        "enum_idx": idx,  # Adding enum_idx as the block_idx can be wrong here.
                    }
                ]
    # Sort the font_freq in descending order.
    sorted_freq = {}
    for key in sorted(font_freq, reverse=True):
        sorted_freq[key] = font_freq[key]

    res = {}
    title_page = (
        sorted_freq[list(sorted_freq.keys())[0]][0]["page"]
        if len(sorted_freq) > 0
        else []
    )
    temp = []
    title_candidates = []

    def retrieve_title_candidates(key_idx):
        temp_ = []
        title_candidates_ = []
        if len(sorted_freq) > 0 and len(list(sorted_freq.keys())) > key_idx:
            for freq_ in sorted_freq[list(sorted_freq.keys())[key_idx]]:
                if (
                    parsed_doc.blocks[freq_["enum_idx"]]["box_style"][0]
                    >= parsed_doc.page_height / 2
                ) or not len(text_only_pattern.sub("", freq_["text"]).strip()):
                    continue
                if (
                    len(temp_) == 0
                    or abs(temp_[-1]["block_idx"] - freq_["block_idx"]) <= 1
                ):
                    temp_.append(freq_)
                if freq_["page"] == title_page:
                    freq_["center_aligned"] = detect_block_center_aligned(
                        parsed_doc.blocks[freq_["enum_idx"]], parsed_doc.page_width
                    )
                    freq_["all_caps"] = parsed_doc.blocks[freq_["enum_idx"]][
                        "block_text"
                    ].isupper()
                    freq_["center_of_page"] = detect_block_center_of_page(
                        parsed_doc.blocks[freq_["enum_idx"]], parsed_doc.page_height
                    )
                    title_candidates_.append(freq_)
        return temp_, title_candidates_

    # Check only the first 2 font_sizes
    for i in range(0, 2):
        temp, title_candidates = retrieve_title_candidates(i)
        if len(temp):
            break
    # Contains candidates from the title page of the same font (probable largest font)
    if title_candidates:
        new_temp = []
        # Preference to center_of_page
        for freq in title_candidates:
            if freq["center_of_page"]:
                new_temp.append(freq)
            elif (
                len(new_temp)
                and abs(new_temp[-1]["block_idx"] - freq["block_idx"]) <= 1
            ):
                new_temp.append(freq)
        # Next Preference to all_caps
        if not new_temp:
            for freq in title_candidates:
                if freq["all_caps"]:
                    new_temp.append(freq)
                elif (
                    len(new_temp)
                    and abs(new_temp[-1]["block_idx"] - freq["block_idx"]) <= 1
                ):
                    new_temp.append(freq)
        # Next Preference to center_aligned
        if not new_temp:
            for freq in title_candidates:
                if freq["center_aligned"]:
                    new_temp.append(freq)
                elif (
                    len(new_temp)
                    and abs(new_temp[-1]["block_idx"] - freq["block_idx"]) <= 1
                ):
                    new_temp.append(freq)
        if new_temp:
            temp = new_temp

    res["first_level"] = [freq["text"] for freq in temp] if len(temp) > 0 else []
    # first level subtitle
    res["first_level_sub"] = []
    for i in range(1, 4):
        # stop after the largest text other than title is found
        if res["first_level_sub"] or i >= len(sorted_freq):
            break
        # loop through next largest texts that's on the title page
        for freq in sorted_freq[list(sorted_freq.keys())[i]]:
            if freq["page"] == title_page:
                res["first_level_sub"].append(freq["text"])

    res["second_level"] = (
        [freq["text"] for freq in sorted_freq[list(sorted_freq.keys())[1]]]
        if len(sorted_freq) > 1
        else []
    )
    res["third_level"] = (
        [freq["text"] for freq in sorted_freq[list(sorted_freq.keys())[2]]]
        if len(sorted_freq) > 2
        else []
    )
    return res
