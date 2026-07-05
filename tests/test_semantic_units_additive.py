import os

from warp_ingest.ingestor.opencontracts_exporter import validate_export
from warp_ingest.ingestor.pdf_ingestor import parse_to_opencontracts

FIX = os.path.join(
    os.path.dirname(__file__), "fixtures", "s1", "exyn_s1__ex1010_de36c8.pdf"
)


def _fine(export):
    return [
        a for a in export["labelled_text"] if a["annotationLabel"] != "Semantic Unit"
    ]


def test_flag_off_is_unchanged_and_flag_on_is_additive():
    off = parse_to_opencontracts(FIX)  # default: flag off
    on = parse_to_opencontracts(FIX, parse_options={"semantic_units": True})
    # flag OFF emits no Semantic Units
    assert not any(
        a["annotationLabel"] == "Semantic Unit" for a in off["labelled_text"]
    )
    # fine layer is byte-identical between off and on
    assert _fine(off) == _fine(on)
    # existing (non-unit) relationships unchanged
    off_rels = [r for r in off["relationships"] if not r["id"].startswith("surel-")]
    on_rels = [r for r in on["relationships"] if not r["id"].startswith("surel-")]
    assert off_rels == on_rels
    # flag ON added Semantic Units and still validates
    assert any(a["annotationLabel"] == "Semantic Unit" for a in on["labelled_text"])
    validate_export(on)  # raises on any invariant break
