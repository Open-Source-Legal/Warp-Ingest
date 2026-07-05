from nltk.metrics.segmentation import pk as nltk_pk
from nltk.metrics.segmentation import windowdiff as nltk_wd

from warp_ingest.ingestor.segmentation_metrics import _k, pk, windowdiff


def _mask_to_str(mask):
    return "".join("1" if b else "0" for b in mask)


def test_identical_masks_score_zero():
    m = [0, 1, 0, 0, 1, 0, 0]
    assert pk(m, m) == 0.0
    assert windowdiff(m, m) == 0.0


def test_matches_nltk_on_a_disagreement():
    ref = [0, 1, 0, 0, 1, 0, 0, 0]
    hyp = [0, 0, 1, 0, 1, 0, 0, 0]
    k = _k(ref)
    assert round(pk(ref, hyp, k), 6) == round(
        nltk_pk(_mask_to_str(ref), _mask_to_str(hyp), k), 6
    )
    assert round(windowdiff(ref, hyp, k), 6) == round(
        nltk_wd(_mask_to_str(ref), _mask_to_str(hyp), k), 6
    )
