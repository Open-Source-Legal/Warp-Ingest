"""Pk (Beeferman 1999) and WindowDiff (Pevzner & Hearst 2002).

A *mask* is a list of 0/1 flags; a 1 marks the start of a new segment at that
position (position 0 is never a boundary). Deterministic and dependency-free;
the window convention (half-open ``[i, i+k)`` window over ``len-k+1`` positions,
``k = round(len / (2 * #boundaries))``) matches ``nltk.metrics.segmentation``
exactly, which the tests assert against (nltk is already a project dependency).
"""


def _k(ref_mask):
    """Default window: half the mean reference segment length (nltk convention)."""
    ones = sum(1 for x in ref_mask if x)
    if ones == 0:
        return max(1, len(ref_mask) // 2)
    return max(1, int(round(len(ref_mask) / (ones * 2.0))))


def pk(ref_mask, hyp_mask, k=None):
    """Pk error: fraction of windows where ref/hyp disagree on 'any boundary?'."""
    assert len(ref_mask) == len(hyp_mask)
    n = len(ref_mask)
    if k is None:
        k = _k(ref_mask)
    if n - k + 1 <= 0:
        return 0.0
    err = 0
    for i in range(n - k + 1):
        if any(ref_mask[i : i + k]) != any(hyp_mask[i : i + k]):
            err += 1
    return round(err / (n - k + 1), 6)


def windowdiff(ref_mask, hyp_mask, k=None):
    """WindowDiff: fraction of windows whose ref/hyp boundary counts differ."""
    assert len(ref_mask) == len(hyp_mask)
    n = len(ref_mask)
    if k is None:
        k = _k(ref_mask)
    if n - k + 1 <= 0:
        return 0.0
    wd = 0
    for i in range(n - k + 1):
        ndiff = abs(sum(ref_mask[i : i + k]) - sum(hyp_mask[i : i + k]))
        wd += min(1, ndiff)
    return round(wd / (n - k + 1), 6)
