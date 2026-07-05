import unittest

from warp_ingest.ingestor_utils.utils import rules, sent_tokenize


class PreProcessingTests(unittest.TestCase):
    def test_sentence_tokenizer(self):
        """
        sentence tokenization tests
        """
        samples = [
            "Effective September 1, 2017, John Smith (“Advisor”) and XYZ, Inc. (“Company”) agree as follows:",
            "Fig. 2 shows a U.S.A. map.",
            "The item at issue is no. 3553.",
            "Computershare Trust Company, N.A. (“Computershare”) is the transfer agent and registrar for our common stock.",
            "valid any day after january 1. not valid on federal holidays, including february 14, or with other in-house events, specials, or happy hour.",
            "LSTM networks, which we preview in Sec. 2, have been successfully",
        ]
        for text in samples:
            sentences = sent_tokenize(text)
            expected = [text]
            self.assertEqual(sentences, expected)

    def test_abbreviation_rules_fire_across_casing(self):
        """The needle prefilter is casefolded, so uppercase/mixed-case
        abbreviations must still be protected from sentence splitting."""
        samples = [
            "FIG. 2 shows a map of the region and its facilities.",
            "See SEC. 4 for the applicable disclosure schedule.",
            "The offices are on Main St. near the harbor.",
        ]
        for text in samples:
            self.assertEqual(sent_tokenize(text), [text])

    def test_prefilter_needle_is_sound(self):
        """A rule's (kind, needle) check must be a *necessary* condition: any
        text its regex matches must also pass the prefilter.  Synthesizes a
        matching text for each rule shape; if the regex matches but the
        prefilter says skip, a live rule would be wrongly pruned."""
        from warp_ingest.ingestor_utils.utils import _space_anchored

        def passes(kind, needle, text):
            cf = text.casefold()
            if kind == "start":
                return cf.startswith(needle)
            if kind == "word":
                return _space_anchored(cf, needle)
            return needle in cf

        for kind, needle, rule, replaced in rules:
            if needle is None:
                continue  # rule always runs; nothing to prove
            abb = replaced.strip().rstrip("_")
            for synth in (f"{abb}. x", f"a {abb}. x", f"\t{abb}. x", f"{abb}."):
                for text in (synth, synth.upper()):
                    if rule.search(text):
                        self.assertTrue(
                            passes(kind, needle, text), (kind, needle, rule.pattern)
                        )
