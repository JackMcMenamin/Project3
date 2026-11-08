import random
import unittest

from rsr_core import data
from rsr_core.sampling import WordPicker
from rsr_core.supervision import (Supervision, SupervisionOptions, batch_composition,
                                  build_supervision, inflect_pairs)
from tests.test_data import _supervision_available


def _toy(pairs: dict) -> Supervision:
    lookup, source, vocab = {}, {}, set()
    for (a, b), src in pairs.items():
        lookup[(a, b)] = lookup[(b, a)] = 0.5
        source[(a, b)] = source[(b, a)] = src
        vocab.update((a, b))
    return Supervision(lookup, source, vocab, SupervisionOptions())


class Inflection(unittest.TestCase):
    def test_forms_keep_the_score(self):
        out = {p for p, _ in inflect_pairs([("run", "jump", 0.4)])}
        self.assertEqual(out, {("run", "jump", 0.4), ("runs", "jumps", 0.4),
                               ("ran", "jumped", 0.4), ("running", "jumping", 0.4)})

    def test_every_copy_remembers_its_base(self):
        self.assertTrue(all(base == ("eat", "drink")
                            for _, base in inflect_pairs([("eat", "drink", 0.9)])))


class Composition(unittest.TestCase):
    def test_counts_by_source(self):
        sup = _toy({("a", "b"): "things", ("a", "c"): "simverb", ("c", "d"): "things"})
        self.assertEqual(batch_composition(["a", "b", "c", "x"], sup.source),
                         {"things": 1, "simverb": 1})


class Picker(unittest.TestCase):
    def setUp(self):
        self.sup = _toy({("a", "b"): "things", ("c", "d"): "things", ("e", "f"): "simverb",
                         ("g", "h"): "simverb_infl", ("a", "c"): "things"})
        self.targets = sorted(self.sup.vocab)

    def test_word_sampler_is_the_old_draw(self):
        random.seed(3)
        got = WordPicker(self.sup, self.targets, 1).candidates()
        random.seed(3)
        self.assertEqual(got, random.sample(self.targets, k=6))

    def test_pair_sampler_keeps_pairs_together(self):
        random.seed(0)
        cand = WordPicker(self.sup, self.targets, 2, mode="pairs").candidates()
        self.assertIn((min(cand[:2]), max(cand[:2])), set(self.sup.pairs()))

    def test_full_simverb_share_draws_only_verb_pairs(self):
        random.seed(0)
        picker = WordPicker(self.sup, self.targets, 2, mode="pairs", simverb_share=1.0)
        for _ in range(20):
            self.assertTrue(set(picker.candidates()) <= {"e", "f", "g", "h"})

    def test_pairs_need_both_words_usable(self):
        picker = WordPicker(self.sup, ["a", "b", "c"], 2, mode="pairs")
        self.assertEqual(sorted(picker.all_pairs), [("a", "b"), ("a", "c")])

    def test_share_needs_pair_sampler(self):
        with self.assertRaises(ValueError):
            WordPicker(self.sup, self.targets, 2, simverb_share=0.5)


@unittest.skipUnless(_supervision_available(), "SimVerb or THINGS not on disk")
class RealSupervision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = build_supervision()
        cls.simlex = data.load_simlex()
        cls.simlex_keys = {(a, b) for a, b, _ in cls.simlex} | {(b, a) for a, b, _ in cls.simlex}

    def test_default_is_the_reported_supervision(self):
        lookup, vocab = data.load_supervision()
        self.assertEqual(self.default.lookup, lookup)
        self.assertEqual(self.default.vocab, vocab)

    def test_exclusion_removes_every_simlex_pair(self):
        sup = build_supervision(SupervisionOptions(exclude_simlex=True))
        self.assertFalse(self.simlex_keys & set(sup.lookup))
        self.assertEqual(sum(sup.dropped_simlex.values()), 274)
        cats = data.partition(self.simlex, sup.vocab, lookup=sup.lookup)
        self.assertEqual(len(cats.get("both_trained", [])), 0)

    def test_exclusion_with_inflection_drops_copies_of_simlex_pairs(self):
        sup = build_supervision(SupervisionOptions(exclude_simlex=True, inflect_verbs=True))
        self.assertFalse(self.simlex_keys & set(sup.lookup))
        self.assertGreater(sup.dropped_simlex["simverb_infl"], 0)

    def test_inflection_adds_verb_pairs(self):
        sup = build_supervision(SupervisionOptions(inflect_verbs=True))
        c = sup.counts()
        self.assertGreater(c["simverb_infl"], 2 * c["simverb"])
        self.assertEqual(c["things"], self.default.counts()["things"])

    def test_things_sample_is_fixed(self):
        a = build_supervision(SupervisionOptions(things_pairs=10_000))
        b = build_supervision(SupervisionOptions(things_pairs=10_000))
        self.assertEqual(a.counts()["things"], 10_000)
        self.assertEqual(a.lookup, b.lookup)


if __name__ == "__main__":
    unittest.main()
