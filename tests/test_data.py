import json
import tempfile
import unittest
from pathlib import Path

from rsr_core import data, paths


class Partition(unittest.TestCase):
    def test_exposure(self):
        vocab = {"cat", "dog"}
        self.assertEqual(data.exposure("cat", "dog", vocab), "both")
        self.assertEqual(data.exposure("cat", "car", vocab), "one")
        self.assertEqual(data.exposure("car", "bus", vocab), "neither")

    def test_every_pair_is_in_all_and_one_partition(self):
        pairs = [("cat", "dog", 1.0), ("cat", "car", 0.5), ("car", "bus", 0.2)]
        cats = data.partition(pairs, {"cat", "dog"})
        self.assertEqual(len(cats["all"]), 3)
        self.assertEqual(sum(len(cats[k]) for k in ("both", "one", "neither")), 3)


def _supervision_available() -> bool:
    return paths.SIMVERB_PATH.exists() and paths.THINGS_SIM_PATH.exists()


@unittest.skipUnless(_supervision_available(), "SimVerb or THINGS not on disk")
class SimLexSplit(unittest.TestCase):
    """The split every reported number rests on. Counts are over all 999 pairs,
    before words without harvested sentences are dropped."""

    @classmethod
    def setUpClass(cls):
        cls.lookup, cls.vocab = data.load_supervision()
        cls.simlex = data.load_simlex()
        cls.cats = data.partition(cls.simlex, cls.vocab)

    def test_supervision_size(self):
        self.assertEqual(len(self.vocab), 2652)

    def test_counts(self):
        self.assertEqual(len(self.simlex), 999)
        self.assertEqual({k: len(v) for k, v in self.cats.items() if k != "all"},
                         {"both": 297, "one": 219, "neither": 483})

    def test_neither_pairs_share_no_word_with_supervision(self):
        for w1, w2, _ in self.cats["neither"]:
            self.assertNotIn(w1, self.vocab)
            self.assertNotIn(w2, self.vocab)

    def test_only_both_pairs_can_be_supervision_pairs(self):
        for name in ("one", "neither"):
            self.assertFalse(any((a, b) in self.lookup for a, b, _ in self.cats[name]))
        trained = sum((a, b) in self.lookup for a, b, _ in self.cats["both"])
        self.assertEqual(trained, 274)


class SentencePool(unittest.TestCase):
    def setUp(self):
        self.saved = paths.sentence_pool()

    def tearDown(self):
        paths._sentence_pool = self.saved

    def test_default_is_strict_pool(self):
        self.assertEqual(paths.DEFAULT_SENTENCE_POOL.name, "sentences_strict")

    def test_switching_pool_changes_what_loads(self):
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "aardvark.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"sentence": "An aardvark digs."}) + "\n")
                f.write("not json\n")
            paths.set_sentence_pool(d)
            self.assertTrue(data.has_sentences("aardvark"))
            self.assertEqual(data.load_sentences_for("aardvark"), ["An aardvark digs."])
            self.assertFalse(data.has_sentences("zebra"))

    def test_missing_pool_raises(self):
        with self.assertRaises(SystemExit):
            paths.set_sentence_pool("no/such/folder")


class Spans(unittest.TestCase):
    def test_find_subword_span(self):
        self.assertEqual(data.find_subword_span([1, 5, 6, 7, 2], [6, 7]), (2, 4))
        self.assertIsNone(data.find_subword_span([1, 5, 2], [6]))
        self.assertIsNone(data.find_subword_span([1], []))


if __name__ == "__main__":
    unittest.main()
