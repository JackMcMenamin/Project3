import unittest


def _tokenizer():
    try:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        return None


TOK = _tokenizer()


@unittest.skipIf(TOK is None, "gpt2 tokenizer not in the local cache")
class GPT2Spans(unittest.TestCase):
    """Byte-level BPE puts the leading space inside the token, so a word has
    different ids mid-sentence, at the start, and capitalised."""

    def span_text(self, sentence, word):
        from gpt2_rsr_continued_pretraining import find_word_span
        ids = TOK(sentence)["input_ids"]
        span = find_word_span(TOK, ids, word)
        return None if span is None else TOK.decode(ids[span[0]:span[1]]).strip()

    def test_mid_sentence(self):
        self.assertEqual(self.span_text("The aardvark dug a hole.", "aardvark"), "aardvark")

    def test_sentence_initial_capitalised(self):
        self.assertEqual(self.span_text("Aardvark burrows are deep.", "aardvark"), "Aardvark")

    def test_absent_word(self):
        self.assertIsNone(self.span_text("Nothing to see here.", "aardvark"))


if __name__ == "__main__":
    unittest.main()
