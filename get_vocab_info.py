"""
Standalone script to compute vocabulary information from a corpus.
This avoids needing to create a placeholder model just to get vocab size.
"""

import re
from collections import Counter
from pathlib import Path


def simple_tokenize(text: str):
    """
    Very basic tokenizer:
    - lowercase
    - keep only alphabetic characters and spaces
    - split on whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    return text.split()


def load_corpus(corpus_path: Path):
    """
    Load corpus as list of token lists (one per line).
    """
    sentences = []
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tokens = simple_tokenize(line)
            if tokens:
                sentences.append(tokens)
    print(f"Loaded {len(sentences)} sentences.")
    return sentences


def build_vocab_info(sentences, min_count=5):
    """
    Build vocabulary from list of token lists.
    Returns word2idx dict, idx2word list, and vocab_size.
    """
    counts = Counter()
    for sent in sentences:
        counts.update(sent)

    # filter by min_count
    filtered = [w for w, c in counts.items() if c >= min_count]

    idx2word = sorted(filtered)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    vocab_size = len(idx2word)
    
    return word2idx, idx2word, vocab_size


if __name__ == "__main__":
    # Example usage
    corpus_path = Path("data/AllCombinedText.txt")
    min_count = 5
    
    print("Loading corpus...")
    sentences = load_corpus(corpus_path)
    
    print("Building vocabulary...")
    word2idx, idx2word, vocab_size = build_vocab_info(sentences, min_count=min_count)
    
    print(f"Vocab size: {vocab_size}")
    print(f"First 10 words: {idx2word[:10]}")
    print(f"Last 10 words: {idx2word[-10:]}")

